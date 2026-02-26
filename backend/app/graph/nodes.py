from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from string import punctuation
from typing import Any, Callable

from app.artifacts import save_briefing_pdf_artifact, save_internal_doc_pdf_artifact
from app.config import Settings
from app.data.locale_maps import (
    canonicalize_company,
    canonicalize_language,
    extract_company_from_prompt,
    extract_language_from_prompt,
)
from app.exceptions import LLMResponseValidationError, LLMRetryExhausted
from app.internal_db.repository import InternalDBRepository
from app.llm.retry import llm_call_with_retry, llm_stream_with_retry
from app.llm.together_client import TogetherClient
from app.schemas import IntentParseOutput, PlannerOutput, PlannerStep, StepEvent, ToolCallRecord
from app.telemetry.langfuse_client import LangfuseTelemetry
from app.tools.real_tools import RealToolbox

from .state import AgentState


# Graph node implementation for the LangGraph runtime.
# Responsibilities:
# - normalize planner/composer outputs from the LLM into strict schemas
# - orchestrate tool execution for each task route
# - maintain progress/telemetry snapshots used by Chainlit and testing UIs
# Boundaries:
# - routing/state concerns only; transport concerns stay in api/runtime modules

# Supported task labels and normalized tool vocabulary used by planner validation.
TASK_TYPES = ("briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat")
ALLOWED_TOOLS = {
    "get_company_info",
    "search_public_web",
    "retrieve_internal_pdf",
    "translate_document",
    "generate_document",
    "security_filter",
}


# ---------------------------------------------------------------------------
# Shared parsing/retry helpers used by multiple nodes.
# ---------------------------------------------------------------------------
def _hash(data: Any) -> str:
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()[:12]


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        return stripped[start : end + 1]
    return stripped


def _extract_json_dict_candidates(text: str) -> list[dict[str, Any]]:
    stripped = _extract_json(text)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        key = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        candidates.append(obj)

    try:
        add_candidate(json.loads(stripped))
    except Exception:
        pass

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(stripped):
        start = stripped.find("{", cursor)
        if start == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        add_candidate(obj)
        cursor = start + max(consumed, 1)

    return candidates


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return cleaned.replace("\u2028", " ").replace("\u2029", " ")


def _validate_nonempty_text(value: str, message: str) -> None:
    if not str(value).strip():
        raise LLMResponseValidationError(message)


def _append_attempts(state: AgentState, attempts: list[Any]) -> None:
    state["llm_attempt_count"] = state.get("llm_attempt_count", 0) + len(attempts)
    status_codes = state.get("provider_status_codes", [])
    errors = state.get("provider_error_chain", [])
    for attempt in attempts:
        status_codes.append(attempt.status_code)
        if attempt.error:
            errors.append(attempt.error)
    state["provider_status_codes"] = status_codes
    state["provider_error_chain"] = errors


def _handle_retry_exhausted(state: AgentState, exc: LLMRetryExhausted) -> AgentState:
    _append_attempts(state, exc.attempts)
    state["llm_retry_exhausted"] = bool(exc.retry_exhausted)
    state["llm_last_error"] = str(exc)
    state["status"] = "failed"
    state["error_message"] = str(exc)
    state["policy_findings"] = state.get("policy_findings", []) + [str(exc)]
    return state


def _guess_company(prompt: str, repository: InternalDBRepository) -> tuple[str, str]:
    alias_match, alias_source = extract_company_from_prompt(prompt)
    if alias_match:
        return alias_match, alias_source

    lower = prompt.lower()
    db_companies: tuple[str, ...]
    list_names = getattr(repository, "list_company_names", None)
    if callable(list_names):
        try:
            db_companies = tuple(str(item) for item in list_names())
        except Exception:
            db_companies = ()
    else:
        db_companies = ()
    for candidate in db_companies:
        if not candidate:
            continue
        if re.search(rf"\b{re.escape(candidate.lower())}\b", lower):
            return candidate, "heuristic"

    known_candidates = (
        "Tencent",
        "Volkswagen",
        "TikTok",
        "Tesla",
        "Siemens",
        "Pfizer",
        "Samsung",
        "Shell",
        "Sony",
        "Grab",
        "OpenAI",
    )
    for candidate in known_candidates:
        if re.search(rf"\b{re.escape(candidate.lower())}\b", lower):
            return candidate, "heuristic"

    # Match common phrasing: "briefing for X", "info about X", "from X"
    m = re.search(
        r"(?:for|about|of|on)\s+([A-Za-z0-9][A-Za-z0-9&.,' \-]{1,80})",
        prompt,
        flags=re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip()
        value = raw.split(" from ")[0].split(" in ")[0].strip()
        value = value.strip(punctuation + " ")
        if value:
            canonical = canonicalize_company(value) or value
            for candidate in db_companies:
                if candidate.lower() == canonical.lower():
                    return candidate, "heuristic"
            return canonical, "heuristic"

    return "Unknown Company", "heuristic"


# ---------------------------------------------------------------------------
# Lightweight intent/language/doc-type heuristics used before LLM planning.
# ---------------------------------------------------------------------------
def _infer_task_type(prompt: str) -> str:
    lower = prompt.lower()
    doc_request = any(
        token in lower
        for token in (
            "proposal",
            "quotation",
            "quote",
            "pdf",
            "document",
            "proposal document",
            "quotation document",
            "文件",
            "文档",
            "文檔",
            "提案",
            "建议书",
            "建議書",
            "报价",
            "報價",
            "angebot",
            "vorschlag",
        )
    )
    briefing_request = any(
        token in lower
        for token in (
            "briefing",
            "brief note",
            "consultant brief",
            "简报",
            "簡報",
            "简讯",
            "briefing文件",
        )
    )
    internal_context = (
        "internal" in lower
        or "database" in lower
        or "internal db" in lower
        or "internal database" in lower
        or "from db" in lower
        or "from database" in lower
        or bool(re.search(r"\bdb\b", lower))
        or "内部" in prompt
        or "数据库" in prompt
    )
    has_translation_intent = (
        "translate" in lower
        or "translation" in lower
        or "translated" in lower
        or "翻译" in prompt
        or "翻譯" in prompt
        or "译成" in prompt
        or "譯成" in prompt
        or "to english" in lower
        or "to german" in lower
        or "to chinese" in lower
        or "to japanese" in lower
        or "into english" in lower
        or "into german" in lower
        or "into chinese" in lower
        or "into japanese" in lower
    )

    if doc_request and has_translation_intent:
        return "translate_only"
    if doc_request and internal_context:
        return "doc_only"
    if (
        "internal database" in lower
        or "from internal db" in lower
        or "from database" in lower
        or "internal record" in lower
        or ("internal" in lower and ("record" in lower or "internal info" in lower or "internal information" in lower))
    ) and ("web" not in lower):
        return "db_only"
    if (
        "from the web" in lower
        or "web only" in lower
        or "from web" in lower
        or ("web" in lower and "internal" not in lower)
        or "互联网" in prompt
        or "网上" in prompt
    ):
        return "web_only"
    if briefing_request:
        return "briefing_full"
    return "general_chat"


def _infer_language(prompt: str) -> str:
    language, _ = _infer_language_with_source(prompt)
    return language


def _infer_language_with_source(prompt: str) -> tuple[str, str]:
    alias_language, source = extract_language_from_prompt(prompt)
    if alias_language:
        return alias_language, source

    lower = prompt.lower()
    for language in ("English", "German", "French", "Spanish", "Chinese", "Japanese"):
        if language.lower() in lower:
            return language, "heuristic"
    if (
        "translate" in lower
        or "translation" in lower
        or "translated" in lower
        or "翻译" in prompt
        or "翻譯" in prompt
    ):
        return "English", "heuristic"
    return "English", "heuristic"


def _infer_internal_doc_type(prompt: str) -> str:
    lower = prompt.lower()
    if any(
        token in lower
        for token in (
            "quotation",
            "quote",
            "pricing",
            "price quote",
            "costing",
            "报价",
            "報價",
            "angebot",
            "offerte",
        )
    ):
        return "quotation"
    return "proposal"


# Stateful node implementations wired by runtime.py into the compiled graph.
class AgentNodes:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: TogetherClient,
        telemetry: LangfuseTelemetry,
        toolbox: RealToolbox,
        repository: InternalDBRepository,
        progress_callback_getter: Callable[[str], Callable[[dict[str, Any]], None] | None] | None = None,
        token_callback_getter: Callable[[str], Callable[[str], None] | None] | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.telemetry = telemetry
        self.toolbox = toolbox
        self.repository = repository
        self.progress_callback_getter = progress_callback_getter
        self.token_callback_getter = token_callback_getter

    # Planner recovery helpers tolerate malformed wrapper payloads emitted by
    # open-source models and recover a best-effort valid plan when possible.
    def _coerce_planner_candidate(
        self,
        candidate: dict[str, Any],
        state: AgentState,
    ) -> PlannerOutput | None:
        company_name = str(state.get("company_name") or "Unknown Company")
        target_language = str(state.get("target_language") or "English")
        internal_doc_type = str(state.get("internal_doc_type") or self.settings.internal_pdf_doc_type_default)

        candidate_task_type = str(candidate.get("task_type") or "").strip()
        if candidate_task_type not in TASK_TYPES:
            candidate_task_type = str(state.get("task_type") or "").strip()
        if candidate_task_type not in TASK_TYPES:
            candidate_task_type = None

        recovered_steps: list[PlannerStep] = []
        if isinstance(candidate.get("tool_name"), str):
            step = self._coerce_step(
                tool_name=str(candidate["tool_name"]),
                raw_args=candidate.get("args"),
                company_name=company_name,
                target_language=target_language,
                internal_doc_type=internal_doc_type,
            )
            if step is not None:
                recovered_steps.append(step)

        for raw_key, raw_value in candidate.items():
            if not isinstance(raw_key, str):
                continue
            tool_name = self._extract_tool_name(raw_key)
            if tool_name is None:
                continue
            step = self._coerce_step(
                tool_name=tool_name,
                raw_args=raw_value,
                company_name=company_name,
                target_language=target_language,
                internal_doc_type=internal_doc_type,
            )
            if step is not None:
                recovered_steps.append(step)

        if not recovered_steps:
            if candidate_task_type == "general_chat":
                return self._planner_output_from_steps(
                    steps=[],
                    company_name=company_name,
                    target_language=target_language,
                    task_type="general_chat",
                )
            if self._looks_like_tool_wrapper_payload(candidate):
                return self._recover_plan_from_task_hint(state)
            return None

        return self._planner_output_from_steps(
            steps=recovered_steps,
            company_name=company_name,
            target_language=target_language,
            task_type=candidate_task_type,
        )

    def _coerce_planner_from_text(self, text: str, state: AgentState) -> PlannerOutput | None:
        company_name = str(state.get("company_name") or "Unknown Company")
        target_language = str(state.get("target_language") or "English")
        internal_doc_type = str(state.get("internal_doc_type") or self.settings.internal_pdf_doc_type_default)
        task_type = str(state.get("task_type") or "")
        if task_type not in TASK_TYPES:
            task_type = ""
        lower = text.lower()

        discovered_tools: list[str] = []
        for tool_name in ALLOWED_TOOLS:
            if tool_name in lower:
                discovered_tools.append(tool_name)
        for match in re.findall(r"to=([a-z_]+)", lower):
            if match in ALLOWED_TOOLS and match not in discovered_tools:
                discovered_tools.append(match)

        if not discovered_tools:
            if task_type == "general_chat":
                return self._planner_output_from_steps(
                    steps=[],
                    company_name=company_name,
                    target_language=target_language,
                    task_type="general_chat",
                )
            if "commentary to=" in lower or "analysis to=" in lower or "repo_browser" in lower:
                return self._recover_plan_from_task_hint(state)
            return None

        steps: list[PlannerStep] = []
        for tool_name in discovered_tools:
            step = self._coerce_step(
                tool_name=tool_name,
                raw_args=None,
                company_name=company_name,
                target_language=target_language,
                internal_doc_type=internal_doc_type,
            )
            if step is not None:
                steps.append(step)

        if not steps:
            return None
        return self._planner_output_from_steps(
            steps=steps,
            company_name=company_name,
            target_language=target_language,
            task_type=task_type or None,
        )

    @staticmethod
    def _extract_tool_name(raw_key: str) -> str | None:
        lowered = raw_key.strip().lower()
        if lowered in ALLOWED_TOOLS:
            return lowered
        match = re.search(r"to=([a-z_]+)", lowered)
        if not match:
            return None
        candidate = match.group(1).strip()
        if candidate in ALLOWED_TOOLS:
            return candidate
        return None

    @staticmethod
    def _normalize_text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            lines = [line.strip(" -•\t") for line in value.splitlines() if line.strip()]
            if lines:
                return lines
            compact = value.strip()
            return [compact] if compact else []
        return []

    def _coerce_composer_candidate(
        self,
        candidate: dict[str, Any],
        *,
        default_internal_summary: str,
    ) -> dict[str, Any] | None:
        public_findings = self._normalize_text_list(candidate.get("public_findings"))
        risk_notes = self._normalize_text_list(candidate.get("risk_notes"))
        sources = self._normalize_text_list(candidate.get("sources"))

        executive_summary = str(candidate.get("executive_summary") or candidate.get("value") or "").strip()
        if not executive_summary:
            for key, value in candidate.items():
                if isinstance(key, str) and "executive_summary" in key.lower():
                    text = str(value).strip()
                    if text:
                        executive_summary = text
                        break

        internal_summary = str(candidate.get("internal_summary") or "").strip()
        if not internal_summary:
            internal_summary = default_internal_summary or "Internal details are confidential."

        recovery_applied = False
        if not executive_summary:
            if public_findings:
                executive_summary = public_findings[0]
            elif risk_notes:
                executive_summary = risk_notes[0]
            elif internal_summary:
                executive_summary = internal_summary
            if executive_summary:
                recovery_applied = True

        if not executive_summary and not (public_findings or risk_notes or sources or internal_summary):
            return None
        if not executive_summary:
            executive_summary = "Summary generated from available evidence with limited detail."
            recovery_applied = True

        if not public_findings:
            public_findings = [
                "No explicit public product/partnership details were provided by the composer output."
            ]
            recovery_applied = True
        if not risk_notes:
            risk_notes = ["No additional risk notes were generated by the composer output."]
            recovery_applied = True
        if not sources:
            sources = ["internal_db", "public_web"]
            recovery_applied = True

        return {
            "executive_summary": executive_summary,
            "public_findings": public_findings,
            "internal_summary": internal_summary,
            "risk_notes": risk_notes,
            "sources": sources,
            "composer_recovery_applied": recovery_applied,
        }

    def _coerce_step(
        self,
        *,
        tool_name: str,
        raw_args: Any,
        company_name: str,
        target_language: str,
        internal_doc_type: str,
    ) -> PlannerStep | None:
        if tool_name not in ALLOWED_TOOLS:
            return None

        args: dict[str, Any]
        if isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            args = {}

        if tool_name in {"get_company_info", "search_public_web"}:
            args["company_name"] = str(args.get("company_name") or company_name)
        elif tool_name == "retrieve_internal_pdf":
            args["company_name"] = str(args.get("company_name") or company_name)
            args["doc_type"] = str(args.get("doc_type") or internal_doc_type)
        elif tool_name == "translate_document":
            args["target_language"] = str(args.get("target_language") or target_language)
        elif tool_name == "generate_document":
            args["template_name"] = str(args.get("template_name") or "consulting_brief.md.j2")

        return PlannerStep(
            tool_name=tool_name,
            args=args,
            rationale_short="Recovered planner step from model tool-style output.",
        )

    def _planner_output_from_steps(
        self,
        *,
        steps: list[PlannerStep],
        company_name: str,
        target_language: str,
        task_type: str | None = None,
    ) -> PlannerOutput | None:
        deduped: list[PlannerStep] = []
        seen: set[str] = set()
        for step in steps:
            key = f"{step.tool_name}:{json.dumps(step.args, sort_keys=True, default=str)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(step)

        payload = {
            "company_name": company_name,
            "target_language": target_language,
            "task_type": task_type if task_type in TASK_TYPES else None,
            "steps": [step.model_dump() for step in deduped],
        }
        try:
            return PlannerOutput.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _looks_like_tool_wrapper_payload(candidate: dict[str, Any]) -> bool:
        for key in candidate.keys():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if any(
                marker in lowered
                for marker in (
                    "commentary to=",
                    "analysis to=",
                    "repo_browser",
                    "print_tree",
                    "tool_call",
                )
            ):
                return True
        return False

    def _recover_plan_from_task_hint(self, state: AgentState) -> PlannerOutput | None:
        task_type = str(state.get("task_type") or "briefing_full")
        company_name = str(state.get("company_name") or "Unknown Company")
        target_language = str(state.get("target_language") or "English")
        internal_doc_type = str(state.get("internal_doc_type") or self.settings.internal_pdf_doc_type_default)

        tool_sequences: dict[str, list[str]] = {
            "briefing_full": ["get_company_info", "search_public_web"],
            "web_only": ["search_public_web"],
            "db_only": ["get_company_info"],
            "doc_only": ["retrieve_internal_pdf"],
            "translate_only": ["retrieve_internal_pdf", "translate_document"],
            "general_chat": [],
        }
        sequence = tool_sequences.get(task_type, tool_sequences["briefing_full"])
        steps: list[PlannerStep] = []
        for tool_name in sequence:
            step = self._coerce_step(
                tool_name=tool_name,
                raw_args=None,
                company_name=company_name,
                target_language=target_language,
                internal_doc_type=internal_doc_type,
            )
            if step is not None:
                step.rationale_short = "Recovered planner step from parse-intent task hint."
                steps.append(step)

        if not steps:
            return None
        return self._planner_output_from_steps(
            steps=steps,
            company_name=company_name,
            target_language=target_language,
            task_type=task_type if task_type in TASK_TYPES else None,
        )

    def _snapshot(self, state: AgentState) -> dict[str, Any]:
        return {
            "run_id": state.get("run_id"),
            "trace_id": state.get("trace_id"),
            "task_type": state.get("task_type"),
            "status": state.get("status", "running"),
            "step_events": [step.model_dump() for step in state.get("step_events", [])],
            "tool_call_records": [record.model_dump() for record in state.get("tool_call_records", [])],
            "policy_findings": list(state.get("policy_findings", [])),
            "llm_tokens": list(state.get("llm_tokens", [])),
        }

    # Progress/token callbacks feed live UI views; they must be best-effort and
    # never break node execution when a callback consumer disappears.
    def _emit_progress(self, state: AgentState) -> None:
        if self.progress_callback_getter is None:
            return
        run_id = state.get("run_id")
        if not run_id:
            return
        callback = self.progress_callback_getter(run_id)
        if callback is None:
            return
        try:
            callback(self._snapshot(state))
        except Exception:
            # Progress callbacks are non-critical and should never fail the graph.
            return

    def _emit_token(self, state: AgentState, token: str) -> None:
        state.setdefault("llm_tokens", []).append(token)
        if self.token_callback_getter is not None:
            run_id = state.get("run_id")
            if run_id:
                callback = self.token_callback_getter(run_id)
                if callback is not None:
                    try:
                        callback(token)
                    except Exception:
                        pass
        self._emit_progress(state)

    def _step_start(self, state: AgentState, step_name: str) -> StepEvent:
        event = StepEvent(step_name=step_name, status="running", started_at=_utc_now())
        state.setdefault("step_events", []).append(event)
        self._emit_progress(state)
        return event

    def _step_end(self, state: AgentState, event: StepEvent, status: str, message: str = "") -> None:
        event.status = status  # type: ignore[assignment]
        event.message = message
        event.ended_at = _utc_now()
        self._emit_progress(state)

    def _preview_payload(self, payload: Any, *, max_items: int = 5, max_chars: int = 1200) -> Any:
        if isinstance(payload, dict):
            preview: dict[str, Any] = {}
            for idx, (key, value) in enumerate(payload.items()):
                if idx >= max_items:
                    preview["_truncated_keys"] = len(payload) - max_items
                    break
                preview[key] = self._preview_payload(value, max_items=max_items, max_chars=max_chars)
            return preview
        if isinstance(payload, list):
            if len(payload) <= max_items:
                return [self._preview_payload(v, max_items=max_items, max_chars=max_chars) for v in payload]
            return [
                *[self._preview_payload(v, max_items=max_items, max_chars=max_chars) for v in payload[:max_items]],
                f"... ({len(payload) - max_items} more items)",
            ]
        text = str(payload)
        if len(text) > max_chars:
            return f"{text[:max_chars]}... ({len(text) - max_chars} chars truncated)"
        return text

    def _record_tool_call(
        self,
        *,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
        result_data: dict[str, Any],
        duration_ms: float,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        trace_id = state["trace_id"]
        input_preview = self._preview_payload(args, max_items=8, max_chars=500)
        output_preview = self._preview_payload(result_data, max_items=8, max_chars=800)
        details = {
            "input_payload": input_preview,
            "output_payload_preview": output_preview,
            "metadata": metadata or {},
        }
        state["tool_call_records"].append(
            ToolCallRecord(
                tool_name=tool_name,
                args_digest=_hash(args),
                duration_ms=duration_ms,
                status="success" if status == "success" else "failed",
                output_hash=_hash(result_data),
                details=details,
            )
        )
        span_id = self.telemetry.start_span(
            trace_id,
            name=f"tool:{tool_name}",
            metadata={
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "status": status,
            },
            input_payload=input_preview,
            observation_type="tool",
        )
        self.telemetry.end_span(
            span_id,
            status="ok" if status == "success" else "error",
            metadata={
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "status": status,
                **(metadata or {}),
            },
            output_payload=output_preview,
        )
        self._emit_progress(state)

    # Evidence pack is the single normalized source for factual evaluation.
    # Keep shape changes deliberate because testing and judge paths depend on it.
    def _build_evidence_pack(self, state: AgentState) -> dict[str, Any]:
        tool_results = state.get("tool_results", {})
        db_data = tool_results.get("get_company_info", {})
        web_data = tool_results.get("search_public_web", {})
        pdf_data = tool_results.get("retrieve_internal_pdf", {})

        db_payload = db_data if isinstance(db_data, dict) else {}
        web_payload = web_data if isinstance(web_data, dict) else {}
        pdf_payload = pdf_data if isinstance(pdf_data, dict) else {}

        web_results: list[dict[str, str]] = []
        raw_results = web_payload.get("results", [])
        if isinstance(raw_results, list):
            for item in raw_results[:8]:
                if not isinstance(item, dict):
                    continue
                web_results.append(
                    {
                        "title": str(item.get("title", "")),
                        "snippet": str(item.get("snippet", "")),
                        "url": str(item.get("url", "")),
                    }
                )

        source_evidence: list[str] = []
        internal_summary = str(db_payload.get("internal_summary", "")).strip()
        if internal_summary:
            source_evidence.append(f"internal_db: {internal_summary}")
        for item in web_results:
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()
            url = item.get("url", "").strip()
            bits = [v for v in (title, snippet, url) if v]
            if bits:
                source_evidence.append("web: " + " | ".join(bits))
        pdf_text = str(pdf_payload.get("sanitized_text", "")).strip()
        if pdf_text:
            source_evidence.append(f"internal_pdf: {pdf_text[:1200]}")

        pack = {
            "company_name": state.get("company_name"),
            "task_type": state.get("task_type"),
            "target_language": state.get("target_language"),
            "internal_db": {
                "record_found": bool(db_payload.get("record_found", False)),
                "industry": str(db_payload.get("industry", "")),
                "public_products": db_payload.get("public_products", []),
                "public_partnerships": db_payload.get("public_partnerships", []),
                "project_risk_level": str(db_payload.get("project_risk_level", "")),
                "internal_summary": internal_summary,
            },
            "web": {
                "search_success": bool(web_payload.get("search_success", False)),
                "answer_preview": str(web_payload.get("answer_preview", "")),
                "query_attempts": web_payload.get("query_attempts", []),
                "results": web_results,
                "source_links": web_payload.get("source_links", []),
                "public_products_candidates": web_payload.get("public_products_candidates", []),
                "public_partnership_candidates": web_payload.get("public_partnership_candidates", []),
            },
            "internal_pdf": {
                "document_found": bool(pdf_payload.get("document_found", False)),
                "doc_type": str(pdf_payload.get("doc_type", "")),
                "language": str(pdf_payload.get("language", "")),
                "file_name": str(pdf_payload.get("file_name", "")),
                "classification": str(pdf_payload.get("classification", "")),
                "sanitized_text": pdf_text,
                "sanitized_text_excerpt": pdf_text[:1200] if pdf_text else "",
                "policy_note": str(pdf_payload.get("policy_note", "")),
            },
            "source_evidence": source_evidence,
        }
        state["source_evidence"] = source_evidence
        state["evidence_pack"] = pack
        return pack

    def _match_company_from_db(self, company_name: str) -> str:
        canonical = canonicalize_company(company_name) or company_name
        if not canonical:
            return "Unknown Company"
        for candidate in self.repository.list_company_names():
            if candidate.lower() == canonical.lower():
                return candidate
        return canonical

    def _looks_non_ascii(self, text: str) -> bool:
        return any(ord(ch) > 127 for ch in text)

    def _llm_parse_intent(
        self,
        *,
        state: AgentState,
        inferred_task_type: str,
        inferred_company: str,
        inferred_language: str,
    ) -> IntentParseOutput | None:
        trace_id = state["trace_id"]
        prompt = state["user_prompt"]
        parser_prompt = f"""
=== ROLE ===
You are an intent parser for a multilingual consultant assistant.

=== GOAL ===
Classify the request into a supported task type and extract company + target language.
If the request does not match supported workflows, choose "general_chat".

=== USER_INPUT ===
{prompt}

=== CURRENT_HEURISTICS ===
task_type_hint: {inferred_task_type}
company_name_hint: {inferred_company}
target_language_hint: {inferred_language}

=== TASK_DEFINITIONS ===
briefing_full: needs both internal company context and public web context, then generate a structured briefing.
web_only: retrieve and summarize public web information only.
db_only: retrieve and summarize internal DB relationship record only.
doc_only: retrieve internal PDF document (sanitized) without translation.
translate_only: retrieve internal PDF document and translate to requested target language.
general_chat: general assistant chat for requests outside the above workflows.

=== OUTPUT_SCHEMA ===
Return ONE strict JSON object only:
{{
  "company_name": "string|null",
  "target_language": "string|null",
  "task_type": "briefing_full|web_only|db_only|doc_only|translate_only|general_chat|null",
  "confidence": 0.0
}}

=== RULES ===
- Output RAW JSON only. No markdown, no prose, no tool wrappers.
- Valid target_language values: English, German, Chinese, Japanese, French, Spanish.
- If uncertain about company_name, return null.
- If uncertain about target_language, return null.
""".strip()

        parsed: IntentParseOutput | None = None

        def validate(text: str) -> None:
            nonlocal parsed
            last_error: Exception | None = None
            for candidate in _extract_json_dict_candidates(text):
                try:
                    payload = IntentParseOutput.model_validate(candidate)
                    parsed = payload
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            reason = str(last_error) if last_error else "No JSON object candidate found"
            raise LLMResponseValidationError(f"Intent parser schema invalid: {reason}")

        result = llm_call_with_retry(
            provider=self.provider,
            telemetry=self.telemetry,
            trace_id=trace_id,
            settings=self.settings,
            node_name="parse_intent_llm_node",
            prompt=parser_prompt,
            max_tokens=min(500, self.settings.planner_max_tokens),
            validate_text=validate,
            reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
            model_override=state.get("model_id"),
            response_format={"type": "json_object"},
        )
        _append_attempts(state, result.attempts)
        return parsed

    # -----------------------------------------------------------------------
    # Graph nodes (ordered roughly by the happy-path runtime route).
    # -----------------------------------------------------------------------
    def parse_intent_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "parse_intent_node", observation_type="chain")
        event = self._step_start(state, "parse_intent")

        if not state.get("reasoning_effort"):
            state["reasoning_effort"] = self.settings.agent_reasoning_effort
        state["tool_results"] = {}
        state["source_evidence"] = []
        state["evidence_pack"] = {}
        state["tool_call_records"] = []
        state["policy_events"] = []
        state["policy_findings"] = []
        state["llm_tokens"] = []
        state["artifacts"] = []
        state["provider_status_codes"] = []
        state["provider_error_chain"] = []
        state["llm_attempt_count"] = 0
        state["llm_retry_exhausted"] = False
        state["llm_last_error"] = None
        state["artifact_document_text"] = ""
        state["status"] = "success"

        prompt = state["user_prompt"]
        requested_task_type = state.get("requested_task_type")
        inferred_task_type = _infer_task_type(prompt)
        task_type = requested_task_type or inferred_task_type

        company_name, company_source = _guess_company(prompt, self.repository)
        target_language, language_source = _infer_language_with_source(prompt)
        intent_resolution_source = "alias" if company_source == "alias" else "heuristic"
        language_resolution_source = "alias" if language_source == "alias" else "heuristic"

        should_use_intent_llm = (
            company_name == "Unknown Company"
            or (language_source != "alias" and self._looks_non_ascii(prompt))
        )
        if should_use_intent_llm:
            try:
                llm_intent = self._llm_parse_intent(
                    state=state,
                    inferred_task_type=inferred_task_type,
                    inferred_company=company_name,
                    inferred_language=target_language,
                )
            except LLMRetryExhausted as exc:
                _append_attempts(state, exc.attempts)
                state["policy_findings"] = state.get("policy_findings", []) + [
                    f"Intent parser fallback failed after retries: {exc}",
                ]
            else:
                if llm_intent is not None:
                    llm_company = self._match_company_from_db(str(llm_intent.company_name or "").strip())
                    if llm_company and llm_company != "Unknown Company" and company_name == "Unknown Company":
                        company_name = llm_company
                        intent_resolution_source = "llm" if company_source == "heuristic" else "hybrid"

                    llm_language = canonicalize_language(str(llm_intent.target_language or "").strip())
                    if llm_language and language_source != "alias":
                        if llm_language != target_language:
                            language_resolution_source = "llm" if language_source == "heuristic" else "hybrid"
                        target_language = llm_language

                    llm_task = str(llm_intent.task_type or "").strip()
                    if (
                        requested_task_type is None
                        and llm_task in TASK_TYPES
                        and inferred_task_type in {"briefing_full", "general_chat"}
                    ):
                        task_type = llm_task

        internal_doc_type = _infer_internal_doc_type(prompt)
        target_language = canonicalize_language(target_language) or target_language or "English"

        state["task_type"] = task_type
        state["requested_task_type"] = requested_task_type
        state["company_name"] = company_name
        state["company_source"] = "prompt_extract"
        state["intent_resolution_source"] = intent_resolution_source  # type: ignore[assignment]
        state["language_resolution_source"] = language_resolution_source  # type: ignore[assignment]
        state["language_fallback_applied"] = False
        state["resolved_target_language"] = target_language
        state["internal_doc_type"] = "quotation" if internal_doc_type == "quotation" else "proposal"  # type: ignore[assignment]
        state["target_language"] = target_language
        state["source_language"] = None
        state["translation_applied"] = False
        state["output_mode"] = "document" if task_type == "briefing_full" else "chat"

        self.telemetry.append_timeline(trace_id, "parse_intent_node")
        self.telemetry.end_span(
            span_id,
            metadata={
                "task_type": task_type,
                "requested_task_type": requested_task_type,
                "inferred_task_type": inferred_task_type,
                "company_name": company_name,
                "company_source": state.get("company_source"),
                "intent_resolution_source": state.get("intent_resolution_source"),
                "language_resolution_source": state.get("language_resolution_source"),
                "target_language": target_language,
                "internal_doc_type": state.get("internal_doc_type"),
            },
        )
        self._step_end(
            state,
            event,
            "completed",
            f"task={task_type}, company={company_name}, language={target_language}",
        )
        return state

    def plan_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "plan_node", observation_type="chain")
        event = self._step_start(state, "plan")

        planner_prompt = f"""
=== ROLE ===
You are a planning engine for a consultant-safe assistant.

=== GOAL ===
Produce a safe, minimal, and executable plan.
If the request is outside supported workflows, return task_type "general_chat" with an empty step list.

=== USER_INPUT ===
{state['user_prompt']}

=== CONTEXT ===
task_type_hint: {state['task_type']}
task_type_override: {state.get('requested_task_type')}
company_name_hint: {state['company_name']}
target_language_hint: {state['target_language']}

=== TOOL_GLOSSARY ===
get_company_info:
- Reads internal company relationship data from internal DB.
- Must not expose confidential fields directly in final output.

search_public_web:
- Retrieves public web evidence for company facts.
- Use this for web-only/public fact requests.

retrieve_internal_pdf:
- Retrieves internal proposal/quotation document text (sanitized copy).
- Use this for internal document retrieval tasks.

translate_document:
- Translates document/text into target language.
- Use only when translation is requested or required by target language.

generate_document:
- Generates template-based long-form briefing output.
- Use for briefing/report-style deliverables.

security_filter:
- Final policy filter to redact restricted details and enforce safe output.

=== WORKFLOW_TASK_TYPES ===
briefing_full, web_only, db_only, doc_only, translate_only, general_chat

=== OUTPUT_SCHEMA ===
Return ONE strict JSON object only:
{{
  "company_name": "string",
  "target_language": "string",
  "task_type": "briefing_full|web_only|db_only|doc_only|translate_only|general_chat|null",
  "steps": [
    {{"tool_name":"...","args":{{}},"rationale_short":"..."}}
  ]
}}

=== RULES ===
- Output RAW JSON only. No markdown, no extra prose.
- Never include role tags or wrapper prefixes (analysis/commentary/to=).
- Keep steps minimal and ordered for execution.
- For general_chat, set "steps": [].
""".strip()

        parsed_plan: PlannerOutput | None = None

        def validate(text: str) -> None:
            nonlocal parsed_plan
            last_error: Exception | None = None
            for candidate in _extract_json_dict_candidates(text):
                try:
                    parsed_plan = PlannerOutput.model_validate(candidate)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    recovered = self._coerce_planner_candidate(candidate, state)
                    if recovered is not None:
                        parsed_plan = recovered
                        return
            recovered_from_text = self._coerce_planner_from_text(text, state)
            if recovered_from_text is not None:
                parsed_plan = recovered_from_text
                return
            reason = str(last_error) if last_error else "No JSON object candidate found"
            raise LLMResponseValidationError(f"Planner schema invalid: {reason}")

        try:
            result = llm_call_with_retry(
                provider=self.provider,
                telemetry=self.telemetry,
                trace_id=trace_id,
                settings=self.settings,
                node_name="plan_node",
                prompt=planner_prompt,
                max_tokens=self.settings.planner_max_tokens,
                validate_text=validate,
                reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
                model_override=state.get("model_id"),
                response_format={"type": "json_object"},
            )
            _append_attempts(state, result.attempts)
        except LLMRetryExhausted as exc:
            self.telemetry.end_span(span_id, status="error", metadata={"error": str(exc)})
            self._step_end(state, event, "failed", "Planner failed after retries")
            return _handle_retry_exhausted(state, exc)

        if parsed_plan is None:
            state["status"] = "failed"
            state["error_message"] = "Planner returned empty/invalid output after validation."
            state["llm_last_error"] = state["error_message"]
            state["policy_findings"] = state.get("policy_findings", []) + [state["error_message"]]
            self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
            self._step_end(state, event, "failed", "planner output invalid")
            return state

        planner_company = parsed_plan.company_name.strip()
        if planner_company:
            current_company = str(state.get("company_name", "")).strip()
            if (not current_company) or current_company == "Unknown Company":
                state["company_name"] = planner_company
                state["company_source"] = "planner"

        planner_task_type = str(parsed_plan.task_type or "").strip()
        if (
            state.get("requested_task_type") is None
            and planner_task_type in TASK_TYPES
            and state.get("task_type") in {"briefing_full", "general_chat"}
        ):
            state["task_type"] = planner_task_type

        state["plan_steps"] = parsed_plan.steps
        self.telemetry.append_timeline(trace_id, "plan_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", f"steps={len(state.get('plan_steps', []))}")
        return state

    def validate_plan_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "validate_plan_node", observation_type="chain")
        event = self._step_start(state, "validate_plan")

        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error", metadata={"error": state.get("error_message")})
            self._step_end(state, event, "failed", "state failed")
            return state

        valid_steps: list[PlannerStep] = []
        for step in state.get("plan_steps", []):
            if step.tool_name not in ALLOWED_TOOLS:
                state["policy_findings"] = state.get("policy_findings", []) + [
                    f"Disallowed tool removed from plan: {step.tool_name}"
                ]
                continue
            valid_steps.append(step)
        if not valid_steps:
            task_hint = str(state.get("task_type") or "")
            if task_hint == "general_chat":
                state["task_type"] = "general_chat"
                state["output_mode"] = "chat"
                state["plan_steps"] = []
                self.telemetry.append_timeline(trace_id, "validate_plan_node")
                self.telemetry.end_span(
                    span_id,
                    metadata={
                        "valid_steps": [],
                        "resolved_task_type": "general_chat",
                    },
                )
                self._step_end(state, event, "completed", "general_chat route accepted with zero tool steps")
                return state

            state["status"] = "failed"
            state["error_message"] = "Planner produced no allowed tool steps."
            state["llm_last_error"] = state["error_message"]
            state["policy_findings"] = state.get("policy_findings", []) + [state["error_message"]]
            self.telemetry.append_timeline(trace_id, "validate_plan_node")
            self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
            self._step_end(state, event, "failed", "no allowed tools in plan")
            return state

        resolved_task_type = self._resolve_task_type_from_plan(
            steps=valid_steps,
            requested_task_type=state.get("requested_task_type"),
        )
        if resolved_task_type is None:
            inferred_task_type = str(state.get("task_type") or "")
            if inferred_task_type in TASK_TYPES:
                resolved_task_type = inferred_task_type
                state["policy_findings"] = state.get("policy_findings", []) + [
                    (
                        "Planner tool set was ambiguous; routing fell back to parse-intent "
                        f"task hint '{inferred_task_type}'."
                    )
                ]
            else:
                state["status"] = "failed"
                state["error_message"] = "Planner tool set is ambiguous and cannot be routed safely."
                state["llm_last_error"] = state["error_message"]
                state["policy_findings"] = state.get("policy_findings", []) + [state["error_message"]]
                self.telemetry.append_timeline(trace_id, "validate_plan_node")
                self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
                self._step_end(state, event, "failed", "ambiguous plan tool set")
                return state

        intent_hint = str(state.get("task_type") or "")
        if (
            state.get("requested_task_type") is None
            and intent_hint in {"web_only", "db_only", "doc_only", "translate_only", "general_chat"}
            and resolved_task_type != intent_hint
        ):
            state["policy_findings"] = state.get("policy_findings", []) + [
                (
                    "Planner task route "
                    f"'{resolved_task_type}' overridden by parse-intent hint '{intent_hint}'."
                )
            ]
            resolved_task_type = intent_hint

        state["task_type"] = resolved_task_type
        state["output_mode"] = "document" if resolved_task_type == "briefing_full" else "chat"
        state["plan_steps"] = valid_steps
        self.telemetry.append_timeline(trace_id, "validate_plan_node")
        self.telemetry.end_span(
            span_id,
            metadata={
                "valid_steps": [s.tool_name for s in valid_steps],
                "resolved_task_type": resolved_task_type,
            },
        )
        self._step_end(
            state,
            event,
            "completed",
            f"valid_steps={len(valid_steps)}, task={resolved_task_type}",
        )
        return state

    # Retrieval fan-out node used by briefing flow; db/web can run concurrently.
    def retrieve_parallel_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "retrieve_parallel_node", observation_type="chain")
        event = self._step_start(state, "retrieve_parallel")

        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state

        company = state["company_name"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_db = pool.submit(self.toolbox.get_company_info, company)
            future_web = pool.submit(self.toolbox.search_public_web, company)
            db_result = future_db.result()
            web_result = future_web.result()

        state["tool_results"]["get_company_info"] = db_result.data
        state["tool_results"]["search_public_web"] = web_result.data
        self._record_tool_call(
            state=state,
            tool_name="get_company_info",
            args={"company_name": company},
            result_data=db_result.data,
            duration_ms=db_result.duration_ms,
            metadata={"source": "internal_db"},
        )
        self._record_tool_call(
            state=state,
            tool_name="search_public_web",
            args={"company_name": company},
            result_data=web_result.data,
            duration_ms=web_result.duration_ms,
            metadata={
                "source": "tavily",
                "query_attempts": web_result.data.get("query_attempts", []),
                "result_count": web_result.data.get("result_count", 0),
                "dedupe_count": web_result.data.get("dedupe_count", 0),
                "answer_preview": web_result.data.get("answer_preview", ""),
            },
        )
        if not bool(web_result.data.get("search_success", False)):
            state["status"] = "failed"
            state["error_message"] = str(
                web_result.data.get("error", "Public web retrieval failed after deterministic retries.")
            )
            state["policy_findings"] = state.get("policy_findings", []) + [state["error_message"]]
            self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
            self._step_end(state, event, "failed", state["error_message"])
            return state
        self.telemetry.append_timeline(trace_id, "retrieve_parallel_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "internal db + web retrieval completed")
        return state

    def retrieve_internal_db_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "retrieve_internal_db_node", observation_type="chain")
        event = self._step_start(state, "retrieve_internal_db")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state
        company = state["company_name"]
        result = self.toolbox.get_company_info(company)
        state["tool_results"]["get_company_info"] = result.data
        self._record_tool_call(
            state=state,
            tool_name="get_company_info",
            args={"company_name": company},
            result_data=result.data,
            duration_ms=result.duration_ms,
            metadata={"source": "internal_db"},
        )
        self.telemetry.append_timeline(trace_id, "retrieve_internal_db_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "internal db retrieval completed")
        return state

    def retrieve_public_web_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "retrieve_public_web_node", observation_type="chain")
        event = self._step_start(state, "retrieve_public_web")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state
        company = state["company_name"]
        result = self.toolbox.search_public_web(company)
        state["tool_results"]["search_public_web"] = result.data
        self._record_tool_call(
            state=state,
            tool_name="search_public_web",
            args={"company_name": company},
            result_data=result.data,
            duration_ms=result.duration_ms,
            metadata={
                "source": "tavily",
                "query_attempts": result.data.get("query_attempts", []),
                "result_count": result.data.get("result_count", 0),
                "dedupe_count": result.data.get("dedupe_count", 0),
                "answer_preview": result.data.get("answer_preview", ""),
            },
        )
        if not bool(result.data.get("search_success", False)):
            state["status"] = "failed"
            state["error_message"] = str(
                result.data.get("error", "Public web retrieval failed after deterministic retries.")
            )
            state["policy_findings"] = state.get("policy_findings", []) + [state["error_message"]]
            self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
            self._step_end(state, event, "failed", state["error_message"])
            return state
        self.telemetry.append_timeline(trace_id, "retrieve_public_web_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "public web retrieval completed")
        return state

    def retrieve_internal_pdf_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "retrieve_internal_pdf_node", observation_type="chain")
        event = self._step_start(state, "retrieve_internal_pdf")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state
        company = state["company_name"]
        doc_type = str(state.get("internal_doc_type") or self.settings.internal_pdf_doc_type_default)
        result = self.toolbox.retrieve_internal_pdf(company, doc_type)
        state["tool_results"]["retrieve_internal_pdf"] = result.data
        self._record_tool_call(
            state=state,
            tool_name="retrieve_internal_pdf",
            args={"company_name": company, "doc_type": doc_type},
            result_data=result.data,
            duration_ms=result.duration_ms,
            metadata={
                "source": "internal_db_pdf",
                "doc_type": doc_type,
                "source_language": str(result.data.get("language", "unknown")),
                "document_found": bool(result.data.get("document_found", False)),
                "translation_applied": False,
            },
        )
        self.telemetry.append_timeline(trace_id, "retrieve_internal_pdf_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "internal pdf retrieval completed")
        return state

    # Composer maps route-specific evidence into either chat text or a
    # structured briefing payload that can be persisted as PDF artifacts.
    def compose_template_document_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(
            trace_id,
            "compose_template_document_node",
            observation_type="chain",
        )
        event = self._step_start(state, "compose_template_document")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state

        task_type = state.get("task_type", "briefing_full")
        data_db = state.get("tool_results", {}).get("get_company_info", {})
        data_web = state.get("tool_results", {}).get("search_public_web", {})
        data_pdf = state.get("tool_results", {}).get("retrieve_internal_pdf", {})

        translated_doc = ""
        if task_type == "doc_only":
            payload = data_pdf if isinstance(data_pdf, dict) else {}
            source_language = str(payload.get("language", "unknown"))
            if source_language.strip().lower() in {"", "unknown"}:
                source_language = self._detect_language_from_text(str(payload.get("sanitized_text", "")))
            state["source_language"] = source_language
            state["translation_applied"] = False
            state["output_mode"] = "chat"

            doc_type = str(payload.get("doc_type") or state.get("internal_doc_type") or self.settings.internal_pdf_doc_type_default)
            sanitized_text = str(payload.get("sanitized_text", "")).strip()
            document_found = bool(payload.get("document_found", False))
            if document_found and sanitized_text:
                state["artifact_document_text"] = sanitized_text
                state["draft_document"] = (
                    f"Retrieved the internal {doc_type} document for {state['company_name']}. "
                    "A sanitized PDF copy is available via the download link."
                )
            elif document_found:
                state["artifact_document_text"] = ""
                state["draft_document"] = (
                    f"Found the internal {doc_type} document for {state['company_name']}, "
                    "but no extractable text was available. A sanitized export was not generated."
                )
            else:
                state["artifact_document_text"] = ""
                state["draft_document"] = (
                    f"No internal {doc_type} document was found for {state['company_name']}."
                )

            self.telemetry.append_timeline(trace_id, "compose_template_document_node")
            self.telemetry.end_span(span_id)
            self._step_end(state, event, "completed", "doc-only output composed")
            return state

        if task_type == "translate_only":
            payload = data_pdf if isinstance(data_pdf, dict) else {}
            source_language = str(payload.get("language", "unknown"))
            if source_language.strip().lower() in {"", "unknown"}:
                source_language = self._detect_language_from_text(str(payload.get("sanitized_text", "")))
            state["source_language"] = source_language
            target_language = str(state.get("target_language", "English"))
            if self._languages_match(source_language, target_language):
                translated_doc = str(payload.get("sanitized_text", "")).strip()
                state["translation_applied"] = False
                state["policy_findings"] = state.get("policy_findings", []) + [
                    f"Translation skipped because source language '{source_language}' matches target '{target_language}'."
                ]
            else:
                if str(payload.get("sanitized_text", "")).strip():
                    translated_doc = self._translate_sanitized_pdf(state, payload)
                    state["translated_document"] = translated_doc
                    state["artifact_document_text"] = translated_doc
                    state["translation_applied"] = True
                else:
                    translated_doc = "No extractable document text was available."
                    state["artifact_document_text"] = ""
                    state["translation_applied"] = False

            prompt_lower = str(state.get("user_prompt", "")).lower()
            wants_document = any(k in prompt_lower for k in ("brief", "template", "report"))
            if wants_document:
                state["output_mode"] = "document"
                generated = self.toolbox.generate_document(
                    "consulting_brief.md.j2",
                    {
                        "company_name": state["company_name"],
                        "executive_summary": "Translated internal document summary.",
                        "public_findings": [],
                        "record_found": False,
                        "project_risk_level": "unknown",
                        "internal_summary": translated_doc or "No extractable internal text found.",
                        "risk_notes": [],
                        "sources": ["internal_document"],
                        "labels": self._brief_labels(target_language),
                    },
                )
                state["draft_document"] = generated.data["document"]
                self._record_tool_call(
                    state=state,
                    tool_name="generate_document",
                    args={"template_name": "consulting_brief.md.j2"},
                    result_data=generated.data,
                    duration_ms=generated.duration_ms,
                    metadata={"artifact_kind": "translation_brief"},
                )
            else:
                state["output_mode"] = "chat"
                state["draft_document"] = translated_doc or "No extractable internal document text was available."

            self.telemetry.append_timeline(trace_id, "compose_template_document_node")
            self.telemetry.end_span(span_id)
            self._step_end(state, event, "completed", "translate-only output composed")
            return state

        if task_type in {"web_only", "db_only"}:
            state["output_mode"] = "chat"
            composer_prompt = self._build_chat_prompt(
                state=state,
                data_db=data_db if isinstance(data_db, dict) else {},
                data_web=data_web if isinstance(data_web, dict) else {},
            )
            try:
                result = llm_stream_with_retry(
                    provider=self.provider,
                    telemetry=self.telemetry,
                    trace_id=trace_id,
                    settings=self.settings,
                    node_name="compose_chat_response_node",
                    prompt=composer_prompt,
                    max_tokens=self.settings.composer_max_tokens,
                    validate_text=lambda text: _validate_nonempty_text(text, "Chat response empty"),
                    on_token=lambda token: self._emit_token(state, token),
                    reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
                    model_override=state.get("model_id"),
                )
                _append_attempts(state, result.attempts)
            except LLMRetryExhausted as exc:
                self.telemetry.end_span(span_id, status="error", metadata={"error": str(exc)})
                self._step_end(state, event, "failed", "chat composer failed")
                return _handle_retry_exhausted(state, exc)

            state["draft_document"] = _clean_text(result.text).strip()
            if task_type == "web_only":
                web_payload = data_web if isinstance(data_web, dict) else {}
                source_links: list[str] = []
                raw_links = web_payload.get("source_links")
                if isinstance(raw_links, list):
                    source_links = [str(item).strip() for item in raw_links if str(item).strip()]
                if not source_links:
                    raw_results = web_payload.get("results", [])
                    if isinstance(raw_results, list):
                        source_links = [
                            str(item.get("url", "")).strip()
                            for item in raw_results
                            if isinstance(item, dict) and str(item.get("url", "")).strip()
                        ]
                source_links = list(dict.fromkeys(source_links))[:6]
                if source_links and not re.search(r"https?://", state["draft_document"]):
                    source_block = "\n".join(f"- {url}" for url in source_links)
                    state["draft_document"] = (
                        f"{state['draft_document']}\n\nSources:\n{source_block}"
                    ).strip()
            self.telemetry.append_timeline(trace_id, "compose_template_document_node")
            self.telemetry.end_span(span_id)
            self._step_end(state, event, "completed", "chat output generated")
            return state

        if task_type == "general_chat":
            state["output_mode"] = "chat"
            composer_prompt = self._build_general_chat_prompt(state=state)
            try:
                result = llm_stream_with_retry(
                    provider=self.provider,
                    telemetry=self.telemetry,
                    trace_id=trace_id,
                    settings=self.settings,
                    node_name="compose_general_chat_node",
                    prompt=composer_prompt,
                    max_tokens=self.settings.composer_max_tokens,
                    validate_text=lambda text: _validate_nonempty_text(text, "General chat response empty"),
                    on_token=lambda token: self._emit_token(state, token),
                    reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
                    model_override=state.get("model_id"),
                )
                _append_attempts(state, result.attempts)
            except LLMRetryExhausted as exc:
                self.telemetry.end_span(span_id, status="error", metadata={"error": str(exc)})
                self._step_end(state, event, "failed", "general chat composer failed")
                return _handle_retry_exhausted(state, exc)

            state["draft_document"] = _clean_text(result.text).strip()
            self.telemetry.append_timeline(trace_id, "compose_template_document_node")
            self.telemetry.end_span(span_id)
            self._step_end(state, event, "completed", "general chat output generated")
            return state

        composer_prompt = self._build_composer_prompt(
            state=state,
            data_db=data_db if isinstance(data_db, dict) else {},
            data_web=data_web if isinstance(data_web, dict) else {},
            data_pdf=data_pdf if isinstance(data_pdf, dict) else {},
            translated_doc=translated_doc,
        )

        parsed: dict[str, Any] | None = None

        def validate(text: str) -> None:
            nonlocal parsed
            last_error: Exception | None = None
            default_internal_summary = (
                str(data_db.get("internal_summary", "")).strip()
                if isinstance(data_db, dict)
                else ""
            )
            for candidate in _extract_json_dict_candidates(text):
                try:
                    normalized = self._coerce_composer_candidate(
                        candidate,
                        default_internal_summary=default_internal_summary,
                    )
                    if normalized is None:
                        raise LLMResponseValidationError("Composer JSON missing recoverable section content")
                    parsed = normalized
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            reason = str(last_error) if last_error else "No JSON object candidate found"
            raise LLMResponseValidationError(f"Composer JSON invalid: {reason}")

        try:
            result = llm_call_with_retry(
                provider=self.provider,
                telemetry=self.telemetry,
                trace_id=trace_id,
                settings=self.settings,
                node_name="compose_template_document_node",
                prompt=composer_prompt,
                max_tokens=self.settings.composer_max_tokens,
                validate_text=validate,
                reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
                model_override=state.get("model_id"),
                response_format={"type": "json_object"},
            )
            _append_attempts(state, result.attempts)
        except LLMRetryExhausted as exc:
            self.telemetry.end_span(span_id, status="error", metadata={"error": str(exc)})
            self._step_end(state, event, "failed", "composer failed")
            return _handle_retry_exhausted(state, exc)

        if parsed is None:
            state["status"] = "failed"
            state["error_message"] = "Composer output parser failed"
            self.telemetry.end_span(span_id, status="error", metadata={"error": state["error_message"]})
            self._step_end(state, event, "failed", "composer parser failed")
            return state

        state["output_mode"] = "document"
        target_language = str(state.get("target_language") or "English")
        content = {
            "company_name": state["company_name"],
            "executive_summary": str(parsed.get("executive_summary", "")).strip(),
            "public_findings": parsed.get("public_findings", []),
            "record_found": bool(data_db.get("record_found", False)) if isinstance(data_db, dict) else False,
            "project_risk_level": str(data_db.get("project_risk_level", "unknown")) if isinstance(data_db, dict) else "unknown",
            "internal_summary": str(parsed.get("internal_summary", "Internal details are confidential.")),
            "risk_notes": parsed.get("risk_notes", []),
            "sources": parsed.get("sources", []),
            "labels": self._brief_labels(target_language),
        }
        web_payload = data_web if isinstance(data_web, dict) else {}
        source_links = web_payload.get("source_links", [])
        if isinstance(source_links, list):
            existing_sources = [str(item) for item in content.get("sources", [])]
            if source_links and not any(re.search(r"https?://", src) for src in existing_sources):
                content["sources"] = [*existing_sources, *[str(url) for url in source_links[:4] if str(url).strip()]]
        product_candidates = web_payload.get("public_products_candidates", [])
        partnership_candidates = web_payload.get("public_partnership_candidates", [])
        if isinstance(product_candidates, list) and isinstance(partnership_candidates, list):
            public_findings = [str(item) for item in content.get("public_findings", []) if str(item).strip()]
            if not any("product" in item.lower() for item in public_findings) and product_candidates:
                public_findings.append(f"Public products: {product_candidates[0]}")
            if not any("partnership" in item.lower() for item in public_findings) and partnership_candidates:
                public_findings.append(f"Public partnerships: {partnership_candidates[0]}")
            content["public_findings"] = public_findings
        state["composer_recovery_applied"] = bool(parsed.get("composer_recovery_applied", False))
        generated = self.toolbox.generate_document("consulting_brief.md.j2", content)
        state["draft_document"] = generated.data["document"]
        self._record_tool_call(
            state=state,
            tool_name="generate_document",
            args={"template_name": "consulting_brief.md.j2"},
            result_data=generated.data,
            duration_ms=generated.duration_ms,
            metadata={
                "artifact_kind": "briefing",
                "composer_recovery_applied": bool(state.get("composer_recovery_applied", False)),
            },
        )

        self.telemetry.append_timeline(trace_id, "compose_template_document_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "template document generated")
        return state

    def enforce_output_language_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "enforce_output_language_node", observation_type="chain")
        event = self._step_start(state, "enforce_output_language")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state

        target_language = canonicalize_language(str(state.get("target_language") or "English")) or "English"
        state["target_language"] = target_language
        state["resolved_target_language"] = target_language
        draft_text = str(state.get("draft_document") or "").strip()
        if not draft_text:
            self.telemetry.append_timeline(trace_id, "enforce_output_language_node")
            self.telemetry.end_span(
                span_id,
                metadata={
                    "target_language": target_language,
                    "language_validator_pass": True,
                    "language_fallback_applied": False,
                },
            )
            self._step_end(state, event, "completed", f"target={target_language}, skipped_empty")
            return state

        validator_pass = self._is_text_in_target_language(draft_text, target_language)
        if not validator_pass:
            try:
                translated = self._translate_text_to_target_language(
                    state=state,
                    text=draft_text,
                    target_language=target_language,
                    node_name="enforce_output_language_node",
                )
            except LLMRetryExhausted as exc:
                self.telemetry.end_span(span_id, status="error", metadata={"error": str(exc)})
                self._step_end(state, event, "failed", "language enforcement translation failed")
                return _handle_retry_exhausted(state, exc)
            state["draft_document"] = translated
            state["language_fallback_applied"] = True
            state["policy_findings"] = state.get("policy_findings", []) + [
                f"Applied language fallback to enforce {target_language} output.",
            ]
        else:
            state["language_fallback_applied"] = False

        self.telemetry.append_timeline(trace_id, "enforce_output_language_node")
        self.telemetry.end_span(
            span_id,
            metadata={
                "target_language": target_language,
                "language_validator_pass": validator_pass,
                "language_fallback_applied": bool(state.get("language_fallback_applied", False)),
            },
        )
        self._step_end(
            state,
            event,
            "completed",
            (
                f"target={target_language}, fallback_applied="
                f"{bool(state.get('language_fallback_applied', False))}"
            ),
        )
        return state

    # Security filter is always the final textual guard before persistence.
    def security_filter_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "security_filter_node", observation_type="guardrail")
        event = self._step_start(state, "security_filter")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state

        filter_result = self.toolbox.security_filter(
            state.get("draft_document", ""),
            company_name=state.get("company_name"),
        )
        filtered = filter_result.data
        state["final_document"] = str(filtered.get("document", ""))
        if state.get("artifact_document_text"):
            artifact_filter_result = self.toolbox.security_filter(
                str(state.get("artifact_document_text", "")),
                company_name=state.get("company_name"),
            )
            artifact_filtered = artifact_filter_result.data
            state["artifact_document_text"] = str(artifact_filtered.get("document", ""))
        state["security_report"] = {
            "pass_fail": bool(filtered.get("pass_fail", False)),
            "redactions_applied": int(filtered.get("redactions_applied", 0)),
            "leaked_terms": list(filtered.get("leaked_terms", [])),
        }
        if filtered.get("leaked_terms"):
            state["policy_findings"] = state.get("policy_findings", []) + [
                f"Confidential terms redacted: {', '.join(filtered['leaked_terms'])}"
            ]
        self._build_evidence_pack(state)
        self._record_tool_call(
            state=state,
            tool_name="security_filter",
            args={"company_name": state.get("company_name")},
            result_data=filtered,
            duration_ms=filter_result.duration_ms,
            metadata={"leaked_terms_count": len(filtered.get("leaked_terms", []))},
        )
        self.telemetry.append_timeline(trace_id, "security_filter_node")
        self.telemetry.end_span(span_id)
        self._step_end(state, event, "completed", "security filtering complete")
        return state

    def persist_artifacts_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "persist_artifacts_node", observation_type="chain")
        event = self._step_start(state, "persist_artifacts")
        if state.get("status") == "failed":
            self.telemetry.end_span(span_id, status="error")
            self._step_end(state, event, "failed", "state failed")
            return state

        run_id = state["run_id"]
        final_doc = state.get("final_document", "")
        artifact_doc_text = str(state.get("artifact_document_text", "")).strip()
        if not artifact_doc_text and str(state.get("task_type", "")) in {"doc_only", "translate_only"}:
            artifact_doc_text = str(final_doc).strip()
        task_type = str(state.get("task_type", "briefing_full"))
        output_mode = state.get("output_mode", "document")
        if task_type == "briefing_full" and output_mode == "document" and final_doc:
            artifact = save_briefing_pdf_artifact(
                root_dir=self.settings.artifacts_dir,
                run_id=run_id,
                company_name=str(state.get("company_name") or "Unknown Company"),
                text=final_doc,
                target_language=str(state.get("target_language") or "English"),
            )
            state.setdefault("artifacts", []).append(artifact)  # type: ignore[arg-type]
        elif task_type in {"doc_only", "translate_only"} and artifact_doc_text:
            pdf_payload = state.get("tool_results", {}).get("retrieve_internal_pdf", {})
            payload = pdf_payload if isinstance(pdf_payload, dict) else {}
            doc_type = str(state.get("internal_doc_type") or payload.get("doc_type") or self.settings.internal_pdf_doc_type_default)
            source_language = str(state.get("source_language") or payload.get("language") or "unknown")
            original_file_name = str(payload.get("file_name") or f"{str(state.get('company_name') or 'document').lower()}_{doc_type}.pdf")
            artifact = save_internal_doc_pdf_artifact(
                root_dir=self.settings.artifacts_dir,
                run_id=run_id,
                company_name=str(state.get("company_name") or "Unknown Company"),
                doc_type=doc_type,
                source_language=source_language,
                original_file_name=original_file_name,
                text=artifact_doc_text,
                classification="confidential",
            )
            state.setdefault("artifacts", []).append(artifact)  # type: ignore[arg-type]

        self.telemetry.append_timeline(trace_id, "persist_artifacts_node")
        renderers = [
            (item.get("metadata") or {}).get("renderer")
            for item in state.get("artifacts", [])
            if isinstance(item, dict)
        ]
        self.telemetry.end_span(
            span_id,
            metadata={"artifact_count": len(state.get("artifacts", [])), "artifact_renderers": [r for r in renderers if r]},
        )
        self._step_end(state, event, "completed", f"artifacts={len(state.get('artifacts', []))}")
        return state

    def finalize_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "finalize_node", observation_type="agent")
        event = self._step_start(state, "finalize")
        if state.get("status") != "failed":
            state["status"] = "success"
        self.telemetry.append_timeline(trace_id, "finalize_node")
        self.telemetry.end_span(span_id, metadata={"status": state.get("status")})
        self._step_end(state, event, "completed", state.get("status", "unknown"))
        return state

    def error_node(self, state: AgentState) -> AgentState:
        trace_id = state["trace_id"]
        span_id = self.telemetry.start_span(trace_id, "error_node", observation_type="agent")
        event = self._step_start(state, "error")
        state["status"] = "failed"
        if not state.get("error_message"):
            state["error_message"] = "Unhandled graph error"
        self.telemetry.append_timeline(trace_id, "error_node")
        self.telemetry.end_span(span_id, status="error", metadata={"error": state.get("error_message")})
        self._step_end(state, event, "failed", state.get("error_message", "error"))
        return state

    @staticmethod
    # -----------------------------------------------------------------------
    # Route selection helpers used by runtime graph edges.
    # -----------------------------------------------------------------------
    def _resolve_task_type_from_plan(
        *,
        steps: list[PlannerStep],
        requested_task_type: str | None,
    ) -> str | None:
        if requested_task_type in TASK_TYPES:
            return requested_task_type
        if not steps:
            return "general_chat"

        tools = {step.tool_name for step in steps}
        has_db = "get_company_info" in tools
        has_web = "search_public_web" in tools
        has_pdf = "retrieve_internal_pdf" in tools
        has_translate = "translate_document" in tools

        if (has_pdf or has_translate) and (has_db or has_web):
            return None
        if has_pdf and has_translate:
            return "translate_only"
        if has_pdf:
            return "doc_only"
        if has_translate:
            return None
        if has_db and has_web:
            return "briefing_full"
        if has_web and not has_db:
            return "web_only"
        if has_db and not has_web:
            return "db_only"
        return None

    def route_after_validation(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        task_type = state.get("task_type", "briefing_full")
        if task_type == "general_chat":
            return "compose_template_document"
        if task_type == "briefing_full":
            return "retrieve_parallel"
        if task_type == "web_only":
            return "retrieve_public_web"
        if task_type == "db_only":
            return "retrieve_internal_db"
        if task_type in {"doc_only", "translate_only"}:
            return "retrieve_internal_pdf"
        return "retrieve_parallel"

    def route_after_parallel(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "compose_template_document"

    def route_after_db(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        if state.get("task_type") == "briefing_full":
            return "retrieve_public_web"
        return "compose_template_document"

    def route_after_web(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "compose_template_document"

    def route_after_pdf(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "compose_template_document"

    def route_after_compose(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "enforce_output_language"

    def route_after_language(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "security_filter"

    def route_after_security(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "error"
        return "persist_artifacts"

    # -----------------------------------------------------------------------
    # Translation and prompt builder helpers.
    # -----------------------------------------------------------------------
    def _translate_sanitized_pdf(self, state: AgentState, pdf_data: dict[str, Any]) -> str:
        text = str(pdf_data.get("sanitized_text", "")).strip()
        if not text:
            return "No extractable document text was available."
        target_language = str(state.get("target_language", "English")).strip() or "English"
        return self._translate_text_to_target_language(
            state=state,
            text=text,
            target_language=target_language,
            node_name="translate_document_node",
        )

    def _translate_text_to_target_language(
        self,
        *,
        state: AgentState,
        text: str,
        target_language: str,
        node_name: str,
    ) -> str:
        prompt = f"""
=== ROLE ===
You are a professional translator.

=== GOAL ===
Translate the source content into {target_language}.

=== RULES ===
- Preserve markdown structure, headings, and bullet formatting.
- Preserve numbers, entities, and factual meaning exactly.
- Keep '[REDACTED]' exactly unchanged.
- Do not add, remove, or reinterpret content.
- Return plain text in {target_language} only.
- Do not include explanations.

=== SOURCE_TEXT ===
{text}
""".strip()

        def validate(value: str) -> None:
            if not value.strip():
                raise LLMResponseValidationError("Translation output empty")

        result = llm_call_with_retry(
            provider=self.provider,
            telemetry=self.telemetry,
            trace_id=state["trace_id"],
            settings=self.settings,
            node_name=node_name,
            prompt=prompt,
            max_tokens=self.settings.composer_max_tokens,
            validate_text=validate,
            reasoning_effort=state.get("reasoning_effort", self.settings.agent_reasoning_effort),
            model_override=state.get("model_id"),
        )
        _append_attempts(state, result.attempts)
        translated = _clean_text(result.text).strip()
        self._record_tool_call(
            state=state,
            tool_name="translate_document",
            args={"target_language": target_language, "input_chars": len(text)},
            result_data={"translated_text_preview": translated[:1200], "translated_chars": len(translated)},
            duration_ms=result.latency_ms,
            metadata={
                "llm_provider": result.provider,
                "llm_model": result.model,
                "translation_reason": node_name,
            },
        )
        return translated

    @staticmethod
    def _languages_match(source_language: str, target_language: str) -> bool:
        src = source_language.strip().lower()
        tgt = target_language.strip().lower()
        if not src or not tgt:
            return False
        aliases = {
            "english": {"english", "en", "eng"},
            "german": {"german", "de", "deutsch"},
            "chinese": {"chinese", "zh", "中文", "mandarin"},
            "japanese": {"japanese", "ja", "日本語"},
            "french": {"french", "fr", "francais", "français"},
            "spanish": {"spanish", "es", "espanol", "español"},
        }

        def normalize(value: str) -> str:
            for canonical, values in aliases.items():
                if value in values:
                    return canonical
            return value

        return normalize(src) == normalize(tgt)

    def _build_chat_prompt(
        self,
        *,
        state: AgentState,
        data_db: dict[str, Any],
        data_web: dict[str, Any],
    ) -> str:
        task_type = str(state.get("task_type") or "web_only")
        safe_payload = {
            "task_type": task_type,
            "company_name": state.get("company_name"),
            "internal_db": data_db,
            "public_web": data_web,
            "target_language": state.get("target_language"),
        }
        task_specific_rules = (
            "- For web_only: explicitly cover BOTH public products and public partnerships when available.\n"
            "- For web_only: include source links from DATA_PAYLOAD.public_web.source_links.\n"
            "- For db_only: summarize only sanitized internal relationship records."
        )
        return f"""
=== ROLE ===
You are a consultant-safe chat assistant.

=== GOAL ===
Answer the user request using only the grounded evidence in DATA_PAYLOAD.

=== TOOL_CONTEXT ===
- internal_db: sanitized relationship signal and risk context.
- public_web: public evidence snippets and URLs.

=== RULES ===
- Use only DATA_PAYLOAD evidence; do not fabricate missing facts.
- If evidence is missing or search failed, explicitly say unknown/unavailable.
- Never reveal confidential project names, exact budgets, or internal product names.
- Preserve '[REDACTED]' exactly when present.
- Follow TASK_SPECIFIC rules.
- Keep the answer concise and actionable.
- Output language must be {state.get("target_language", "English")}.

=== TASK_SPECIFIC ===
{task_specific_rules}

=== DATA_PAYLOAD ===
{json.dumps(safe_payload, ensure_ascii=False)}
""".strip()

    def _build_general_chat_prompt(self, *, state: AgentState) -> str:
        payload = {
            "user_request": state.get("user_prompt"),
            "target_language": state.get("target_language"),
            "known_company_hint": state.get("company_name"),
        }
        return f"""
=== ROLE ===
You are a consultant-safe general assistant.

=== GOAL ===
Provide a helpful response for a general request that may be outside predefined workflows.

=== SAFETY_RULES ===
- Do not fabricate internal database records, internal documents, or web evidence.
- Do not reveal confidential project names, exact budgets, or internal product names.
- If a request needs unavailable internal evidence, state limitations clearly.
- Keep '[REDACTED]' unchanged when present.
- Keep response concise and practical.

=== OUTPUT_REQUIREMENTS ===
- Respond in {state.get("target_language", "English")}.
- Plain text only.

=== INPUT_CONTEXT ===
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    @staticmethod
    def _detect_language_from_text(text: str) -> str:
        value = text.strip()
        if not value:
            return "unknown"
        if re.search(r"[\u3040-\u30ff]", value):
            return "Japanese"
        if re.search(r"[\u4e00-\u9fff]", value):
            return "Chinese"
        lower = value.lower()
        german_markers = (" und ", " der ", " die ", " das ", " nicht ", " für ", "über ", "vertraulich")
        if any(marker in lower for marker in german_markers):
            return "German"
        return "English"

    @staticmethod
    def _is_text_in_target_language(text: str, target_language: str) -> bool:
        target = (canonicalize_language(target_language) or target_language or "English").lower()
        normalized = text.strip()
        if not normalized:
            return True
        scrubbed = re.sub(r"\[REDACTED\]", " ", normalized, flags=re.IGNORECASE)
        scrubbed = re.sub(r"https?://\S+", " ", scrubbed)
        scrubbed = re.sub(r"[`*_>#-]", " ", scrubbed)

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", scrubbed))
        latin_count = len(re.findall(r"[A-Za-z]", scrubbed))
        lower = f" {scrubbed.lower()} "

        if target == "chinese":
            return cjk_count >= max(4, latin_count // 3)
        if target == "english":
            return cjk_count <= 2
        if target == "german":
            markers = (
                " der ",
                " die ",
                " das ",
                " und ",
                " für ",
                " nicht ",
                " zusammenfassung ",
                " quellen ",
                " vertraulich ",
                " risiko ",
            )
            marker_hits = sum(1 for marker in markers if marker in lower)
            has_umlaut = bool(re.search(r"[äöüß]", lower))
            return marker_hits >= 2 or has_umlaut
        if target == "japanese":
            hira_kata = len(re.findall(r"[\u3040-\u30ff]", scrubbed))
            return hira_kata >= 3
        if target == "french":
            markers = (" le ", " la ", " les ", " et ", " des ", " résumé ", " source ")
            return sum(1 for marker in markers if marker in lower) >= 2
        if target == "spanish":
            markers = (" el ", " la ", " los ", " las ", " y ", " de ", " resumen ", " fuentes ")
            return sum(1 for marker in markers if marker in lower) >= 2
        return True

    @staticmethod
    def _brief_labels(target_language: str) -> dict[str, str]:
        canonical = canonicalize_language(target_language) or "English"
        if canonical == "German":
            return {
                "title_prefix": "Beratungs-Briefing",
                "executive_summary": "Management-Zusammenfassung",
                "public_findings": "Öffentliche Erkenntnisse",
                "internal_relationship_signal": "Interne Beziehungssignale",
                "internal_record_found": "Interner Datensatz gefunden",
                "project_risk_level": "Projekt-Risikostufe",
                "internal_note_sanitized": "Interner Hinweis (bereinigt)",
                "confidential_fields": "Vertrauliche Felder (Berateransicht)",
                "risk_notes": "Risikohinweise",
                "sources": "Quellen",
                "redaction_notes": "Hinweise zu Schwärzung und Vertraulichkeit",
                "redaction_note_1": "Projektname, exaktes Budget, interner Produktname und vertrauliche Inhalte wurden geschwärzt.",
                "redaction_note_2": "Dieses Dokument ist beratersicher und enthält keine eingeschränkten Kundendetails.",
                "yes": "Ja",
                "no": "Nein",
                "project_name_label": "Projektname",
                "exact_budget_label": "Exaktes Budget",
                "internal_product_name_label": "Interner Produktname",
                "proposal_lines_label": "Vertrauliche Angebots-/Vorschlagszeilen",
            }
        if canonical == "Chinese":
            return {
                "title_prefix": "顾问简报",
                "executive_summary": "执行摘要",
                "public_findings": "公开信息发现",
                "internal_relationship_signal": "内部关系信号",
                "internal_record_found": "是否存在内部记录",
                "project_risk_level": "项目风险等级",
                "internal_note_sanitized": "内部说明（已脱敏）",
                "confidential_fields": "敏感字段（顾问视图）",
                "risk_notes": "风险说明",
                "sources": "信息来源",
                "redaction_notes": "脱敏与保密说明",
                "redaction_note_1": "项目名称、精确预算、内部产品名称及提案/报价敏感内容均已脱敏。",
                "redaction_note_2": "该文档为顾问安全版本，不包含受限的客户交付细节。",
                "yes": "是",
                "no": "否",
                "project_name_label": "项目名称",
                "exact_budget_label": "精确预算",
                "internal_product_name_label": "内部产品名称",
                "proposal_lines_label": "提案/报价敏感内容",
            }
        return {
            "title_prefix": "Consultant Briefing Note",
            "executive_summary": "Executive Summary",
            "public_findings": "Public Findings",
            "internal_relationship_signal": "Internal Relationship Signal",
            "internal_record_found": "Internal record found",
            "project_risk_level": "Project risk level",
            "internal_note_sanitized": "Internal note (sanitized)",
            "confidential_fields": "Confidential Fields (Consultant View)",
            "risk_notes": "Risk Notes",
            "sources": "Sources",
            "redaction_notes": "Redaction and Confidentiality Notes",
            "redaction_note_1": "Project name, exact budget, internal product name, and confidential proposal/quotation details are redacted.",
            "redaction_note_2": "This document is consultant-safe and excludes restricted internal client-delivery details.",
            "yes": "Yes",
            "no": "No",
            "project_name_label": "Project name",
            "exact_budget_label": "Exact budget",
            "internal_product_name_label": "Internal product name",
            "proposal_lines_label": "Proposal/quotation confidential lines",
        }

    def _build_composer_prompt(
        self,
        *,
        state: AgentState,
        data_db: dict[str, Any],
        data_web: dict[str, Any],
        data_pdf: dict[str, Any],
        translated_doc: str,
    ) -> str:
        safe_payload = {
            "task_type": state.get("task_type"),
            "company_name": state.get("company_name"),
            "target_language": state.get("target_language"),
            "internal_db": data_db,
            "public_web": data_web,
            "internal_pdf_summary": {
                "document_found": data_pdf.get("document_found"),
                "policy_note": data_pdf.get("policy_note"),
                "sanitized_excerpt": str(data_pdf.get("sanitized_text", ""))[:800],
            },
            "translated_document_excerpt": translated_doc[:800],
        }
        return f"""
=== ROLE ===
You are a consultant-safe briefing writer.

=== GOAL ===
Generate structured briefing content from grounded evidence only.

=== TOOL_CONTEXT ===
- get_company_info: internal DB relationship/risk signals (sanitized).
- search_public_web: public snippets and source URLs.
- retrieve_internal_pdf: sanitized internal document excerpt.
- translate_document: translated internal excerpt when applicable.

=== HARD_CONSTRAINTS ===
- Use only DATA_PAYLOAD evidence.
- Never reconstruct or infer confidential project names, exact budgets, or internal product names.
- Never include verbatim confidential excerpts.
- Preserve '[REDACTED]' exactly (do not translate it).
- Write all text values in {state.get("target_language", "English")} only.
- In public findings, explicitly include both products and partnerships when available.
- Include source links in "sources" from retrieved URLs when available.

=== OUTPUT_SCHEMA ===
Return ONE strict JSON object only:
{{
  "executive_summary": "string",
  "public_findings": ["string"],
  "internal_summary": "string",
  "risk_notes": ["string"],
  "sources": ["string"]
}}

=== DATA_PAYLOAD ===
{json.dumps(safe_payload, ensure_ascii=False)}
""".strip()
