from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.exceptions import LLMResponseValidationError, LLMRetryExhausted
from app.llm.retry import llm_call_with_retry
from app.llm.together_client import TogetherClient
from app.telemetry.langfuse_client import LangfuseTelemetry

# LLM-based factual grounding judge.
# Responsibilities:
# - score candidate output (1..5) against provided tool evidence
# - return structured verdict/reason for accuracy and simulation pages
# - keep parser tolerant to minor wrapper/noise in model outputs
# Boundaries:
# - applicability gating is handled by runner/api before calling this module


@dataclass(frozen=True)
class FactJudgeResult:
    score_1_5: float | None
    verdict: str | None
    reason: str | None
    source: str = "llm_json"
    applicable: bool = True
    error: str | None = None
    raw: str | None = None


# Prompt templates intentionally mirror the structured style used by agent prompts.
FACT_JUDGE_SYSTEM_TEMPLATE = """
=== ROLE ===
You are a strict factual-grounding evaluator for a white-box agent test.

=== NON-NEGOTIABLE EVIDENCE POLICY ===
1) Use ONLY the provided evidence. Never use your own background knowledge.
2) Treat internal_db evidence and web evidence as equal-trust sources.
3) Treat internal_pdf evidence as part of the same provided evidence pool.
4) A claim is supported if it is grounded by any provided evidence source.
5) Penalize claims only when they are unsupported by provided evidence.
6) Output strict JSON only.
""".strip()


FACT_JUDGE_USER_TEMPLATE = """
=== SCENARIO PROMPT ===
{{scenario_prompt}}

=== EVIDENCE TRUST MODEL ===
internal_db and web_results are equal-trust evidence sources.
internal_pdf is also trusted evidence from the same provided pool.

=== INTERNAL_DB ===
{{internal_db_json}}

=== INTERNAL_PDF ===
{{internal_pdf_json}}

=== WEB_RESULTS ===
{{web_json}}

=== CANDIDATE OUTPUT ===
{{candidate_output}}

=== RUBRIC ===
1 = Major factual errors or mostly unsupported statements.
2 = Many unsupported claims; only limited evidence alignment.
3 = Mixed quality; some key claims supported, others unsupported/unclear.
4 = Mostly evidence-grounded; minor unsupported details at most.
5 = Fully grounded; claims are clearly supported by evidence.

=== DECISION PROTOCOL ===
1) Extract concrete factual claims from candidate output.
2) For each claim, mark: supported_by_internal_db | supported_by_web | supported_by_internal_pdf | unsupported.
3) If any supported_by_* flag is true, treat the claim as supported.
4) Score using the rubric.

=== DECISION RULE ===
pass if score_1_5 >= 4, otherwise fail.

=== OUTPUT JSON SCHEMA ===
{
  "score_1_5": <number between 1 and 5>,
  "verdict": "pass" | "fail",
  "reason": "<concise reason>"
}
""".strip()


def render_fact_judge_user_prompt(*, scenario_prompt: str, candidate_output: str, evidence_pack: dict[str, Any]) -> str:
    internal_db = evidence_pack.get("internal_db", {}) if isinstance(evidence_pack, dict) else {}
    internal_pdf = evidence_pack.get("internal_pdf", {}) if isinstance(evidence_pack, dict) else {}
    web = evidence_pack.get("web", {}) if isinstance(evidence_pack, dict) else {}

    return (
        FACT_JUDGE_USER_TEMPLATE
        .replace("{{scenario_prompt}}", scenario_prompt)
        .replace("{{candidate_output}}", candidate_output)
        .replace("{{internal_db_json}}", json.dumps(internal_db, ensure_ascii=False))
        .replace("{{internal_pdf_json}}", json.dumps(internal_pdf, ensure_ascii=False))
        .replace("{{web_json}}", json.dumps(web, ensure_ascii=False))
    )


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


def _parse_fact_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not text.strip():
        return None, "Fact judge output empty"
    candidate = _extract_json(text).strip()
    if not candidate:
        return None, "Fact judge output empty after JSON extraction"
    try:
        payload = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "Fact judge output is not a JSON object"
    if "score_1_5" not in payload or "verdict" not in payload:
        return None, "Fact judge output missing required keys"
    return payload, None


def _collect_evidence_texts(evidence_pack: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if not isinstance(evidence_pack, dict):
        return texts
    source_evidence = evidence_pack.get("source_evidence")
    if isinstance(source_evidence, list):
        for item in source_evidence:
            text = str(item).strip()
            if text:
                texts.append(text)

    internal_db = evidence_pack.get("internal_db")
    if isinstance(internal_db, dict):
        summary = str(internal_db.get("internal_summary", "")).strip()
        if summary:
            texts.append(summary)

    web = evidence_pack.get("web")
    if isinstance(web, dict):
        answer_preview = str(web.get("answer_preview", "")).strip()
        if answer_preview:
            texts.append(answer_preview)
        results = web.get("results")
        if isinstance(results, list):
            for result in results[:8]:
                if not isinstance(result, dict):
                    continue
                title = str(result.get("title", "")).strip()
                snippet = str(result.get("snippet", "")).strip()
                if title:
                    texts.append(title)
                if snippet:
                    texts.append(snippet)

    internal_pdf = evidence_pack.get("internal_pdf")
    if isinstance(internal_pdf, dict):
        pdf_text = str(internal_pdf.get("sanitized_text", "")).strip()
        if pdf_text:
            texts.append(pdf_text)
    return texts


def is_fact_check_applicable(*, task_type: str | None, evidence_pack: dict[str, Any] | None) -> bool:
    if (task_type or "").strip().lower() == "general_chat":
        return False
    if not isinstance(evidence_pack, dict):
        return False
    return bool(_collect_evidence_texts(evidence_pack))


def _normalize_fact_payload(parsed: dict[str, Any], *, raw: str | None, source: str, error: str | None = None) -> FactJudgeResult:
    try:
        score = float(parsed.get("score_1_5"))
    except (TypeError, ValueError):
        score = 1.0
    score = max(1.0, min(5.0, score))
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "fail"}:
        verdict = "pass" if score >= 4.0 else "fail"
    reason = str(parsed.get("reason", "")).strip() or "No reason provided."
    return FactJudgeResult(
        score_1_5=score,
        verdict=verdict,
        reason=reason,
        source=source,
        applicable=True,
        error=error,
        raw=raw,
    )


def _conservative_fact_default(error: str) -> FactJudgeResult:
    return FactJudgeResult(
        score_1_5=2.0,
        verdict="fail",
        reason="Fact judge unavailable; conservative fail fallback applied.",
        source="deterministic_fallback",
        applicable=True,
        error=error,
        raw=None,
    )


def build_not_applicable_fact_result() -> FactJudgeResult:
    return FactJudgeResult(
        score_1_5=None,
        verdict=None,
        reason=None,
        source="not_applicable",
        applicable=False,
        error=None,
        raw=None,
    )


def run_fact_judge(
    *,
    provider: TogetherClient,
    telemetry: LangfuseTelemetry,
    settings: Settings,
    trace_id: str,
    scenario_prompt: str,
    candidate_output: str,
    evidence_pack: dict[str, Any],
) -> FactJudgeResult:
    parsed: dict[str, Any] | None = None
    judge_user_prompt = render_fact_judge_user_prompt(
        scenario_prompt=scenario_prompt,
        candidate_output=candidate_output,
        evidence_pack=evidence_pack,
    )
    judge_prompt = (
        "=== FACT JUDGE REQUEST START ===\n"
        f"{judge_user_prompt}\n"
        "=== FACT JUDGE REQUEST END ==="
    )

    def validate(text: str) -> None:
        nonlocal parsed
        payload, parse_error = _parse_fact_payload(text)
        if payload is None:
            raise LLMResponseValidationError(f"Fact judge output invalid: {parse_error or 'unknown'}")
        parsed = payload

    try:
        result = llm_call_with_retry(
            provider=provider,
            telemetry=telemetry,
            trace_id=trace_id,
            settings=settings,
            node_name="fact_judge_node",
            prompt=f"[System]\n{FACT_JUDGE_SYSTEM_TEMPLATE}\n\n[User]\n{judge_prompt}",
            max_tokens=settings.judge_max_tokens,
            validate_text=validate,
            reasoning_effort=settings.llm_judge_reasoning_effort,
            model_override=settings.llm_judge_model,
            response_format={"type": "json_object"},
        )
    except (LLMRetryExhausted, Exception) as exc:  # noqa: BLE001
        return _conservative_fact_default(str(exc))

    if parsed is None:
        return _conservative_fact_default("Fact judge parser returned empty result")
    return _normalize_fact_payload(parsed, raw=result.text, source="llm_json")
