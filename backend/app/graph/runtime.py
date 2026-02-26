from __future__ import annotations

import time
from contextlib import AbstractContextManager
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.internal_db.repository import InternalDBRepository
from app.llm.together_client import TogetherClient
from app.schemas import RunRequest, RunResponse, SecurityReport
from app.telemetry.langfuse_client import LangfuseTelemetry
from app.tools.real_tools import RealToolbox

from .nodes import AgentNodes
from .state import AgentState

# Runtime wrapper around compiled LangGraph.
# Responsibilities:
# - build and hold the compiled state graph
# - execute synchronous runs with optional progress/token callbacks
# - expose trace/run helpers used by API background jobs
# Boundaries:
# - node-level business logic lives in graph/nodes.py


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: TogetherClient,
        telemetry: LangfuseTelemetry,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.telemetry = telemetry
        self.repository = InternalDBRepository(settings.internal_db_path)
        self.toolbox = RealToolbox(settings=settings, repository=self.repository)
        self._progress_callback_lock = Lock()
        self._progress_callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._token_callback_lock = Lock()
        self._token_callbacks: dict[str, Callable[[str], None]] = {}
        self._checkpointer_ctx: AbstractContextManager[Any] | None = None
        self._checkpointer: Any = self._initialize_checkpointer()

        context_window = self.provider.discover_context_window()
        if context_window <= 0:
            raise RuntimeError("Together context window discovery returned invalid value")

        self.nodes = AgentNodes(
            settings=settings,
            provider=provider,
            telemetry=telemetry,
            toolbox=self.toolbox,
            repository=self.repository,
            progress_callback_getter=self._get_progress_callback,
            token_callback_getter=self._get_token_callback,
        )
        self.graph = self._compile_graph()

    # Per-run callback registry powers live token/progress updates in UIs.
    def _get_progress_callback(self, run_id: str) -> Callable[[dict[str, Any]], None] | None:
        with self._progress_callback_lock:
            return self._progress_callbacks.get(run_id)

    def _set_progress_callback(self, run_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._progress_callback_lock:
            self._progress_callbacks[run_id] = callback

    def _clear_progress_callback(self, run_id: str) -> None:
        with self._progress_callback_lock:
            self._progress_callbacks.pop(run_id, None)

    def _get_token_callback(self, run_id: str) -> Callable[[str], None] | None:
        with self._token_callback_lock:
            return self._token_callbacks.get(run_id)

    def _set_token_callback(self, run_id: str, callback: Callable[[str], None]) -> None:
        with self._token_callback_lock:
            self._token_callbacks[run_id] = callback

    def _clear_token_callback(self, run_id: str) -> None:
        with self._token_callback_lock:
            self._token_callbacks.pop(run_id, None)

    def _initialize_checkpointer(self) -> Any:
        if not self.settings.require_postgres_checkpointer:
            return None

        if not self.settings.langgraph_postgres_uri:
            raise RuntimeError(
                "LANGGRAPH_POSTGRES_URI is required when REQUIRE_POSTGRES_CHECKPOINTER=true"
            )

        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Postgres checkpointer required but langgraph checkpoint postgres package is missing"
            ) from exc

        ctx = PostgresSaver.from_conn_string(self.settings.langgraph_postgres_uri)
        saver = ctx.__enter__()
        saver.setup()
        self._checkpointer_ctx = ctx
        return saver

    # Graph topology definition and conditional routing wiring.
    def _compile_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("parse_intent", self.nodes.parse_intent_node)
        builder.add_node("plan", self.nodes.plan_node)
        builder.add_node("validate_plan", self.nodes.validate_plan_node)
        builder.add_node("retrieve_parallel", self.nodes.retrieve_parallel_node)
        builder.add_node("retrieve_internal_db", self.nodes.retrieve_internal_db_node)
        builder.add_node("retrieve_public_web", self.nodes.retrieve_public_web_node)
        builder.add_node("retrieve_internal_pdf", self.nodes.retrieve_internal_pdf_node)
        builder.add_node("compose_template_document", self.nodes.compose_template_document_node)
        builder.add_node("enforce_output_language", self.nodes.enforce_output_language_node)
        builder.add_node("security_filter", self.nodes.security_filter_node)
        builder.add_node("persist_artifacts", self.nodes.persist_artifacts_node)
        builder.add_node("finalize", self.nodes.finalize_node)
        builder.add_node("error", self.nodes.error_node)

        builder.add_edge(START, "parse_intent")
        builder.add_edge("parse_intent", "plan")
        builder.add_edge("plan", "validate_plan")
        builder.add_conditional_edges(
            "validate_plan",
            self.nodes.route_after_validation,
            {
                "compose_template_document": "compose_template_document",
                "retrieve_parallel": "retrieve_parallel",
                "retrieve_public_web": "retrieve_public_web",
                "retrieve_internal_db": "retrieve_internal_db",
                "retrieve_internal_pdf": "retrieve_internal_pdf",
                "error": "error",
            },
        )
        builder.add_conditional_edges(
            "retrieve_parallel",
            self.nodes.route_after_parallel,
            {"compose_template_document": "compose_template_document", "error": "error"},
        )
        builder.add_conditional_edges(
            "retrieve_internal_db",
            self.nodes.route_after_db,
            {
                "retrieve_public_web": "retrieve_public_web",
                "compose_template_document": "compose_template_document",
                "error": "error",
            },
        )
        builder.add_conditional_edges(
            "retrieve_public_web",
            self.nodes.route_after_web,
            {"compose_template_document": "compose_template_document", "error": "error"},
        )
        builder.add_conditional_edges(
            "retrieve_internal_pdf",
            self.nodes.route_after_pdf,
            {"compose_template_document": "compose_template_document", "error": "error"},
        )
        builder.add_conditional_edges(
            "compose_template_document",
            self.nodes.route_after_compose,
            {"enforce_output_language": "enforce_output_language", "error": "error"},
        )
        builder.add_conditional_edges(
            "enforce_output_language",
            self.nodes.route_after_language,
            {"security_filter": "security_filter", "error": "error"},
        )
        builder.add_conditional_edges(
            "security_filter",
            self.nodes.route_after_security,
            {"persist_artifacts": "persist_artifacts", "error": "error"},
        )
        builder.add_edge("persist_artifacts", "finalize")
        builder.add_edge("finalize", END)
        builder.add_edge("error", END)

        return builder.compile(checkpointer=self._checkpointer)

    def create_trace(self, run_id: str, request: RunRequest, *, trace_id: str | None = None) -> str:
        return self.telemetry.start_trace(
            run_id,
            trace_id=trace_id,
            session_id=request.session_id or run_id,
            user_id="agent-demo-user",
            input_payload={"prompt": request.prompt},
            metadata={
                "model": self.provider.model,
                "request_model_override": request.model_id,
                "reasoning_effort": request.reasoning_effort,
                "scenario_id": request.scenario_id,
            },
        )

    def create_trace_id(self) -> str:
        return self.telemetry.create_trace_id()

    def run(self, request: RunRequest) -> RunResponse:
        run_id = str(uuid4())
        trace_id = self.create_trace_id()
        return self.run_with_ids(request=request, run_id=run_id, trace_id=trace_id, start_trace=True)

    # Main execution entrypoint used by API sync and async run paths.
    def run_with_ids(
        self,
        *,
        request: RunRequest,
        run_id: str,
        trace_id: str,
        start_trace: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> RunResponse:
        started = time.perf_counter()
        if start_trace:
            self.create_trace(run_id, request, trace_id=trace_id)
        if progress_callback is not None:
            self._set_progress_callback(run_id, progress_callback)
        if token_callback is not None:
            self._set_token_callback(run_id, token_callback)

        try:
            initial: AgentState = {
                "run_id": run_id,
                "trace_id": trace_id,
                "user_prompt": request.prompt,
                "requested_task_type": request.task_type,
                "task_type": request.task_type or "briefing_full",
                "reasoning_effort": request.reasoning_effort,
                "model_id": request.model_id,
            }

            final_state = self.graph.invoke(initial, config={"configurable": {"thread_id": run_id}})

            if isinstance(final_state.get("security_report"), dict):
                security_report = SecurityReport.model_validate(final_state["security_report"])
            else:
                security_report = None

            response = RunResponse(
                run_id=run_id,
                status=final_state.get("status", "failed"),
                task_type=final_state.get("task_type", request.task_type or "briefing_full"),
                final_document=final_state.get("final_document"),
                evidence_pack=final_state.get("evidence_pack"),
                resolved_company_name=final_state.get("company_name"),
                company_source=final_state.get("company_source"),
                resolved_target_language=final_state.get("resolved_target_language"),
                intent_resolution_source=final_state.get("intent_resolution_source"),
                language_resolution_source=final_state.get("language_resolution_source"),
                language_fallback_applied=bool(final_state.get("language_fallback_applied", False)),
                output_mode=final_state.get("output_mode", "document"),
                source_language=final_state.get("source_language"),
                target_language=final_state.get("target_language"),
                translation_applied=bool(final_state.get("translation_applied", False)),
                security_report=security_report,
                tool_call_records=final_state.get("tool_call_records", []),
                artifacts=final_state.get("artifacts", []),
                step_events=final_state.get("step_events", []),
                policy_findings=final_state.get("policy_findings", []),
                llm_attempt_count=final_state.get("llm_attempt_count", 0),
                llm_retry_exhausted=final_state.get("llm_retry_exhausted", False),
                provider_error_chain=final_state.get("provider_error_chain", []),
                provider_status_codes=final_state.get("provider_status_codes", []),
                langfuse_trace_id=trace_id,
                run_duration_ms=(time.perf_counter() - started) * 1000,
            )
            self.telemetry.end_trace(
                trace_id,
                status="ok" if response.status == "success" else "error",
                metadata={"status": response.status, "run_id": run_id},
                output_payload={
                    "status": response.status,
                    "final_document": response.final_document,
                    "llm_attempt_count": response.llm_attempt_count,
                    "provider_status_codes": response.provider_status_codes,
                    "task_type": response.task_type,
                },
            )
            self.telemetry.flush()
            return response
        except Exception as exc:
            self.telemetry.end_trace(
                trace_id,
                status="error",
                metadata={"status": "failed", "run_id": run_id, "error": str(exc)},
                output_payload={"status": "failed", "error": str(exc)},
            )
            self.telemetry.flush()
            raise
        finally:
            if progress_callback is not None:
                self._clear_progress_callback(run_id)
            if token_callback is not None:
                self._clear_token_callback(run_id)

    def get_trace_payload(self, run_id: str, trace_id: str) -> dict[str, Any]:
        trace = self.telemetry.get_trace(trace_id)
        if trace is None:
            return {
                "run_id": run_id,
                "langfuse_trace_id": trace_id,
                "timeline_steps": [],
                "tool_spans": [],
                "policy_events": [],
            }

        return {
            "run_id": run_id,
            "langfuse_trace_id": trace_id,
            "timeline_steps": trace.timeline_steps,
            "tool_spans": trace.spans,
            "policy_events": trace.policy_events,
        }

    def close(self) -> None:
        if self._checkpointer_ctx is not None:
            self._checkpointer_ctx.__exit__(None, None, None)
            self._checkpointer_ctx = None
        self.provider.close()
