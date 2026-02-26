from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from app.schemas import TraceSpan

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional runtime dependency behavior
    Langfuse = None  # type: ignore[misc]

# Unified tracing adapter with in-memory fallback.
# Responsibilities:
# - emit spans/events to Langfuse when configured
# - keep minimal in-memory traces for local/demo resilience
# - provide trace snapshots consumed by API/UI diagnostics
# Boundaries:
# - evaluation scoring is handled outside this module


@dataclass
class _InMemoryTrace:
    trace_id: str
    run_id: str
    timeline_steps: list[str] = field(default_factory=list)
    policy_events: list[str] = field(default_factory=list)
    spans: list[TraceSpan] = field(default_factory=list)


class LangfuseTelemetry:
    _VALID_OBSERVATION_TYPES = {
        "span",
        "agent",
        "tool",
        "chain",
        "retriever",
        "evaluator",
        "guardrail",
    }

    def __init__(
        self,
        *,
        enabled: bool,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._enabled = enabled and Langfuse is not None and bool(host and public_key and secret_key)
        self._lock = Lock()
        self._traces: dict[str, _InMemoryTrace] = {}
        self._span_index: dict[str, tuple[str, int]] = {}

        self._langfuse = None
        self._use_legacy_api = False
        self._external_traces: dict[str, Any] = {}
        self._external_trace_contexts: dict[str, Any] = {}
        self._external_spans: dict[str, Any] = {}

        if self._enabled:
            self._langfuse = Langfuse(host=host, public_key=public_key, secret_key=secret_key)
            self._use_legacy_api = hasattr(self._langfuse, "trace")

    @property
    def client(self) -> Any | None:
        return self._langfuse

    def create_trace_id(self) -> str:
        if self._langfuse is not None and hasattr(self._langfuse, "create_trace_id"):
            try:
                return str(self._langfuse.create_trace_id())
            except Exception:
                pass
        return uuid4().hex

    def _normalize_observation_type(self, observation_type: str | None, *, fallback: str = "span") -> str:
        candidate = str(observation_type or fallback).strip().lower()
        if candidate == "generation":
            return "generation"
        if candidate in self._VALID_OBSERVATION_TYPES:
            return candidate
        return fallback

    def _resolve_external_parent(self, *, trace_id: str, parent_span_id: str | None) -> Any | None:
        if parent_span_id:
            parent = self._external_spans.get(parent_span_id)
            if parent is not None:
                return parent
        return self._external_traces.get(trace_id)

    def start_trace(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_payload: Any | None = None,
    ) -> str:
        resolved_trace_id = trace_id or uuid4().hex
        with self._lock:
            self._traces[resolved_trace_id] = _InMemoryTrace(trace_id=resolved_trace_id, run_id=run_id)

        if self._langfuse is not None:
            if self._use_legacy_api:
                ext_trace = self._langfuse.trace(
                    id=resolved_trace_id,
                    name="agent_run",
                    metadata=metadata or {},
                    input=input_payload,
                    session_id=session_id,
                    user_id=user_id,
                )
                self._external_traces[resolved_trace_id] = ext_trace
            else:
                if hasattr(self._langfuse, "start_as_current_observation"):
                    cm = self._langfuse.start_as_current_observation(
                        name="agent_run",
                        as_type="agent",
                        trace_context={"trace_id": resolved_trace_id},
                        input=input_payload,
                        metadata=metadata or {},
                        end_on_exit=False,
                    )
                else:
                    cm = self._langfuse.start_as_current_span(
                        name="agent_run",
                        trace_context={"trace_id": resolved_trace_id},
                        input=input_payload,
                        metadata=metadata or {},
                        end_on_exit=False,
                    )
                ext_root_span = cm.__enter__()
                self._langfuse.update_current_trace(
                    name="agent_run",
                    session_id=session_id,
                    user_id=user_id,
                    input=input_payload,
                    metadata=metadata or {},
                )
                self._external_traces[resolved_trace_id] = ext_root_span
                self._external_trace_contexts[resolved_trace_id] = cm

        return resolved_trace_id

    def append_timeline(self, trace_id: str, step: str) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                trace.timeline_steps.append(step)

    def append_policy_event(self, trace_id: str, event: str) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                trace.policy_events.append(event)

    def start_span(
        self,
        trace_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        *,
        input_payload: Any | None = None,
        observation_type: str = "span",
        parent_span_id: str | None = None,
    ) -> str:
        normalized_type = self._normalize_observation_type(observation_type)
        local_metadata = dict(metadata or {})
        local_metadata.setdefault("observation_type", normalized_type)
        if parent_span_id:
            local_metadata["parent_span_id"] = parent_span_id

        span_id = str(uuid4())
        span = TraceSpan(
            span_id=span_id,
            name=name,
            start_time=datetime.now(timezone.utc),
            metadata=local_metadata,
        )

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                raise KeyError(f"Trace '{trace_id}' not found")
            trace.spans.append(span)
            self._span_index[span_id] = (trace_id, len(trace.spans) - 1)

        if self._langfuse is not None:
            parent_observation = self._resolve_external_parent(
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
            ext_span = None
            if parent_observation is not None:
                if self._use_legacy_api:
                    if hasattr(parent_observation, "span"):
                        ext_span = parent_observation.span(
                            id=span_id,
                            name=name,
                            metadata=local_metadata,
                            input=input_payload,
                        )
                else:
                    if hasattr(parent_observation, "start_observation"):
                        ext_span = parent_observation.start_observation(
                            name=name,
                            as_type=normalized_type,
                            metadata=local_metadata,
                            input=input_payload,
                        )
                    elif hasattr(parent_observation, "start_span"):
                        ext_span = parent_observation.start_span(
                            name=name,
                            metadata=local_metadata,
                            input=input_payload,
                        )
            if ext_span is not None:
                self._external_spans[span_id] = ext_span

        return span_id

    def start_generation(
        self,
        *,
        trace_id: str,
        name: str,
        input_payload: Any,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> str:
        local_metadata = dict(metadata or {})
        local_metadata.setdefault("observation_type", "generation")
        if parent_span_id:
            local_metadata["parent_span_id"] = parent_span_id

        generation_id = str(uuid4())
        local_span = TraceSpan(
            span_id=generation_id,
            name=name,
            start_time=datetime.now(timezone.utc),
            metadata={
                **local_metadata,
                "input_preview": str(input_payload)[:500],
            },
        )
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                raise KeyError(f"Trace '{trace_id}' not found")
            trace.spans.append(local_span)
            self._span_index[generation_id] = (trace_id, len(trace.spans) - 1)

        if self._langfuse is None:
            return generation_id

        parent_observation = self._resolve_external_parent(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        if parent_observation is None:
            return generation_id

        if self._use_legacy_api:
            if hasattr(parent_observation, "generation"):
                ext_gen = parent_observation.generation(
                    id=generation_id,
                    name=name,
                    input=input_payload,
                    metadata=local_metadata,
                    model=model,
                    model_parameters=model_parameters or {},
                )
                self._external_spans[generation_id] = ext_gen
            return generation_id

        ext_gen = None
        if hasattr(parent_observation, "start_observation"):
            ext_gen = parent_observation.start_observation(
                name=name,
                as_type="generation",
                input=input_payload,
                metadata=local_metadata,
                model=model,
                model_parameters=model_parameters or {},
            )
        elif hasattr(parent_observation, "start_generation"):
            ext_gen = parent_observation.start_generation(
                name=name,
                input=input_payload,
                metadata=local_metadata,
                model=model,
                model_parameters=model_parameters or {},
            )

        if ext_gen is None:
            return generation_id

        self._external_spans[generation_id] = ext_gen
        return generation_id

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
        output_payload: Any | None = None,
    ) -> None:
        with self._lock:
            trace_id, span_index = self._span_index[span_id]
            trace = self._traces[trace_id]
            span = trace.spans[span_index]
            span.end_time = datetime.now(timezone.utc)
            if status in {"ok", "error"}:
                span.status = status
            if metadata:
                span.metadata.update(metadata)
            if output_payload is not None:
                span.metadata["output_preview"] = str(output_payload)[:500]

        ext_span = self._external_spans.get(span_id)
        if ext_span is None:
            return

        if self._use_legacy_api:
            ext_span.end(
                level="ERROR" if status == "error" else "DEFAULT",
                output=output_payload if output_payload is not None else (metadata or {}),
            )
            return

        ext_span.update(
            level="ERROR" if status == "error" else "DEFAULT",
            metadata=metadata or {},
            output=output_payload,
        )
        ext_span.end()

    def end_generation(
        self,
        generation_id: str,
        *,
        status: str = "ok",
        output_payload: Any | None = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
    ) -> None:
        ext_generation = self._external_spans.get(generation_id)
        if ext_generation is not None and not self._use_legacy_api:
            ext_generation.update(
                usage_details=usage_details or {},
                metadata=metadata or {},
                output=output_payload,
                level="ERROR" if status == "error" else "DEFAULT",
            )
            ext_generation.end()
        elif ext_generation is not None and self._use_legacy_api:
            ext_generation.end(
                level="ERROR" if status == "error" else "DEFAULT",
                output=output_payload if output_payload is not None else (metadata or {}),
            )

        with self._lock:
            trace_id, span_index = self._span_index[generation_id]
            trace = self._traces[trace_id]
            span = trace.spans[span_index]
            span.end_time = datetime.now(timezone.utc)
            if status in {"ok", "error"}:
                span.status = status
            if metadata:
                span.metadata.update(metadata)
            if usage_details:
                span.metadata.update(
                    {
                        "usage_input_tokens": usage_details.get("input", 0),
                        "usage_output_tokens": usage_details.get("output", 0),
                    }
                )
            if output_payload is not None:
                span.metadata["output_preview"] = str(output_payload)[:500]

        if ext_generation is None:
            return

    def flush(self) -> None:
        if self._langfuse is not None:
            self._langfuse.flush()

    def end_trace(
        self,
        trace_id: str,
        *,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
        output_payload: Any | None = None,
    ) -> None:
        if self._langfuse is None:
            return

        ext_trace = self._external_traces.get(trace_id)
        if ext_trace is None:
            return

        if self._use_legacy_api:
            return

        ext_trace.update(
            level="ERROR" if status == "error" else "DEFAULT",
            metadata=metadata or {},
            output=output_payload,
        )
        ext_trace.end()

        cm = self._external_trace_contexts.pop(trace_id, None)
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                # Context close can fail when trace lifecycle spans threads; trace is already ended.
                pass

    def create_score(
        self,
        *,
        trace_id: str | None,
        name: str,
        value: float,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._langfuse is None or trace_id is None:
            return
        self._langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment,
            metadata=metadata or {},
        )

    def get_trace(self, trace_id: str) -> _InMemoryTrace | None:
        with self._lock:
            return self._traces.get(trace_id)
