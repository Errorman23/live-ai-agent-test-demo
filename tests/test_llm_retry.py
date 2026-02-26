from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.exceptions import LLMHTTPError, LLMResponseValidationError, LLMRetryExhausted
from app.llm.retry import llm_call_with_retry
from app.llm.together_client import TogetherRawResponse
from app.telemetry.langfuse_client import LangfuseTelemetry

# Retry suite validates:
# - retryable vs non-retryable provider failures
# - context-window budgeting/compression behavior
# - telemetry attempt accounting contracts


class StubProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.context_window = 8192
        self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    def chat_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        timeout_seconds: float,
        model_override: str | None = None,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
    ):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MinimalProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    def chat_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        timeout_seconds: float,
        model_override: str | None = None,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
    ):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _settings() -> Settings:
    return Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        LLM_MAX_RETRIES=2,
        LLM_TIMEOUT_SECONDS=0.1,
    )


def _telemetry_with_trace():
    telemetry = LangfuseTelemetry(enabled=False)
    trace_id = telemetry.start_trace("run-test")
    return telemetry, trace_id


# ---------------------------------------------------------------------------
# Retry and retry-exhaustion invariants.
# ---------------------------------------------------------------------------
def test_retry_429_then_success(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = StubProvider(
        [
            LLMHTTPError(429, "rate limited"),
            TogetherRawResponse(text="ok", usage={"completion_tokens": 5}, status_code=200),
        ]
    )
    telemetry, trace_id = _telemetry_with_trace()

    result = llm_call_with_retry(
        provider=provider,
        telemetry=telemetry,
        trace_id=trace_id,
        settings=_settings(),
        node_name="plan_node",
        prompt="hello",
        max_tokens=1200,
        validate_text=lambda text: None,
        reasoning_effort="low",
    )

    assert result.text == "ok"
    assert len(result.attempts) == 2
    assert result.attempts[0].status_code == 429
    assert result.attempts[1].status_code == 200

    trace = telemetry.get_trace(trace_id)
    assert trace is not None
    llm_call_spans = [s for s in trace.spans if s.name == "llm_call_attempt"]
    llm_generation_spans = [s for s in trace.spans if s.name == "llm_generation_attempt"]
    assert len(llm_call_spans) == 2
    assert len(llm_generation_spans) == 2
    assert llm_call_spans[0].metadata.get("observation_type") == "chain"
    assert llm_generation_spans[0].metadata.get("observation_type") == "generation"
    assert llm_generation_spans[0].metadata.get("parent_span_id") == llm_call_spans[0].span_id


def test_retry_timeout_then_5xx_then_success(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = StubProvider(
        [
            httpx.ReadTimeout("timed out"),
            LLMHTTPError(503, "unavailable"),
            TogetherRawResponse(text="third attempt ok", usage={}, status_code=200),
        ]
    )
    telemetry, trace_id = _telemetry_with_trace()

    result = llm_call_with_retry(
        provider=provider,
        telemetry=telemetry,
        trace_id=trace_id,
        settings=_settings(),
        node_name="compose_document_node",
        prompt="hello",
        max_tokens=2000,
        validate_text=lambda text: None,
        reasoning_effort="low",
    )

    assert result.text == "third attempt ok"
    assert len(result.attempts) == 3
    assert result.attempts[0].retry_reason == "readtimeout"
    assert result.attempts[1].status_code == 503


def test_three_retryable_failures_exhaust(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = StubProvider(
        [
            LLMHTTPError(500, "a"),
            LLMHTTPError(503, "b"),
            LLMHTTPError(429, "c"),
        ]
    )
    telemetry, trace_id = _telemetry_with_trace()

    with pytest.raises(LLMRetryExhausted) as err:
        llm_call_with_retry(
            provider=provider,
            telemetry=telemetry,
            trace_id=trace_id,
            settings=_settings(),
            node_name="plan_node",
            prompt="hello",
            max_tokens=1200,
            validate_text=lambda text: None,
            reasoning_effort="low",
        )

    assert err.value.retry_exhausted is True
    assert len(err.value.attempts) == 3


def test_non_retryable_401_immediate_fail(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = StubProvider([LLMHTTPError(401, "unauthorized")])
    telemetry, trace_id = _telemetry_with_trace()

    with pytest.raises(LLMRetryExhausted) as err:
        llm_call_with_retry(
            provider=provider,
            telemetry=telemetry,
            trace_id=trace_id,
            settings=_settings(),
            node_name="plan_node",
            prompt="hello",
            max_tokens=1200,
            validate_text=lambda text: None,
            reasoning_effort="low",
        )

    assert len(err.value.attempts) == 1
    assert err.value.attempts[0].status_code == 401


def test_invalid_response_counts_as_retry(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = StubProvider(
        [
            TogetherRawResponse(text="{}", usage={}, status_code=200),
            TogetherRawResponse(text='{"ok": true}', usage={}, status_code=200),
        ]
    )
    telemetry, trace_id = _telemetry_with_trace()
    calls = {"n": 0}

    def validate(text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMResponseValidationError("invalid schema")

    result = llm_call_with_retry(
        provider=provider,
        telemetry=telemetry,
        trace_id=trace_id,
        settings=_settings(),
        node_name="plan_node",
        prompt="hello",
        max_tokens=1200,
        validate_text=validate,
        reasoning_effort="low",
    )
    assert result.text == '{"ok": true}'
    assert len(result.attempts) == 2


# Compatibility fallback for older provider stubs lacking context methods.
def test_provider_without_context_methods_uses_default_context(monkeypatch):
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _x: None)
    provider = MinimalProvider([TogetherRawResponse(text="ok", usage={}, status_code=200)])
    telemetry, trace_id = _telemetry_with_trace()

    result = llm_call_with_retry(
        provider=provider,
        telemetry=telemetry,
        trace_id=trace_id,
        settings=_settings(),
        node_name="plan_node",
        prompt="hello",
        max_tokens=1200,
        validate_text=lambda text: None,
        reasoning_effort="low",
    )

    assert result.text == "ok"
    assert len(result.attempts) == 1
