from __future__ import annotations

import random
import time
from typing import Any, Callable

import httpx

from app.config import Settings
from app.exceptions import LLMHTTPError, LLMResponseValidationError, LLMRetryExhausted
from app.schemas import AttemptLog, LLMCallResult, PromptBudget
from app.telemetry.langfuse_client import LangfuseTelemetry
from app.token_utils import compress_prompt, estimate_tokens

from .together_client import TogetherClient

# Retry/backoff wrapper around model calls used by graph and evaluators.
# Responsibilities:
# - bounded retries with provider-aware backoff
# - prompt budgeting/compression under context limits
# - consistent attempt logs for telemetry and failure diagnostics
# Boundaries:
# - raw provider transport remains in together_client.py


def _safe_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value <= 0:
            return None
        return int(value)
    return None


def _resolve_context_window(provider: object, resolved_model: str, default_value: int = 8192) -> int:
    # Compatibility order keeps older test doubles/providers working:
    # get_context_window -> discover_context_window -> context_window attr -> default.
    """Resolve context window compatibly across real provider and lightweight test doubles."""
    get_context_window = getattr(provider, "get_context_window", None)
    if callable(get_context_window):
        try:
            value = _safe_positive_int(get_context_window(resolved_model))
            if value is not None:
                return value
        except Exception:  # noqa: BLE001
            pass

    discover_context_window = getattr(provider, "discover_context_window", None)
    if callable(discover_context_window):
        try:
            value = _safe_positive_int(discover_context_window(model_override=resolved_model))
            if value is not None:
                return value
        except TypeError:
            try:
                value = _safe_positive_int(discover_context_window())
                if value is not None:
                    return value
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    context_window = _safe_positive_int(getattr(provider, "context_window", None))
    if context_window is not None:
        return context_window

    return default_value


def build_prompt_budget(*, context_window: int, max_tokens: int, safety_margin: int) -> PromptBudget:
    input_budget = context_window - max_tokens - safety_margin
    return PromptBudget(
        context_window=context_window,
        max_tokens=max_tokens,
        safety_margin=safety_margin,
        input_budget=input_budget,
    )


def is_retryable_exception(exc: Exception) -> tuple[bool, int | None, str]:
    if isinstance(exc, LLMHTTPError):
        code = exc.status_code
        if code == 429 or 500 <= code <= 599:
            return True, code, f"http_{code}"
        return False, code, f"http_{code}"

    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True, None, exc.__class__.__name__.lower()

    if isinstance(exc, LLMResponseValidationError):
        return True, None, "invalid_response"

    return False, None, exc.__class__.__name__.lower()


# Non-streaming retry path used by parser/planner/composer and judge calls.
def llm_call_with_retry(
    *,
    provider: TogetherClient,
    telemetry: LangfuseTelemetry,
    trace_id: str,
    settings: Settings,
    node_name: str,
    prompt: str,
    max_tokens: int,
    validate_text: Callable[[str], None],
    reasoning_effort: str,
    model_override: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> LLMCallResult:
    resolved_model = model_override or provider.model
    context_window = _resolve_context_window(provider, resolved_model)

    budget = build_prompt_budget(
        context_window=context_window,
        max_tokens=max_tokens,
        safety_margin=settings.llm_safety_margin_tokens,
    )

    compressed_prompt = compress_prompt(prompt, budget.input_budget)
    input_tokens = estimate_tokens(compressed_prompt)

    attempts: list[AttemptLog] = []
    total_trials = settings.llm_max_retries + 1

    for attempt_index in range(1, total_trials + 1):
        provider_status_code: int | None = None
        span_id = telemetry.start_span(
            trace_id,
            name="llm_call_attempt",
            metadata={
                "node_name": node_name,
                "model": resolved_model,
                "attempt_index": attempt_index,
                "context_window": budget.context_window,
                "input_tokens": input_tokens,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            },
            observation_type="chain",
        )
        generation_id = telemetry.start_generation(
            trace_id=trace_id,
            name="llm_generation_attempt",
            input_payload=compressed_prompt,
            metadata={
                "node_name": node_name,
                "attempt_index": attempt_index,
                "context_window": budget.context_window,
            },
            model=resolved_model,
            model_parameters={
                "temperature": 0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            },
            parent_span_id=span_id,
        )

        started = time.perf_counter()
        try:
            raw = provider.chat_completion(
                compressed_prompt,
                max_tokens=max_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
                model_override=model_override,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
            provider_status_code = raw.status_code
            validate_text(raw.text)
            latency_ms = (time.perf_counter() - started) * 1000
            output_tokens = int(raw.usage.get("completion_tokens") or estimate_tokens(raw.text))

            attempt_log = AttemptLog(
                attempt_index=attempt_index,
                status_code=raw.status_code,
                latency_ms=latency_ms,
                retry_reason=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            attempts.append(attempt_log)

            telemetry.end_span(
                span_id,
                status="ok",
                metadata={
                    "status_code": raw.status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": None,
                    "output_tokens": output_tokens,
                    "input_tokens": input_tokens,
                    "model": resolved_model,
                    "attempt_index": attempt_index,
                    "context_window": budget.context_window,
                    "reasoning_effort": reasoning_effort,
                },
            )
            telemetry.end_generation(
                generation_id,
                status="ok",
                output_payload=raw.text,
                metadata={
                    "status_code": raw.status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": None,
                    "attempt_index": attempt_index,
                    "reasoning_effort": reasoning_effort,
                },
                usage_details={"input": input_tokens, "output": output_tokens},
            )

            return LLMCallResult(
                text=raw.text,
                provider="together",
                model=resolved_model,
                latency_ms=latency_ms,
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            retryable, status_code, retry_reason = is_retryable_exception(exc)
            if status_code is None and provider_status_code is not None:
                status_code = provider_status_code
            attempt_log = AttemptLog(
                attempt_index=attempt_index,
                status_code=status_code,
                latency_ms=latency_ms,
                retry_reason=retry_reason,
                error=str(exc),
                input_tokens=input_tokens,
                output_tokens=0,
            )
            attempts.append(attempt_log)

            telemetry.end_span(
                span_id,
                status="error",
                metadata={
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": retry_reason,
                    "error": str(exc),
                    "output_tokens": 0,
                    "input_tokens": input_tokens,
                    "model": resolved_model,
                    "attempt_index": attempt_index,
                    "context_window": budget.context_window,
                    "reasoning_effort": reasoning_effort,
                },
            )
            telemetry.end_generation(
                generation_id,
                status="error",
                output_payload=str(exc),
                metadata={
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": retry_reason,
                    "attempt_index": attempt_index,
                    "reasoning_effort": reasoning_effort,
                },
                usage_details={"input": input_tokens, "output": 0},
            )

            if (not retryable) or attempt_index >= total_trials:
                raise LLMRetryExhausted(
                    message=(
                        f"LLM call failed for node '{node_name}' after {attempt_index} trial(s): {exc}"
                    ),
                    attempts=attempts,
                    retry_exhausted=attempt_index >= total_trials,
                ) from exc

            backoff = 1.0 if attempt_index == 1 else 2.0
            jitter = random.uniform(0.01, 0.25)
            time.sleep(backoff + jitter)

    raise LLMRetryExhausted(
        message=f"LLM call failed for node '{node_name}' with unknown retry state",
        attempts=attempts,
        retry_exhausted=True,
    )


# Streaming retry path used for final chat outputs.
def llm_stream_with_retry(
    *,
    provider: TogetherClient,
    telemetry: LangfuseTelemetry,
    trace_id: str,
    settings: Settings,
    node_name: str,
    prompt: str,
    max_tokens: int,
    validate_text: Callable[[str], None],
    on_token: Callable[[str], None] | None,
    reasoning_effort: str,
    model_override: str | None = None,
) -> LLMCallResult:
    resolved_model = model_override or provider.model
    context_window = _resolve_context_window(provider, resolved_model)

    budget = build_prompt_budget(
        context_window=context_window,
        max_tokens=max_tokens,
        safety_margin=settings.llm_safety_margin_tokens,
    )
    compressed_prompt = compress_prompt(prompt, budget.input_budget)
    input_tokens = estimate_tokens(compressed_prompt)
    attempts: list[AttemptLog] = []
    total_trials = settings.llm_max_retries + 1

    for attempt_index in range(1, total_trials + 1):
        span_id = telemetry.start_span(
            trace_id,
            name="llm_stream_attempt",
            metadata={
                "node_name": node_name,
                "model": resolved_model,
                "attempt_index": attempt_index,
                "context_window": budget.context_window,
                "input_tokens": input_tokens,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "stream": True,
            },
            observation_type="chain",
        )
        generation_id = telemetry.start_generation(
            trace_id=trace_id,
            name="llm_stream_generation_attempt",
            input_payload=compressed_prompt,
            metadata={
                "node_name": node_name,
                "attempt_index": attempt_index,
                "context_window": budget.context_window,
                "stream": True,
            },
            model=resolved_model,
            model_parameters={
                "temperature": 0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "stream": True,
            },
            parent_span_id=span_id,
        )
        started = time.perf_counter()
        collected: list[str] = []
        try:
            for token in provider.chat_completion_stream(
                compressed_prompt,
                max_tokens=max_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
                model_override=model_override,
                reasoning_effort=reasoning_effort,
            ):
                collected.append(token)
                if on_token is not None:
                    on_token(token)
            text = "".join(collected).strip()
            validate_text(text)
            latency_ms = (time.perf_counter() - started) * 1000
            output_tokens = estimate_tokens(text)
            attempt_log = AttemptLog(
                attempt_index=attempt_index,
                status_code=200,
                latency_ms=latency_ms,
                retry_reason=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            attempts.append(attempt_log)
            telemetry.end_span(
                span_id,
                status="ok",
                metadata={
                    "status_code": 200,
                    "latency_ms": latency_ms,
                    "retry_reason": None,
                    "output_tokens": output_tokens,
                    "input_tokens": input_tokens,
                    "model": resolved_model,
                    "attempt_index": attempt_index,
                    "context_window": budget.context_window,
                    "reasoning_effort": reasoning_effort,
                    "stream": True,
                },
            )
            telemetry.end_generation(
                generation_id,
                status="ok",
                output_payload=text,
                metadata={
                    "status_code": 200,
                    "latency_ms": latency_ms,
                    "retry_reason": None,
                    "attempt_index": attempt_index,
                    "reasoning_effort": reasoning_effort,
                    "stream": True,
                },
                usage_details={"input": input_tokens, "output": output_tokens},
            )
            return LLMCallResult(
                text=text,
                provider="together",
                model=resolved_model,
                latency_ms=latency_ms,
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            retryable, status_code, retry_reason = is_retryable_exception(exc)
            attempt_log = AttemptLog(
                attempt_index=attempt_index,
                status_code=status_code,
                latency_ms=latency_ms,
                retry_reason=retry_reason,
                error=str(exc),
                input_tokens=input_tokens,
                output_tokens=0,
            )
            attempts.append(attempt_log)
            telemetry.end_span(
                span_id,
                status="error",
                metadata={
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": retry_reason,
                    "error": str(exc),
                    "output_tokens": 0,
                    "input_tokens": input_tokens,
                    "model": resolved_model,
                    "attempt_index": attempt_index,
                    "context_window": budget.context_window,
                    "reasoning_effort": reasoning_effort,
                    "stream": True,
                },
            )
            telemetry.end_generation(
                generation_id,
                status="error",
                output_payload=str(exc),
                metadata={
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "retry_reason": retry_reason,
                    "attempt_index": attempt_index,
                    "reasoning_effort": reasoning_effort,
                    "stream": True,
                },
                usage_details={"input": input_tokens, "output": 0},
            )
            if (not retryable) or attempt_index >= total_trials:
                raise LLMRetryExhausted(
                    message=(
                        f"LLM streaming call failed for node '{node_name}' after {attempt_index} trial(s): {exc}"
                    ),
                    attempts=attempts,
                    retry_exhausted=attempt_index >= total_trials,
                ) from exc
            backoff = 1.0 if attempt_index == 1 else 2.0
            jitter = random.uniform(0.01, 0.25)
            time.sleep(backoff + jitter)

    raise LLMRetryExhausted(
        message=f"LLM streaming call failed for node '{node_name}' with unknown retry state",
        attempts=attempts,
        retry_exhausted=True,
    )
