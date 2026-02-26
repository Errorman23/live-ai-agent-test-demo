from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from app.config import Settings
from app.exceptions import LLMHTTPError

# Thin Together API transport wrapper.
# Responsibilities:
# - perform chat completion and streaming calls
# - discover effective context-window metadata from model info endpoints
# - normalize raw provider errors into app-level exceptions
# Boundaries:
# - retry/budget logic is handled by llm/retry.py


_CONTEXT_KEYS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_sequence_length",
)
_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Follow the requested output format exactly. "
    "If JSON is requested, return raw JSON only without markdown fences, role tags, or tool-call wrappers."
)


@dataclass
class TogetherRawResponse:
    text: str
    usage: dict[str, Any]
    status_code: int


class TogetherClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = settings.together_model
        self.base_url = settings.together_base_url.rstrip("/")
        self.timeout_seconds = settings.llm_timeout_seconds
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.together_api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )
        self.context_window: int | None = None
        self._context_windows: dict[str, int] = {}

    def close(self) -> None:
        self._client.close()

    def discover_context_window(self, model_override: str | None = None) -> int:
        target_model = model_override or self.model
        cached = self._context_windows.get(target_model)
        if isinstance(cached, int) and cached > 0:
            if target_model == self.model:
                self.context_window = cached
            return cached

        response = self._client.get("/models")
        if response.status_code >= 400:
            raise LLMHTTPError(response.status_code, response.text)

        payload = response.json()
        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            candidates = [p for p in payload["data"] if isinstance(p, dict)]
        elif isinstance(payload, list):
            candidates = [p for p in payload if isinstance(p, dict)]

        model_record: dict[str, Any] | None = None
        for entry in candidates:
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == target_model:
                model_record = entry
                break

        if model_record is None:
            raise RuntimeError(f"Model metadata for '{target_model}' not found in Together /models response")

        for key in _CONTEXT_KEYS:
            value = model_record.get(key)
            if isinstance(value, int) and value > 0:
                self._context_windows[target_model] = value
                if target_model == self.model:
                    self.context_window = value
                return value
            if isinstance(value, str) and value.isdigit():
                parsed = int(value)
                self._context_windows[target_model] = parsed
                if target_model == self.model:
                    self.context_window = parsed
                return parsed

        raise RuntimeError(
            f"Context window for '{target_model}' missing. Supported keys: {', '.join(_CONTEXT_KEYS)}"
        )

    def get_context_window(self, model_override: str | None = None) -> int:
        target_model = model_override or self.model
        cached = self._context_windows.get(target_model)
        if isinstance(cached, int) and cached > 0:
            return cached
        discovered = self.discover_context_window(model_override=target_model)
        if discovered <= 0:
            raise RuntimeError(f"Context window discovery returned invalid value for '{target_model}'")
        return discovered

    def chat_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        timeout_seconds: float,
        model_override: str | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> TogetherRawResponse:
        resolved_model = model_override or self.model
        payload = {
            "model": resolved_model,
            "messages": [
                {
                    "role": "system",
                    "content": _DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "max_tokens": max_tokens,
        }
        if reasoning_effort in {"low", "medium", "high"}:
            payload["reasoning_effort"] = reasoning_effort
        if response_format:
            payload["response_format"] = response_format

        response = self._client.post(
            "/chat/completions",
            content=json.dumps(payload),
            timeout=timeout_seconds,
        )

        if response.status_code >= 400:
            raise LLMHTTPError(response.status_code, response.text)

        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Invalid Together completion response: missing choices")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None

        if not isinstance(content, str):
            raise RuntimeError("Invalid Together completion response: missing message.content")

        usage = data.get("usage") if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}

        return TogetherRawResponse(
            text=content,
            usage=usage,
            status_code=response.status_code,
        )

    def chat_completion_stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        timeout_seconds: float,
        model_override: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        resolved_model = model_override or self.model
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning_effort in {"low", "medium", "high"}:
            payload["reasoning_effort"] = reasoning_effort

        with self._client.stream(
            "POST",
            "/chat/completions",
            content=json.dumps(payload),
            timeout=timeout_seconds,
        ) as response:
            if response.status_code >= 400:
                raise LLMHTTPError(response.status_code, response.text)

            for line in response.iter_lines():
                if not line:
                    continue
                text_line = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                if not text_line.startswith("data:"):
                    continue
                chunk = text_line.split("data:", 1)[1].strip()
                if chunk == "[DONE]":
                    break
                try:
                    payload_chunk = json.loads(chunk)
                except Exception:
                    continue
                choices = payload_chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0]
                delta = first.get("delta") if isinstance(first, dict) else None
                token = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(token, str) and token:
                    yield token
