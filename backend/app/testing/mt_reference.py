from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from app.config import Settings


@dataclass(frozen=True)
class MTReferenceResult:
    translated_text: str
    model: str
    latency_ms: float


MT_REFERENCE_SYSTEM_TEMPLATE = (
    "You are a professional translator.\n"
    "Translate faithfully, preserve line/section structure, and keep [REDACTED] unchanged.\n"
    "Return only translated text."
)

MT_REFERENCE_USER_TEMPLATE = (
    "Translate from {{source_language}} to {{target_language}}.\n\n"
    "{{source_text}}"
)


def render_mt_reference_user_prompt(*, source_language: str, target_language: str, source_text: str) -> str:
    return (
        MT_REFERENCE_USER_TEMPLATE
        .replace("{{source_language}}", source_language)
        .replace("{{target_language}}", target_language)
        .replace("{{source_text}}", source_text)
    )


class SiliconFlowMTClient:
    def __init__(self, settings: Settings) -> None:
        self.api_key = (settings.siliconflow_api_key or "").strip()
        self.base_url = settings.siliconflow_base_url.rstrip("/")
        self.model = settings.siliconflow_mt_model
        self.timeout_seconds = settings.siliconflow_timeout_seconds

    def translate(self, *, text: str, source_language: str, target_language: str) -> MTReferenceResult:
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not configured")

        user_prompt = render_mt_reference_user_prompt(
            source_language=source_language,
            target_language=target_language,
            source_text=text,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": MT_REFERENCE_SYSTEM_TEMPLATE},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "top_p": 0.7,
            "thinking_budget": 4096,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                content=json.dumps(payload),
            )

        if response.status_code >= 400:
            body = response.text.strip()
            short_body = body[:260] + ("..." if len(body) > 260 else "")
            raise RuntimeError(f"SiliconFlow HTTP {response.status_code}: {short_body}")

        raw = response.json()
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("SiliconFlow response missing choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("SiliconFlow response missing message.content")

        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        latency_ms = 0.0
        if isinstance(usage, dict):
            for key in ("latency_ms", "total_latency_ms"):
                value = usage.get(key)
                if isinstance(value, int | float):
                    latency_ms = float(value)
                    break

        return MTReferenceResult(
            translated_text=content.strip(),
            model=self.model,
            latency_ms=latency_ms,
        )
