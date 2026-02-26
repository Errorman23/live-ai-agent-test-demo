from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.llm.together_client import TogetherClient
from app.token_utils import compress_prompt, estimate_tokens


def test_deterministic_inference_params_sent_to_together():
    observed_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_payload
        if request.url.path.endswith("/chat/completions"):
            observed_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 2},
                },
            )
        return httpx.Response(
            status_code=200,
            json={"data": [{"id": "openai/gpt-oss-20b", "context_length": 8192}]},
        )

    transport = httpx.MockTransport(handler)
    settings = Settings(TOGETHER_API_KEY="test", TOGETHER_BASE_URL="https://example.com", REQUIRE_POSTGRES_CHECKPOINTER=False)
    client = TogetherClient(settings, transport=transport)

    client.chat_completion("hello", max_tokens=1200, timeout_seconds=30, reasoning_effort="low")

    assert observed_payload["model"] == "openai/gpt-oss-20b"
    assert observed_payload["temperature"] == 0
    assert observed_payload["top_p"] == 1
    assert observed_payload["frequency_penalty"] == 0
    assert observed_payload["presence_penalty"] == 0
    assert observed_payload["n"] == 1
    assert observed_payload["max_tokens"] == 1200
    assert observed_payload["reasoning_effort"] == "low"


def test_context_compression_drops_trace_and_fits_budget():
    prompt = "\n".join(
        [
            "TASK: Build a briefing.",
            "TRACE: this should be dropped",
            "EVIDENCE: line one",
            "EVIDENCE: line one",  # duplicate
            "EVIDENCE: line two",
            "[trace] also dropped",
        ]
        + [f"NOISE line {i}" for i in range(200)]
    )

    compressed = compress_prompt(prompt, budget_tokens=50)

    assert "TRACE:" not in compressed
    assert "[trace]" not in compressed.lower()
    assert compressed.count("EVIDENCE: line one") == 1
    assert estimate_tokens(compressed) <= 50
