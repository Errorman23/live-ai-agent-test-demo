from __future__ import annotations

from app.config import Settings
from app.exceptions import LLMHTTPError
from app.graph.runtime import AgentRuntime
from app.llm.together_client import TogetherRawResponse
from app.schemas import RunRequest
from app.telemetry.langfuse_client import LangfuseTelemetry


class AlwaysFailProvider:
    def __init__(self):
        self.model = "openai/gpt-oss-20b"
        self.context_window = None

    def discover_context_window(self) -> int:
        self.context_window = 8192
        return 8192

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
        raise LLMHTTPError(503, "upstream unavailable")

    def close(self) -> None:
        return None


class InvalidPlannerProvider:
    def __init__(self):
        self.model = "openai/gpt-oss-20b"
        self.context_window = None

    def discover_context_window(self) -> int:
        self.context_window = 8192
        return 8192

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
        return TogetherRawResponse(text="{}", usage={}, status_code=200)

    def close(self) -> None:
        return None


def test_graph_routes_to_error_after_retry_exhaustion():
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        LLM_MAX_RETRIES=2,
        LLM_TIMEOUT_SECONDS=0.1,
    )
    telemetry = LangfuseTelemetry(enabled=False)
    provider = AlwaysFailProvider()
    runtime = AgentRuntime(settings=settings, provider=provider, telemetry=telemetry)

    response = runtime.run(RunRequest(prompt="Generate a company briefing on Tesla in German"))

    assert response.status == "failed"
    assert response.llm_retry_exhausted is True
    assert response.llm_attempt_count == 3
    assert response.provider_status_codes == [503, 503, 503]


def test_graph_fails_when_planner_never_returns_valid_schema():
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        LLM_MAX_RETRIES=2,
        LLM_TIMEOUT_SECONDS=0.1,
    )
    telemetry = LangfuseTelemetry(enabled=False)
    provider = InvalidPlannerProvider()
    runtime = AgentRuntime(settings=settings, provider=provider, telemetry=telemetry)

    response = runtime.run(RunRequest(prompt="Generate a company briefing on Tesla"))

    assert response.status == "failed"
    assert response.llm_retry_exhausted is True
    assert response.llm_attempt_count == 3
    assert response.provider_status_codes == [200, 200, 200]
