from __future__ import annotations

from app.config import Settings
from app.graph.nodes import AgentNodes
from app.internal_db.repository import InternalDBRepository
from app.telemetry.langfuse_client import LangfuseTelemetry


class _DummyProvider:
    model = "openai/gpt-oss-20b"


class _DummyToolbox:
    pass


def _build_nodes(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace(
        "run-lang",
        trace_id="trace-lang",
        session_id="session-lang",
        user_id="user-lang",
        input_payload={},
    )
    return AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )


def _base_state(target_language: str, draft_document: str) -> dict:
    return {
        "run_id": "run-lang",
        "trace_id": "trace-lang",
        "status": "success",
        "target_language": target_language,
        "draft_document": draft_document,
        "step_events": [],
        "tool_call_records": [],
        "policy_findings": [],
        "llm_tokens": [],
    }


def test_language_enforcement_skips_when_already_aligned(tmp_path):
    nodes = _build_nodes(tmp_path)
    state = _base_state("Chinese", "这是顾问简报的执行摘要。")
    updated = nodes.enforce_output_language_node(state)
    assert updated["status"] == "success"
    assert updated["language_fallback_applied"] is False
    assert updated["draft_document"] == "这是顾问简报的执行摘要。"


def test_language_enforcement_applies_translation_fallback(tmp_path):
    nodes = _build_nodes(tmp_path)
    state = _base_state("German", "This is an executive summary.")

    def fake_translate(*, state, text, target_language, node_name):  # noqa: ANN001
        assert target_language == "German"
        assert node_name == "enforce_output_language_node"
        return "Dies ist eine Management-Zusammenfassung."

    nodes._translate_text_to_target_language = fake_translate  # type: ignore[method-assign]
    updated = nodes.enforce_output_language_node(state)
    assert updated["status"] == "success"
    assert updated["language_fallback_applied"] is True
    assert updated["draft_document"] == "Dies ist eine Management-Zusammenfassung."


def test_language_enforcement_preserves_redacted_token(tmp_path):
    nodes = _build_nodes(tmp_path)
    state = _base_state("Chinese", "Project name: [REDACTED]")

    def fake_translate(*, state, text, target_language, node_name):  # noqa: ANN001
        assert "[REDACTED]" in text
        return "项目名称：[REDACTED]"

    nodes._is_text_in_target_language = lambda text, target: False  # type: ignore[method-assign]
    nodes._translate_text_to_target_language = fake_translate  # type: ignore[method-assign]
    updated = nodes.enforce_output_language_node(state)
    assert updated["status"] == "success"
    assert updated["language_fallback_applied"] is True
    assert "[REDACTED]" in updated["draft_document"]
