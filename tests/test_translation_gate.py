from __future__ import annotations

from app.config import Settings
from app.graph.nodes import AgentNodes
from app.internal_db.repository import InternalDBRepository
from app.telemetry.langfuse_client import LangfuseTelemetry


class _DummyProvider:
    model = "openai/gpt-oss-20b"


class _DummyToolbox:
    pass


def _base_state(language: str, text: str) -> dict:
    return {
        "run_id": "r-1",
        "trace_id": "t-1",
        "status": "success",
        "task_type": "translate_only",
        "user_prompt": "Translate Tencent proposal to English.",
        "company_name": "Tencent",
        "target_language": "English",
        "tool_results": {
            "retrieve_internal_pdf": {
                "document_found": True,
                "company_name": "Tencent",
                "doc_type": "proposal",
                "language": language,
                "sanitized_text": text,
                "policy_note": "ok",
            }
        },
        "step_events": [],
        "tool_call_records": [],
        "policy_findings": [],
        "llm_tokens": [],
        "artifacts": [],
    }


def test_translation_skips_when_source_matches_target(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-1", trace_id="t-1", session_id="s-1", user_id="u-1", input_payload={})
    repository = InternalDBRepository(settings.internal_db_path)
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=repository,
    )

    state = _base_state("English", "Confidential content sanitized.")
    updated = nodes.compose_template_document_node(state)
    assert updated["translation_applied"] is False
    assert updated["source_language"] == "English"
    assert updated["output_mode"] == "chat"
    assert "Confidential content sanitized." in updated["draft_document"]


def test_translation_runs_when_source_differs(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-1", trace_id="t-1", session_id="s-1", user_id="u-1", input_payload={})
    repository = InternalDBRepository(settings.internal_db_path)
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=repository,
    )

    state = _base_state("German", "Dies ist ein internes Dokument.")
    called = {"value": False}

    def fake_translate(_state, _payload):  # noqa: ANN001
        called["value"] = True
        return "This is an internal document."

    nodes._translate_sanitized_pdf = fake_translate  # type: ignore[method-assign]
    updated = nodes.compose_template_document_node(state)
    assert called["value"] is True
    assert updated["translation_applied"] is True
    assert updated["source_language"] == "German"
    assert updated["draft_document"] == "This is an internal document."


def test_translation_runs_when_language_detected_as_japanese(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-1", trace_id="t-1", session_id="s-1", user_id="u-1", input_payload={})
    repository = InternalDBRepository(settings.internal_db_path)
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=repository,
    )

    state = _base_state("unknown", "機密プロジェクトの概要です。")
    called = {"value": False}

    def fake_translate(_state, _payload):  # noqa: ANN001
        called["value"] = True
        return "This is a confidential project summary."

    nodes._translate_sanitized_pdf = fake_translate  # type: ignore[method-assign]
    updated = nodes.compose_template_document_node(state)
    assert called["value"] is True
    assert updated["translation_applied"] is True
    assert updated["source_language"] == "Japanese"
    assert updated["draft_document"] == "This is a confidential project summary."
