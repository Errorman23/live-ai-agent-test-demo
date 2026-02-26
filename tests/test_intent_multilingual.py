from __future__ import annotations

from app.config import Settings
from app.graph.nodes import AgentNodes, _guess_company, _infer_language, _infer_task_type
from app.internal_db.repository import InternalDBRepository
from app.telemetry.langfuse_client import LangfuseTelemetry


class _DummyProvider:
    model = "openai/gpt-oss-20b"


class _DummyToolbox:
    pass


class _RepoStub:
    def list_company_names(self) -> tuple[str, ...]:
        return ()


def _build_nodes(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("run-intent", trace_id="trace-intent", session_id="session-intent", user_id="user-intent", input_payload={})
    return AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )


def test_multilingual_company_and_language_helpers():
    company, source = _guess_company("生成腾讯的briefing文件，文件最终需要是中文", _RepoStub())  # type: ignore[arg-type]
    assert company == "Tencent"
    assert source == "alias"
    assert _infer_language("生成腾讯的briefing文件，文件最终需要是中文") == "Chinese"
    assert _infer_task_type("生成腾讯的briefing文件，文件最终需要是中文") == "briefing_full"


def test_parse_intent_resolves_tesla_german(tmp_path):
    nodes = _build_nodes(tmp_path)
    state = {
        "run_id": "run-1",
        "trace_id": "trace-intent",
        "user_prompt": "Generate a company briefing on Tesla in German",
        "requested_task_type": None,
    }
    updated = nodes.parse_intent_node(state)
    assert updated["company_name"] == "Tesla"
    assert updated["target_language"] == "German"
    assert updated["task_type"] == "briefing_full"
    assert updated["intent_resolution_source"] in {"alias", "heuristic"}


def test_parse_intent_resolves_chinese_tencent_prompt(tmp_path):
    nodes = _build_nodes(tmp_path)
    state = {
        "run_id": "run-2",
        "trace_id": "trace-intent",
        "user_prompt": "生成腾讯的briefing文件，文件最终需要是中文",
        "requested_task_type": None,
    }
    updated = nodes.parse_intent_node(state)
    assert updated["company_name"] == "Tencent"
    assert updated["target_language"] == "Chinese"
    assert updated["resolved_target_language"] == "Chinese"
    assert updated["task_type"] == "briefing_full"
    assert updated["language_resolution_source"] == "alias"
