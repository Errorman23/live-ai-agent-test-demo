from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.graph.nodes import AgentNodes, _guess_company, _infer_task_type
from app.internal_db.repository import InternalDBRepository
from app.promptfoo.service import PromptfooServiceManager
from app.schemas import PlannerStep
from app.telemetry.langfuse_client import LangfuseTelemetry
from app.testing.runner import TestingRunner as _TestingRunner

# Coverage focus:
# - intent/routing heuristics for multilingual and OOD prompts
# - promptfoo report normalization and summary contract stability
# - runner-level denominator and policy-gate invariants

class _RepoStub:
    pass


class _DummyProvider:
    model = "openai/gpt-oss-20b"


class _DummyToolbox:
    pass


def _step(tool_name: str) -> PlannerStep:
    return PlannerStep(tool_name=tool_name, args={}, rationale_short="")


# Intent extraction / task inference checks.
# ---------------------------------------------------------------------------
# Intent/routing heuristics and planner coercion coverage.
# ---------------------------------------------------------------------------
def test_openai_prompt_never_falls_back_to_tencent():
    company, source = _guess_company("Get OpenAI information from web only.", _RepoStub())
    assert company == "OpenAI"
    assert source in {"alias", "heuristic"}


def test_unknown_company_does_not_default_to_tencent():
    company, _ = _guess_company("Please help me with that company profile.", _RepoStub())
    assert company != "Tencent"


def test_intent_inference_covers_internal_record_and_translation():
    assert _infer_task_type("Get TikTok internal record.") == "db_only"
    assert _infer_task_type("Retrieve Sony proposal from internal DB.") == "doc_only"
    assert _infer_task_type("Translate Tencent proposal to English.") == "translate_only"


def test_ood_prompt_defaults_to_general_chat():
    assert _infer_task_type("Give me practical consultant onboarding tips for my first week.") == "general_chat"


# Plan-to-route resolution checks.
def test_plan_toolset_resolves_route_types():
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[_step("get_company_info"), _step("search_public_web")],
            requested_task_type=None,
        )
        == "briefing_full"
    )
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[_step("search_public_web")],
            requested_task_type=None,
        )
        == "web_only"
    )
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[_step("get_company_info")],
            requested_task_type=None,
        )
        == "db_only"
    )
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[_step("retrieve_internal_pdf")],
            requested_task_type=None,
        )
        == "doc_only"
    )
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[_step("retrieve_internal_pdf"), _step("translate_document")],
            requested_task_type=None,
        )
        == "translate_only"
    )
    assert (
        AgentNodes._resolve_task_type_from_plan(
            steps=[],
            requested_task_type=None,
        )
        == "general_chat"
    )


def test_plan_toolset_detects_ambiguous_routes():
    resolved = AgentNodes._resolve_task_type_from_plan(
        steps=[_step("retrieve_internal_pdf"), _step("search_public_web")],
        requested_task_type=None,
    )
    assert resolved is None


def test_requested_task_type_override_wins():
    resolved = AgentNodes._resolve_task_type_from_plan(
        steps=[_step("search_public_web")],
        requested_task_type="db_only",
    )
    assert resolved == "db_only"


def test_route_after_validation_for_general_chat(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-route-gc", trace_id="t-route-gc", session_id="s-route-gc", user_id="u-route-gc", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )
    next_node = nodes.route_after_validation({"status": "success", "task_type": "general_chat"})
    assert next_node == "compose_template_document"


def test_validate_plan_prefers_non_briefing_intent_hint(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-1", trace_id="t-1", session_id="s-1", user_id="u-1", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )
    state = {
        "run_id": "r-1",
        "trace_id": "t-1",
        "status": "success",
        "task_type": "db_only",
        "requested_task_type": None,
        "plan_steps": [_step("retrieve_internal_pdf")],
        "policy_findings": [],
        "step_events": [],
        "tool_call_records": [],
        "llm_tokens": [],
    }
    updated = nodes.validate_plan_node(state)
    assert updated["status"] == "success"
    assert updated["task_type"] == "db_only"
    assert updated["output_mode"] == "chat"


def test_validate_plan_falls_back_to_briefing_hint_when_ambiguous(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-1b", trace_id="t-1b", session_id="s-1b", user_id="u-1b", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )
    state = {
        "run_id": "r-1b",
        "trace_id": "t-1b",
        "status": "success",
        "task_type": "briefing_full",
        "requested_task_type": None,
        "plan_steps": [_step("retrieve_internal_pdf"), _step("search_public_web")],
        "policy_findings": [],
        "step_events": [],
        "tool_call_records": [],
        "llm_tokens": [],
    }
    updated = nodes.validate_plan_node(state)
    assert updated["status"] == "success"
    assert updated["task_type"] == "briefing_full"
    assert updated["output_mode"] == "document"


def test_validate_plan_accepts_general_chat_with_empty_steps(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-gc", trace_id="t-gc", session_id="s-gc", user_id="u-gc", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )
    state = {
        "run_id": "r-gc",
        "trace_id": "t-gc",
        "status": "success",
        "task_type": "general_chat",
        "requested_task_type": None,
        "plan_steps": [],
        "policy_findings": [],
        "step_events": [],
        "tool_call_records": [],
        "llm_tokens": [],
    }
    updated = nodes.validate_plan_node(state)
    assert updated["status"] == "success"
    assert updated["task_type"] == "general_chat"
    assert updated["output_mode"] == "chat"


# Planner/composer coercion recovery checks for malformed model outputs.
def test_planner_candidate_coercion_handles_tool_style_output(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-2", trace_id="t-2", session_id="s-2", user_id="u-2", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )

    recovered = nodes._coerce_planner_candidate(
        {"commentary to=search_public_web code": "OpenAI company information"},
        {"company_name": "OpenAI", "target_language": "English"},
    )
    assert recovered is not None
    assert recovered.company_name == "OpenAI"
    assert recovered.target_language == "English"
    assert recovered.steps
    assert recovered.steps[0].tool_name == "search_public_web"
    assert recovered.steps[0].args["company_name"] == "OpenAI"


def test_planner_candidate_coercion_recovers_repo_browser_wrapper_from_task_hint(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-3", trace_id="t-3", session_id="s-3", user_id="u-3", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )

    recovered = nodes._coerce_planner_candidate(
        {"commentary to=repo_browser.print_tree code{": 1},
        {"company_name": "OpenAI", "target_language": "English", "task_type": "web_only"},
    )
    assert recovered is not None
    assert recovered.company_name == "OpenAI"
    assert recovered.target_language == "English"
    assert recovered.steps


def test_planner_candidate_coercion_accepts_general_chat_with_no_steps(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-4", trace_id="t-4", session_id="s-4", user_id="u-4", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )

    recovered = nodes._coerce_planner_candidate(
        {"task_type": "general_chat", "steps": []},
        {"company_name": "Unknown Company", "target_language": "English", "task_type": "general_chat"},
    )
    assert recovered is not None
    assert recovered.task_type == "general_chat"
    assert recovered.steps == []


def test_composer_candidate_coercion_accepts_value_field(tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("r-compose", trace_id="t-compose", session_id="s-compose", user_id="u-compose", input_payload={})
    nodes = AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )
    normalized = nodes._coerce_composer_candidate(
        {
            "commentary to=commentary jsonWe need executive_summary": "noise",
            "value": "Executive summary text.",
            "public_findings": "One\nTwo",
            "risk_notes": ["Low risk"],
            "sources": "https://example.com",
        },
        default_internal_summary="Default internal summary.",
    )
    assert normalized is not None
    assert normalized["executive_summary"] == "Executive summary text."
    assert normalized["internal_summary"] == "Default internal summary."
    assert normalized["public_findings"] == ["One", "Two"]


# ---------------------------------------------------------------------------
# Promptfoo service lifecycle and campaign execution coverage.
# ---------------------------------------------------------------------------
def test_promptfoo_service_health_and_restart(monkeypatch, tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        PROMPTFOO_ENABLED=True,
        PROMPTFOO_OUTPUT_DIR=str(tmp_path / "out"),
        PROMPTFOO_COMMAND="promptfoo",
        PROMPTFOO_PORT=16699,
        PROMPTFOO_LOG_PATH=str(tmp_path / "promptfoo.log"),
    )
    manager = PromptfooServiceManager(settings=settings, project_root=tmp_path)

    class FakeProc:
        def __init__(self) -> None:
            self.pid = 4321
            self.returncode = None

        def poll(self):  # noqa: ANN001
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout=None) -> int:  # noqa: ANN001
            return 0

        def kill(self) -> None:
            self.returncode = -9

    proc_holder: dict[str, FakeProc] = {}

    def fake_popen(*_args, **_kwargs):  # noqa: ANN001
        proc = FakeProc()
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr("app.promptfoo.service.subprocess.Popen", fake_popen)
    monkeypatch.setattr("app.promptfoo.service._port_open", lambda *_args, **_kwargs: True)

    manager.start()
    health = manager.health()
    assert health["enabled"] is True
    assert health["healthy"] is True
    assert health["process_alive"] is True
    assert health["port_open"] is True

    manager.restart()
    health_after = manager.health()
    assert health_after["healthy"] is True
    assert health_after["pid"] == 4321

    manager.stop()
    assert proc_holder["proc"].returncode in {0, -9}


def test_promptfoo_campaign_uses_service_health(monkeypatch, tmp_path):
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        PROMPTFOO_ENABLED=True,
        PROMPTFOO_OUTPUT_DIR=str(tmp_path / "artifacts"),
        PROMPTFOO_COMMAND="promptfoo",
        PROMPTFOO_PORT=15500,
    )

    class DummyService:
        def ensure_running(self) -> None:
            return None

        def health(self) -> dict[str, object]:
            return {"healthy": True, "ui_url": "http://127.0.0.1:15500"}

    runner = _TestingRunner(runtime=SimpleNamespace(), settings=settings, promptfoo_service=DummyService())

    def fake_run(cmd, check, capture_output, text, timeout, env):  # noqa: ANN001
        assert "eval" in cmd
        report_index = cmd.index("-o") + 1
        report_path = Path(cmd[report_index])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"results": []}), encoding="utf-8")

        class Completed:
            returncode = 0
            stderr = ""
            stdout = ""

        return Completed()

    monkeypatch.setattr("app.testing.runner.subprocess.run", fake_run)

    report, promptfoo_url, error = runner._run_promptfoo_campaign(
        test_id="t-1",
        suite="functional",
        test_domain="functional",
    )
    assert error is None
    assert report is not None
    assert Path(report).exists()
    assert promptfoo_url == "http://127.0.0.1:15500"
