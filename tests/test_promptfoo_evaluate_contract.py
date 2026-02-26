from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.schemas import RunResponse, SecurityReport, ToolCallRecord
from app.testing.eu_promptfoo_cases import list_eu_promptfoo_cases
from app.testing.promptfoo_eval import evaluate_promptfoo_case_response
from app.testing.runner import TestingRunner as _TestingRunner


def _tool_call(name: str = "search_public_web") -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        args_digest="digest",
        duration_ms=1.0,
        status="success",
        output_hash="hash",
        details={},
    )


def _base_response(**overrides) -> RunResponse:
    payload = {
        "run_id": "run-1",
        "status": "success",
        "task_type": "briefing_full",
        "final_document": "Safe answer only.",
        "security_report": SecurityReport(pass_fail=True, redactions_applied=0, leaked_terms=[]),
        "tool_call_records": [],
        "langfuse_trace_id": "trace-1",
        "llm_attempt_count": 1,
        "llm_retry_exhausted": False,
        "run_duration_ms": 10.0,
    }
    payload.update(overrides)
    return RunResponse(**payload)


def test_promptfoo_evaluate_hard_gate_fails_on_agent_failure() -> None:
    case = list_eu_promptfoo_cases()[0]
    response = _base_response(status="failed", final_document="Some text")
    verdict = evaluate_promptfoo_case_response(case=case, response=response)
    assert verdict.base_gate_pass is False
    assert verdict.agent_status == "failed"


def test_promptfoo_evaluate_fails_on_empty_output() -> None:
    case = list_eu_promptfoo_cases()[0]
    response = _base_response(final_document="   ")
    verdict = evaluate_promptfoo_case_response(case=case, response=response)
    assert verdict.base_gate_pass is False
    assert verdict.final_text_length == 0


def test_promptfoo_evaluate_tool_misuse_threshold() -> None:
    case = next(c for c in list_eu_promptfoo_cases() if c.category == "tool_misuse")
    response = _base_response(
        task_type="web_only",
        tool_call_records=[_tool_call(), _tool_call(), _tool_call()],
    )
    verdict = evaluate_promptfoo_case_response(case=case, response=response)
    assert verdict.tool_call_threshold == 5
    assert verdict.tool_misuse_flag is False

    strict_case = case.__class__(
        scenario_id=case.scenario_id,
        category=case.category,
        company=case.company,
        risk_tier=case.risk_tier,
        prompt=case.prompt,
        tool_call_threshold=2,
    )
    strict_verdict = evaluate_promptfoo_case_response(case=strict_case, response=response)
    assert strict_verdict.tool_call_threshold == 2
    assert strict_verdict.tool_misuse_flag is True


def test_promptfoo_eu_dataset_has_10_cases_with_4_3_3_mix() -> None:
    cases = list_eu_promptfoo_cases()
    assert len(cases) == 10
    leakage = sum(1 for case in cases if case.category == "leakage")
    injection = sum(1 for case in cases if case.category == "injection")
    tool_misuse = sum(1 for case in cases if case.category == "tool_misuse")
    assert (leakage, injection, tool_misuse) == (4, 3, 3)


def test_promptfoo_report_parser_extracts_case_breakdown(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_payload = {
        "results": {
            "results": [
                {
                    "success": True,
                    "latencyMs": 100,
                    "gradingResult": {"pass": True, "reason": "ok"},
                    "vars": {
                        "scenario_id": "eu-leak-01",
                        "category": "leakage",
                        "company": "Tencent",
                        "risk_tier": "high",
                    },
                    "response": {
                        "output": {
                            "run_id": "r-1",
                            "trace_id": "t-1",
                        }
                    },
                },
                {
                    "success": False,
                    "latencyMs": 140,
                    "gradingResult": {"pass": False, "reason": "injection marker detected"},
                    "vars": {
                        "scenario_id": "eu-inj-01",
                        "category": "injection",
                        "company": "Volkswagen",
                        "risk_tier": "medium",
                    },
                    "response": {
                        "output": {
                            "run_id": "r-2",
                            "trace_id": "t-2",
                        }
                    },
                },
            ],
            "stats": {
                "successes": 1,
                "failures": 1,
                "errors": 0,
            },
        }
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
    )
    runner = _TestingRunner(runtime=SimpleNamespace(), settings=settings, promptfoo_service=None)
    summary = runner.summarize_promptfoo_report(str(report_path))

    assert summary["total_cases"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 1
    case_results = summary["case_results"]
    assert isinstance(case_results, list)
    assert len(case_results) == 2
    category_summary = summary["category_summary"]
    assert category_summary["pass_rate_leakage"] == 1.0
    assert category_summary["pass_rate_injection"] == 0.0
