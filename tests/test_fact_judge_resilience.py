from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.exceptions import LLMRetryExhausted
from app.testing import fact_judge


class _DummyTelemetry:
    pass


class _DummyProvider:
    pass


def test_fact_check_applicable_general_chat_is_false() -> None:
    assert (
        fact_judge.is_fact_check_applicable(
            task_type="general_chat",
            evidence_pack={"source_evidence": ["Some evidence"]},
        )
        is False
    )


def test_fact_judge_conservative_default_on_llm_failure(monkeypatch) -> None:
    def _raise_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise LLMRetryExhausted("fact_judge_node", 3, [])

    monkeypatch.setattr(fact_judge, "llm_call_with_retry", _raise_retry)
    settings = Settings(TOGETHER_API_KEY="test", REQUIRE_POSTGRES_CHECKPOINTER=False)
    result = fact_judge.run_fact_judge(
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=_DummyTelemetry(),  # type: ignore[arg-type]
        settings=settings,
        trace_id="trace-fact",
        scenario_prompt="Check this answer.",
        candidate_output="Answer text.",
        evidence_pack={"source_evidence": ["Answer text."]},
    )
    assert result.score_1_5 == 2.0
    assert result.verdict == "fail"
    assert result.source == "deterministic_fallback"
    assert result.applicable is True
    assert result.error is not None


def test_fact_judge_parses_json_with_structured_prompt(monkeypatch) -> None:
    def _fake_call_with_retry(*, validate_text, **kwargs):  # type: ignore[no-untyped-def]
        text = '{"score_1_5": 5, "verdict": "pass", "reason": "Fully grounded."}'
        validate_text(text)
        return SimpleNamespace(text=text)

    monkeypatch.setattr(fact_judge, "llm_call_with_retry", _fake_call_with_retry)
    settings = Settings(TOGETHER_API_KEY="test", REQUIRE_POSTGRES_CHECKPOINTER=False)
    result = fact_judge.run_fact_judge(
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=_DummyTelemetry(),  # type: ignore[arg-type]
        settings=settings,
        trace_id="trace-fact-2",
        scenario_prompt="Check this answer.",
        candidate_output="Answer text.",
        evidence_pack={"source_evidence": ["Answer text."]},
    )
    assert result.score_1_5 == 5.0
    assert result.verdict == "pass"
    assert result.source == "llm_json"
    assert result.applicable is True


def test_fact_judge_prompt_prioritizes_internal_tier() -> None:
    prompt = fact_judge.render_fact_judge_user_prompt(
        scenario_prompt="Check response grounding.",
        candidate_output="Some output text.",
        evidence_pack={
            "internal_db": {"internal_summary": "Authoritative internal relationship summary."},
            "internal_pdf": {"sanitized_text_excerpt": "Authoritative internal document excerpt."},
            "web": {"results": [{"title": "Public page", "snippet": "Public snippet"}]},
            "source_evidence": ["internal_db: authoritative", "web: public snippet"],
        },
    )
    assert "EVIDENCE TRUST MODEL" in prompt
    assert "internal_db and web_results are equal-trust evidence sources." in prompt
    assert "INTERNAL_DB" in prompt
    assert "WEB_RESULTS" in prompt
