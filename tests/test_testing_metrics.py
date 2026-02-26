from __future__ import annotations

from app.testing.metrics import detect_output_language, language_heuristic_match, structure_score


def test_output_language_dominant_english_allows_minor_cjk_tokens() -> None:
    text = (
        "Consultant briefing for Tencent with concise bullets and source references. "
        "Operational modernization, governance optimization, and risk notes are summarized. 腾讯"
    )
    assert detect_output_language(text) == "English"
    assert language_heuristic_match(text, "English") is True


def test_output_language_returns_mix_when_no_language_dominates() -> None:
    text = "Tencent briefing 腾讯 テンセント"
    assert detect_output_language(text) == "Mix"
    assert language_heuristic_match(text, "English") is False


def test_output_language_english_not_misclassified_by_im_acronym() -> None:
    text = (
        "Tencent Cloud Chat (IM) supports global artist messaging and operations modernization "
        "with partner integrations and governance controls."
    )
    assert detect_output_language(text) == "English"
    assert language_heuristic_match(text, "English") is True


def test_structure_score_accepts_internal_relationship_signal_heading() -> None:
    text = """
# Consultant Briefing Note: Tencent

## Executive Summary
Summary line.

## Public Findings
- Finding.

## Internal Relationship Signal
- Internal record found: Yes
- Internal note (sanitized): Note.

## Risk Notes
- Risk.

## Sources
- https://example.com

## Redaction and Confidentiality Notes
- Project name: [REDACTED]
""".strip()
    score, violations = structure_score(text)
    assert score == 1.0
    assert "missing_or_empty:internal_summary" not in violations
