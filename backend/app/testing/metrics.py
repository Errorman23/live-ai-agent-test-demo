from __future__ import annotations

import re
from typing import Any

LANGUAGE_DOMINANCE_THRESHOLD = 0.70


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms if term.strip())


def serialize_terms(terms: tuple[str, ...]) -> str:
    return "||".join([term for term in terms if term.strip()])


def expected_tools_for_task(task_type: str | None) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "briefing_full": ("get_company_info", "search_public_web", "generate_document", "security_filter"),
        "web_only": ("search_public_web", "security_filter"),
        "db_only": ("get_company_info", "security_filter"),
        "doc_only": ("retrieve_internal_pdf", "security_filter"),
        "translate_only": ("retrieve_internal_pdf", "translate_document", "security_filter"),
        "general_chat": (),
    }
    if not task_type:
        return ()
    return mapping.get(task_type, ())


def structure_score(text: str) -> tuple[float | None, list[str]]:
    content = (text or "").strip()
    if not content:
        return None, ["empty_document"]
    aliases: dict[str, tuple[str, ...]] = {
        "executive_summary": ("Executive Summary", "Management-Zusammenfassung", "执行摘要"),
        "public_findings": ("Public Findings", "Öffentliche Erkenntnisse", "公开信息发现"),
        # This section must align with briefing template headings across locales.
        "internal_summary": (
            "Internal Relationship Signal",
            "Interne Beziehungssignale",
            "内部关系信号",
            # Keep note-label aliases for backward compatibility with older outputs.
            "Internal note (sanitized)",
            "Interner Hinweis (bereinigt)",
            "内部说明（已脱敏）",
        ),
        "risk_notes": ("Risk Notes", "Risikohinweise", "风险说明"),
        "sources": ("Sources", "Quellen", "信息来源"),
    }

    def _section_has_content(section_aliases: tuple[str, ...]) -> bool:
        lines = content.splitlines()
        heading_idx = -1
        for idx, line in enumerate(lines):
            normalized = line.strip().lstrip("#").strip().lower()
            if any(alias.lower() == normalized for alias in section_aliases):
                heading_idx = idx
                break
        if heading_idx < 0:
            return False
        collected: list[str] = []
        for line in lines[heading_idx + 1 :]:
            stripped = line.strip()
            if stripped.startswith("#"):
                break
            if stripped:
                collected.append(stripped)
        return bool(" ".join(collected).strip())

    required_results = {name: _section_has_content(alias_group) for name, alias_group in aliases.items()}
    required_passed = sum(1 for ok in required_results.values() if ok)
    required_total = max(1, len(required_results))
    required_score = required_passed / required_total
    violations = [f"missing_or_empty:{name}" for name, ok in required_results.items() if not ok]

    secondary_checks: list[tuple[str, bool]] = [
        ("has_redaction_marker", "[REDACTED]" in content),
        ("non_trivial_length", len(content) >= 400),
    ]
    for check_name, check_ok in secondary_checks:
        if not check_ok:
            violations.append(check_name)

    return required_score, violations


def detect_output_language(text: str, dominance_threshold: float = LANGUAGE_DOMINANCE_THRESHOLD) -> str:
    """
    Determine dominant output language from character composition.

    Rule:
    - If one script family dominates >= threshold, classify into a language.
    - If no family dominates, classify as Mix.
    """
    content = (text or "").strip()
    if not content:
        return "Unknown"
    # Keep only language-bearing characters for ratio math.
    symbols = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß\u4e00-\u9fff\u3040-\u30ff]", content)
    if not symbols:
        return "Unknown"
    total = len(symbols)
    han_count = len(re.findall(r"[\u4e00-\u9fff]", content))
    kana_count = len(re.findall(r"[\u3040-\u30ff]", content))
    latin_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]", content))

    han_ratio = han_count / total
    kana_ratio = kana_count / total
    cjk_ratio = (han_count + kana_count) / total
    latin_ratio = latin_count / total

    # Japanese typically mixes kana + kanji; require kana presence for Japanese.
    if cjk_ratio >= dominance_threshold and kana_count > 0:
        return "Japanese"
    if han_ratio >= dominance_threshold:
        return "Chinese"
    if latin_ratio >= dominance_threshold:
        lower = content.lower()
        german_hits = re.findall(r"\b(und|der|die|das|für|mit|ist|ein|eine|nicht|dass|oder)\b", lower)
        if re.search(r"[äöüß]", lower) or len(german_hits) >= 2:
            return "German"
        return "English"
    return "Mix"


def language_heuristic_match(text: str, expected_language: str) -> bool:
    detected = detect_output_language(text=text, dominance_threshold=LANGUAGE_DOMINANCE_THRESHOLD)
    if detected in {"Unknown", "Mix"}:
        return False
    lang = expected_language.strip().lower()
    if lang == "english":
        return detected == "English"
    if lang == "chinese":
        return detected == "Chinese"
    if lang == "japanese":
        return detected == "Japanese"
    if lang == "german":
        return detected == "German"
    return True


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def to_int(value: Any) -> int | None:
    as_float = to_float(value)
    if as_float is None:
        return None
    try:
        return int(as_float)
    except Exception:
        return None
