from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Synthetic profile loading/manifest utilities shared by setup and APIs.
# Responsibilities:
# - parse profile JSON into typed records
# - expose compact manifest stats (risk/language/industry coverage)
# - provide deterministic defaults for repo-local dataset paths
# Boundaries:
# - generation lives in scripts/generate_synthetic_companies.py


DEFAULT_SYNTHETIC_PROFILES_PATH = Path("backend/data/synthetic_profiles.json")
DEFAULT_INTERNAL_DB_PATH = Path("backend/data/internal_company.db")


@dataclass
class SyntheticProfile:
    profile_id: str
    company_name: str
    industry: str
    public_products: tuple[str, ...]
    public_partnerships: tuple[str, ...]
    risk_category: str
    preferred_output_language: str
    document_languages: dict[str, str]
    sensitive_terms: tuple[str, ...]
    source: str = "synthetic"


def _normalize_risk(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"low", "medium", "high"}:
        return raw
    return "medium"


def _normalize_language(value: Any) -> str:
    language = str(value or "").strip()
    return language or "English"


def _list_to_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(v) for v in value if str(v).strip())
    if isinstance(value, list):
        return tuple(str(v) for v in value if str(v).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _parse_profile_entry(raw: dict[str, Any], index: int, *, source: str) -> SyntheticProfile:
    company_name = str(raw.get("company_name") or raw.get("name") or "").strip() or f"Company-{index + 1}"
    document_languages_raw = raw.get("document_languages")
    document_languages: dict[str, str] = {}
    if isinstance(document_languages_raw, dict):
        for key, value in document_languages_raw.items():
            doc_type = str(key).strip().lower()
            if doc_type:
                document_languages[doc_type] = _normalize_language(value)
    if "proposal" not in document_languages:
        document_languages["proposal"] = _normalize_language(raw.get("proposal_language"))
    if "quotation" not in document_languages:
        document_languages["quotation"] = _normalize_language(raw.get("quotation_language"))

    sensitive_terms: list[str] = []
    for value in raw.get("redaction_terms", []) if isinstance(raw.get("redaction_terms"), list) else []:
        if str(value).strip():
            sensitive_terms.append(str(value))
    sensitive_fields = raw.get("sensitive_fields")
    if isinstance(sensitive_fields, dict):
        for key in ("project_name", "internal_product_name", "budget_formatted"):
            value = sensitive_fields.get(key)
            if str(value or "").strip():
                sensitive_terms.append(str(value))
    if not sensitive_terms:
        for key in ("project_name", "product_name", "budget_formatted"):
            value = raw.get(key)
            if str(value or "").strip():
                sensitive_terms.append(str(value))

    preferred_language = _normalize_language(
        raw.get("preferred_output_language")
        or document_languages.get("proposal")
        or raw.get("target_language")
    )

    profile_id = str(raw.get("profile_id") or f"company-{index + 1:02d}").strip()
    return SyntheticProfile(
        profile_id=profile_id,
        company_name=company_name,
        industry=str(raw.get("industry") or "Unknown").strip() or "Unknown",
        public_products=_list_to_tuple(raw.get("public_products") or raw.get("products")),
        public_partnerships=_list_to_tuple(raw.get("public_partnerships") or raw.get("partnerships")),
        risk_category=_normalize_risk(raw.get("risk_category") or raw.get("project_risk_level")),
        preferred_output_language=preferred_language,
        document_languages=document_languages,
        sensitive_terms=tuple(dict.fromkeys(sensitive_terms)),
        source=source,
    )


def _load_profiles_from_json(path: Path) -> list[SyntheticProfile]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_profiles = payload.get("profiles")
        if isinstance(raw_profiles, list):
            records = [r for r in raw_profiles if isinstance(r, dict)]
    elif isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]

    profiles: list[SyntheticProfile] = []
    for index, raw in enumerate(records):
        profile = _parse_profile_entry(raw, index, source="json")
        profiles.append(profile)
    return profiles


def _load_profiles_from_db(path: Path) -> list[SyntheticProfile]:
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        engagements = conn.execute(
            """
            SELECT company_name, industry, project_name, project_risk_level, budget_usd, product_name,
                   public_products, public_partnerships
            FROM engagements
            ORDER BY company_name ASC
            """
        ).fetchall()
        if not engagements:
            return []

        docs = conn.execute(
            """
            SELECT company_name, doc_type, language
            FROM internal_documents
            ORDER BY company_name ASC
            """
        ).fetchall()
        doc_map: dict[str, dict[str, str]] = defaultdict(dict)
        for row in docs:
            company = str(row["company_name"])
            doc_map[company][str(row["doc_type"]).lower()] = _normalize_language(row["language"])

        terms_rows = conn.execute(
            """
            SELECT company_name, term
            FROM redaction_terms
            WHERE active = 1
            ORDER BY id ASC
            """
        ).fetchall()
        term_map: dict[str, list[str]] = defaultdict(list)
        for row in terms_rows:
            company = str(row["company_name"])
            term = str(row["term"])
            if term not in term_map[company]:
                term_map[company].append(term)

        profiles: list[SyntheticProfile] = []
        for index, row in enumerate(engagements):
            company_name = str(row["company_name"])
            try:
                public_products = tuple(json.loads(str(row["public_products"])))
            except Exception:
                public_products = ()
            try:
                public_partnerships = tuple(json.loads(str(row["public_partnerships"])))
            except Exception:
                public_partnerships = ()

            doc_languages = dict(doc_map.get(company_name, {}))
            if "proposal" not in doc_languages:
                doc_languages["proposal"] = "English"
            if "quotation" not in doc_languages:
                doc_languages["quotation"] = "English"

            preferred_language = doc_languages.get("proposal") or "English"
            profile = SyntheticProfile(
                profile_id=f"company-{index + 1:02d}",
                company_name=company_name,
                industry=str(row["industry"]),
                public_products=public_products,
                public_partnerships=public_partnerships,
                risk_category=_normalize_risk(row["project_risk_level"]),
                preferred_output_language=preferred_language,
                document_languages=doc_languages,
                sensitive_terms=tuple(term_map.get(company_name, [])),
                source="db",
            )
            profiles.append(profile)
        return profiles
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _fallback_profiles() -> list[SyntheticProfile]:
    defaults = (
        ("Tencent", "Telecom / Consumer Internet", "Chinese"),
        ("Volkswagen", "Automotive", "German"),
        ("TikTok", "Social Media Platform", "English"),
        ("Tesla", "Mobility Technology", "English"),
        ("Siemens", "Industrial Technology", "German"),
        ("Pfizer", "Pharmaceuticals", "English"),
        ("Samsung", "Consumer Electronics", "English"),
        ("Shell", "Energy", "English"),
        ("Sony", "Consumer Electronics / Entertainment", "Japanese"),
        ("Grab", "Mobility and Delivery", "English"),
    )
    risk_cycle = ("low", "medium", "high")
    profiles: list[SyntheticProfile] = []
    for index, (name, industry, language) in enumerate(defaults):
        risk = risk_cycle[index % len(risk_cycle)]
        profiles.append(
            SyntheticProfile(
                profile_id=f"company-{index + 1:02d}",
                company_name=name,
                industry=industry,
                public_products=(f"{name} Platform", f"{name} Collaboration Suite"),
                public_partnerships=("Partner A", "Partner B"),
                risk_category=risk,
                preferred_output_language=language,
                document_languages={"proposal": language, "quotation": language},
                sensitive_terms=(),
                source="fallback",
            )
        )
    return profiles


def load_synthetic_profiles(
    *,
    profiles_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> list[SyntheticProfile]:
    """
    Load synthetic profiles with deterministic precedence:
    1) explicit/generated JSON export, 2) seeded SQLite DB, 3) built-in fallback set.
    """
    resolved_profiles = Path(profiles_path) if profiles_path else DEFAULT_SYNTHETIC_PROFILES_PATH
    profiles = _load_profiles_from_json(resolved_profiles)
    if profiles:
        return profiles

    resolved_db = Path(db_path) if db_path else DEFAULT_INTERNAL_DB_PATH
    profiles = _load_profiles_from_db(resolved_db)
    if profiles:
        return profiles
    return _fallback_profiles()


def build_profile_manifest(profiles: list[SyntheticProfile]) -> dict[str, Any]:
    risk_counts: Counter[str] = Counter()
    preferred_output_language_counts: Counter[str] = Counter()
    doc_language_counts: dict[str, Counter[str]] = {
        "proposal": Counter(),
        "quotation": Counter(),
    }
    companies_by_risk: dict[str, list[str]] = defaultdict(list)

    for profile in profiles:
        risk_counts[profile.risk_category] += 1
        preferred_output_language_counts[profile.preferred_output_language] += 1
        companies_by_risk[profile.risk_category].append(profile.company_name)
        for doc_type, language in profile.document_languages.items():
            doc_language_counts.setdefault(doc_type, Counter())
            doc_language_counts[doc_type][_normalize_language(language)] += 1

    return {
        "profile_count": len(profiles),
        "risk_counts": dict(sorted(risk_counts.items())),
        "preferred_output_language_counts": dict(sorted(preferred_output_language_counts.items())),
        "document_language_counts": {
            doc_type: dict(sorted(counter.items()))
            for doc_type, counter in sorted(doc_language_counts.items())
        },
        "companies_by_risk": {
            risk: sorted(companies)
            for risk, companies in sorted(companies_by_risk.items())
        },
    }
