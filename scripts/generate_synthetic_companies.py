#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpdf import FPDF

# Make script runnable directly from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.synthetic_dataset import SyntheticProfile, build_profile_manifest
from backend.app.internal_db import InternalDBRepository, ensure_internal_db
from backend.app.internal_db.pdf_utils import sha256_bytes

# Deterministic synthetic-data generator used for assignment setup and testing.
# Responsibilities:
# - create 10 company profiles with multilingual internal docs
# - seed SQLite tables and PDF blobs for runtime retrieval flows
# - emit optional profile/manifest exports for transparency
# Boundaries:
# - runtime retrieval/evaluation logic stays in backend modules


COMPANIES = [
    ("Tencent", "Telecom / Consumer Internet"),
    ("Volkswagen", "Automotive"),
    ("TikTok", "Social Media Platform"),
    ("Tesla", "Mobility Technology"),
    ("Siemens", "Industrial Technology"),
    ("Pfizer", "Pharmaceuticals"),
    ("Samsung", "Consumer Electronics"),
    ("Shell", "Energy"),
    ("Sony", "Consumer Electronics / Entertainment"),
    ("Grab", "Mobility and Delivery"),
]

PROFILE_EXPORT_DEFAULT = "backend/data/synthetic_profiles.json"
MANIFEST_EXPORT_DEFAULT = "backend/data/synthetic_manifest.json"


# Font fallback keeps multilingual PDF generation deterministic across macOS setups.
def _pick_unicode_font() -> str | None:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    )
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            return str(path)
    return None


def _pdf_blob(title: str, paragraphs: list[str], language: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    unicode_font = _pick_unicode_font()

    if unicode_font:
        try:
            pdf.add_font("Unicode", style="", fname=unicode_font)
            pdf.set_font("Unicode", size=14)
        except Exception:
            unicode_font = None

    if unicode_font is None:
        pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)
    if unicode_font:
        pdf.set_font("Unicode", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
    for text in paragraphs:
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)
    raw = pdf.output()
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, bytes):
        return raw
    return raw.encode("latin-1")


def generate_profile_records(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    risk_levels = ["low", "medium", "high"]
    partners = [
        ["Microsoft", "Oracle"],
        ["SAP", "AWS"],
        ["Google Cloud", "Salesforce"],
        ["NVIDIA", "IBM"],
        ["Accenture", "Capgemini"],
        ["Bain", "McKinsey"],
        ["Cisco", "ServiceNow"],
        ["Adobe", "Meta"],
        ["Intel", "Palantir"],
        ["Snowflake", "Databricks"],
    ]
    records: list[dict[str, Any]] = []

    # Fixed 10-company set: deterministic for reproducible evaluation batches.
    for idx, (name, industry) in enumerate(COMPANIES):
        risk = risk_levels[idx % len(risk_levels)]
        budget = rng.randint(300_000, 4_500_000)
        project_name = f"{name} Strategic Transformation {idx + 1}"
        internal_product = f"{name} Internal Delivery Stack {idx + 1}"
        public_products = [f"{name} Platform", f"{name} Collaboration Suite"]
        public_partnerships = partners[idx]
        budget_formatted = f"USD {budget:,}"

        proposal_language = "English"
        quotation_language = "English"
        proposal_title = f"{name} - Confidential Proposal"
        quotation_title = f"{name} - Confidential Quotation"
        # Language policy: selected companies intentionally use non-English source docs
        # so translation gating and multilingual workflows can be tested.
        if name == "Tencent":
            proposal_language = "Chinese"
            quotation_language = "Chinese"
            proposal_title = f"{name} - 机密项目建议书"
            quotation_title = f"{name} - 机密项目报价单"
            proposal_text = [
                f"客户公司：{name}。",
                f"机密项目名称：{project_name}。",
                f"预算金额：{budget_formatted}。",
                f"内部产品线：{internal_product}。",
                "公开可用摘要：聚焦运营现代化与治理优化。",
            ]
            quotation_text = [
                f"本报价文件属于咨询团队与{name}之间的机密资料。",
                f"预算区间（机密）：{budget_formatted}。",
                "详细交付范围与排期均为机密，不得外传。",
                "对外可公开摘要：分阶段实施，关注效率与风险控制。",
            ]
        elif name == "Volkswagen":
            proposal_language = "German"
            quotation_language = "German"
            proposal_title = f"{name} - Vertraulicher Projektvorschlag"
            quotation_title = f"{name} - Vertrauliches Angebot"
            proposal_text = [
                f"Kundenunternehmen: {name}.",
                f"Vertraulicher Projektname: {project_name}.",
                f"Vorgeschlagenes Budget: {budget_formatted}.",
                f"Interner Produktstrom: {internal_product}.",
                "Öffentlich sichere Zusammenfassung: betriebliche Modernisierung und Governance.",
            ]
            quotation_text = [
                f"Dieses Angebotsdokument ist vertraulich zwischen Beratungsteam und {name}.",
                f"Vertraulicher Budgetrahmen: {budget_formatted}.",
                "Lieferumfang und Ressourcenplanung sind streng vertraulich.",
                "Öffentlich sichere Zusammenfassung: gestufte Umsetzung mit messbaren Ergebnissen.",
            ]
        elif name == "Siemens":
            proposal_language = "German"
            quotation_language = "German"
            proposal_title = f"{name} - Vertraulicher Projektvorschlag"
            quotation_title = f"{name} - Vertrauliches Angebot"
            proposal_text = [
                f"Kunde: {name}.",
                f"Interner Projektcodename: {project_name}.",
                f"Budgetrahmen: {budget_formatted}.",
                f"Interner Produktpfad: {internal_product}.",
                "Öffentlich sichere Zusammenfassung: Prozessdigitalisierung und Qualitätssteuerung.",
            ]
            quotation_text = [
                f"Dieses Angebotsdokument ist vertraulich zwischen Beratungsteam und {name}.",
                f"Vertraulicher Budgetrahmen: {budget_formatted}.",
                "Detaillierte Arbeitspakete sind intern und dürfen nicht offengelegt werden.",
                "Öffentlich sichere Zusammenfassung: schrittweise Einführung mit Governance-Kontrollen.",
            ]
        elif name == "Sony":
            proposal_language = "Japanese"
            quotation_language = "Japanese"
            proposal_title = f"{name} - 機密提案書"
            quotation_title = f"{name} - 機密見積書"
            proposal_text = [
                f"顧客企業：{name}。",
                f"機密プロジェクト名：{project_name}。",
                f"予算：{budget_formatted}。",
                f"社内製品ライン：{internal_product}。",
                "公開可能な要約：運用最適化とガバナンス強化を目的とする。",
            ]
            quotation_text = [
                f"本見積書はコンサルティングチームと{name}の機密資料です。",
                f"機密予算範囲：{budget_formatted}。",
                "詳細な納品項目および人員計画は社外秘です。",
                "公開可能な要約：段階的導入により価値創出を加速する。",
            ]
        else:
            proposal_text = [
                f"Client company: {name}.",
                f"Confidential project name: {project_name}.",
                f"Proposed budget: {budget_formatted}.",
                f"Internal product stream: {internal_product}.",
                "Key public-safe theme: operational modernization and platform governance.",
            ]
            quotation_text = [
                f"This quotation is between consulting team and {name}.",
                f"Total confidential budget estimate: {budget_formatted}.",
                "Resource model and detailed deliverables are confidential.",
                "Public-safe summary: phased implementation with medium-term value delivery.",
            ]

        proposal_blob = _pdf_blob(proposal_title, proposal_text, proposal_language)
        quotation_blob = _pdf_blob(quotation_title, quotation_text, quotation_language)

        # Each profile packs: public facts, confidential fields, and PDF BLOBs.
        records.append(
            {
                "profile_id": f"company-{idx + 1:02d}",
                "company_name": name,
                "industry": industry,
                "risk_category": risk,
                "public_products": public_products,
                "public_partnerships": public_partnerships,
                "document_languages": {
                    "proposal": proposal_language,
                    "quotation": quotation_language,
                },
                "preferred_output_language": proposal_language or "English",
                "sensitive_fields": {
                    "project_name": project_name,
                    "budget_usd": budget,
                    "budget_formatted": budget_formatted,
                    "internal_product_name": internal_product,
                },
                "redaction_terms": [project_name, internal_product, budget_formatted],
                "proposal_document": {
                    "doc_type": "proposal",
                    "language": proposal_language,
                    "file_name": f"{name.lower()}_proposal.pdf",
                    "title": proposal_title,
                    "text_lines": proposal_text,
                    "pdf_blob": proposal_blob,
                    "sha256": sha256_bytes(proposal_blob),
                },
                "quotation_document": {
                    "doc_type": "quotation",
                    "language": quotation_language,
                    "file_name": f"{name.lower()}_quotation.pdf",
                    "title": quotation_title,
                    "text_lines": quotation_text,
                    "pdf_blob": quotation_blob,
                    "sha256": sha256_bytes(quotation_blob),
                },
            }
        )
    return records


# Writes generated profiles/documents into the SQLite store consumed by the app.
def _seed_internal_db(*, db_path: str, overwrite: bool, profile_records: list[dict[str, Any]]) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and db_file.exists():
        db_file.unlink()

    ensure_internal_db(db_path)
    repo = InternalDBRepository(db_path)

    for record in profile_records:
        company_name = str(record["company_name"])
        sensitive = record["sensitive_fields"]
        project_name = str(sensitive["project_name"])
        internal_product_name = str(sensitive["internal_product_name"])
        budget_formatted = str(sensitive["budget_formatted"])

        repo.upsert_engagement(
            company_name=company_name,
            industry=str(record["industry"]),
            project_name=project_name,
            project_risk_level=str(record["risk_category"]),
            budget_usd=int(sensitive["budget_usd"]),
            product_name=internal_product_name,
            public_products=[str(item) for item in record["public_products"]],
            public_partnerships=[str(item) for item in record["public_partnerships"]],
        )

        proposal = record["proposal_document"]
        quotation = record["quotation_document"]
        repo.upsert_document(
            company_name=company_name,
            doc_type="proposal",
            language=str(proposal["language"]),
            file_name=str(proposal["file_name"]),
            pdf_blob=bytes(proposal["pdf_blob"]),
            sha256=str(proposal["sha256"]),
            classification="confidential",
        )
        repo.upsert_document(
            company_name=company_name,
            doc_type="quotation",
            language=str(quotation["language"]),
            file_name=str(quotation["file_name"]),
            pdf_blob=bytes(quotation["pdf_blob"]),
            sha256=str(quotation["sha256"]),
            classification="confidential",
        )
        repo.upsert_redaction_term(company_name=company_name, term=project_name, term_type="project_name")
        repo.upsert_redaction_term(company_name=company_name, term=internal_product_name, term_type="product_name")
        repo.upsert_redaction_term(company_name=company_name, term=budget_formatted, term_type="budget")


def _profile_export_payload(profile_records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    for record in profile_records:
        proposal = record["proposal_document"]
        quotation = record["quotation_document"]
        profiles.append(
            {
                "profile_id": record["profile_id"],
                "company_name": record["company_name"],
                "industry": record["industry"],
                "risk_category": record["risk_category"],
                "public_products": record["public_products"],
                "public_partnerships": record["public_partnerships"],
                "preferred_output_language": record["preferred_output_language"],
                "document_languages": {
                    "proposal": proposal["language"],
                    "quotation": quotation["language"],
                },
                "sensitive_fields": record["sensitive_fields"],
                "redaction_terms": record["redaction_terms"],
            }
        )
    return {
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_count": len(profiles),
        "profiles": profiles,
    }


def _build_manifest(profile_payload: dict[str, Any]) -> dict[str, Any]:
    typed_profiles: list[SyntheticProfile] = []
    for raw in profile_payload.get("profiles", []):
        if not isinstance(raw, dict):
            continue
        document_languages = raw.get("document_languages")
        typed_profiles.append(
            SyntheticProfile(
                profile_id=str(raw.get("profile_id", "")),
                company_name=str(raw.get("company_name", "")),
                industry=str(raw.get("industry", "Unknown")),
                public_products=tuple(str(item) for item in raw.get("public_products", [])),
                public_partnerships=tuple(str(item) for item in raw.get("public_partnerships", [])),
                risk_category=str(raw.get("risk_category", "medium")),
                preferred_output_language=str(raw.get("preferred_output_language", "English")),
                document_languages={
                    "proposal": str((document_languages or {}).get("proposal", "English")),
                    "quotation": str((document_languages or {}).get("quotation", "English")),
                },
                sensitive_terms=tuple(str(item) for item in raw.get("redaction_terms", [])),
                source="generator",
            )
        )
    manifest = build_profile_manifest(typed_profiles)
    return {
        "seed": profile_payload.get("seed"),
        "generated_at": profile_payload.get("generated_at"),
        **manifest,
    }


def _write_json(path: str | None, payload: dict[str, Any]) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(file_path)


# Public script entrypoint used by README quickstart.
def build_dataset(
    *,
    db_path: str,
    seed: int,
    overwrite: bool,
    emit_profiles_json: str | None,
    emit_manifest_json: str | None,
) -> tuple[str | None, str | None]:
    profile_records = generate_profile_records(seed=seed)
    _seed_internal_db(db_path=db_path, overwrite=overwrite, profile_records=profile_records)

    profile_payload = _profile_export_payload(profile_records, seed=seed)
    profiles_path = _write_json(emit_profiles_json, profile_payload)
    manifest_path = _write_json(emit_manifest_json, _build_manifest(profile_payload))
    return profiles_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10 synthetic companies and seed SQLite DB.")
    parser.add_argument("--db-path", default="backend/data/internal_company.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--emit-profiles-json",
        default=PROFILE_EXPORT_DEFAULT,
        help="Optional JSON export path for generated synthetic profiles.",
    )
    parser.add_argument(
        "--emit-manifest-json",
        default=MANIFEST_EXPORT_DEFAULT,
        help="Optional JSON export path for generated synthetic manifest.",
    )
    args = parser.parse_args()

    profiles_path, manifest_path = build_dataset(
        db_path=args.db_path,
        seed=args.seed,
        overwrite=args.overwrite,
        emit_profiles_json=args.emit_profiles_json,
        emit_manifest_json=args.emit_manifest_json,
    )
    print(f"Seeded synthetic dataset at {args.db_path}")
    if profiles_path:
        print(f"Wrote synthetic profiles JSON: {profiles_path}")
    if manifest_path:
        print(f"Wrote synthetic manifest JSON: {manifest_path}")


if __name__ == "__main__":
    main()
