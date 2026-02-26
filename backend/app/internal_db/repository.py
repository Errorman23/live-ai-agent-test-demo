from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# SQLite repository for synthetic internal company records and PDF blobs.
# Responsibilities:
# - schema creation/migration for demo data
# - CRUD/query helpers consumed by tools and admin explorer APIs
# - deterministic access to confidential/public fields used in evaluation
# Boundaries:
# - content generation lives in scripts/generate_synthetic_companies.py


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS engagements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company_name TEXT NOT NULL UNIQUE,
  industry TEXT NOT NULL,
  project_name TEXT NOT NULL,
  project_risk_level TEXT NOT NULL CHECK(project_risk_level IN ('low', 'medium', 'high')),
  budget_usd INTEGER NOT NULL,
  product_name TEXT NOT NULL,
  public_products TEXT NOT NULL,
  public_partnerships TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS internal_documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company_name TEXT NOT NULL,
  doc_type TEXT NOT NULL,
  language TEXT NOT NULL,
  file_name TEXT NOT NULL,
  pdf_blob BLOB NOT NULL,
  sha256 TEXT NOT NULL,
  classification TEXT NOT NULL DEFAULT 'confidential',
  created_at TEXT NOT NULL,
  UNIQUE(company_name, doc_type, file_name)
);

CREATE TABLE IF NOT EXISTS redaction_terms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company_name TEXT NOT NULL,
  term TEXT NOT NULL,
  term_type TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1
);
"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Row-level containers used by repository query methods.
@dataclass(frozen=True)
class Engagement:
    company_name: str
    industry: str
    project_name: str
    project_risk_level: str
    budget_usd: int
    product_name: str
    public_products: tuple[str, ...]
    public_partnerships: tuple[str, ...]


@dataclass(frozen=True)
class InternalDocument:
    id: int
    company_name: str
    doc_type: str
    language: str
    file_name: str
    pdf_blob: bytes
    sha256: str
    classification: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: str) -> Path:
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_internal_db(db_path: str) -> None:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


# Main SQLite repository wrapper used by runtime tools and admin APIs.
class InternalDBRepository:
    def __init__(self, db_path: str) -> None:
        resolved = _resolve_db_path(db_path)
        self.db_path = str(resolved)
        ensure_internal_db(self.db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def health(self) -> dict[str, object]:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM engagements").fetchone()
        return {"ok": True, "engagement_count": int(row["c"] if row else 0)}

    def list_company_names(self) -> tuple[str, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT company_name
                FROM engagements
                ORDER BY company_name ASC
                """
            ).fetchall()
        return tuple(str(row["company_name"]) for row in rows)

    def upsert_engagement(
        self,
        *,
        company_name: str,
        industry: str,
        project_name: str,
        project_risk_level: str,
        budget_usd: int,
        product_name: str,
        public_products: list[str],
        public_partnerships: list[str],
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO engagements (
                  company_name, industry, project_name, project_risk_level, budget_usd, product_name,
                  public_products, public_partnerships, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_name) DO UPDATE SET
                  industry=excluded.industry,
                  project_name=excluded.project_name,
                  project_risk_level=excluded.project_risk_level,
                  budget_usd=excluded.budget_usd,
                  product_name=excluded.product_name,
                  public_products=excluded.public_products,
                  public_partnerships=excluded.public_partnerships,
                  updated_at=excluded.updated_at
                """,
                (
                    company_name,
                    industry,
                    project_name,
                    project_risk_level,
                    budget_usd,
                    product_name,
                    json.dumps(public_products),
                    json.dumps(public_partnerships),
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_engagement(self, company_name: str) -> Engagement | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT company_name, industry, project_name, project_risk_level, budget_usd, product_name,
                       public_products, public_partnerships
                FROM engagements
                WHERE lower(company_name) = lower(?)
                LIMIT 1
                """,
                (company_name,),
            ).fetchone()

        if row is None:
            return None
        return Engagement(
            company_name=str(row["company_name"]),
            industry=str(row["industry"]),
            project_name=str(row["project_name"]),
            project_risk_level=str(row["project_risk_level"]),
            budget_usd=int(row["budget_usd"]),
            product_name=str(row["product_name"]),
            public_products=tuple(json.loads(str(row["public_products"]))),
            public_partnerships=tuple(json.loads(str(row["public_partnerships"]))),
        )

    def upsert_document(
        self,
        *,
        company_name: str,
        doc_type: str,
        language: str,
        file_name: str,
        pdf_blob: bytes,
        sha256: str,
        classification: str = "confidential",
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO internal_documents (
                  company_name, doc_type, language, file_name, pdf_blob, sha256, classification, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_name, doc_type, file_name) DO UPDATE SET
                  language=excluded.language,
                  pdf_blob=excluded.pdf_blob,
                  sha256=excluded.sha256,
                  classification=excluded.classification
                """,
                (company_name, doc_type, language, file_name, pdf_blob, sha256, classification, now),
            )
            conn.commit()

    def get_document(self, company_name: str, doc_type: str) -> InternalDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, company_name, doc_type, language, file_name, pdf_blob, sha256, classification
                FROM internal_documents
                WHERE lower(company_name) = lower(?) AND lower(doc_type) = lower(?)
                ORDER BY id DESC
                LIMIT 1
                """,
                (company_name, doc_type),
            ).fetchone()

        if row is None:
            return None
        return InternalDocument(
            id=int(row["id"]),
            company_name=str(row["company_name"]),
            doc_type=str(row["doc_type"]),
            language=str(row["language"]),
            file_name=str(row["file_name"]),
            pdf_blob=bytes(row["pdf_blob"]),
            sha256=str(row["sha256"]),
            classification=str(row["classification"]),
        )

    def upsert_redaction_term(self, *, company_name: str, term: str, term_type: str) -> None:
        with self._connect() as conn:
            exists = conn.execute(
                """
                SELECT id FROM redaction_terms
                WHERE lower(company_name) = lower(?) AND lower(term) = lower(?) AND lower(term_type) = lower(?)
                LIMIT 1
                """,
                (company_name, term, term_type),
            ).fetchone()
            if exists is None:
                conn.execute(
                    """
                    INSERT INTO redaction_terms (company_name, term, term_type, active)
                    VALUES (?, ?, ?, 1)
                    """,
                    (company_name, term, term_type),
                )
            conn.commit()

    def list_redaction_terms(self, company_name: str | None = None) -> list[dict[str, str]]:
        with self._connect() as conn:
            if company_name:
                rows = conn.execute(
                    """
                    SELECT company_name, term, term_type
                    FROM redaction_terms
                    WHERE active = 1 AND lower(company_name) = lower(?)
                    ORDER BY id ASC
                    """,
                    (company_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT company_name, term, term_type
                    FROM redaction_terms
                    WHERE active = 1
                    ORDER BY id ASC
                    """
                ).fetchall()
        return [
            {
                "company_name": str(row["company_name"]),
                "term": str(row["term"]),
                "term_type": str(row["term_type"]),
            }
            for row in rows
        ]

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def list_table_columns(self, table_name: str) -> list[str]:
        safe_table = self._validated_table_name(table_name)
        with self._connect() as conn:
            rows = conn.execute(f"PRAGMA table_info({safe_table})").fetchall()
        return [str(row["name"]) for row in rows]

    def list_table_rows(
        self,
        table_name: str,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        safe_table = self._validated_table_name(table_name)
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        columns = self.list_table_columns(safe_table)
        if not columns:
            return {"table": safe_table, "columns": [], "rows": [], "total": 0}

        text_columns = [col for col in columns if col.lower().endswith(("name", "type", "industry", "language", "classification", "term"))]
        with self._connect() as conn:
            if search and text_columns:
                where_parts = [f"lower(COALESCE({col}, '')) LIKE lower(?)" for col in text_columns]
                where_sql = " OR ".join(where_parts)
                params: list[Any] = [f"%{search}%"] * len(text_columns)
                total_row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {safe_table} WHERE {where_sql}",
                    tuple(params),
                ).fetchone()
                row_items = conn.execute(
                    f"SELECT * FROM {safe_table} WHERE {where_sql} LIMIT ? OFFSET ?",
                    tuple(params + [limit, offset]),
                ).fetchall()
            else:
                total_row = conn.execute(f"SELECT COUNT(*) AS c FROM {safe_table}").fetchone()
                row_items = conn.execute(
                    f"SELECT * FROM {safe_table} LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()

        rows: list[dict[str, Any]] = []
        for row in row_items:
            payload: dict[str, Any] = {}
            for col in columns:
                value = row[col]
                if isinstance(value, (bytes, bytearray)):
                    payload[col] = f"<BLOB:{len(value)} bytes>"
                else:
                    payload[col] = value
            rows.append(payload)
        return {
            "table": safe_table,
            "columns": columns,
            "rows": rows,
            "total": int(total_row["c"] if total_row else 0),
            "limit": limit,
            "offset": offset,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, company_name, doc_type, language, file_name, sha256, classification, created_at
                FROM internal_documents
                ORDER BY id DESC
                """
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "company_name": str(row["company_name"]),
                "doc_type": str(row["doc_type"]),
                "language": str(row["language"]),
                "file_name": str(row["file_name"]),
                "sha256": str(row["sha256"]),
                "classification": str(row["classification"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def get_document_by_id(self, document_id: int) -> InternalDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, company_name, doc_type, language, file_name, pdf_blob, sha256, classification
                FROM internal_documents
                WHERE id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return InternalDocument(
            id=int(row["id"]),
            company_name=str(row["company_name"]),
            doc_type=str(row["doc_type"]),
            language=str(row["language"]),
            file_name=str(row["file_name"]),
            pdf_blob=bytes(row["pdf_blob"]),
            sha256=str(row["sha256"]),
            classification=str(row["classification"]),
        )

    def sqlite_file_path(self) -> str:
        return str(Path(self.db_path).resolve())

    def _validated_table_name(self, table_name: str) -> str:
        known = set(self.list_tables())
        if table_name not in known:
            raise ValueError(f"Unknown table '{table_name}'")
        return table_name
