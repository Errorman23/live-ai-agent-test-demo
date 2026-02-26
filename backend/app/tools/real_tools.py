from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import Settings
from app.internal_db import InternalDBRepository, extract_text_from_pdf_blob

# Runtime tool implementations invoked by graph nodes.
# Responsibilities:
# - internal DB and PDF retrieval with sanitized payload previews
# - real web retrieval (Tavily) with deterministic query strategy
# - template composition and final security filtering helpers
# Boundaries:
# - tool call sequencing is controlled by planner/routes in graph/nodes.py


@dataclass
class ToolResult:
    data: dict[str, Any]
    duration_ms: float


def _hash(data: Any) -> str:
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()[:12]


def _compile_budget_pattern() -> re.Pattern[str]:
    return re.compile(r"(?:USD|US\\$|\\$)\\s?\\d[\\d,]*(?:\\.\\d+)?", re.IGNORECASE)


def _strip_control_chars(text: str) -> str:
    # Keep common whitespace while removing control bytes that can break JSON/rendering.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return cleaned.replace("\u2028", " ").replace("\u2029", " ")


_PARTNERSHIP_HINTS = (
    "partner",
    "partnership",
    "collaboration",
    "alliance",
    "joint venture",
    "agreement",
)

_PRODUCT_HINTS = (
    "product",
    "platform",
    "service",
    "feature",
    "device",
    "model",
    "offering",
)


def _contains_any_hint(text: str, hints: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in hints)


# Tool collection exposed to planner steps.
class RealToolbox:
    def __init__(self, settings: Settings, repository: InternalDBRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._budget_pattern = _compile_budget_pattern()
        self._jinja = Environment(
            loader=FileSystemLoader(self.settings.templates_dir),
            autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_company_info(self, company_name: str) -> ToolResult:
        started = perf_counter()
        record = self.repository.get_engagement(company_name)
        if record is None:
            return ToolResult(
                data={
                    "record_found": False,
                    "company_name": company_name,
                    "industry": "Unknown",
                    "public_products": [],
                    "public_partnerships": [],
                    "project_risk_level": "unknown",
                    "internal_summary": "No internal relationship record found.",
                },
                duration_ms=(perf_counter() - started) * 1000,
            )

        return ToolResult(
            data={
                "record_found": True,
                "company_name": record.company_name,
                "industry": record.industry,
                "public_products": list(record.public_products),
                "public_partnerships": list(record.public_partnerships),
                "project_risk_level": record.project_risk_level,
                "internal_summary": (
                    "Internal engagement exists. Project name: [REDACTED]. "
                    "Budget: [REDACTED]. Internal product line: [REDACTED]."
                ),
            },
            duration_ms=(perf_counter() - started) * 1000,
        )

    # Deterministic multi-query retrieval for products + partnerships coverage.
    def search_public_web(self, company_name: str) -> ToolResult:
        started = perf_counter()
        queries = [
            f"{company_name} products partnerships latest news",
            f"{company_name} official site products",
            f"{company_name} wikipedia partnerships",
        ]
        snippets: list[dict[str, str]] = []
        query_attempts: list[dict[str, Any]] = []
        dedupe: set[str] = set()
        source_links: list[str] = []
        product_candidates: list[str] = []
        partnership_candidates: list[str] = []
        answer_preview = ""
        for idx, query in enumerate(queries, start=1):
            raw_count = 0
            added_count = 0
            error = None
            try:
                tavily_payload = self._tavily_search(query)
                if not answer_preview:
                    answer_preview = str(tavily_payload.get("answer", "")).strip()
                results = tavily_payload.get("results", [])
                if isinstance(results, list):
                    raw_count = len(results)
                    for row in results:
                        if not isinstance(row, dict):
                            continue
                        title = _strip_control_chars(str(row.get("title", "")).strip())
                        body = _strip_control_chars(
                            str(row.get("content") or row.get("snippet") or "").strip()
                        )
                        href = str(row.get("url", "")).strip()
                        if not (title or body):
                            continue
                        key = href.lower() or f"{title.lower()}|{body[:80].lower()}"
                        if key in dedupe:
                            continue
                        dedupe.add(key)
                        snippets.append({"title": title, "snippet": body, "url": href})
                        if href and href not in source_links:
                            source_links.append(href)
                        merged = f"{title} {body}".strip()
                        if merged:
                            if _contains_any_hint(merged, _PRODUCT_HINTS):
                                product_candidates.append(merged[:220])
                            if _contains_any_hint(merged, _PARTNERSHIP_HINTS):
                                partnership_candidates.append(merged[:220])
                        added_count += 1
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            query_attempts.append(
                {
                    "attempt_index": idx,
                    "query": query,
                    "raw_result_count": raw_count,
                    "added_result_count": added_count,
                    "error": error,
                }
            )

        success = len(snippets) > 0
        unique_products = list(dict.fromkeys(product_candidates))[:8]
        unique_partnerships = list(dict.fromkeys(partnership_candidates))[:8]
        unique_links = list(dict.fromkeys(source_links))[:12]
        payload: dict[str, Any] = {
            "company_name": company_name,
            "results": snippets,
            "answer_preview": answer_preview,
            "query_attempts": query_attempts,
            "result_count": len(snippets),
            "dedupe_count": len(dedupe),
            "search_success": success,
            "public_products_candidates": unique_products,
            "public_partnership_candidates": unique_partnerships,
            "source_links": unique_links,
        }
        if not success:
            payload["error"] = (
                "Web search produced no usable public results after deterministic retries. "
                "Treat as web retrieval failure."
            )

        return ToolResult(
            data=payload,
            duration_ms=(perf_counter() - started) * 1000,
        )

    def _tavily_search(self, query: str) -> dict[str, Any]:
        api_key = (self.settings.tavily_api_key or "").strip()
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")

        request_payload: dict[str, Any] = {
            "query": query,
            "topic": self.settings.tavily_topic,
            "search_depth": self.settings.tavily_search_depth,
            "max_results": self.settings.tavily_max_results,
            "include_answer": True,
            "include_raw_content": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                self.settings.tavily_search_url,
                headers=headers,
                json=request_payload,
            )

        if response.status_code >= 400:
            body = response.text.strip()
            short_body = body[:260] + ("..." if len(body) > 260 else "")
            raise RuntimeError(f"Tavily HTTP {response.status_code}: {short_body}")

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Tavily response was not a JSON object")
        return payload

    # Internal document retrieval always returns sanitized text payloads.
    def retrieve_internal_pdf(self, company_name: str, doc_type: str) -> ToolResult:
        started = perf_counter()
        doc = self.repository.get_document(company_name, doc_type)
        if doc is None:
            return ToolResult(
                data={
                    "document_found": False,
                    "company_name": company_name,
                    "doc_type": doc_type,
                    "language": "unknown",
                    "file_name": "",
                    "classification": "unknown",
                    "sanitized_text": "",
                    "policy_note": "No internal document found for requested type.",
                },
                duration_ms=(perf_counter() - started) * 1000,
            )

        raw_text = extract_text_from_pdf_blob(doc.pdf_blob)
        if not raw_text.strip():
            return ToolResult(
                data={
                    "document_found": True,
                    "company_name": company_name,
                    "doc_type": doc_type,
                    "language": doc.language,
                    "file_name": doc.file_name,
                    "classification": doc.classification,
                    "sanitized_text": "",
                    "policy_note": (
                        "Document exists but contains no extractable text. "
                        "Scanned-image OCR is unsupported in this demo."
                    ),
                },
                duration_ms=(perf_counter() - started) * 1000,
            )

        sanitized_text = self._sanitize_confidential_text(raw_text, company_name)
        return ToolResult(
            data={
                "document_found": True,
                "company_name": company_name,
                "doc_type": doc_type,
                "language": doc.language,
                "file_name": doc.file_name,
                "classification": doc.classification,
                "sanitized_text": sanitized_text,
                "policy_note": "Confidential details were removed before model usage.",
            },
            duration_ms=(perf_counter() - started) * 1000,
        )

    def generate_document(self, template_name: str, content_dict: dict[str, Any]) -> ToolResult:
        started = perf_counter()
        template = self._jinja.get_template(template_name)
        rendered = _strip_control_chars(template.render(**content_dict)).strip()
        return ToolResult(
            data={
                "document": rendered,
                "template_name": template_name,
                "document_hash": _hash(rendered),
            },
            duration_ms=(perf_counter() - started) * 1000,
        )

    def security_filter(self, document: str, company_name: str | None = None) -> ToolResult:
        started = perf_counter()
        redacted = _strip_control_chars(document)
        terms = self.repository.list_redaction_terms(company_name)
        leaked_terms: list[str] = []

        for term_row in terms:
            term = term_row["term"]
            if not term:
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(redacted):
                leaked_terms.append(term)
                redacted = pattern.sub("[REDACTED]", redacted)

        budget_hits = self._budget_pattern.findall(redacted)
        if budget_hits:
            leaked_terms.extend(budget_hits)
            redacted = self._budget_pattern.sub("[REDACTED]", redacted)

        leaked_terms = sorted(set(leaked_terms), key=str.lower)
        return ToolResult(
            data={
                "document": redacted,
                "pass_fail": len(leaked_terms) == 0,
                "redactions_applied": len(leaked_terms),
                "leaked_terms": leaked_terms,
                "document_hash": _hash(redacted),
            },
            duration_ms=(perf_counter() - started) * 1000,
        )

    def _sanitize_confidential_text(self, text: str, company_name: str) -> str:
        sanitized = _strip_control_chars(text)
        for term_row in self.repository.list_redaction_terms(company_name):
            term = term_row["term"]
            if not term:
                continue
            sanitized = re.compile(re.escape(term), re.IGNORECASE).sub("[REDACTED]", sanitized)
        sanitized = self._budget_pattern.sub("[REDACTED]", sanitized)
        lines = [line.strip() for line in sanitized.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        normalized_lines: list[str] = []
        prior_blank = False
        for line in lines:
            if not line:
                if not prior_blank:
                    normalized_lines.append("")
                prior_blank = True
                continue
            normalized_lines.append(line)
            prior_blank = False
        return "\n".join(normalized_lines).strip()

    @staticmethod
    def digest_args(args: dict[str, Any]) -> str:
        return _hash(json.dumps(args, sort_keys=True))


def ensure_template_exists(templates_dir: str, template_name: str) -> None:
    path = Path(templates_dir) / template_name
    if not path.exists():
        raise FileNotFoundError(f"Template '{template_name}' not found at {path}")
