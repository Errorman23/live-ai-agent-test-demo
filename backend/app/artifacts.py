from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fpdf import FPDF
from fpdf.errors import FPDFException

# Artifact rendering/persistence utilities.
# Responsibilities:
# - render briefing/internal-document PDFs with deterministic styling
# - write artifacts to backend/data/artifacts and return metadata pointers
# - keep confidential placeholders explicit in generated outputs
# Boundaries:
# - artifact selection decisions are made by graph nodes, not here


def _canonical_language(value: str | None) -> str:
    if not value:
        return "English"
    lower = value.strip().lower()
    mapping = {
        "english": "English",
        "en": "English",
        "german": "German",
        "de": "German",
        "deutsch": "German",
        "chinese": "Chinese",
        "zh": "Chinese",
        "中文": "Chinese",
        "japanese": "Japanese",
        "ja": "Japanese",
    }
    return mapping.get(lower, value.strip().title())


def _brief_pdf_labels(target_language: str) -> dict[str, str]:
    canonical = _canonical_language(target_language)
    if canonical == "German":
        return {
            "title": "Beratungs-Briefing",
            "subtitle": "Professionelles Briefing (bereinigte Ansicht)",
            "meta_company": "Unternehmen",
            "meta_generated": "Erstellt am",
            "meta_classification": "Klassifikation",
            "classification_value": "Beratersicher (geschwärzt)",
            "footer": "Beratersicheres Briefing",
        }
    if canonical == "Chinese":
        return {
            "title": "顾问简报",
            "subtitle": "专业简报（脱敏视图）",
            "meta_company": "公司",
            "meta_generated": "生成日期",
            "meta_classification": "密级",
            "classification_value": "顾问安全版（已脱敏）",
            "footer": "顾问安全简报",
        }
    return {
        "title": "Consultant Briefing Note",
        "subtitle": "Professional consultant briefing (sanitized view)",
        "meta_company": "Company",
        "meta_generated": "Generated",
        "meta_classification": "Classification",
        "classification_value": "Consultant-safe (redacted)",
        "footer": "Consultant-Safe Briefing",
    }


def ensure_artifacts_dir(root_dir: str) -> Path:
    path = Path(root_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Generic text artifact writer used for markdown/debug outputs.
def save_text_artifact(*, root_dir: str, run_id: str, kind: str, text: str, suffix: str = "md") -> dict[str, str]:
    root = ensure_artifacts_dir(root_dir)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{kind}-{uuid4().hex[:12]}"
    filename = f"{artifact_id}.{suffix}"
    path = run_dir / filename
    path.write_text(text, encoding="utf-8")
    return {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "kind": kind,
        "filename": filename,
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class _StyledPDF(FPDF):
    def __init__(self, footer_label: str) -> None:
        super().__init__()
        self._footer_label = footer_label
        self._footer_use_unicode = any(ord(ch) > 127 for ch in footer_label)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_text_color(120, 126, 136)
        if self._footer_use_unicode:
            self.set_font("Unicode", size=8)
        else:
            self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"{self._footer_label} | Page {self.page_no()}/{{nb}}", align="C")


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


def _normalized_text(text: str) -> str:
    return (
        text.replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\t", "    ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _register_unicode_font(pdf: FPDF, *, requires_unicode: bool) -> bool:
    if not requires_unicode:
        return False
    unicode_font = _pick_unicode_font()
    if unicode_font is None:
        return False
    try:
        pdf.add_font("Unicode", style="", fname=unicode_font)
        return True
    except Exception:
        return False


def _set_font(pdf: FPDF, *, use_unicode: bool, size: int, bold: bool = False, italic: bool = False) -> None:
    if use_unicode:
        # Unicode fallback font has no style variants in this demo.
        pdf.set_font("Unicode", size=size)
        return
    style = ""
    if bold:
        style += "B"
    if italic:
        style += "I"
    pdf.set_font("Helvetica", style, size)


def _safe_multicell(pdf: FPDF, *, width: float, line_height: float, text: str) -> None:
    content = text if text.strip() else " "
    try:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(width, line_height, content, wrapmode="CHAR")
    except FPDFException:
        safe = content.encode("ascii", "replace").decode("ascii")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(width, line_height, safe, wrapmode="CHAR")


def _encode_pdf(pdf: FPDF) -> bytes:
    raw = pdf.output()
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, bytes):
        return raw
    return raw.encode("latin-1")


# Typed PDF renderers (briefing vs. sanitized internal docs).
def _render_briefing_pdf_bytes(*, company_name: str, text: str, target_language: str) -> bytes:
    labels = _brief_pdf_labels(target_language)
    title = labels["title"]
    normalized = _normalized_text(text)
    pdf = _StyledPDF(labels["footer"])
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    requires_unicode = any(ord(ch) > 127 for ch in normalized) or any(
        ord(ch) > 127 for ch in " ".join(labels.values())
    )
    use_unicode = _register_unicode_font(pdf, requires_unicode=requires_unicode)
    pdf._footer_use_unicode = bool(use_unicode and any(ord(ch) > 127 for ch in labels["footer"]))
    line_width = max(float(pdf.w - pdf.l_margin - pdf.r_margin), 20.0)

    # Corporate-neutral title band.
    pdf.set_fill_color(31, 52, 74)
    pdf.rect(pdf.l_margin, 12, line_width, 24, style="F")
    pdf.set_text_color(255, 255, 255)
    _set_font(pdf, use_unicode=use_unicode, size=16, bold=True)
    pdf.set_xy(pdf.l_margin + 3, 16)
    pdf.cell(line_width - 6, 6, title)
    pdf.set_xy(pdf.l_margin + 3, 24)
    _set_font(pdf, use_unicode=use_unicode, size=10, italic=True)
    pdf.cell(line_width - 6, 5, labels["subtitle"])

    # Metadata stripe.
    meta_y = 40
    pdf.set_fill_color(238, 241, 245)
    pdf.rect(pdf.l_margin, meta_y, line_width, 12, style="F")
    pdf.set_text_color(48, 56, 68)
    _set_font(pdf, use_unicode=use_unicode, size=9, bold=True)
    pdf.set_xy(pdf.l_margin + 3, meta_y + 2)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pdf.cell(
        line_width - 6,
        5,
        (
            f"{labels['meta_company']}: {company_name}   |   "
            f"{labels['meta_generated']}: {generated}   |   "
            f"{labels['meta_classification']}: {labels['classification_value']}"
        ),
    )

    pdf.set_y(meta_y + 18)
    pdf.set_text_color(26, 34, 46)

    # Render markdown-like structure.
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            pdf.ln(2)
            continue
        if line.startswith("# "):
            # Main title is already rendered in banner.
            continue
        if line.startswith("## "):
            _set_font(pdf, use_unicode=use_unicode, size=12, bold=True)
            pdf.set_text_color(33, 67, 107)
            _safe_multicell(pdf, width=line_width, line_height=7, text=line[3:].strip())
            pdf.set_draw_color(213, 220, 230)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(1)
            pdf.set_text_color(26, 34, 46)
            continue
        if line.startswith("- "):
            _set_font(pdf, use_unicode=use_unicode, size=10)
            _safe_multicell(pdf, width=line_width, line_height=6, text=f"- {line[2:].strip()}")
            continue
        _set_font(pdf, use_unicode=use_unicode, size=10)
        _safe_multicell(pdf, width=line_width, line_height=6, text=line)

    return _encode_pdf(pdf)


def _render_internal_doc_pdf_bytes(
    *,
    company_name: str,
    doc_type: str,
    source_language: str,
    original_file_name: str,
    classification: str,
    text: str,
) -> bytes:
    normalized = _normalized_text(text)
    pdf = _StyledPDF("Sanitized Internal Document")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    use_unicode = _register_unicode_font(pdf, requires_unicode=any(ord(ch) > 127 for ch in normalized))
    pdf._footer_use_unicode = False
    line_width = max(float(pdf.w - pdf.l_margin - pdf.r_margin), 20.0)

    pdf.set_fill_color(39, 46, 56)
    pdf.rect(pdf.l_margin, 12, line_width, 22, style="F")
    pdf.set_text_color(255, 255, 255)
    _set_font(pdf, use_unicode=False, size=14, bold=True)
    pdf.set_xy(pdf.l_margin + 3, 17)
    pdf.cell(line_width - 6, 6, f"Sanitized Copy - {doc_type.title()} Document")
    _set_font(pdf, use_unicode=False, size=9, italic=True)
    pdf.set_xy(pdf.l_margin + 3, 25)
    pdf.cell(line_width - 6, 5, "Sensitive values replaced with [REDACTED]")

    pdf.set_y(40)
    pdf.set_text_color(45, 54, 66)
    _set_font(pdf, use_unicode=False, size=9, bold=True)
    meta_lines = [
        f"Company: {company_name}",
        f"Document Type: {doc_type}",
        f"Source Language: {source_language}",
        f"Original File: {original_file_name}",
        f"Classification: {classification}",
    ]
    for row in meta_lines:
        _safe_multicell(pdf, width=line_width, line_height=5, text=row)

    pdf.ln(1)
    pdf.set_draw_color(213, 220, 230)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(2)

    pdf.set_text_color(33, 67, 107)
    _set_font(pdf, use_unicode=False, size=11, bold=True)
    _safe_multicell(pdf, width=line_width, line_height=6, text="Sanitized Content (source language preserved)")
    pdf.ln(1)

    pdf.set_text_color(26, 34, 46)
    _set_font(pdf, use_unicode=use_unicode, size=10)
    if not normalized:
        _safe_multicell(pdf, width=line_width, line_height=6, text="No extractable text was available.")
    else:
        for raw_line in normalized.split("\n"):
            _safe_multicell(pdf, width=line_width, line_height=6, text=raw_line.strip() if raw_line.strip() else " ")

    return _encode_pdf(pdf)


# Shared persistence wrapper used by the typed PDF artifact helpers.
def _save_pdf_payload(
    *,
    root_dir: str,
    run_id: str,
    kind: str,
    pdf_bytes: bytes,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = ensure_artifacts_dir(root_dir)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{kind}-{uuid4().hex[:12]}"
    filename = f"{artifact_id}.pdf"
    path = run_dir / filename
    path.write_bytes(pdf_bytes)
    return {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "kind": kind,
        "filename": filename,
        "path": str(path),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


def save_briefing_pdf_artifact(
    *,
    root_dir: str,
    run_id: str,
    company_name: str,
    text: str,
    target_language: str = "English",
) -> dict[str, Any]:
    pdf_bytes = _render_briefing_pdf_bytes(company_name=company_name, text=text, target_language=target_language)
    return _save_pdf_payload(
        root_dir=root_dir,
        run_id=run_id,
        kind="briefing-pdf",
        pdf_bytes=pdf_bytes,
        metadata={
            "renderer": "briefing_corporate_v1",
            "company_name": company_name,
            "target_language": _canonical_language(target_language),
        },
    )


def save_internal_doc_pdf_artifact(
    *,
    root_dir: str,
    run_id: str,
    company_name: str,
    doc_type: str,
    source_language: str,
    original_file_name: str,
    text: str,
    classification: str = "confidential",
) -> dict[str, Any]:
    kind = f"sanitized-{doc_type}-pdf"
    pdf_bytes = _render_internal_doc_pdf_bytes(
        company_name=company_name,
        doc_type=doc_type,
        source_language=source_language,
        original_file_name=original_file_name,
        classification=classification,
        text=text,
    )
    return _save_pdf_payload(
        root_dir=root_dir,
        run_id=run_id,
        kind=kind,
        pdf_bytes=pdf_bytes,
        metadata={
            "renderer": "internal_sanitized_v1",
            "company_name": company_name,
            "doc_type": doc_type,
            "source_language": source_language,
            "original_file_name": original_file_name,
            "classification": classification,
        },
    )
