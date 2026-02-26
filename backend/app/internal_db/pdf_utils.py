from __future__ import annotations

import hashlib
from io import BytesIO

from pypdf import PdfReader


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def extract_text_from_pdf_blob(blob: bytes) -> str:
    reader = PdfReader(BytesIO(blob))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text.strip())
    return "\n\n".join(chunks).strip()

