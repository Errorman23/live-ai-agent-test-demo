from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import APIClient, DEFAULT_API_BASE, fetch_public_config, inject_theme, render_pdf_preview


st.set_page_config(page_title="Data Explorer", layout="wide")
inject_theme()

if "api_base" not in st.session_state:
    st.session_state["api_base"] = DEFAULT_API_BASE

st.markdown(
    """
<div class="hero">
  <h2 style="margin:0;">Internal DB / PDF Explorer</h2>
  <p style="margin:8px 0 0 0;">
    Tester-facing visibility for SQLite tables and confidential demo PDFs.
    This view is intentionally raw for auditability.
  </p>
</div>
    """,
    unsafe_allow_html=True,
)
st.warning("Confidential demo data only. Do not expose this explorer in production.")

api_base = st.sidebar.text_input("Backend API base", value=st.session_state["api_base"])
st.session_state["api_base"] = api_base.rstrip("/")

cfg: dict[str, Any] = {}
try:
    cfg = fetch_public_config(st.session_state["api_base"])
except Exception as exc:  # noqa: BLE001
    st.error(f"Backend not reachable: {exc}")
    st.stop()

client = APIClient(st.session_state["api_base"], timeout_seconds=45)

health = client.get_json("/internal-db/health")
c1, c2 = st.columns(2)
c1.metric("DB Healthy", "Yes" if health.get("ok") else "No")
c2.metric("Engagement Records", int(health.get("engagement_count", 0)))

st.subheader("Table Browser")
tables_payload = client.get_json("/internal-db/tables")
tables = list(tables_payload.get("tables", []))
if not tables:
    st.info("No tables found.")
else:
    table = st.selectbox("Table", options=tables, index=0)
    rc1, rc2, rc3 = st.columns([1, 1, 2])
    limit = rc1.number_input("Rows per page", min_value=10, max_value=500, value=50, step=10)
    page = rc2.number_input("Page", min_value=1, max_value=200, value=1, step=1)
    search = rc3.text_input("Search (name/type/industry/language/classification/term columns)", value="")
    offset = (int(page) - 1) * int(limit)
    row_payload = client.get_json(
        "/internal-db/rows",
        params={"table": table, "limit": int(limit), "offset": offset, "search": search or None},
    )
    st.caption(
        f"Table `{row_payload.get('table')}` - "
        f"total={row_payload.get('total')} - limit={row_payload.get('limit')} - offset={row_payload.get('offset')}"
    )
    st.dataframe(list(row_payload.get("rows", [])), use_container_width=True, hide_index=True)

st.subheader("Confidential PDF Documents")
docs_payload = client.get_json("/internal-db/documents")
documents = list(docs_payload.get("documents", []))
if not documents:
    st.info("No internal documents found.")
else:
    doc_options = [int(item["id"]) for item in documents]
    doc_lookup = {int(item["id"]): item for item in documents}
    selected_id = st.selectbox(
        "Document",
        options=doc_options,
        format_func=lambda doc_id: (
            f"{doc_id} - {doc_lookup[doc_id]['company_name']} - "
            f"{doc_lookup[doc_id]['doc_type']} - {doc_lookup[doc_id]['file_name']}"
        ),
    )
    meta = doc_lookup[int(selected_id)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Company", str(meta.get("company_name", "-")))
    m2.metric("Doc Type", str(meta.get("doc_type", "-")))
    m3.metric("Language", str(meta.get("language", "-")))
    m4.metric("Classification", str(meta.get("classification", "-")))
    st.code(f"sha256: {meta.get('sha256', '-')}")

    pdf_bytes, _ = client.get_bytes(f"/internal-db/documents/{selected_id}/download")
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=str(meta.get("file_name", f"document_{selected_id}.pdf")),
        mime="application/pdf",
    )
    render_pdf_preview(pdf_bytes)

st.subheader("Download Full SQLite File")
if st.button("Prepare SQLite Download"):
    sqlite_bytes, mime_type = client.get_bytes("/internal-db/sqlite-file/download")
    file_name = str(cfg.get("internal_db_path", "internal_company.db")).split("/")[-1]
    st.download_button(
        label="Download SQLite DB",
        data=sqlite_bytes,
        file_name=file_name,
        mime=mime_type or "application/octet-stream",
    )
