from __future__ import annotations

import base64
import json
import math
import os
from typing import Any

import httpx
import streamlit as st

# Shared Streamlit UI utilities for all testing pages.
# Responsibilities:
# - centralized theming/styling
# - small rendering helpers for cards/tables/timelines
# - thin HTTP client wrapper for backend APIs
# Boundaries:
# - page-specific control flow belongs in frontend/testing_ui/pages/*


DEFAULT_API_BASE = os.getenv("AGENT_API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")


def inject_theme() -> None:
    """Inject a single global theme for all Streamlit testing pages."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --bg:#f2f6ff;
  --ink:#0f2238;
  --ink-muted:#53657f;
  --panel:#ffffff;
  --panel-soft:#f8fbff;
  --line:#d3e1f3;
  --line-strong:#a9c3e4;
  --brand:#1f5fae;
  --brand-2:#0f766e;
  --brand-3:#d97706;
  --accent-func:#2463eb;
  --accent-acc:#7c3aed;
  --accent-sec:#c2410c;
  --accent-sim:#0f766e;
  --warn:#a84e1e;
  --danger:#b4232f;
  --shadow:0 14px 30px rgba(16, 41, 74, 0.11);
}
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
h1, h2, h3, h4 {
  font-family:'Space Grotesk', 'Manrope', sans-serif !important;
  letter-spacing: -0.01em;
}
.stApp {
  background:
    radial-gradient(circle at 100% -10%, rgba(36,99,235,0.18), rgba(36,99,235,0) 34%),
    radial-gradient(circle at -6% 8%, rgba(124,58,237,0.14), rgba(124,58,237,0) 30%),
    radial-gradient(circle at 72% 110%, rgba(15,118,110,0.10), rgba(15,118,110,0) 30%),
    linear-gradient(180deg, #f9fbff 0%, #eef4ff 50%, #edf6ff 100%);
  color:var(--ink);
}
[data-testid="block-container"]{
  padding-top:1.2rem;
  max-width: 1320px;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stAppViewContainer"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"]{
  color:var(--ink) !important;
}
[data-testid="stMetricDelta"]{
  color:var(--ink-muted) !important;
}
/* Sidebar and navigation readability */
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 88% -10%, rgba(56,189,248,0.20), rgba(56,189,248,0) 34%),
    radial-gradient(circle at 20% -18%, rgba(34,197,94,0.16), rgba(34,197,94,0) 36%),
    linear-gradient(180deg, #132a4a 0%, #0b1a31 100%);
  border-right:1px solid rgba(255,255,255,0.10);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #dce8fb !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: rgba(248,252,255,0.97) !important;
  color: #111827 !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
  fill: #111827 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
  color: #dce7f6 !important;
}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNav"] p {
  color: #e6edf8 !important;
  opacity: 1 !important;
}
[data-testid="stSidebarNav"] a:hover {
  color: #ffffff !important;
  background: rgba(255,255,255,0.07) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebarNav"] [aria-current="page"] {
  background: linear-gradient(90deg, rgba(56,189,248,0.31), rgba(16,185,129,0.22)) !important;
  border-left: 3px solid rgba(226, 247, 255, 0.95) !important;
  border-radius: 10px !important;
}
/* Hero and section headers */
.hero {
  position: relative;
  border:1px solid var(--line);
  border-radius:18px;
  background:
    linear-gradient(120deg, rgba(31,95,174,0.15), rgba(15,118,110,0.12) 46%, rgba(217,119,6,0.10) 100%),
    var(--panel);
  box-shadow: var(--shadow);
  padding:18px 20px 16px 22px;
  overflow: hidden;
}
.hero::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 6px;
  height: 100%;
  background: linear-gradient(180deg, var(--brand), var(--brand-2), var(--brand-3));
}
.hero h2, .hero h3{
  margin: 0;
}
.hero p{
  margin: 8px 0 0 0;
}
.surface{
  border:1px solid var(--line);
  border-radius:16px;
  padding:14px 16px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,252,255,0.92) 100%);
  box-shadow: 0 6px 16px rgba(16,41,74,0.07);
  margin-top: 12px;
}
.section-head{
  border-radius: 12px;
  border:1px solid var(--line-strong);
  padding: 10px 12px;
  margin: 8px 0 10px 0;
  background: linear-gradient(90deg, rgba(31,95,174,0.10), rgba(15,118,110,0.08));
}
.section-head h4{
  margin:0;
}
.section-head p{
  margin:4px 0 0 0;
  color: var(--ink-muted);
  font-size: 13px;
}
.section-functional{
  background: linear-gradient(90deg, rgba(36,99,235,0.14), rgba(59,130,246,0.05));
}
.section-accuracy{
  background: linear-gradient(90deg, rgba(124,58,237,0.16), rgba(99,102,241,0.06));
}
.section-security{
  background: linear-gradient(90deg, rgba(194,65,12,0.16), rgba(245,158,11,0.05));
}
.section-simulation{
  background: linear-gradient(90deg, rgba(15,118,110,0.16), rgba(20,184,166,0.05));
}
[data-baseweb="tab-list"] {
  gap: 8px;
  margin-top: 12px;
}
[data-baseweb="tab"] {
  border: 1px solid var(--line-strong);
  border-radius: 12px;
  background: #f3f8ff;
  color: var(--ink);
  padding: 8px 14px;
  font-weight: 600;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
  border-color: #8ab0d8;
  box-shadow: 0 6px 14px rgba(31,95,174,0.14);
}
/* KPI cards and button states */
.kpi {
  border:1px solid var(--line);
  background:
    linear-gradient(180deg, #ffffff 0%, var(--panel-soft) 100%);
  box-shadow:0 5px 16px rgba(16,41,74,0.07);
  border-radius:14px;
  padding:12px 14px;
  transition: transform 120ms ease, box-shadow 120ms ease;
  position: relative;
  overflow: hidden;
}
.kpi::after{
  content:"";
  position:absolute;
  top:0;
  right:0;
  width:46px;
  height:4px;
  background: linear-gradient(90deg, var(--brand), var(--brand-2));
  border-bottom-left-radius: 8px;
}
.kpi:hover {
  transform: translateY(-1px);
  box-shadow:0 9px 18px rgba(16,41,74,0.12);
}
[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: 10px 12px;
  background:
    linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  box-shadow: 0 4px 14px rgba(16,41,74,0.06);
}
.stButton > button {
  border-radius:11px;
  border:1px solid #9fb8d7;
  background:linear-gradient(180deg, #ffffff 0%, #edf5ff 100%);
  color:#16325a;
  font-weight:600;
  box-shadow: 0 3px 10px rgba(31,95,174,0.12);
}
.stButton > button:hover {
  border-color:var(--brand);
  color:var(--brand);
  box-shadow:0 0 0 2px rgba(31,95,174,0.10);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #d62d2d 0%, #b51f1f 100%) !important;
  border: 1px solid #921919 !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 18px rgba(182, 32, 32, 0.22) !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(180deg, #e03a3a 0%, #bd2525 100%) !important;
  border-color: #7f1515 !important;
  color: #ffffff !important;
  box-shadow: 0 10px 20px rgba(182, 32, 32, 0.28) !important;
}
.stButton > button[kind="primary"]:focus {
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.65), 0 0 0 6px rgba(182, 32, 32, 0.55) !important;
}
/* Inputs/tables/expanders */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  border:1px solid var(--line-strong) !important;
  border-radius:10px !important;
  background:#ffffff !important;
  color:var(--ink) !important;
}
[data-testid="stDataFrame"] {
  border:1px solid var(--line);
  border-radius:12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(16,41,74,0.06);
  background: #ffffff;
}
[data-testid="stDataFrame"] [role="columnheader"] {
  background: #e9f1ff !important;
  color: #183b68 !important;
  font-weight: 700 !important;
}
[data-testid="stDataFrame"] [role="row"] [role="gridcell"] {
  border-bottom: 1px solid #ebf2fb !important;
}
[data-testid="stExpander"]{
  border:1px solid var(--line);
  border-radius:12px;
  background:linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  box-shadow: 0 4px 12px rgba(16,41,74,0.05);
}
[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--line) !important;
  border-radius: 16px !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,251,255,0.94)) !important;
  box-shadow: 0 8px 18px rgba(16,41,74,0.07);
}
[data-testid="stExpander"] summary {
  color: #14335e !important;
  font-weight: 700 !important;
}
/* Status and timeline utility classes */
.status-chip{
  display:inline-block;
  padding:3px 11px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  letter-spacing:0.02em;
  border: 1px solid transparent;
}
.status-running {
  background: linear-gradient(180deg, #e6f3ff 0%, #d7ebff 100%);
  color:#1b5d7f;
  border-color:#9bc5ea;
}
.status-completed, .status-success {
  background: linear-gradient(180deg, #e8f9f2 0%, #d5f3e5 100%);
  color:#12613f;
  border-color:#9dd7bc;
}
.status-failed, .status-error {
  background: linear-gradient(180deg, #ffecec 0%, #ffdede 100%);
  color:#8c2222;
  border-color:#e7aaaa;
}
.timeline-item{
  border-left:3px solid #4f87be;
  padding:8px 11px;
  margin:6px 0 6px 4px;
  background:linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  border:1px solid #d9e5f2;
  border-radius:10px;
  box-shadow: 0 2px 8px rgba(16,41,74,0.05);
}
.timeline-head{
  display:flex;
  justify-content:space-between;
  gap:8px;
  font-weight:600;
}
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.subtle { color: var(--ink-muted); font-size: 12px; }
a, a:visited { color: var(--brand); text-decoration: none; font-weight: 600; }
a:hover { text-decoration: underline; }
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid var(--line) !important;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border-radius:999px;
  padding:6px 10px;
  font-size:12px;
  font-weight:700;
  letter-spacing:0.01em;
  color:#124873;
  border:1px solid #9bc5ea;
  background:linear-gradient(180deg, #ebf5ff, #dcf0ff);
}
.grid2{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Backend transport helpers used across all testing pages.
# ---------------------------------------------------------------------------
class APIClient:
    def __init__(self, api_base: str, timeout_seconds: float = 60.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.api_base}{path}"

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self._url(path), params=params)
        response.raise_for_status()
        return response.json()

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self._url(path), json=payload)
        response.raise_for_status()
        return response.json()

    def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> tuple[bytes, str]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self._url(path), params=params)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "application/octet-stream")


@st.cache_data(ttl=10, show_spinner=False)
def fetch_public_config(api_base: str) -> dict[str, Any]:
    return APIClient(api_base).get_json("/config/public")


# ---------------------------------------------------------------------------
# Shared rendering primitives for timelines, tool cards, and value formatting.
# ---------------------------------------------------------------------------
def status_chip(status: str) -> str:
    safe = (status or "unknown").lower()
    return (
        f"<span class='status-chip status-{safe}'>"
        f"{safe.upper()}"
        "</span>"
    )


def render_step_timeline(step_events: list[dict[str, Any]]) -> None:
    if not step_events:
        st.info("No steps yet.")
        return
    for step in step_events:
        name = str(step.get("step_name", "unknown"))
        status = str(step.get("status", "unknown"))
        msg = str(step.get("message", "")).strip()
        started = str(step.get("started_at", "")).replace("T", " ").replace("+00:00", "Z")
        st.markdown(
            (
                "<div class='timeline-item'>"
                f"<div class='timeline-head'><span>{name}</span>{status_chip(status)}</div>"
                f"<div class='subtle'>{started}</div>"
                f"<div>{msg or 'No message.'}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_tool_cards(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        st.info("No tool calls yet.")
        return
    for idx, call in enumerate(tool_calls, start=1):
        tool = str(call.get("tool_name", "tool"))
        status = str(call.get("status", "unknown"))
        duration = float(call.get("duration_ms", 0.0))
        details = call.get("details", {}) or {}
        with st.expander(f"{idx}. {tool} - {status.upper()} - {duration:.1f} ms", expanded=False):
            st.caption("Input payload")
            st.json(details.get("input_payload", {}))
            st.caption("Output payload preview")
            st.json(details.get("output_payload_preview", {}))
            metadata = details.get("metadata", {})
            if metadata:
                st.caption("Metadata")
                st.json(metadata)


def render_policy_findings(policy_findings: list[str]) -> None:
    if not policy_findings:
        return
    for finding in policy_findings:
        st.warning(finding)


def render_llm_attempt_cards(trace_payload: dict[str, Any]) -> None:
    spans = trace_payload.get("tool_spans", [])
    llm_spans = [s for s in spans if str(s.get("name", "")).startswith("llm_")]
    if not llm_spans:
        st.info("No LLM attempt spans available yet.")
        return

    for idx, span in enumerate(llm_spans, start=1):
        name = str(span.get("name", "llm_span"))
        status = str(span.get("status", "ok"))
        metadata = span.get("metadata", {}) or {}
        attempt = metadata.get("attempt_index", "-")
        model = metadata.get("model", "-")
        with st.expander(f"{idx}. {name} - attempt {attempt} - {status.upper()}", expanded=False):
            st.markdown(f"**Model:** `{model}`")
            st.json(metadata)


def trace_ui_hint(config: dict[str, Any], trace_id: str | None) -> None:
    host = str(config.get("langfuse_host", "") or "").strip().rstrip("/")
    if not host:
        st.caption("Langfuse host not configured.")
        return
    if trace_id:
        trace_url = f"{host}/trace/{trace_id}"
        st.markdown(
            f"[Open Trace in Langfuse]({trace_url}) - trace id: `{trace_id}`",
            help="Use the trace id in Langfuse search to open this exact run.",
        )
    else:
        st.markdown(f"[Open Langfuse UI]({host})")


def render_pdf_preview(pdf_bytes: bytes, height: int = 620) -> None:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    html = (
        "<iframe "
        "src='data:application/pdf;base64,"
        f"{encoded}"
        f"' width='100%' height='{height}' type='application/pdf'></iframe>"
    )
    st.markdown(html, unsafe_allow_html=True)


def as_json_text(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        return str(payload)


def normalize_tabular_na(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    if isinstance(value, str) and not value.strip():
        return ""
    if isinstance(value, list) and not value:
        return "N/A"
    return value


def section_header(title: str, subtitle: str = "", *, tone: str = "functional") -> None:
    tone_class = {
        "functional": "section-functional",
        "accuracy": "section-accuracy",
        "security": "section-security",
        "simulation": "section-simulation",
    }.get(tone, "section-functional")
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class="section-head {tone_class}">
  <h4>{title}</h4>
  {subtitle_html}
</div>
""",
        unsafe_allow_html=True,
    )
