from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import APIClient, DEFAULT_API_BASE, fetch_public_config, inject_theme, section_header  # noqa: E402

# Testing dashboard landing page.
# Responsibilities:
# - summarize platform/runtime status and model configuration
# - link users to domain-specific testing pages
# - display high-level architecture aids (including graph image)


st.set_page_config(page_title="Testing Dashboard", layout="wide")
inject_theme()

if "api_base" not in st.session_state:
    st.session_state["api_base"] = DEFAULT_API_BASE

api_base = st.sidebar.text_input("Backend API base", value=st.session_state["api_base"])
st.session_state["api_base"] = api_base.rstrip("/")

cfg_error = None
cfg: dict[str, Any] = {}
api = APIClient(st.session_state["api_base"], timeout_seconds=30)
try:
    cfg = dict(fetch_public_config(st.session_state["api_base"]))
except Exception as exc:  # noqa: BLE001
    cfg_error = str(exc)

st.markdown(
    """
<div class="hero">
  <h2 style="margin:0;">Agent Safety Test Control Center</h2>
  <p style="margin:8px 0 0 0;">
    Structured testing workspace for functional completion, factual/translation accuracy, security red teaming,
    and multilingual simulation robustness.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if cfg_error:
    st.error(f"Backend not reachable: {cfg_error}")
    st.stop()

catalog_error = None
catalog: dict[str, Any] = {}
try:
    catalog = dict(api.get_json("/tests/catalog"))
except Exception as exc:  # noqa: BLE001
    catalog_error = str(exc)

# Home section order is intentional:
# Start Here -> Platform Health -> Agent Graph -> Evaluation Stack -> Catalog/Links.
section_header(
    "Start Here",
    "Choose a domain page based on the capability you want to validate.",
    tone="functional",
)
start_cards = [
    (
        "Functional Test",
        "Validate task completion, route correctness, and artifact delivery.",
        "Use sidebar page: Functional Test",
    ),
    (
        "Accuracy Test",
        "Score factual grounding, translation faithfulness, and document structure quality.",
        "Use sidebar page: Accuracy Test",
    ),
    (
        "Security Test",
        "Run Promptfoo red-team minimal set for leakage, injection, and tool misuse.",
        "Use sidebar page: Security Test",
    ),
    (
        "Simulation Test",
        "Stress multilingual and instruction-style variants with route + output checks.",
        "Use sidebar page: Simulation Test",
    ),
]
with st.container(border=True):
    start_cols = st.columns(4)
    for idx, (title, desc, hint) in enumerate(start_cards):
        with start_cols[idx]:
            st.markdown(
                f"""
<div class="kpi" style="min-height:150px;">
  <div style="font-weight:700;">{title}</div>
  <div class="subtle" style="margin-top:8px;">{desc}</div>
  <div class="mono" style="margin-top:12px;">{hint}</div>
</div>
""",
                unsafe_allow_html=True,
            )

section_header(
    "Platform Health",
    "Runtime status and model stack for backend, observability, and evaluation services.",
    tone="accuracy",
)
with st.container(border=True):
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Cases / Run", int(cfg.get("default_cases_per_run", 10)))
    s2.metric("Evaluator Modes", len(cfg.get("supported_evaluator_modes", [])))
    s3.metric("Promptfoo Port", int(cfg.get("promptfoo_port", 15500)))
    s4.metric("SiliconFlow", "Enabled" if cfg.get("siliconflow_enabled") else "Disabled")
    s5.metric("Langfuse", "Enabled" if cfg.get("langfuse_enabled") else "Disabled")

    m1, m2 = st.columns(2)
    m1.markdown(
        f"""
<div class="kpi">
  <div style="font-weight:700;">Agent Core Model</div>
  <div class="mono" style="margin-top:6px;">{cfg.get("together_model", "-")}</div>
  <div class="subtle" style="margin-top:6px;">Context window: {cfg.get("together_model_context_window", "unknown")}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"""
<div class="kpi">
  <div style="font-weight:700;">Judge Model</div>
  <div class="mono" style="margin-top:6px;">{cfg.get("llm_judge_model", "-")}</div>
  <div class="subtle" style="margin-top:6px;">Context window: {cfg.get("llm_judge_context_window", "unknown")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

section_header(
    "Agent Graph",
    "Compiled LangGraph runtime map used by the backend orchestration engine.",
    tone="simulation",
)
with st.container(border=True):
    api_root = st.session_state["api_base"].rstrip("/")
    graph_png_url = f"{api_root}/graph/langgraph.png"
    st.image(graph_png_url, caption="Compiled LangGraph workflow", use_container_width=True)

section_header(
    "Evaluation Stack",
    "How this test platform combines deterministic checks, judge scoring, and red teaming.",
    tone="security",
)
stack_items = [
    ("Deterministic Checks", "Route checks, artifact checks, threshold-based scoring."),
    ("LLM-as-Judge", "Factual scoring and reasoning-aware verdicts with explicit rubric."),
    ("Promptfoo Red Team", "Batch adversarial execution and per-case findings."),
]
with st.container(border=True):
    stack_cols = st.columns(3)
    for idx, (title, desc) in enumerate(stack_items):
        with stack_cols[idx]:
            st.markdown(
                f"""
<div class="kpi" style="min-height:110px;">
  <div style="font-weight:700;">{title}</div>
  <div class="subtle" style="margin-top:8px;">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )

section_header(
    "Domain Catalog and Links",
    "Available test domains plus direct navigation to runtime tools and dashboards.",
    tone="functional",
)
with st.container(border=True):
    if catalog_error:
        st.warning(f"Catalog unavailable: {catalog_error}")
    else:
        domains = catalog.get("domains", [])
        for row in domains:
            if not isinstance(row, dict):
                continue
            st.markdown(
                f"""
<div class="kpi">
  <div style="display:flex;justify-content:space-between;gap:8px;">
    <b>{row.get("title", row.get("id", "Domain"))}</b>
    <span class="mono">{row.get("evaluator", "-")}</span>
  </div>
  <div class="subtle" style="margin-top:4px;">{row.get("description", "")}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("### Quick Links")
    links = [
        ("Backend API Docs", f"{st.session_state['api_base'].replace('/api/v1', '')}/docs"),
        ("Chainlit Chat UI", cfg.get("chainlit_ui_url") or "http://127.0.0.1:8501"),
        ("Testing UI", cfg.get("testing_ui_url") or "http://127.0.0.1:8502"),
        ("Promptfoo UI", f"http://127.0.0.1:{cfg.get('promptfoo_port', 15500)}"),
        ("Langfuse UI", cfg.get("langfuse_host") or "http://127.0.0.1:3000"),
    ]
    for label, url in links:
        st.markdown(f"- [{label}]({url})")
