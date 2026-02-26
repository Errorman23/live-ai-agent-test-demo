from __future__ import annotations

import pandas as pd
import streamlit as st

from common import normalize_tabular_na, render_step_timeline, render_tool_cards, section_header, status_chip
from shared import (
    auto_refresh_controls,
    fetch_history,
    filter_history,
    fmt_float,
    fmt_pct,
    history_table,
    init_page,
    maybe_auto_refresh,
    render_eval_method_badge,
    render_model_strip,
    select_batch,
    start_test,
)

# Security page focuses on Promptfoo-driven adversarial evaluation batches.
# Responsibilities:
# - launch minimal red-team suite
# - surface batch/runtime diagnostics and case-level outcomes
# - expose links to Promptfoo viewer artifacts for selected completed batches


api, cfg = init_page("Security Test")
reasoning_effort = st.sidebar.selectbox("Reasoning effort", ["low", "medium", "high"], index=1)
auto_refresh_enabled, auto_refresh_interval = auto_refresh_controls(key="security")

st.markdown(
    """
<div class="hero">
  <h3 style="margin:0;">Security Test</h3>
  <p style="margin:8px 0 0 0;">
    Promptfoo-first security run (10 cases): leakage, injection, and tool misuse.
    This version uses Promptfoo Red Team <b>Minimal Set</b> plugins for faster iteration.
    This page reports attack success and resilience only.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
render_model_strip(cfg)

with st.expander("Methodology", expanded=False):
    methodology = api.get_json("/evaluators/methodology")
    redteam = methodology.get("security_redteam", {}) if isinstance(methodology, dict) else {}
    st.markdown("**Promptfoo Native Red-Team Execution Path**")
    st.json(
        {
            "enabled_path": redteam.get("enabled_path"),
            "target_endpoint": redteam.get("target_endpoint"),
            "target_output_used_by_plugins": redteam.get("target_output_used_by_plugins"),
            "plugin_set_name": redteam.get("plugin_set_name"),
            "plugins": redteam.get("plugins"),
            "strategies": redteam.get("strategies"),
            "num_tests_per_batch": redteam.get("num_tests_per_batch"),
            "available_profiles": redteam.get("available_profiles"),
        }
    )
    st.markdown("**Plugin Assertion Logic (actual security pass/fail checks)**")
    st.json(redteam.get("plugin_logic", {}))
    st.markdown("**Security Batch Metrics**")
    st.json(redteam.get("batch_metrics", {}))
    st.caption(
        "This page is scored by Promptfoo native red-team plugin assertions. "
        "The deterministic /promptfoo/evaluate security gates are not the active path for this run button."
    )

run_evaluator_mode = "deterministic"
section_header(
    "Run Security Suite",
    "Promptfoo Red Team minimal set with native Promptfoo scoring.",
    tone="security",
)
with st.container(border=True):
    render_eval_method_badge(
        "Promptfoo native Red Team (Minimal Set)",
        "This page runs native Promptfoo redteam generation/eval using the configured minimal plugin set.",
    )

    run_col, disabled_col = st.columns(2)
    if run_col.button("Run Security Minimal Set (Promptfoo Red Team, 10 cases)", width="stretch", type="primary"):
        test_id = start_test(
            api,
            domain="security",
            test_type="security-minimal",
            reasoning_effort=reasoning_effort,
            evaluator_mode=run_evaluator_mode,
        )
        st.session_state["selected_batch_security"] = test_id
        st.session_state["select_security"] = test_id
        st.rerun()

    disabled_col.button(
        "Run Security EU AI Act Pack (Disabled)",
        width="stretch",
        disabled=True,
        help="CLI-only in this demo. See README for command.",
    )

rows = filter_history(fetch_history(api), domain="security")
section_header(
    "Batch Selection",
    "Review historical security batches and select one for runtime and case-level diagnostics.",
    tone="security",
)
with st.container(border=True):
    history_table(rows)
    selected_id = select_batch(rows, key="security")
    if not selected_id:
        st.info("No security batch selected yet.")
        st.stop()

live = dict(api.get_json(f"/tests/{selected_id}/live"))
all_cases = list(live.get("all_cases", []))
payload = dict(api.get_json(f"/tests/{selected_id}"))
summary = payload.get("summary") or {}

promptfoo_summary = payload.get("promptfoo_summary") or {}
asr = None
if promptfoo_summary:
    try:
        asr = 1.0 - float(promptfoo_summary.get("pass_rate", 0.0))
    except Exception:
        asr = None

section_header(
    "Security KPI Dashboard",
    "Leakage, injection, and attack-success signals for completed security cases.",
    tone="security",
)
with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Success Rate", fmt_pct(summary.get("success_rate")))
    k2.metric("Error Rate", fmt_pct(summary.get("runtime_error_rate")))
    k3.metric("Leakage Rate", fmt_pct(summary.get("leakage_rate")))
    k4.metric("Injection Resilience", fmt_pct(summary.get("injection_resilience")))
    k5.metric("Attack Success Rate (ASR)", fmt_pct(asr))
    k6, k7 = st.columns(2)
    k6.metric("Retry Exhaustion", fmt_pct(summary.get("retry_exhaustion_rate")))
    k7.metric("Avg Attempts", fmt_float(summary.get("avg_attempts")))
    st.caption(
        "Denominators - completion: "
        f"{summary.get('completion_denominator', 0)}, quality/completed: {summary.get('quality_denominator', 0)}, "
        f"leakage: {summary.get('leakage_denominator', 0)}, injection: {summary.get('injection_denominator', 0)}"
    )

section_header(
    "Batch Runtime and Promptfoo Output",
    "Track current run progress and open report artifacts for deeper attack analysis.",
    tone="security",
)
with st.container(border=True):
    meta = dict(api.get_json(f"/tests/{selected_id}/promptfoo-meta"))
    promptfoo_result_url = meta.get("promptfoo_result_url")
    if promptfoo_result_url:
        st.markdown(f"[Open Promptfoo View Result]({promptfoo_result_url})")
    if meta.get("promptfoo_report_path"):
        st.caption(f"Report path: `{meta['promptfoo_report_path']}`")
    if meta.get("parser_warning"):
        parser_warning = str(meta["parser_warning"])
        lower_warning = parser_warning.lower()
        if (
            "fallback" in lower_warning
            or "email verification" in lower_warning
            or "non-zero" in lower_warning
            or "version of promptfoo" in lower_warning
        ):
            st.info(parser_warning)
        else:
            st.warning(parser_warning)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Status", str(live.get("status", "-")).upper())
    r2.metric("Progress", fmt_pct(float(live.get("progress", 0.0))))
    r3.metric("Completed", str(live.get("completed_display") or f"{int(live.get('completed_cases', 0))}/{int(live.get('total_cases', 0))}"))
    r4.metric("ETA (s)", fmt_float(live.get("eta_seconds"), 1))
    st.progress(float(live.get("progress", 0.0)))

section_header(
    "Case Inspector",
    "Inspect a specific case to review runtime timeline and tool call behavior.",
    tone="security",
)
case_options = [str(c.get("case_id") or c.get("scenario_id")) for c in all_cases if c.get("case_id") or c.get("scenario_id")]
if case_options:
    case_state_key = "selected_case_security"
    if st.session_state.get(case_state_key) not in case_options:
        st.session_state[case_state_key] = case_options[0]
    chosen_case = st.selectbox(
        "Select case to inspect",
        options=case_options,
        key="security_case_selector",
        index=case_options.index(st.session_state[case_state_key]),
    )
    st.session_state[case_state_key] = chosen_case
    live = dict(api.get_json(f"/tests/{selected_id}/live", params={"selected_case_id": chosen_case}))

current_case = live.get("current_case") or {}
with st.container(border=True):
    if current_case:
        header_cols = st.columns([3, 2, 2])
        header_cols[0].markdown(f"Scenario `{current_case.get('scenario_id', '-')}`")
        header_cols[1].markdown(status_chip(str(current_case.get("status", "running"))), unsafe_allow_html=True)
        if current_case.get("trace_url"):
            header_cols[2].markdown(f"[Open Trace]({current_case['trace_url']})")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Step Timeline")
            render_step_timeline(list(current_case.get("step_events", [])))
        with right:
            st.markdown("#### Tool Input/Output")
            render_tool_cards(list(current_case.get("tool_call_records", [])))
    else:
        st.caption("No case details available yet.")

section_header(
    "Case Results",
    "Promptfoo security case outcomes with pass/fail reasons and trace links.",
    tone="security",
)
with st.container(border=True):
    promptfoo_cases = payload.get("promptfoo_case_results") or []
    if promptfoo_cases:
        pdf = pd.DataFrame(promptfoo_cases)
        keep_cols = [
            "scenario_id",
            "category",
            "task_type",
            "company",
            "risk_tier",
            "execution_status",
            "passed",
            "reason",
            "latency_ms",
            "run_id",
            "langfuse_trace_url",
        ]
        existing = [c for c in keep_cols if c in pdf.columns]
        pdf = pdf[existing].copy()
        for col in pdf.columns:
            pdf[col] = pdf[col].map(normalize_tabular_na)
        st.dataframe(
            pdf,
            width="stretch",
            hide_index=True,
            column_config={
                "task_type": st.column_config.TextColumn("Recognized Route"),
                "langfuse_trace_url": st.column_config.LinkColumn("Trace", display_text="Open Trace"),
            },
        )
    else:
        st.caption("No parsed Promptfoo case rows for this batch.")

maybe_auto_refresh(
    enabled=auto_refresh_enabled,
    interval_seconds=auto_refresh_interval,
    run_status=str(live.get("status") or payload.get("status") or ""),
    key="security",
)
