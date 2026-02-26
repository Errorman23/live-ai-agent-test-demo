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

# Simulation page stresses robustness across language/style/input variations.
# It reuses factual checks but reports language compliance and route robustness KPIs.


api, cfg = init_page("Simulation Test")
reasoning_effort = st.sidebar.selectbox("Reasoning effort", ["low", "medium", "high"], index=0)
auto_refresh_enabled, auto_refresh_interval = auto_refresh_controls(key="simulation")

st.markdown(
    """
<div class="hero">
  <h3 style="margin:0;">Simulation Test</h3>
  <p style="margin:8px 0 0 0;">
    10-case multilingual/style stress set (English, Chinese, German, Japanese, mixed prompts)
    with language compliance and robustness scoring.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
render_model_strip(cfg)

with st.expander("Methodology", expanded=False):
    methodology = api.get_json("/evaluators/methodology")
    det = methodology.get("deterministic", {}) if isinstance(methodology, dict) else {}
    judge = methodology.get("llm_judge", {}) if isinstance(methodology, dict) else {}
    sim_rules = det.get("simulation_rules", {}) if isinstance(det, dict) else {}
    st.markdown("**Simulation Rules**")
    st.json(sim_rules)
    st.markdown("**Factual Judge Prompt Template**")
    st.code(str(judge.get("system_template", "")), language="text")
    st.code(str(judge.get("user_template", "")), language="text")
    st.markdown("**Batch Pass/Fail Gates (Simulation domain)**")
    st.json(
        {
            "base_gate": "agent_status == success AND final_text_length > 0",
            "language_gate": sim_rules.get("language_compliance"),
            "factual_gate": sim_rules.get("factual_rule"),
            "intent_route_gate": sim_rules.get("intent_route_rule"),
            "robustness_ratio": sim_rules.get("robustness_ratio"),
        }
    )

run_evaluator_mode = "llm_judge"
section_header(
    "Run Simulation Suite",
    "Execute multilingual and instruction-style robustness tests across 10 curated scenarios.",
    tone="simulation",
)
with st.container(border=True):
    render_eval_method_badge(
        "Hybrid (LLM judge + deterministic simulation checks)",
        "This page applies LLM-judge factual scoring plus deterministic language/style compliance checks.",
    )
    if st.button("Run Simulation Suite (10 cases)", width="stretch", type="primary"):
        test_id = start_test(
            api,
            domain="simulation",
            test_type="simulation",
            reasoning_effort=reasoning_effort,
            evaluator_mode=run_evaluator_mode,
        )
        st.session_state["selected_batch_simulation"] = test_id
        st.session_state["select_simulation"] = test_id
        st.rerun()

rows = filter_history(fetch_history(api), domain="simulation")
section_header(
    "Batch Selection",
    "Choose a simulation batch to inspect runtime behavior and per-case robustness outcomes.",
    tone="simulation",
)
with st.container(border=True):
    history_table(rows)
    selected_id = select_batch(rows, key="simulation")
    if not selected_id:
        st.info("No simulation batch selected yet.")
        st.stop()

live = dict(api.get_json(f"/tests/{selected_id}/live"))
all_cases = list(live.get("all_cases", []))
payload = dict(api.get_json(f"/tests/{selected_id}"))
summary = payload.get("summary") or {}
cases = list(payload.get("cases", []))

section_header(
    "Simulation KPI Dashboard",
    "Track multilingual compliance, robustness ratio, factual quality, and route intent accuracy.",
    tone="simulation",
)
with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Success Rate", fmt_pct(summary.get("success_rate")))
    k2.metric("Error Rate", fmt_pct(summary.get("runtime_error_rate")))
    k3.metric("Language Compliance", fmt_pct(summary.get("simulation_language_compliance")))
    k4.metric("Robustness Ratio", fmt_float(summary.get("simulation_robustness_ratio"), 2))
    k5.metric("Factual Avg (1-5)", fmt_float(summary.get("factual_score_avg_1_5"), 2))
    st.metric("Intent Route Accuracy", fmt_pct(summary.get("intent_route_accuracy")))
    st.caption(
        "Denominators - completion: "
        f"{summary.get('completion_denominator', 0)}, quality/completed: {summary.get('quality_denominator', 0)}, "
        f"simulation: {summary.get('simulation_denominator', 0)}, factual: {summary.get('factual_denominator', 0)}"
    )

section_header(
    "Batch Runtime",
    "Monitor current simulation run progress and review Promptfoo report references.",
    tone="simulation",
)
with st.container(border=True):
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Status", str(live.get("status", "-")).upper())
    r2.metric("Progress", fmt_pct(float(live.get("progress", 0.0))))
    r3.metric("Completed", str(live.get("completed_display") or f"{int(live.get('completed_cases', 0))}/{int(live.get('total_cases', 0))}"))
    r4.metric("ETA (s)", fmt_float(live.get("eta_seconds"), 1))
    st.progress(float(live.get("progress", 0.0)))

    meta = dict(api.get_json(f"/tests/{selected_id}/promptfoo-meta"))
    promptfoo_result_url = meta.get("promptfoo_result_url")
    if promptfoo_result_url:
        st.markdown(f"[Open Promptfoo View Result]({promptfoo_result_url})")
    if meta.get("promptfoo_report_path"):
        st.caption(f"Report path: `{meta['promptfoo_report_path']}`")
    if meta.get("parser_warning"):
        parser_warning = str(meta["parser_warning"])
        lower_warning = parser_warning.lower()
        if "fallback" in lower_warning or "non-zero" in lower_warning or "version of promptfoo" in lower_warning:
            st.info(parser_warning)
        else:
            st.warning(parser_warning)

section_header(
    "Case Inspector",
    "Inspect each simulation case timeline and tool traces with direct Langfuse links.",
    tone="simulation",
)
case_ids = [str(c.get("case_id")) for c in all_cases if c.get("case_id")]
if case_ids:
    case_state_key = "selected_case_simulation"
    if st.session_state.get(case_state_key) not in case_ids:
        st.session_state[case_state_key] = case_ids[0]
    chosen_case = st.selectbox(
        "Select case to inspect",
        options=case_ids,
        key="simulation_case_selector",
        index=case_ids.index(st.session_state[case_state_key]),
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
    "Simulation outcomes with language/style fields, factual scores, and trace links.",
    tone="simulation",
)
with st.container(border=True):
    if cases:
        cdf = pd.DataFrame(cases)
        keep_cols = [
            "scenario_id",
            "task_type",
            "resolved_task_type",
            "intent_route_match",
            "company_name",
            "input_language",
            "expected_output_language",
            "instruction_style",
            "execution_status",
            "evaluation_status",
            "fact_score_1_5",
            "fact_verdict",
            "output_language_match",
            "latency_ms",
            "langfuse_trace_url",
            "notes",
        ]
        existing = [c for c in keep_cols if c in cdf.columns]
        cdf = cdf[existing].copy()
        for col in cdf.columns:
            cdf[col] = cdf[col].map(normalize_tabular_na)
        st.dataframe(
            cdf,
            width="stretch",
            hide_index=True,
            column_config={
                "langfuse_trace_url": st.column_config.LinkColumn("Trace", display_text="Open Trace"),
            },
        )
    else:
        st.caption("No case rows yet.")

maybe_auto_refresh(
    enabled=auto_refresh_enabled,
    interval_seconds=auto_refresh_interval,
    run_status=str(live.get("status") or payload.get("status") or ""),
    key="simulation",
)
