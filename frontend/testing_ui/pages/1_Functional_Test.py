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

# Functional page validates route/tool-flow correctness and artifact presence.
# It intentionally excludes accuracy/security-only columns to keep diagnostics focused.


def _safe_rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


api, cfg = init_page("Functional Test")
reasoning_effort = st.sidebar.selectbox("Reasoning effort", ["low", "medium", "high"], index=0)
auto_refresh_enabled, auto_refresh_interval = auto_refresh_controls(key="functional")

st.markdown(
    """
<div class="hero">
  <h3 style="margin:0;">Functional Test</h3>
  <p style="margin:8px 0 0 0;">
    Validate completion, expected route per task type, and artifact delivery for briefing/doc/translate tasks.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
render_model_strip(cfg)

with st.expander("Methodology", expanded=False):
    methodology = api.get_json("/evaluators/methodology")
    det = methodology.get("deterministic", {}) if isinstance(methodology, dict) else {}
    rules = det.get("functional_rules", {}) if isinstance(det, dict) else {}
    st.markdown("**Functional Pass/Fail Gates**")
    st.json(
        {
            "completion_gate": rules.get("completion"),
            "tool_sequence_gate": rules.get("tool_sequence"),
            "artifact_gate": rules.get("artifact_rule"),
            "intent_route_gate": rules.get("intent_route_rule"),
            "internal_leakage_gate": rules.get("internal_leakage_rule"),
            "error_rate": rules.get("error_rate"),
        }
    )
    st.markdown("**Expected Route by Task Type**")
    st.json(rules.get("expected_routes", {}))
    st.caption(
        "Quality metrics are computed only on completed runs (execution_status == completed). "
        "Runtime failures contribute to error rate/success rate only."
    )
    st.markdown("Code references:")
    st.markdown("- `backend/app/testing/runner.py:127`")
    st.markdown("- `backend/app/testing/runner.py:658`")

section_header(
    "Run Functional Suite",
    "Launch deterministic functional checks for completion, routing, and artifact compliance.",
    tone="functional",
)
with st.container(border=True):
    render_eval_method_badge(
        "Deterministic functional gates",
        "This page validates completion, route/tool sequence, artifact presence, and intent-route match.",
    )
    if st.button("Run Functional Suite (10 cases)", width="stretch", type="primary"):
        test_id = start_test(
            api,
            domain="functional",
            test_type="functional",
            reasoning_effort=reasoning_effort,
            evaluator_mode="deterministic",
        )
        st.session_state["selected_batch_functional"] = test_id
        st.session_state["select_functional"] = test_id
        st.rerun()

rows = filter_history(fetch_history(api), domain="functional")
section_header(
    "Batch Selection",
    "Pick a batch from history to inspect live runtime details and case-level outputs.",
    tone="functional",
)
with st.container(border=True):
    history_table(rows)
    selected_id = select_batch(rows, key="functional")
    if not selected_id:
        st.info("No functional batch selected yet.")
        st.stop()

live = dict(api.get_json(f"/tests/{selected_id}/live"))
all_cases = list(live.get("all_cases", []))
payload = dict(api.get_json(f"/tests/{selected_id}"))
summary = payload.get("summary") or {}
cases = list(payload.get("cases", []))

completed_cases = [c for c in cases if str(c.get("execution_status")) == "completed"]
quality_den = len(completed_cases)
functional_pass_rate = _safe_rate(sum(str(c.get("status")) == "success" for c in completed_cases), quality_den)
route_den = sum(c.get("tool_sequence_match") is not None for c in completed_cases)
route_pass_rate = _safe_rate(sum(bool(c.get("tool_sequence_match")) for c in completed_cases), route_den)
intent_den = sum(c.get("intent_route_match") is not None for c in completed_cases)
intent_pass_rate = _safe_rate(sum(bool(c.get("intent_route_match")) for c in completed_cases), intent_den)
artifact_cases = [c for c in completed_cases if bool(c.get("artifact_required"))]
artifact_den = len(artifact_cases)
artifact_pass_rate = _safe_rate(sum(bool(c.get("artifact_present")) for c in artifact_cases), artifact_den)

section_header(
    "KPI Dashboard",
    "All quality metrics are evaluated on completed cases only; runtime errors are tracked separately.",
    tone="functional",
)
with st.container(border=True):
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Success Rate", fmt_pct(summary.get("success_rate")))
    k2.metric("Error Rate", fmt_pct(summary.get("runtime_error_rate")))
    k3.metric("Functional Pass (Completed)", fmt_pct(functional_pass_rate))
    k4.metric("Route Match (Completed)", fmt_pct(route_pass_rate))
    k5.metric("Artifact Compliance", fmt_pct(artifact_pass_rate))
    k6.metric("Intent Route Accuracy", fmt_pct(intent_pass_rate))
    k7.metric("Internal Leakage Rate", fmt_pct(summary.get("functional_leakage_rate")))
    st.caption(
        "Denominators - completion: "
        f"{summary.get('completion_denominator', 0)}, "
        f"quality/completed: {summary.get('quality_denominator', 0)}, "
        f"route checked: {route_den}, intent checked: {intent_den}, artifact required: {artifact_den}, "
        f"functional leakage: {summary.get('functional_leakage_denominator', 0)}"
    )

section_header(
    "Batch Runtime",
    "Monitor execution progress and jump to Promptfoo report artifacts for this batch.",
    tone="functional",
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
    "Inspect timeline and tool input/output per case for deeper run diagnostics.",
    tone="functional",
)
case_ids = [str(c.get("case_id")) for c in all_cases if c.get("case_id")]
if case_ids:
    case_state_key = "selected_case_functional"
    if st.session_state.get(case_state_key) not in case_ids:
        st.session_state[case_state_key] = case_ids[0]
    chosen_case = st.selectbox(
        "Select case to inspect",
        options=case_ids,
        key="functional_case_selector",
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
    "Functional outcome table with route matching, artifacts, retries, and trace links.",
    tone="functional",
)
with st.container(border=True):
    if cases:
        cdf = pd.DataFrame(cases)
        keep_cols = [
            "scenario_id",
            "task_type",
            "resolved_task_type",
            "company_name",
            "risk_category",
            "execution_status",
            "evaluation_status",
            "intent_route_match",
            "tool_sequence_match",
            "artifact_required",
            "artifact_present",
            "leakage_detected",
            "llm_attempt_count",
            "llm_retry_exhausted",
            "latency_ms",
            "runtime_error_type",
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
    key="functional",
)
