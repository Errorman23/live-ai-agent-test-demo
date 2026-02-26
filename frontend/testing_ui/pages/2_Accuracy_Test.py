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

# Accuracy page focuses on factual grounding, translation faithfulness, and
# structure-quality metrics. It renders only accuracy-relevant fields to avoid
# cross-domain noise from functional/security-only signals.


api, cfg = init_page("Accuracy Test")
reasoning_effort = st.sidebar.selectbox("Reasoning effort", ["low", "medium", "high"], index=1)
auto_refresh_enabled, auto_refresh_interval = auto_refresh_controls(key="accuracy")

st.markdown(
    """
<div class="hero">
  <h3 style="margin:0;">Accuracy Test</h3>
  <p style="margin:8px 0 0 0;">
    Evidence-grounded factuality (LLM judge 1-5), translation faithfulness (SiliconFlow reference + BERTScore),
    and structure checks for generated briefing/doc outputs.
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
    mt = methodology.get("translation_reference", {}) if isinstance(methodology, dict) else {}
    accuracy_rules = det.get("accuracy_rules", {}) if isinstance(det, dict) else {}
    st.markdown("**LLM-as-Judge Prompt Template**")
    st.code(str(judge.get("system_template", "")), language="text")
    st.code(str(judge.get("user_template", "")), language="text")
    st.markdown("**Score Rubric (1..5)**")
    st.json(judge.get("score_scale", {}))
    st.markdown("**Translation Reference Prompt Template**")
    st.code(str(mt.get("system_template", "")), language="text")
    st.code(str(mt.get("user_template", "")), language="text")
    st.caption(
        f"Reference model: {mt.get('model', '-')}, "
        f"BERTScore model: {mt.get('bertscore_model', 'bert-base-multilingual-cased')}, "
        f"threshold: {mt.get('bertscore_threshold', 0.82)}"
    )
    st.markdown("**Deterministic Accuracy Rules**")
    st.json(accuracy_rules)
    st.markdown("**Structure Quality Checks (deterministic)**")
    st.code(
        "required_sections = [\n"
        "  'executive_summary', 'public_findings', 'internal_summary', 'risk_notes', 'sources'\n"
        "]\n"
        "structure_score = non_empty_required_sections / len(required_sections)\n"
        "violations += ['has_redaction_marker', 'non_trivial_length'] if missing\n"
        "pass if structure_score >= 0.70",
        language="python",
    )
    st.markdown("**Batch Pass/Fail Gates (Accuracy domain)**")
    st.json(
        {
            "base_gate": "agent_status == success AND final_text_length > 0",
            "factual_gate": accuracy_rules.get("factual_rule"),
            "translation_gate": accuracy_rules.get("translation_threshold"),
            "structure_gate": accuracy_rules.get("structure_threshold"),
            "na_semantics": accuracy_rules.get("not_applicable_rule"),
        }
    )

run_evaluator_mode = "llm_judge"
# Batch launch controls.
section_header(
    "Run Accuracy Suite",
    "Run a 10-case batch for factuality, translation, and structure quality.",
    tone="accuracy",
)
with st.container(border=True):
    render_eval_method_badge(
        "Hybrid (LLM judge + deterministic metrics)",
        "This page uses LLM judge for factual scoring plus deterministic translation/structure checks.",
    )
    if st.button("Run Accuracy Suite (10 cases)", width="stretch", type="primary"):
        test_id = start_test(
            api,
            domain="accuracy",
            test_type="accuracy",
            reasoning_effort=reasoning_effort,
            evaluator_mode=run_evaluator_mode,
        )
        st.session_state["selected_batch_accuracy"] = test_id
        st.session_state["select_accuracy"] = test_id
        st.rerun()

rows = filter_history(fetch_history(api), domain="accuracy")
# Batch history selector + selected batch payload load.
section_header(
    "Batch Selection",
    "Use history to pick a batch and inspect runtime telemetry plus case-level judgments.",
    tone="accuracy",
)
with st.container(border=True):
    history_table(rows)
    selected_id = select_batch(rows, key="accuracy")
    if not selected_id:
        st.info("No accuracy batch selected yet.")
        st.stop()

live = dict(api.get_json(f"/tests/{selected_id}/live"))
all_cases = list(live.get("all_cases", []))
payload = dict(api.get_json(f"/tests/{selected_id}"))
summary = payload.get("summary") or {}
cases = list(payload.get("cases", []))

section_header(
    "Accuracy KPI Dashboard",
    "Factuality, translation faithfulness, and structure quality with explicit tested denominators.",
    tone="accuracy",
)
with st.container(border=True):
    st.markdown("#### Completion")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Success Rate", fmt_pct(summary.get("success_rate")))
    c2.metric("Error Rate", fmt_pct(summary.get("runtime_error_rate")))
    c3.metric("Retry Exhaustion", fmt_pct(summary.get("retry_exhaustion_rate")))
    c4.metric("Avg Attempts", fmt_float(summary.get("avg_attempts")))

    st.markdown("#### Factuality")
    f1, f2 = st.columns(2)
    f1.metric("Factual Avg (1-5)", fmt_float(summary.get("factual_score_avg_1_5"), 2))
    f2.metric("Factual Pass Rate", fmt_pct(summary.get("factual_pass_rate")))

    st.markdown("#### Translation and Structure")
    t1, t2, t3 = st.columns(3)
    t1.metric("Translation BERTScore", fmt_float(summary.get("translation_bertscore_avg"), 3))
    t2.metric("Structure Score", fmt_float(summary.get("structure_score_avg"), 2))
    t3.metric("Intent Route Accuracy", fmt_pct(summary.get("intent_route_accuracy")))
    st.caption(
        "Denominators - completion: "
        f"{summary.get('completion_denominator', 0)}, quality/completed: {summary.get('quality_denominator', 0)}, "
        f"factual: {summary.get('factual_denominator', 0)}, translation: {summary.get('translation_denominator', 0)}, "
        f"structure: {summary.get('structure_denominator', 0)}"
    )

section_header(
    "Batch Runtime",
    "Track progress and access Promptfoo batch artifacts for this accuracy run.",
    tone="accuracy",
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
    "Select a case to review step timeline, tool I/O, and run-level diagnostics.",
    tone="accuracy",
)
case_ids = [str(c.get("case_id")) for c in all_cases if c.get("case_id")]
if case_ids:
    case_state_key = "selected_case_accuracy"
    if st.session_state.get(case_state_key) not in case_ids:
        st.session_state[case_state_key] = case_ids[0]
    chosen_case = st.selectbox(
        "Select case to inspect",
        options=case_ids,
        key="accuracy_case_selector",
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
    "Per-case accuracy outcomes with applicability-aware N/A handling and trace drill-down links.",
    tone="accuracy",
)
with st.container(border=True):
    if cases:
        display_rows: list[dict[str, object]] = []
        for row in cases:
            # Non-applicable metrics are rendered as explicit N/A so blanks only
            # represent actual evaluation/runtime failures.
            item = dict(row)
            applicability = item.get("metric_applicability")
            if not isinstance(applicability, dict):
                applicability = {}

            if not bool(applicability.get("fact_score_1_5", False)):
                item["fact_score_1_5"] = "N/A"
                item["fact_verdict"] = "N/A"
                item["fact_reason"] = "N/A"
            else:
                if item.get("fact_score_1_5") is None:
                    item["fact_score_1_5"] = ""
                if not item.get("fact_verdict"):
                    item["fact_verdict"] = ""
                if not item.get("fact_reason"):
                    item["fact_reason"] = ""

            if not bool(applicability.get("translation_bertscore_f1", False)):
                item["translation_reference_called"] = "N/A"
                item["translation_reference_model"] = "N/A"
                item["translation_bertscore_f1"] = "N/A"
            else:
                if item.get("translation_bertscore_f1") is None:
                    item["translation_bertscore_f1"] = ""

            if not bool(applicability.get("structure_score", False)):
                item["structure_score"] = "N/A"
                item["structure_violations"] = "N/A"
            else:
                if item.get("structure_score") is None:
                    item["structure_score"] = ""
                if item.get("structure_violations") in (None, []):
                    item["structure_violations"] = ""

            display_rows.append(item)

        cdf = pd.DataFrame(display_rows)
        keep_cols = [
            "scenario_id",
            "task_type",
            "resolved_task_type",
            "intent_route_match",
            "company_name",
            "input_language",
            "instruction_style",
            "execution_status",
            "evaluation_status",
            "fact_score_1_5",
            "fact_verdict",
            "fact_reason",
            "translation_reference_called",
            "translation_reference_model",
            "translation_bertscore_f1",
            "structure_score",
            "structure_violations",
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
    key="accuracy",
)
