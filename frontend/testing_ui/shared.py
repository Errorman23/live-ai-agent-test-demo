from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import APIClient, DEFAULT_API_BASE, fetch_public_config, inject_theme  # noqa: E402

# Shared page utilities for testing UI tabs/pages.
# Responsibilities:
# - bootstrap page/session defaults and API client wiring
# - normalize history filtering, selectors, and run-trigger payloads
# - keep auto-refresh behavior consistent across domain pages
# Boundaries:
# - visual components live in common.py; domain specifics live in page modules


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def init_page(title: str) -> tuple[APIClient, dict[str, Any]]:
    st.set_page_config(page_title=title, layout="wide")
    inject_theme()
    if "api_base" not in st.session_state:
        st.session_state["api_base"] = DEFAULT_API_BASE
    api_base = st.sidebar.text_input("Backend API base", value=st.session_state["api_base"])
    st.session_state["api_base"] = api_base.rstrip("/")
    api = APIClient(st.session_state["api_base"], timeout_seconds=50)
    cfg = dict(fetch_public_config(st.session_state["api_base"]))
    return api, cfg


def render_model_strip(cfg: dict[str, Any]) -> None:
    agent_model = str(cfg.get("together_model") or "-")
    judge_model = str(cfg.get("llm_judge_model") or "-")
    agent_ctx = cfg.get("together_model_context_window")
    judge_ctx = cfg.get("llm_judge_context_window")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""
<div class="kpi">
  <div style="font-weight:700;">Agent Model</div>
  <div class="mono" style="margin-top:6px;">{agent_model}</div>
  <div class="subtle" style="margin-top:6px;">Context window: {agent_ctx if agent_ctx else "unknown"}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
<div class="kpi">
  <div style="font-weight:700;">Judge Model</div>
  <div class="mono" style="margin-top:6px;">{judge_model}</div>
  <div class="subtle" style="margin-top:6px;">Context window: {judge_ctx if judge_ctx else "unknown"}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_eval_method_badge(label: str, detail: str | None = None) -> None:
    st.markdown(
        f"<span class='status-chip status-running'>Evaluation Method: {label}</span>",
        unsafe_allow_html=True,
    )
    if detail:
        st.caption(detail)


def start_test(
    api: APIClient,
    *,
    domain: str,
    test_type: str,
    reasoning_effort: str,
    evaluator_mode: str,
    execution_mode: str = "promptfoo",
) -> str:
    payload = {
        "test_type": test_type,
        "test_domain": domain,
        "reasoning_effort": reasoning_effort,
        "execution_mode": execution_mode,
        "evaluator_mode": evaluator_mode,
        "repeat_count": 1,
    }
    started = api.post_json("/tests/start", payload)
    return str(started.get("test_id"))


def fetch_history(api: APIClient) -> list[dict[str, Any]]:
    payload = api.get_json("/tests/history", params={"limit": 80})
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    return []


def filter_history(rows: list[dict[str, Any]], *, domain: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        row_domain = str(row.get("test_domain") or "").strip().lower()
        suite = str(row.get("suite") or "").strip().lower()
        if row_domain == domain:
            output.append(row)
            continue
        if domain == "security" and suite in {"security", "eu", "leakage", "injection", "hallucination", "nist", "owasp", "all"}:
            output.append(row)
            continue
        if suite == domain:
            output.append(row)
    return output


def history_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.caption("No batches yet.")
        return
    table = []
    for row in rows:
        table.append(
            {
                "test_id": row.get("test_id"),
                "suite": row.get("suite"),
                "domain": row.get("test_domain") or "-",
                "status": row.get("status"),
                "started_at": row.get("started_at"),
                "ended_at": row.get("ended_at"),
                "completed": f"{row.get('completed_cases', 0)}/{row.get('total_cases', 0)}",
                "pass_rate": fmt_pct(row.get("pass_rate")),
            }
        )
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)


def select_batch(rows: list[dict[str, Any]], *, key: str) -> str | None:
    state_key = f"selected_batch_{key}"
    select_key = f"select_{key}"
    if not rows:
        return str(st.session_state.get(state_key) or "") or None

    ids = [str(row.get("test_id") or "") for row in rows if row.get("test_id")]
    if not ids:
        return None

    labels: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for row in rows:
        test_id = str(row.get("test_id") or "")
        if not test_id:
            continue
        status = str(row.get("status", "running")).upper()
        suite = str(row.get("suite", "-"))
        labels[test_id] = f"{test_id} | {suite} | {status}"
        statuses[test_id] = status

    preferred = st.session_state.get(state_key)
    if not isinstance(preferred, str) or preferred not in ids:
        running_id = next((test_id for test_id in ids if statuses.get(test_id) == "RUNNING"), None)
        preferred = running_id or ids[0]
        st.session_state[state_key] = preferred

    current_widget_value = st.session_state.get(select_key)
    if not isinstance(current_widget_value, str) or current_widget_value not in ids:
        st.session_state[select_key] = preferred

    selected_id = st.selectbox(
        "Select batch",
        options=ids,
        key=select_key,
        format_func=lambda test_id: labels.get(test_id, test_id),
    )
    st.session_state[state_key] = selected_id
    return selected_id


def auto_refresh_controls(*, key: str) -> tuple[bool, int]:
    enabled_key = f"auto_refresh_enabled_{key}"
    interval_key = f"auto_refresh_interval_{key}"
    enabled_default = bool(st.session_state.get(enabled_key, True))
    interval_default = int(st.session_state.get(interval_key, 3))
    enabled = st.sidebar.checkbox(
        "Auto-refresh while running",
        value=enabled_default,
        help="When a selected batch is running, automatically refresh this page.",
    )
    options = [2, 3, 5, 10, 15]
    current = interval_default if interval_default in options else 3
    interval = int(st.sidebar.selectbox("Refresh interval (sec)", options=options, index=options.index(current)))
    st.session_state[enabled_key] = enabled
    st.session_state[interval_key] = interval
    return enabled, interval


def maybe_auto_refresh(
    *,
    enabled: bool,
    interval_seconds: int,
    run_status: str | None,
    key: str = "default",
) -> None:
    normalized = (run_status or "").strip().lower()
    is_running = normalized in {"running", "queued"}
    if not enabled or not is_running:
        return
    st.caption(f"Auto-refreshing every {interval_seconds}s while this batch is running.")
    interval_ms = max(interval_seconds, 1) * 1000
    st_autorefresh(interval=interval_ms, key=f"autorefresh_{key}")
