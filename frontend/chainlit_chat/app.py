from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import chainlit as cl
import httpx
from chainlit.input_widget import Select, TextInput

# Chainlit consultant-facing chat UI.
# Responsibilities:
# - send natural-language prompts to backend run APIs
# - stream runtime status and final responses to the chat session
# - render artifact links and lightweight evidence summaries
# Boundaries:
# - orchestration logic remains in backend graph nodes; this file is client presentation glue


API_BASE = os.getenv("AGENT_API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
RUN_TIMEOUT_SECONDS = float(os.getenv("CHAINLIT_RUN_TIMEOUT_SECONDS", "300"))
POLL_INTERVAL_SECONDS = float(os.getenv("CHAINLIT_RUN_POLL_INTERVAL_SECONDS", "1.0"))

STEP_LABELS: dict[str, str] = {
    "parse_intent": "Understand Request",
    "plan": "Plan Workflow",
    "validate_plan": "Validate Plan",
    "retrieve_parallel": "Retrieve (Parallel)",
    "retrieve_internal_db": "Retrieve Internal DB",
    "retrieve_public_web": "Retrieve Public Web",
    "retrieve_internal_pdf": "Retrieve Internal PDF",
    "compose_template_document": "Compose Response",
    "enforce_output_language": "Enforce Output Language",
    "security_filter": "Apply Security Filter",
    "persist_artifacts": "Persist Artifacts",
    "finalize": "Finalize",
    "error": "Handle Error",
}


# ---------------------------------------------------------------------------
# Quick actions and payload normalization helpers.
# ---------------------------------------------------------------------------
def _quick_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="preset_brief_tencent",
            label="Brief Tencent",
            payload={"prompt": "Generate briefing notes for Tencent."},
        ),
        cl.Action(
            name="preset_web_openai",
            label="Web: OpenAI",
            payload={"prompt": "Get OpenAI information from web only."},
        ),
        cl.Action(
            name="preset_db_tiktok",
            label="DB: TikTok",
            payload={"prompt": "Get TikTok internal record from the internal database."},
        ),
        cl.Action(
            name="preset_translate_tencent",
            label="Translate Tencent",
            payload={"prompt": "Translate Tencent proposal document to English."},
        ),
        cl.Action(
            name="preset_doc_sony",
            label="Doc: Sony Proposal",
            payload={"prompt": "Retrieve Sony proposal document from internal database."},
        ),
        cl.Action(
            name="clear_chat",
            label="🗑",
            payload={},
        ),
    ]


def _build_run_payload(prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "reasoning_effort": str(settings.get("reasoning_effort") or "low"),
    }
    model_id = str(settings.get("model_id") or "").strip()
    if model_id:
        payload["model_id"] = model_id
    session_id = str(settings.get("session_id") or "").strip()
    if session_id:
        payload["session_id"] = session_id
    return payload


def _short(text: str, *, limit: int = 88) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 3]}..."


def _task_status(raw_status: str) -> cl.TaskStatus:
    if raw_status == "running":
        return cl.TaskStatus.RUNNING
    if raw_status == "completed":
        return cl.TaskStatus.DONE
    if raw_status == "failed":
        return cl.TaskStatus.FAILED
    return cl.TaskStatus.READY


# ---------------------------------------------------------------------------
# Tool-output adapters used for optional tabular rendering in chat.
# ---------------------------------------------------------------------------
def _extract_tool_output_preview(final_response: dict[str, Any], tool_name: str) -> dict[str, Any]:
    tool_records = final_response.get("tool_call_records")
    if not isinstance(tool_records, list):
        return {}
    for record in tool_records:
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "") != tool_name:
            continue
        details = record.get("details")
        if not isinstance(details, dict):
            continue
        payload = details.get("output_payload_preview")
        if isinstance(payload, dict):
            return payload
    return {}


def _build_web_results_rows(final_response: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    payload = _extract_tool_output_preview(final_response, "search_public_web")
    rows: list[dict[str, str]] = []
    links: list[str] = []
    raw_results = payload.get("results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            rows.append({"Title": title, "Snippet": snippet, "URL": url})
            if url:
                links.append(url)
    raw_links = payload.get("source_links")
    if isinstance(raw_links, list):
        for item in raw_links:
            url = str(item or "").strip()
            if url:
                links.append(url)
    deduped_links = list(dict.fromkeys(links))[:8]
    return rows, deduped_links


def _build_db_summary_rows(final_response: dict[str, Any]) -> list[dict[str, str]]:
    payload = _extract_tool_output_preview(final_response, "get_company_info")
    if not payload:
        return []
    products = payload.get("public_products")
    partnerships = payload.get("public_partnerships")
    return [
        {
            "Company": str(payload.get("company_name") or ""),
            "Record Found": "Yes" if bool(payload.get("record_found", False)) else "No",
            "Industry": str(payload.get("industry") or ""),
            "Risk Level": str(payload.get("project_risk_level") or ""),
            "Public Products": ", ".join(str(item) for item in products) if isinstance(products, list) else "",
            "Public Partnerships": ", ".join(str(item) for item in partnerships) if isinstance(partnerships, list) else "",
            "Internal Summary": str(payload.get("internal_summary") or ""),
        }
    ]


async def _send_tabular_supplement(task_type: str, final_response: dict[str, Any]) -> None:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        pd = None  # type: ignore[assignment]

    if task_type == "web_only":
        _, links = _build_web_results_rows(final_response)
        if links:
            await cl.Message(content="Sources:\n" + "\n".join(f"- {url}" for url in links)).send()

    if task_type == "db_only":
        rows = _build_db_summary_rows(final_response)
        if rows and pd is not None:
            try:
                await cl.Message(
                    content="Retrieved internal DB summary table (sanitized).",
                    elements=[cl.Dataframe(data=pd.DataFrame(rows), display="inline", name="db_summary")],
                ).send()
            except Exception:
                await cl.Message(content="Retrieved internal DB summary (sanitized).").send()


async def _ensure_task_list() -> cl.TaskList:
    existing = cl.user_session.get("task_list")
    if isinstance(existing, cl.TaskList):
        return existing

    task_list = cl.TaskList(display="side", status="Waiting for request", tasks=[])
    await task_list.send()
    cl.user_session.set("task_list", task_list)
    cl.user_session.set("task_index", {})
    return task_list


async def _reset_task_list(status: str = "Waiting for request") -> None:
    task_list = await _ensure_task_list()
    task_list.tasks = []
    task_list.status = status
    await task_list.update()
    cl.user_session.set("task_index", {})


async def _upsert_step_task(step_event: dict[str, Any]) -> None:
    task_list = await _ensure_task_list()
    task_index = dict(cl.user_session.get("task_index") or {})

    step_name = str(step_event.get("step_name") or "").strip()
    if not step_name:
        return

    status = str(step_event.get("status") or "pending")
    message = str(step_event.get("message") or "").strip()
    label = STEP_LABELS.get(step_name, step_name.replace("_", " ").title())

    idx = int(task_index.get(step_name, -1))
    if idx < 0 or idx >= len(task_list.tasks):
        await task_list.add_task(cl.Task(title=label, status=_task_status(status)))
        task_index[step_name] = len(task_list.tasks) - 1
        idx = int(task_index[step_name])

    task = task_list.tasks[idx]
    task.status = _task_status(status)
    if message:
        task.title = f"{label}: {_short(message)}"
    else:
        task.title = label

    if status == "running":
        task_list.status = f"Running: {label}"
    elif status == "failed":
        task_list.status = f"Failed: {label}"

    await task_list.update()
    cl.user_session.set("task_index", task_index)


async def _append_tool_task(tool_event: dict[str, Any]) -> None:
    task_list = await _ensure_task_list()
    tool_name = str(tool_event.get("tool_name") or "").strip()
    if not tool_name:
        return
    status = str(tool_event.get("status") or "success")
    duration_ms = float(tool_event.get("duration_ms") or 0.0)
    await task_list.add_task(
        cl.Task(
            title=f"Tool: {tool_name} ({duration_ms:.0f} ms)",
            status=cl.TaskStatus.DONE if status == "success" else cl.TaskStatus.FAILED,
        )
    )
    await task_list.update()


async def _send_welcome_message() -> None:
    cfg = cl.user_session.get("public_config") or {}
    agent_model = str((cfg or {}).get("together_model") or "openai/gpt-oss-20b")
    await cl.Message(
        content=(
            "Consultant Research Assistant is ready.\n\n"
            "Use natural language only.\n\n"
            f"Agent model: `{agent_model}`"
        ),
    ).send()


async def _load_public_config() -> None:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)) as client:
            response = await client.get(f"{API_BASE}/config/public")
            response.raise_for_status()
            payload = response.json()
    except Exception:
        payload = {}
    cl.user_session.set("public_config", payload if isinstance(payload, dict) else {})


async def _clear_chat_history() -> None:
    messages = list(cl.chat_context.get())
    for message in reversed(messages):
        try:
            cl.chat_context.remove(message)
            await message.remove()
        except Exception:
            continue
    cl.chat_context.clear()
    cl.user_session.set("quick_actions_msg", None)


async def _show_quick_actions_toolbar() -> None:
    prior = cl.user_session.get("quick_actions_msg")
    if isinstance(prior, cl.Message):
        try:
            await prior.remove()
        except Exception:
            pass
    toolbar = cl.Message(
        content="",
        actions=_quick_actions(),
    )
    await toolbar.send()
    cl.user_session.set("quick_actions_msg", toolbar)


async def _run_agent(prompt: str) -> None:
    # Phase 1: gather user settings and initialize a fresh run/task context.
    settings = cl.user_session.get("chat_settings") or {}
    payload = _build_run_payload(prompt, settings)

    await _reset_task_list(status="Running")

    final_response: dict[str, Any] | None = None
    final_status = "running"
    backend_error: str | None = None
    policy_findings: list[str] = []
    artifact_lines: list[str] = []
    streamed_any_token = False

    streaming_msg = cl.Message(content="")
    await streaming_msg.send()

    last_step_idx = 0
    last_tool_idx = 0
    last_token_idx = 0
    run_id = ""
    timed_out = False

    timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=15.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Phase 2: create backend run and poll until terminal status.
            start = await client.post(f"{API_BASE}/runs/start", json=payload)
            start.raise_for_status()
            start_payload = start.json()
            run_id = str(start_payload.get("run_id") or "")
            if not run_id:
                raise RuntimeError("Backend did not return run_id")

            started_at = time.monotonic()
            while True:
                status_resp = await client.get(f"{API_BASE}/runs/{run_id}")
                status_resp.raise_for_status()
                status_payload = status_resp.json()

                # Incremental step timeline updates.
                steps = status_payload.get("step_events")
                if isinstance(steps, list):
                    while last_step_idx < len(steps):
                        step = steps[last_step_idx]
                        if isinstance(step, dict):
                            await _upsert_step_task(step)
                        last_step_idx += 1

                # Incremental tool call updates.
                tool_calls = status_payload.get("tool_call_records")
                if isinstance(tool_calls, list):
                    while last_tool_idx < len(tool_calls):
                        tool_event = tool_calls[last_tool_idx]
                        if isinstance(tool_event, dict):
                            await _append_tool_task(tool_event)
                        last_tool_idx += 1

                # Incremental token streaming updates.
                llm_tokens = status_payload.get("llm_tokens")
                if isinstance(llm_tokens, list):
                    while last_token_idx < len(llm_tokens):
                        token = str(llm_tokens[last_token_idx] or "")
                        if token:
                            streamed_any_token = True
                            await streaming_msg.stream_token(token)
                        last_token_idx += 1

                findings = status_payload.get("policy_findings")
                if isinstance(findings, list):
                    policy_findings = [str(item) for item in findings]

                final_status = str(status_payload.get("status") or final_status)
                response_payload = status_payload.get("response")
                if isinstance(response_payload, dict):
                    final_response = response_payload
                backend_error = str(status_payload.get("error") or backend_error or "")

                if final_status != "running":
                    break

                # Timeout guard to prevent stuck client-side polling loops.
                if (time.monotonic() - started_at) > RUN_TIMEOUT_SECONDS:
                    timed_out = True
                    backend_error = (
                        f"Run timed out after {int(RUN_TIMEOUT_SECONDS)}s. "
                        f"run_id={run_id}. Check backend logs or retry."
                    )
                    final_status = "failed"
                    break

                await asyncio.sleep(POLL_INTERVAL_SECONDS)
    except Exception as exc:  # noqa: BLE001
        final_status = "failed"
        backend_error = str(exc)

    # Phase 3: finalize display output (tokens/text), artifact links, and status.
    final_text = ""
    if final_response is not None:
        final_text = str(final_response.get("final_document") or "").strip()
        if not policy_findings:
            raw_findings = final_response.get("policy_findings", [])
            if isinstance(raw_findings, list):
                policy_findings = [str(item) for item in raw_findings]
        raw_artifacts = final_response.get("artifacts")
        if isinstance(raw_artifacts, list):
            for item in raw_artifacts:
                if not isinstance(item, dict):
                    continue
                artifact_id = str(item.get("artifact_id") or "").strip()
                filename = str(item.get("filename") or artifact_id or "artifact")
                if not artifact_id:
                    continue
                artifact_lines.append(f"- [{filename}]({API_BASE}/artifacts/{artifact_id}/download)")

    if not streamed_any_token:
        if final_text:
            await streaming_msg.stream_token(final_text)
        else:
            reason = backend_error or (policy_findings[0] if policy_findings else "No response generated.")
            if timed_out and run_id:
                reason = f"{reason} (You can start a fresh request or clear chat.)"
            await streaming_msg.stream_token(f"Run failed: {reason}")
    elif final_status == "completed" and final_text:
        rendered = str(streaming_msg.content or "").strip()
        expected = final_text.strip()
        if rendered != expected:
            streaming_msg.content = final_text
    await streaming_msg.update()

    if artifact_lines and final_status == "completed":
        await cl.Message(
            content="Download generated file(s):\n" + "\n".join(artifact_lines),
        ).send()

    if final_status == "completed" and isinstance(final_response, dict):
        task_type = str(final_response.get("task_type") or "")
        if task_type in {"web_only", "db_only"}:
            await _send_tabular_supplement(task_type, final_response)

    task_list = await _ensure_task_list()
    task_list.status = "Completed" if final_status == "completed" else "Failed"
    await task_list.update()
    await _show_quick_actions_toolbar()


@cl.on_chat_start
async def on_chat_start() -> None:
    await _load_public_config()
    settings = await cl.ChatSettings(
        [
            Select(
                id="reasoning_effort",
                label="Reasoning Effort",
                values=["low", "medium", "high"],
                initial_index=0,
            ),
            TextInput(
                id="model_id",
                label="Model Override (optional)",
                initial="",
            ),
            TextInput(
                id="session_id",
                label="Session ID (optional)",
                initial="",
            ),
        ]
    ).send()
    cl.user_session.set("chat_settings", settings)
    await _reset_task_list()
    await _send_welcome_message()
    await _show_quick_actions_toolbar()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    cl.user_session.set("chat_settings", settings)


async def _run_preset_action(action: cl.Action) -> None:
    prompt = str((action.payload or {}).get("prompt") or "").strip()
    if not prompt:
        return
    await cl.Message(content=prompt, author="user").send()
    try:
        await _run_agent(prompt)
    except Exception as exc:  # noqa: BLE001
        task_list = await _ensure_task_list()
        task_list.status = "Failed"
        await task_list.update()
        await cl.Message(content=f"Execution failed: {exc}").send()


@cl.action_callback("preset_brief_tencent")
async def on_preset_brief(action: cl.Action) -> None:
    await _run_preset_action(action)


@cl.action_callback("preset_web_openai")
async def on_preset_web(action: cl.Action) -> None:
    await _run_preset_action(action)


@cl.action_callback("preset_db_tiktok")
async def on_preset_db(action: cl.Action) -> None:
    await _run_preset_action(action)


@cl.action_callback("preset_translate_tencent")
async def on_preset_translate(action: cl.Action) -> None:
    await _run_preset_action(action)


@cl.action_callback("preset_doc_sony")
async def on_preset_doc_sony(action: cl.Action) -> None:
    await _run_preset_action(action)


@cl.action_callback("clear_chat")
async def on_clear_chat(_action: cl.Action) -> None:
    await _clear_chat_history()
    await _reset_task_list()
    await _send_welcome_message()
    await _show_quick_actions_toolbar()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    prompt = str(message.content or "").strip()
    if not prompt:
        await cl.Message(content="Please enter a request.").send()
        return
    try:
        await _run_agent(prompt)
    except Exception as exc:  # noqa: BLE001
        task_list = await _ensure_task_list()
        task_list.status = "Failed"
        await task_list.update()
        await cl.Message(content=f"Execution failed: {exc}").send()
        await _show_quick_actions_toolbar()
