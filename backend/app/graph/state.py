from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.schemas import ArtifactRef, PlannerStep, StepEvent, ToolCallRecord


class AgentState(TypedDict, total=False):
    run_id: str
    trace_id: str
    user_prompt: str
    requested_task_type: str | None
    task_type: str
    company_name: str
    company_source: Literal["prompt_extract", "planner", "user_override"]
    intent_resolution_source: Literal["heuristic", "alias", "llm", "hybrid"]
    language_resolution_source: Literal["heuristic", "alias", "llm", "hybrid"]
    language_fallback_applied: bool
    resolved_target_language: str
    internal_doc_type: Literal["proposal", "quotation"]
    target_language: str
    source_language: str | None
    translation_applied: bool
    output_mode: Literal["chat", "document"]
    reasoning_effort: str
    model_id: str | None

    plan_steps: list[PlannerStep]
    current_step_idx: int
    tool_results: dict[str, Any]
    tool_call_records: list[ToolCallRecord]
    step_events: list[StepEvent]
    artifacts: list[ArtifactRef]
    policy_findings: list[str]
    expected_facts: tuple[str, ...]
    source_evidence: list[str]
    evidence_pack: dict[str, Any]
    translated_document: str
    artifact_document_text: str

    llm_attempt_count: int
    llm_retry_exhausted: bool
    llm_last_error: str | None
    provider_status_codes: list[int | None]
    provider_error_chain: list[str]

    draft_document: str
    final_document: str
    security_report: dict[str, Any]
    llm_tokens: list[str]

    policy_events: list[str]
    status: Literal["success", "failed"]
    error_message: str | None
