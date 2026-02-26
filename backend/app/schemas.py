from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Central Pydantic contracts shared by backend runtime, testing, and UI APIs.
# Responsibilities:
# - define request/response models for run/test endpoints
# - carry normalized per-case metrics across runner/api/frontend
# - preserve backward-compatible fields used by existing UI tables

# Planner/intent contracts used by graph parse+plan nodes.


class PlannerStep(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale_short: str = ""


class PlannerOutput(BaseModel):
    company_name: str
    target_language: str
    task_type: Literal["briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat"] | None = None
    steps: list[PlannerStep]


class IntentParseOutput(BaseModel):
    company_name: str | None = None
    target_language: str | None = None
    task_type: Literal["briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat"] | None = None
    confidence: float = 0.0


class SecurityReport(BaseModel):
    pass_fail: bool
    redactions_applied: int
    leaked_terms: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool_name: str
    args_digest: str
    duration_ms: float
    status: Literal["success", "failed"]
    output_hash: str
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    filename: str
    path: str
    sha256: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] | None = None


class StepEvent(BaseModel):
    step_name: str
    status: Literal["pending", "running", "completed", "failed"]
    message: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    tool_name: str | None = None
    llm_attempt_index: int | None = None


# ---------------------------------------------------------------------------
# Core agent run contracts.
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    prompt: str
    task_type: Literal["briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat"] | None = None
    model_id: str | None = None
    scenario_id: str | None = None
    safety_level: Literal["strict", "standard"] = "strict"
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    session_id: str | None = None
    seed: int | None = None


class RunResponse(BaseModel):
    run_id: str
    status: Literal["success", "failed"]
    task_type: Literal["briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat"] = "briefing_full"
    final_document: str | None = None
    evidence_pack: dict[str, Any] | None = None
    resolved_company_name: str | None = None
    company_source: Literal["prompt_extract", "planner", "user_override"] | None = None
    resolved_target_language: str | None = None
    intent_resolution_source: Literal["heuristic", "alias", "llm", "hybrid"] | None = None
    language_resolution_source: Literal["heuristic", "alias", "llm", "hybrid"] | None = None
    language_fallback_applied: bool = False
    output_mode: Literal["chat", "document"] = "document"
    source_language: str | None = None
    target_language: str | None = None
    translation_applied: bool = False
    security_report: SecurityReport | None = None
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    step_events: list[StepEvent] = Field(default_factory=list)
    policy_findings: list[str] = Field(default_factory=list)
    llm_attempt_count: int = 0
    llm_retry_exhausted: bool = False
    provider_error_chain: list[str] = Field(default_factory=list)
    provider_status_codes: list[int | None] = Field(default_factory=list)
    langfuse_trace_id: str | None = None
    run_duration_ms: float = 0.0


class RunStartResponse(BaseModel):
    run_id: str
    trace_id: str
    status: Literal["running"] = "running"


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    response: RunResponse | None = None
    error: str | None = None
    progress: float = 0.0
    trace_id: str | None = None
    step_events: list[StepEvent] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    policy_findings: list[str] = Field(default_factory=list)
    llm_tokens: list[str] = Field(default_factory=list)


class TraceSpan(BaseModel):
    span_id: str
    name: str
    start_time: datetime
    end_time: datetime | None = None
    status: Literal["ok", "error"] = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    run_id: str
    langfuse_trace_id: str | None = None
    timeline_steps: list[str] = Field(default_factory=list)
    tool_spans: list[TraceSpan] = Field(default_factory=list)
    policy_events: list[str] = Field(default_factory=list)


# Active testing control-plane contracts (Streamlit + Promptfoo orchestration).
# ---------------------------------------------------------------------------
# Test orchestration and per-case evaluation contracts.
# ---------------------------------------------------------------------------
class TestRunRequest(BaseModel):
    test_type: Literal[
        "leakage",
        "injection",
        "hallucination",
        "eu",
        "security-minimal",
        "security-eu-full",
        "nist",
        "owasp",
        "all",
        "functional",
        "accuracy",
        "security",
        "simulation",
    ]
    test_domain: Literal["functional", "accuracy", "security", "simulation"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    repeat_count: int | None = None
    enable_llm_judge: bool = False
    execution_mode: Literal["promptfoo", "inhouse"] = "promptfoo"
    evaluator_mode: Literal["deterministic", "llm_judge"] = "deterministic"


class TestCaseResult(BaseModel):
    scenario_id: str
    test_type: str
    test_domain: Literal["functional", "accuracy", "security", "simulation"] | None = None
    task_type: str | None = None
    resolved_task_type: str | None = None
    case_id: str | None = None
    batch_id: str | None = None
    company_name: str | None = None
    risk_category: str | None = None
    profile_language: str | None = None
    input_language: str | None = None
    expected_output_language: str | None = None
    instruction_style: str | None = None
    profile_id: str | None = None
    status: Literal["success", "failed"]
    execution_status: Literal["completed", "runtime_failed"] = "completed"
    evaluation_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    runtime_error_type: str | None = None
    tool_sequence_expected: list[str] = Field(default_factory=list)
    tool_sequence_observed: list[str] = Field(default_factory=list)
    tool_sequence_match: bool | None = None
    intent_route_match: bool | None = None
    artifact_required: bool = False
    artifact_present: bool = False
    leakage_detected: bool
    injection_detected: bool
    hallucination_detected: bool
    llm_attempt_count: int
    llm_retry_exhausted: bool = False
    latency_ms: float
    run_id: str | None = None
    trace_id: str | None = None
    langfuse_trace_url: str | None = None
    fact_score_1_5: float | None = None
    fact_verdict: Literal["pass", "fail"] | None = None
    fact_reason: str | None = None
    fact_eval_source: str | None = None
    fact_eval_applicable: bool | None = None
    fact_eval_error: str | None = None
    translation_bertscore_f1: float | None = None
    translation_reference_called: bool = False
    translation_reference_model: str | None = None
    output_language_match: bool | None = None
    structure_score: float | None = None
    structure_violations: list[str] = Field(default_factory=list)
    metric_applicability: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# Promptfoo adapter request/response contracts used by /promptfoo/evaluate.
class PromptfooEvaluateRequest(BaseModel):
    prompt: str
    test_domain: Literal["functional", "accuracy", "security", "simulation"] = "security"
    suite: Literal["eu", "functional", "accuracy", "security", "simulation"] = "eu"
    scenario_id: str
    category: Literal[
        "functional",
        "accuracy",
        "simulation",
        "leakage",
        "injection",
        "tool_misuse",
        "hallucination",
    ]
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    evaluator_mode: Literal["deterministic", "llm_judge"] = "deterministic"
    batch_id: str | None = None
    case_id: str | None = None
    company: str | None = None
    risk_tier: str | None = None
    task_type: str | None = None
    expected_tool_sequence: list[str] | str = Field(default_factory=list)
    artifact_required: bool | str = False
    input_language: str | None = None
    expected_output_language: str | None = None
    instruction_style: str | None = None
    forbidden_terms: list[str] | str = Field(default_factory=list)
    injection_markers: list[str] | str = Field(default_factory=list)
    unsupported_claims: list[str] | str = Field(default_factory=list)
    tool_call_threshold: int | None = None


class PromptfooEvaluateResponse(BaseModel):
    scenario_id: str
    test_domain: Literal["functional", "accuracy", "security", "simulation"] = "security"
    suite: Literal["eu", "functional", "accuracy", "security", "simulation"] = "eu"
    category: Literal[
        "functional",
        "accuracy",
        "simulation",
        "leakage",
        "injection",
        "tool_misuse",
        "hallucination",
    ]
    task_type: str | None = None
    resolved_task_type: str | None = None
    run_id: str
    batch_id: str | None = None
    case_id: str | None = None
    trace_id: str | None = None
    langfuse_trace_url: str | None = None
    agent_status: Literal["success", "failed"]
    evaluator_mode: Literal["deterministic", "llm_judge"] = "deterministic"
    base_gate_pass: bool
    final_text: str
    final_text_length: int
    leakage_hit_count: int
    leakage_hits: list[str] = Field(default_factory=list)
    injection_marker_hit_count: int
    injection_marker_hits: list[str] = Field(default_factory=list)
    hallucination_hit_count: int = 0
    hallucination_hits: list[str] = Field(default_factory=list)
    tool_call_count: int
    tool_call_threshold: int
    tool_misuse_flag: bool
    tool_sequence_expected: list[str] = Field(default_factory=list)
    tool_sequence_observed: list[str] = Field(default_factory=list)
    tool_sequence_match: bool | None = None
    artifact_required: bool = False
    artifact_present: bool = False
    llm_attempt_count: int
    llm_retry_exhausted: bool = False
    latency_ms: float = 0.0
    error_summary: str | None = None
    runtime_error_type: str | None = None
    judge_score: float | None = None
    judge_verdict: Literal["pass", "fail"] | None = None
    judge_reason: str | None = None
    fact_score_1_5: float | None = None
    fact_verdict: Literal["pass", "fail"] | None = None
    fact_reason: str | None = None
    fact_eval_source: str | None = None
    fact_eval_applicable: bool | None = None
    fact_eval_error: str | None = None
    translation_bertscore_f1: float | None = None
    translation_reference_called: bool = False
    translation_reference_model: str | None = None
    structure_score: float | None = None
    structure_violations: list[str] = Field(default_factory=list)
    output_language_match: bool | None = None
    metric_breakdown: dict[str, Any] = Field(default_factory=dict)
    assertion_gate_pass: bool | None = None
    assertion_reason: str | None = None


class PromptfooCaseSummary(BaseModel):
    scenario_id: str
    test_domain: Literal["functional", "accuracy", "security", "simulation"] | None = None
    category: str
    task_type: str | None = None
    resolved_task_type: str | None = None
    company: str | None = None
    risk_tier: str | None = None
    input_language: str | None = None
    expected_output_language: str | None = None
    instruction_style: str | None = None
    case_id: str | None = None
    batch_id: str | None = None
    execution_status: Literal["queued", "running", "completed", "failed", "runtime_failed"] | None = None
    passed: bool | None = None
    reason: str | None = None
    latency_ms: float | None = None
    run_id: str | None = None
    trace_id: str | None = None
    langfuse_trace_url: str | None = None
    llm_attempt_count: int | None = None
    llm_retry_exhausted: bool | None = None
    agent_status: Literal["success", "failed"] | None = None
    runtime_error_type: str | None = None
    step_events: list[StepEvent] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    policy_findings: list[str] = Field(default_factory=list)
    tool_sequence_expected: list[str] = Field(default_factory=list)
    tool_sequence_observed: list[str] = Field(default_factory=list)
    tool_sequence_match: bool | None = None
    intent_route_match: bool | None = None
    artifact_required: bool = False
    artifact_present: bool = False
    metric_breakdown: dict[str, Any] = Field(default_factory=dict)
    fact_eval_source: str | None = None
    fact_eval_applicable: bool | None = None
    fact_eval_error: str | None = None


# Aggregated test metrics; denominators are explicit to avoid misleading rates.
class TestSummary(BaseModel):
    success_rate: float
    completion_denominator: int = 0
    quality_denominator: int = 0
    runtime_error_rate: float = 0.0
    leakage_rate: float | None = None
    functional_leakage_rate: float | None = None
    injection_resilience: float | None = None
    hallucination_rate: float | None = None
    factual_score_avg_1_5: float | None = None
    factual_pass_rate: float | None = None
    translation_bertscore_avg: float | None = None
    structure_score_avg: float | None = None
    simulation_language_compliance: float | None = None
    simulation_robustness_ratio: float | None = None
    intent_route_accuracy: float | None = None
    intent_route_denominator: int = 0
    retry_exhaustion_rate: float
    avg_attempts: float
    avg_latency_ms: float
    leakage_tested: bool = False
    leakage_denominator: int = 0
    functional_leakage_tested: bool = False
    functional_leakage_denominator: int = 0
    injection_tested: bool = False
    injection_denominator: int = 0
    hallucination_tested: bool = False
    hallucination_denominator: int = 0
    factual_tested: bool = False
    factual_denominator: int = 0
    translation_tested: bool = False
    translation_denominator: int = 0
    structure_tested: bool = False
    structure_denominator: int = 0
    simulation_tested: bool = False
    simulation_denominator: int = 0


class CurrentCaseStatus(BaseModel):
    scenario_id: str
    run_id: str
    trace_id: str | None = None
    trace_url: str | None = None
    started_at: str
    status: Literal["queued", "running", "completed", "failed"]
    step_events: list[StepEvent] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    policy_findings: list[str] = Field(default_factory=list)


class TestHistoryRow(BaseModel):
    test_id: str
    suite: str
    test_domain: str | None = None
    status: Literal["running", "completed", "failed"]
    started_at: str
    ended_at: str | None = None
    total_cases: int = 0
    completed_cases: int = 0
    pass_rate: float | None = None


class TestLiveResponse(BaseModel):
    test_id: str
    status: Literal["running", "completed", "failed"]
    progress: float
    eta_seconds: float | None = None
    total_cases: int = 0
    completed_cases: int = 0
    planned_cases: int = 0
    observed_cases: int = 0
    completed_display: str | None = None
    current_case: CurrentCaseStatus | None = None
    all_cases: list[PromptfooCaseSummary] = Field(default_factory=list)
    trace_links: dict[str, str] = Field(default_factory=dict)
    promptfoo_batch_ui_url: str | None = None
    promptfoo_report_path: str | None = None


class TestRunStatusResponse(BaseModel):
    test_id: str
    status: Literal["running", "completed", "failed"]
    progress: float
    summary: TestSummary | None = None
    cases: list[TestCaseResult] = Field(default_factory=list)
    current_case: CurrentCaseStatus | None = None
    promptfoo_report_path: str | None = None
    promptfoo_ui_url: str | None = None
    promptfoo_summary: dict[str, object] | None = None
    promptfoo_case_results: list[PromptfooCaseSummary] | None = None
    promptfoo_category_summary: dict[str, float] | None = None
    metric_coverage: dict[str, bool | int | float | None] | None = None
    history_rows: list[TestHistoryRow] | None = None
    error: str | None = None


class TaskPreset(BaseModel):
    task_id: str
    label: str
    task_type: Literal["briefing_full", "web_only", "db_only", "doc_only", "translate_only", "general_chat"]
    prompt: str


class AttemptLog(BaseModel):
    attempt_index: int
    status_code: int | None = None
    latency_ms: float
    retry_reason: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMCallResult(BaseModel):
    text: str
    provider: str
    model: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    attempts: list[AttemptLog] = Field(default_factory=list)


class PromptBudget(BaseModel):
    context_window: int
    max_tokens: int
    safety_margin: int
    input_budget: int

    @model_validator(mode="after")
    def validate_budget(self) -> "PromptBudget":
        if self.input_budget <= 0:
            raise ValueError("input_budget must be > 0")
        return self
