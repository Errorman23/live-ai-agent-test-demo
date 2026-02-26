from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.graph.nodes import AgentNodes, _infer_internal_doc_type
from app.internal_db import InternalDBRepository, ensure_internal_db
from app.telemetry.langfuse_client import LangfuseTelemetry
from app.tools.real_tools import RealToolbox
from app.artifacts import save_briefing_pdf_artifact


class _DummyProvider:
    model = "openai/gpt-oss-20b"


class _DummyToolbox:
    pass


def _build_nodes(tmp_path: Path, *, artifacts_dir: str) -> AgentNodes:
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
        ARTIFACTS_DIR=artifacts_dir,
    )
    telemetry = LangfuseTelemetry(enabled=False)
    telemetry.start_trace("run-1", trace_id="trace-1", session_id="session-1", user_id="user-1", input_payload={})
    ensure_internal_db(settings.internal_db_path)
    return AgentNodes(
        settings=settings,
        provider=_DummyProvider(),  # type: ignore[arg-type]
        telemetry=telemetry,
        toolbox=_DummyToolbox(),  # type: ignore[arg-type]
        repository=InternalDBRepository(settings.internal_db_path),
    )


def test_security_filter_uses_redacted_placeholder(tmp_path):
    db_path = str(tmp_path / "redact.db")
    ensure_internal_db(db_path)
    repo = InternalDBRepository(db_path)
    repo.upsert_redaction_term(company_name="Tencent", term="Project Phoenix", term_type="project_name")
    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=db_path,
    )
    toolbox = RealToolbox(settings=settings, repository=repo)
    result = toolbox.security_filter(
        "Project Phoenix has budget USD 1,000,000 for launch.",
        company_name="Tencent",
    )
    text = result.data["document"]
    assert "[REDACTED]" in text
    assert "[SENSITIVE]" not in text
    assert "Project Phoenix" not in text


def test_persist_artifacts_creates_briefing_pdf(tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")
    nodes = _build_nodes(tmp_path, artifacts_dir=artifacts_dir)
    state = {
        "run_id": "run-brief",
        "trace_id": "trace-1",
        "status": "success",
        "task_type": "briefing_full",
        "output_mode": "document",
        "company_name": "Tencent",
        "final_document": "# Briefing\nSensitive item [REDACTED]\n",
        "internal_doc_type": "proposal",
        "step_events": [],
        "tool_call_records": [],
        "policy_findings": [],
        "llm_tokens": [],
        "artifacts": [],
    }
    updated = nodes.persist_artifacts_node(state)
    artifacts = updated.get("artifacts", [])
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["kind"] == "briefing-pdf"
    assert artifact["filename"].endswith(".pdf")
    assert (artifact.get("metadata") or {}).get("renderer") == "briefing_corporate_v1"
    assert Path(artifact["path"]).exists()


def test_save_briefing_pdf_artifact_supports_chinese_target(tmp_path):
    artifact = save_briefing_pdf_artifact(
        root_dir=str(tmp_path / "artifacts"),
        run_id="run-cn",
        company_name="Tencent",
        text="# 顾问简报: Tencent\n\n## 执行摘要\n这是一个测试文档。\n",
        target_language="Chinese",
    )
    assert artifact["kind"] == "briefing-pdf"
    assert Path(artifact["path"]).exists()
    assert (artifact.get("metadata") or {}).get("target_language") == "Chinese"


def test_persist_artifacts_creates_sanitized_internal_pdf_for_translate_chat(tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")
    nodes = _build_nodes(tmp_path, artifacts_dir=artifacts_dir)
    state = {
        "run_id": "run-doc",
        "trace_id": "trace-1",
        "status": "success",
        "task_type": "translate_only",
        "output_mode": "chat",
        "company_name": "Tencent",
        "final_document": "Sanitized internal quotation. [REDACTED]",
        "internal_doc_type": "quotation",
        "step_events": [],
        "tool_call_records": [],
        "policy_findings": [],
        "llm_tokens": [],
        "artifacts": [],
    }
    updated = nodes.persist_artifacts_node(state)
    artifacts = updated.get("artifacts", [])
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["kind"] == "sanitized-quotation-pdf"
    assert artifact["filename"].endswith(".pdf")
    assert (artifact.get("metadata") or {}).get("renderer") == "internal_sanitized_v1"
    assert Path(artifact["path"]).exists()


def test_persist_artifacts_creates_sanitized_internal_pdf_for_doc_only(tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")
    nodes = _build_nodes(tmp_path, artifacts_dir=artifacts_dir)
    state = {
        "run_id": "run-doc-only",
        "trace_id": "trace-1",
        "status": "success",
        "task_type": "doc_only",
        "output_mode": "chat",
        "company_name": "Sony",
        "source_language": "Japanese",
        "internal_doc_type": "proposal",
        "final_document": "Retrieved a sanitized copy.",
        "artifact_document_text": "機密プロジェクト名: [REDACTED]",
        "tool_results": {
            "retrieve_internal_pdf": {
                "doc_type": "proposal",
                "language": "Japanese",
                "file_name": "sony_proposal.pdf",
            }
        },
        "step_events": [],
        "tool_call_records": [],
        "policy_findings": [],
        "llm_tokens": [],
        "artifacts": [],
    }
    updated = nodes.persist_artifacts_node(state)
    artifacts = updated.get("artifacts", [])
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["kind"] == "sanitized-proposal-pdf"
    assert artifact["filename"].endswith(".pdf")
    assert (artifact.get("metadata") or {}).get("renderer") == "internal_sanitized_v1"
    assert Path(artifact["path"]).exists()


def test_infer_internal_doc_type_detects_quotation_keyword():
    assert _infer_internal_doc_type("Get Tencent quotation from internal db.") == "quotation"
    assert _infer_internal_doc_type("Get Tencent proposal from internal db.") == "proposal"
