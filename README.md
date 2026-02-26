# Live AI Agent Test Demo

End-to-end demo for an agentic consultant assistant with:

- LangGraph orchestration (real tools, no synthetic fallback)
- Together inference
  - agent model: `openai/gpt-oss-20b`
  - judge model: `Qwen/Qwen3-235B-A22B-Instruct-2507-tput`
- Internal SQLite + confidential PDF simulation
- Public retrieval via Tavily
- Confidentiality redaction policy in chat + documents
- Chainlit chat UI + Streamlit multi-page testing UI + admin explorer
- Langfuse trace telemetry
- Promptfoo-first batch testing (Security Minimal via UI + Full-EU via CLI)

## What Is Implemented

### Agent workflow
LangGraph nodes:

- `parse_intent`
- `plan`
- `validate_plan`
- `retrieve_parallel`
- `retrieve_internal_db`
- `retrieve_public_web`
- `retrieve_internal_pdf`
- `compose_template_document`
- `security_filter`
- `persist_artifacts`
- `finalize`
- `error`

Supported task types:

- `briefing_full`
- `web_only`
- `db_only`
- `doc_only`
- `translate_only`
- `general_chat` (OOD-safe route, no forced workflow tools)

Behavior summary:

- `briefing_full`: internal DB + public web retrieval in parallel, then briefing generation.
- `web_only`: public web retrieval with source links.
- `db_only`: sanitized internal DB summary.
- `doc_only`: sanitized internal document retrieval artifact.
- `translate_only`: translation path when source language differs from target.
- `general_chat`: safe chat fallback for unsupported intents.

### Reliability contract

- Provider: Together
- Deterministic defaults (`temperature=0`, `top_p=1`, `n=1`)
- Max trials per LLM call: 3 total (initial + retries)
- Retryable classes: timeout/transport, `429`, `5xx`, schema-invalid output
- Non-retryable: non-`429` `4xx`
- No fake output fallback

### Internal data model

SQLite file: `backend/data/internal_company.db`

Tables:

- `engagements`
- `internal_documents`
- `redaction_terms`

Confidential fields are redacted in consultant-facing outputs, including generated PDFs.

## Repository Structure

- Backend (API, graph, tools): `backend/app/`
- Chainlit chat UI: `frontend/chainlit_chat/app.py`
- Streamlit testing dashboard: `frontend/testing_ui/Home.py`
- Streamlit admin explorer: `frontend/testing_ui/pages/Admin_Explorer.py`
- Promptfoo configs: `promptfoo/`
- Synthetic data generator: `scripts/generate_synthetic_companies.py`
- Lifecycle scripts: `scripts/start_demo.sh`, `scripts/stop_demo.sh`
- Latest test plan snapshot: `TEST_PLAN_20260226.md`
- Latest results write-up: `RESULTS_20260226.md`

## Assignment Reviewer Guide

Use these files to locate assignment deliverables quickly:

- Q3 (test plan): `TEST_PLAN_20260226.md`
- Q4 (synthetic data generation): `scripts/generate_synthetic_companies.py` (Data already populated, no need to run again)
- Q5 (evaluation write-up): `RESULTS_20260226.md`
- Latest evidence bundle: `public/evidence/task5_20260226/`
- Latest Promptfoo batch artifacts (used by Q5): `backend/data/artifacts/promptfoo/`

## Prerequisites

For reviewer-grade full experience, all of the following are required:

- Python 3.13+
- Node.js 20+ (required by Promptfoo CLI)
- Docker + Docker Compose (required; Langfuse/Postgres are not optional in reviewer flow)
- Together API key (required)
- Tavily API key (required)
- SiliconFlow API key (required for translation-faithfulness scoring against Hunyuan MT reference)
- Internet access for first BERTScore run (downloads Hugging Face model `bert-base-multilingual-cased` used by translation-faithfulness scoring)
- Promptfoo CLI (`npx`) runtime available; Promptfoo account login may be requested in some environments for native red-team generation/eval

Pinned runtime defaults in this repo:

- Promptfoo CLI: `0.120.25`
- Langfuse Python SDK: `>=3.14.4`

## Setup

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp infra/langfuse/infra.env.example infra/langfuse/infra.env
```

BERTScore runtime dependency note:
- The first Accuracy/Simulation run that evaluates translation faithfulness triggers a one-time Hugging Face model download (`bert-base-multilingual-cased`) via `bert-score`.
- Default cache location is the user Hugging Face cache (for example `~/.cache/huggingface/`).
- If offline, translation-faithfulness metric will be unavailable until the model is cached.

If upgrading an existing environment:

```bash
.venv313/bin/pip install -U -r requirements.txt
npx -y promptfoo@0.120.25 --version
```

## First-Run Checklist (Clean Clone)

Run these steps in order for a fully usable demo environment:

1. Create and activate `.venv313`
2. Install dependencies
3. Configure `.env` and `infra/langfuse/infra.env`
4. Start stack
5. Verify internal DB is available (pre-seeded DB is included in repo)
6. Optionally regenerate synthetic data (only if you want to refresh/reset dataset)

## Environment Configuration

Use `.env` as your base config. Important fields:

```env
TOGETHER_API_KEY=...
TAVILY_API_KEY=...
TOGETHER_MODEL=openai/gpt-oss-20b
LLM_JUDGE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-tput

# Disable Langfuse native auto-evaluator bootstrap in this project
LANGFUSE_NATIVE_EVALUATOR_BOOTSTRAP_ENABLED=false

# Optional translation reference scoring
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.com/v1
SILICONFLOW_MT_MODEL=tencent/Hunyuan-MT-7B

# Promptfoo
PROMPTFOO_ENABLED=true
PROMPTFOO_COMMAND='npx -y promptfoo@0.120.25'
PROMPTFOO_PORT=15500
```

Reviewer mode (required for assignment verification):

```env
REQUIRE_POSTGRES_CHECKPOINTER=true
LANGFUSE_ENABLED=true
LANGGRAPH_POSTGRES_URI=postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@127.0.0.1:5432/langgraph
LANGFUSE_HOST=http://127.0.0.1:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PROJECT_ID=agent-demo-project
```

Important:
- Ensure `<POSTGRES_USER>` / `<POSTGRES_PASSWORD>` match your local Langfuse Docker env settings in `infra/langfuse/infra.env`.

Reviewer note:
- For assignment verification, use reviewer mode above.
- Security testing is Promptfoo native red-team only (no deterministic security fallback).
- Translation-faithfulness scoring is enabled only when SiliconFlow is configured.

Local development fallback mode (not reviewer-equivalent):

```env
REQUIRE_POSTGRES_CHECKPOINTER=false
LANGFUSE_ENABLED=false
```

`scripts/start_demo.sh` uses:

1. Root app env: `.env` (required)
2. Langfuse docker env: `infra/langfuse/infra.env` (required when docker stack is started)

The legacy `.run/backend_langfuse.env` override path is intentionally removed.

## Synthetic Data (Assignment Deliverable)

Important:
- This repository includes a pre-seeded SQLite file at `backend/data/internal_company.db` for reviewer convenience.
- Startup does **not** auto-seed synthetic records/documents.
- You only need to run the generator if you want to refresh/reset the dataset.

Optional reseed command (10 company profiles + SQLite/PDF data):

```bash
.venv313/bin/python scripts/generate_synthetic_companies.py \
  --db-path backend/data/internal_company.db \
  --seed 7 \
  --emit-profiles-json backend/data/synthetic_profiles.json \
  --emit-manifest-json backend/data/synthetic_manifest.json \
  --overwrite
```

Outputs:

- `backend/data/internal_company.db`
- `backend/data/synthetic_profiles.json`
- `backend/data/synthetic_manifest.json`

Quick readiness checks after generation:

```bash
curl -sS http://127.0.0.1:8000/api/v1/internal-db/health
curl -sS http://127.0.0.1:8000/api/v1/internal-db/documents
```

## Precomputed Artifacts Included (Latest Run)

This repo includes latest-run artifacts so reviewers can inspect outputs without rerunning everything first:

- Promptfoo batch artifacts for the latest run IDs under:
  - `backend/data/artifacts/promptfoo/7c378072-6d3e-4cc5-9a7d-dba20905722b/`
  - `backend/data/artifacts/promptfoo/6d9bb07c-f1e9-4766-9ec8-8292c0138de1/`
  - `backend/data/artifacts/promptfoo/24e9b29d-1a26-485c-86a4-ae4af1b089df/`
  - `backend/data/artifacts/promptfoo/8b3eadef-26f2-4cdf-bdec-2eb76868d743/`
  - `backend/data/artifacts/promptfoo/00403a81-5fff-4b19-9a59-d72093a7a63a/`
- Consolidated screenshots + JSON evidence:
  - `public/evidence/task5_20260226/`

## Start / Stop

### Start full stack (recommended)

```bash
scripts/start_demo.sh --with-docker
```

What this script does:

- validates Python UI/backend binaries
- validates Node runtime (`>=20`) for Promptfoo
- starts Langfuse/Postgres docker stack when needed
- starts backend (`8000`), Chainlit (`8501`), testing UI (`8502`)
- starts/ensures Promptfoo viewer (`15500`)
- fails fast with logs if startup is incomplete

Important for stable restarts during review:
- Start the stack from a persistent terminal session and keep that terminal open while testing.
- If the launcher terminal is interrupted/closed, background services may terminate and pages may become unreachable.
- Quick health check after restart:
  - `lsof -n -P -iTCP:8000 -sTCP:LISTEN`
  - `lsof -n -P -iTCP:8501 -sTCP:LISTEN`
  - `lsof -n -P -iTCP:8502 -sTCP:LISTEN`

### Stop everything

```bash
scripts/stop_demo.sh --with-docker
```

### Clean restart

```bash
scripts/stop_demo.sh --with-docker
PROMPTFOO_COMMAND="$(which npx) -y promptfoo@0.120.25" scripts/start_demo.sh --with-docker
```

## URLs

- Backend docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Chainlit chat: [http://127.0.0.1:8501](http://127.0.0.1:8501)
- Testing UI: [http://127.0.0.1:8502](http://127.0.0.1:8502)
- Promptfoo viewer: [http://127.0.0.1:15500](http://127.0.0.1:15500)
- Langfuse: [http://127.0.0.1:3000](http://127.0.0.1:3000)

## Testing UI Domains

The Streamlit test control includes dedicated pages:

1. Functional Test
- completion/error rate
- expected tool route match
- artifact compliance
- intent-route accuracy
- internal sensitive-data leakage rate (deterministic redaction check against synthetic DB/PDF sensitive terms)

2. Accuracy Test
- hybrid evaluator mode (LLM judge + deterministic metrics)
- factual grounding (LLM judge 1-5)
- translation faithfulness (SiliconFlow reference + BERTScore)
- structure completeness/quality checks

3. Security Test
- Promptfoo-first security run
- Red Team Minimal Set (2 plugins):
  - `pii:direct`
  - `system-prompt-override`
- 10 cases per run
- native Promptfoo red-team generation/eval is required (no deterministic security fallback)
- Full EU profile is available via CLI-only trigger (`security-eu-full`), kept disabled in UI to avoid accidental expensive runs.

### Security full-EU CLI mode (tracked in Testing UI)

The Security page keeps the EU button disabled by design. You can still run the full profile via backend API, and it will appear in Security batch history.

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/v1/tests/start" \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "security-eu-full",
    "test_domain": "security",
    "reasoning_effort": "medium",
    "execution_mode": "promptfoo",
    "evaluator_mode": "deterministic",
    "repeat_count": 1
  }'
```

Expected behavior:
- batch is visible in Security page `Select batch`
- `Open Promptfoo View Result` resolves for the selected completed batch
- case-level `Open Trace` links resolve in Langfuse from the case table

Runtime estimate for full-EU profile:
- typical: ~25–90 minutes
- depends on provider latency, plugin grading path, and API rate limits

Full-EU profile plugin set (14):
- `hijacking`
- `excessive-agency`
- `imitation`
- `harmful:misinformation-disinformation`
- `overreliance`
- `pii:direct`
- `pii:session`
- `harmful:privacy`
- `pii:api-db`
- `shell-injection`
- `sql-injection`
- `ssrf`
- `hallucination`
- `harmful:hate`

4. Simulation Test
- language/style variation robustness (English/Chinese/German/Japanese/mixed)
- output-language compliance against scenario-level expected output language (gold labels), using dominant-language detection with a 70% character-ratio threshold (otherwise treated as `Mix` and fails)
- route + factual robustness checks

5. Admin Explorer (tester/reviewer view)
- browse raw SQLite tables (`engagements`, `internal_documents`, `redaction_terms`)
- inspect row-level content with pagination/filtering
- preview and download confidential demo PDF documents stored in DB
- download the SQLite file for audit/reproducibility checks
- intended for test/review only (not consultant-facing)

Quality metric semantics in UI:

- Completion/error metrics use all scheduled cases.
- Quality metrics use completed cases only.
- `N/A` means metric not applicable for that case.

## Promptfoo + Langfuse

### Promptfoo service endpoints

- `GET /api/v1/promptfoo/health`
- `POST /api/v1/promptfoo/restart`
- `GET /api/v1/promptfoo/log-tail?lines=200`
- `POST /api/v1/promptfoo/evaluate` (functional / accuracy / simulation eval target)
- `POST /api/v1/promptfoo/redteam-run` (security red-team target; returns plain final text for plugin scoring)

### Test orchestration endpoints

- `POST /api/v1/tests/start`
- `GET /api/v1/tests/history`
- `GET /api/v1/tests/{test_id}`
- `GET /api/v1/tests/{test_id}/live`
- `GET /api/v1/tests/{test_id}/promptfoo-meta`
- `GET /api/v1/tests/catalog`
- `GET /api/v1/evaluators/methodology`
- `GET /api/v1/evaluators/config`

Langfuse trace URLs are emitted per case and use `/trace/{trace_id}` format.

## LangGraph Diagram

A compiled graph PNG is exposed at:

- `GET /api/v1/graph/langgraph.png`

The testing Home page renders this image directly.

## Troubleshooting

1. `Promptfoo eval failed ... Node.js v18 is not supported`
- Use Node 20+ (22 recommended), then restart stack.

2. Missing env files at startup
- Missing app env: create `.env` from `.env.example`.
- Missing Langfuse env (docker mode): create `infra/langfuse/infra.env` from `infra/langfuse/infra.env.example`.

3. Services started but pages unreachable
- Check logs:
  - `.run/server.log`
  - `.run/chainlit.log`
  - `.run/testing_ui.log`
  - `.run/promptfoo_view.log`
- Then run a clean restart.

4. Promptfoo security redteam generation blocked by account/email verification
- Security run will fail until Promptfoo CLI identity is verified.
- Verify Promptfoo account/email, then rerun security batch.

5. Trace opens but page appears missing
- Ensure `LANGFUSE_HOST` and credentials point to the same running instance.
- Confirm trace ID exists in current environment and project.

6. Postgres/checkpointer errors on backend startup
- If you do not need full checkpointing, set `REQUIRE_POSTGRES_CHECKPOINTER=false`.
- Otherwise start with `--with-docker`.

7. Promptfoo native module crash (`better-sqlite3`, `ERR_DLOPEN_FAILED`, `NODE_MODULE_VERSION ...`)
- Cause: mixed local Node runtimes reusing stale `npx` cache artifacts.
- Use the project startup script (it resolves Node 20+ and pins Promptfoo runtime).
- If it still occurs, clear stale npx cache and restart:
  - `rm -rf ~/.npm/_npx`
  - `scripts/stop_demo.sh --with-docker`
  - `PROMPTFOO_COMMAND="$(which npx) -y promptfoo@0.120.25" scripts/start_demo.sh --with-docker`

## Developer Validation

Run tests:

```bash
.venv313/bin/pytest -q
```

Optional API smoke checks:

```bash
curl -sS http://127.0.0.1:8000/api/v1/config/public
curl -sS http://127.0.0.1:8000/api/v1/tests/catalog
curl -sS http://127.0.0.1:8000/api/v1/promptfoo/health
```
