#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv313}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
CHAINLIT_HOST="${CHAINLIT_HOST:-127.0.0.1}"
CHAINLIT_PORT="${CHAINLIT_PORT:-8501}"
TESTING_UI_HOST="${TESTING_UI_HOST:-127.0.0.1}"
TESTING_UI_PORT="${TESTING_UI_PORT:-8502}"
PROMPTFOO_PORT="${PROMPTFOO_PORT:-15500}"

# Demo stack bootstrap script.
# Responsibilities:
# - validate env/venv prerequisites
# - start backend + Chainlit + Streamlit with stable local ports
# - optionally start dockerized infra (Langfuse/Postgres) when requested
# Boundaries:
# - long-running app logic remains in Python services; this is orchestration glue

WITH_DOCKER="auto"
if [[ $# -gt 0 ]]; then
  case "${1:-}" in
    --with-docker)
      WITH_DOCKER="true"
      ;;
    --no-docker)
      WITH_DOCKER="false"
      ;;
    *)
      echo "Usage: scripts/start_demo.sh [--with-docker|--no-docker]" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Virtualenv not found at $VENV_DIR" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
  echo "uvicorn not found in $VENV_DIR. Run: $VENV_DIR/bin/pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/chainlit" ]]; then
  echo "chainlit not found in $VENV_DIR. Run: $VENV_DIR/bin/pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/streamlit" ]]; then
  echo "streamlit not found in $VENV_DIR. Run: $VENV_DIR/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Local helpers for env loading, port checks, and promptfoo runtime detection.
# ---------------------------------------------------------------------------
load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
    echo "Loaded env from $file"
  fi
}

is_listening() {
  local port="$1"
  lsof -n -P -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local timeout_seconds="$2"
  local label="$3"
  local elapsed=0
  while (( elapsed < timeout_seconds )); do
    if is_listening "$port"; then
      echo "$label is ready on port $port"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "Warning: $label did not become ready on port $port within ${timeout_seconds}s" >&2
  return 1
}

resolve_promptfoo_runtime() {
  if [[ "${PROMPTFOO_ENABLED:-true}" != "true" ]]; then
    return 0
  fi

  # If nvm is active, prioritize its binaries so node/npx resolve consistently.
  if [[ -n "${NVM_BIN:-}" && -d "${NVM_BIN}" ]]; then
    export PATH="$NVM_BIN:$PATH"
  fi

  local node_bin
  local npx_bin
  node_bin="$(command -v node || true)"
  npx_bin="$(command -v npx || true)"
  if [[ -z "$node_bin" || -z "$npx_bin" ]]; then
    echo "Promptfoo requires Node.js 20+ with npx, but node/npx were not found in PATH." >&2
    echo "Install Node.js 22 (recommended) and retry." >&2
    echo "Example: nvm install 22 && nvm use 22" >&2
    exit 1
  fi

  local node_major
  node_major="$("$node_bin" -e 'process.stdout.write(process.versions.node.split(".")[0])' 2>/dev/null || true)"
  if [[ -z "$node_major" || ! "$node_major" =~ ^[0-9]+$ || "$node_major" -lt 20 ]]; then
    echo "Promptfoo requires Node.js 20+, but resolved node is $("$node_bin" -v 2>/dev/null || echo unknown) at $node_bin" >&2
    echo "Fix PATH so Node 20+ is first, then restart." >&2
    echo "Example: nvm install 22 && nvm use 22" >&2
    exit 1
  fi

  local configured_cmd="${PROMPTFOO_COMMAND:-}"
  local configured_head=""
  if [[ -n "$configured_cmd" ]]; then
    configured_head="${configured_cmd%% *}"
  fi

  # Preserve configured promptfoo version when present.
  local promptfoo_pkg="promptfoo@0.120.25"
  if [[ "$configured_cmd" == *"promptfoo@"* ]]; then
    local detected_pkg
    detected_pkg="$(printf "%s" "$configured_cmd" | sed -nE 's/.*(promptfoo@[0-9.]+).*/\1/p')"
    if [[ -n "$detected_pkg" ]]; then
      promptfoo_pkg="$detected_pkg"
    fi
  fi

  # If command is unset or npx-based, pin to absolute npx path from current Node runtime.
  if [[ -z "$configured_cmd" || "$configured_head" == "npx" || "$(basename "$configured_head" 2>/dev/null || true)" == "npx" ]]; then
    PROMPTFOO_COMMAND="$npx_bin -y $promptfoo_pkg"
    export PROMPTFOO_COMMAND
    echo "Resolved PROMPTFOO_COMMAND: $PROMPTFOO_COMMAND"
  fi
}

# Optional infra startup used when --with-docker/auto resolves true.
start_langfuse_stack() {
  local compose_file="$ROOT_DIR/infra/langfuse/docker-compose.yml"
  local compose_env="$ROOT_DIR/infra/langfuse/infra.env"
  if [[ ! -f "$compose_file" ]]; then
    echo "Langfuse compose file missing: $compose_file" >&2
    return 1
  fi
  if [[ ! -f "$compose_env" ]]; then
    echo "Langfuse env file missing: $compose_env" >&2
    echo "Create it from: infra/langfuse/infra.env.example" >&2
    return 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for Langfuse/Postgres startup but was not found." >&2
    return 1
  fi

  docker compose --env-file "$compose_env" -f "$compose_file" up -d >"$RUN_DIR/langfuse_docker.log" 2>&1 || {
    tail -n 80 "$RUN_DIR/langfuse_docker.log" >&2 || true
    return 1
  }
  echo "Langfuse docker stack started."
}

# Load app secrets/config from root .env only.
if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing required env file: $ROOT_DIR/.env" >&2
  echo "Create it from: .env.example" >&2
  exit 1
fi
load_env_file "$ROOT_DIR/.env"

resolve_promptfoo_runtime

START_DOCKER_STACK="false"
if [[ "$WITH_DOCKER" == "true" ]]; then
  START_DOCKER_STACK="true"
elif [[ "$WITH_DOCKER" == "auto" ]]; then
  if [[ "${REQUIRE_POSTGRES_CHECKPOINTER:-false}" == "true" || "${LANGFUSE_ENABLED:-false}" == "true" ]]; then
    START_DOCKER_STACK="true"
  fi
fi

if [[ "$START_DOCKER_STACK" == "true" ]]; then
  start_langfuse_stack || {
    echo "Failed to start Langfuse docker stack." >&2
    exit 1
  }
  if [[ "${REQUIRE_POSTGRES_CHECKPOINTER:-false}" == "true" ]]; then
    wait_for_port 5432 120 "Postgres" || {
      echo "Postgres is required but unavailable; aborting startup." >&2
      exit 1
    }
  fi
  if [[ "${LANGFUSE_ENABLED:-false}" == "true" ]]; then
    wait_for_port 3000 120 "Langfuse UI" || {
      echo "Langfuse UI did not become ready; aborting startup." >&2
      exit 1
    }
  fi
fi

if is_listening "$BACKEND_PORT"; then
  sleep 1
fi
if is_listening "$BACKEND_PORT"; then
  echo "Backend already listening on port $BACKEND_PORT"
else
  nohup env PYTHONPATH=backend "$VENV_DIR/bin/uvicorn" app.main:app \
    --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
    >"$RUN_DIR/server.log" 2>&1 &
  echo $! >"$RUN_DIR/server.pid"
  echo "Started backend (pid $(cat "$RUN_DIR/server.pid"))"
fi
wait_for_port "$BACKEND_PORT" 90 "Backend" || {
  tail -n 120 "$RUN_DIR/server.log" >&2 || true
  echo "Backend failed to start; aborting startup." >&2
  exit 1
}

if is_listening "$CHAINLIT_PORT"; then
  sleep 1
fi
if is_listening "$CHAINLIT_PORT"; then
  echo "Chainlit already listening on port $CHAINLIT_PORT"
else
  nohup env AGENT_API_BASE="http://$BACKEND_HOST:$BACKEND_PORT/api/v1" \
    "$VENV_DIR/bin/chainlit" run "$ROOT_DIR/frontend/chainlit_chat/app.py" \
    --host "$CHAINLIT_HOST" \
    --port "$CHAINLIT_PORT" \
    >"$RUN_DIR/chainlit.log" 2>&1 &
  echo $! >"$RUN_DIR/chainlit.pid"
  echo "Started chainlit (pid $(cat "$RUN_DIR/chainlit.pid"))"
fi
wait_for_port "$CHAINLIT_PORT" 30 "Chainlit" || {
  tail -n 120 "$RUN_DIR/chainlit.log" >&2 || true
  echo "Chainlit failed to start; aborting startup." >&2
  exit 1
}

if is_listening "$TESTING_UI_PORT"; then
  sleep 1
fi
if is_listening "$TESTING_UI_PORT"; then
  echo "Testing Streamlit already listening on port $TESTING_UI_PORT"
else
  nohup env AGENT_API_BASE="http://$BACKEND_HOST:$BACKEND_PORT/api/v1" \
    "$VENV_DIR/bin/streamlit" run "$ROOT_DIR/frontend/testing_ui/Home.py" \
    --server.address "$TESTING_UI_HOST" \
    --server.port "$TESTING_UI_PORT" \
    >"$RUN_DIR/testing_ui.log" 2>&1 &
  echo $! >"$RUN_DIR/testing_ui.pid"
  echo "Started testing UI (pid $(cat "$RUN_DIR/testing_ui.pid"))"
fi
wait_for_port "$TESTING_UI_PORT" 30 "Testing UI" || {
  tail -n 120 "$RUN_DIR/testing_ui.log" >&2 || true
  echo "Testing UI failed to start; aborting startup." >&2
  exit 1
}

echo
echo "Backend API:   http://$BACKEND_HOST:$BACKEND_PORT/docs"
echo "Chainlit Chat: http://$CHAINLIT_HOST:$CHAINLIT_PORT"
echo "Testing UI:    http://$TESTING_UI_HOST:$TESTING_UI_PORT"
echo "Promptfoo UI:  http://127.0.0.1:$PROMPTFOO_PORT"
if [[ "${LANGFUSE_ENABLED:-false}" == "true" || "$START_DOCKER_STACK" == "true" ]]; then
  echo "Langfuse UI:   http://127.0.0.1:3000"
fi
