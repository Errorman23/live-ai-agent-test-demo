#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

BACKEND_PORT="${BACKEND_PORT:-8000}"
CHAINLIT_PORT="${CHAINLIT_PORT:-8501}"
TESTING_UI_PORT="${TESTING_UI_PORT:-8502}"
PROMPTFOO_PORT="${PROMPTFOO_PORT:-15500}"

stop_pid_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$file" || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 0.2
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "Stopped pid $pid from $(basename "$file")"
  fi
  rm -f "$file"
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:$port" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "$pids" | xargs -I{} kill {} >/dev/null 2>&1 || true
  sleep 0.2
  pids="$(lsof -ti "tcp:$port" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -I{} kill -9 {} >/dev/null 2>&1 || true
  fi
  echo "Stopped listeners on port $port"
}

stop_pid_file "$RUN_DIR/server.pid"
stop_pid_file "$RUN_DIR/chainlit.pid"
stop_pid_file "$RUN_DIR/testing_ui.pid"
stop_port "$BACKEND_PORT"
stop_port "$CHAINLIT_PORT"
stop_port "$TESTING_UI_PORT"
stop_port "$PROMPTFOO_PORT"

if [[ "${1:-}" == "--with-docker" ]]; then
  if command -v docker >/dev/null 2>&1; then
    COMPOSE_FILE="$ROOT_DIR/infra/langfuse/docker-compose.yml"
    COMPOSE_ENV="$ROOT_DIR/infra/langfuse/infra.env"
    if [[ -f "$COMPOSE_ENV" ]]; then
      docker compose --env-file "$COMPOSE_ENV" -f "$COMPOSE_FILE" down >/dev/null 2>&1 || true
    else
      docker compose -f "$COMPOSE_FILE" down >/dev/null 2>&1 || true
    fi
    echo "Langfuse docker stack stopped."
  else
    echo "Docker not found; skipped docker shutdown."
  fi
fi

echo "Demo apps stopped."
