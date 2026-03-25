#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${WORKBENCH_PORT:-8000}"

cd "$ROOT_DIR"
export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

if lsof -ti TCP:"$PORT" >/dev/null 2>&1; then
  echo "Stopping existing process on port $PORT"
  lsof -ti TCP:"$PORT" | xargs -r kill
  sleep 1
fi

exec "$ROOT_DIR/.venv/bin/uvicorn" shipyard.api:app \
  --reload \
  --reload-dir "$ROOT_DIR/shipyard" \
  --port "$PORT"
