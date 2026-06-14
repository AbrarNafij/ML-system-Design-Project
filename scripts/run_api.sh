#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
fi

# Default API address for local demo
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"
OPEN_HOST="$HOST"
if [[ "$HOST" == "0.0.0.0" ]]; then
  OPEN_HOST="127.0.0.1"
fi

RELOAD="${API_RELOAD:-0}"
UVICORN_ARGS=(api.main:app --host "$HOST" --port "$PORT")
if [[ "$RELOAD" == "1" ]]; then
  UVICORN_ARGS+=(--reload --reload-dir api --reload-dir src)
fi

echo "Starting API on http://$OPEN_HOST:$PORT"
exec uvicorn "${UVICORN_ARGS[@]}"
