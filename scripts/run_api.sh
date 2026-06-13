#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
