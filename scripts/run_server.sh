#!/usr/bin/env bash
# Launch the perception server on the DGX, bound to the LAN.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
source "$VENV/bin/activate"
cd "$ROOT"
HOST="${AP_HOST:-0.0.0.0}"
PORT="${AP_PORT:-8000}"
echo "[run] http://$(hostname -I | awk '{print $1}'):$PORT  (binding $HOST)"
exec uvicorn server.app:app --host "$HOST" --port "$PORT" "$@"
