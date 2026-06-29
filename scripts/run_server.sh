#!/usr/bin/env bash
# Launch the perception server on the DGX, bound to the LAN.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
source "$VENV/bin/activate"
cd "$ROOT"
# Enable the Contact-GraspNet backend if it was installed into cgn_repo/.
if [ -d "$ROOT/cgn_repo" ]; then
  export CGN_CKPT="${CGN_CKPT:-cgn_repo/checkpoints/contact_graspnet}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"   # headless pyrender import
fi
HOST="${AP_HOST:-0.0.0.0}"
PORT="${AP_PORT:-8000}"
echo "[run] http://$(hostname -I | awk '{print $1}'):$PORT  (binding $HOST)"
exec uvicorn server.app:app --host "$HOST" --port "$PORT" "$@"
