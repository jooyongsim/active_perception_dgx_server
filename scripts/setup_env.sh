#!/usr/bin/env bash
# Set up the DGX Spark (aarch64 / GB10 / CUDA 13) Python env for the perception server.
# Idempotent-ish: re-running re-installs into the same venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"

echo "[setup] project root: $ROOT"
echo "[setup] venv:         $VENV"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- PyTorch for aarch64 + CUDA 13 (GB10 / Blackwell, sm_121) ---------------
# The cu130 index ships CUDA-enabled aarch64 (sbsa) wheels. If this machine ever
# has no GPU, swap the index-url for the default PyPI (CPU) wheel.
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision

# --- Core server + perception deps ------------------------------------------
pip install \
  "fastapi>=0.115" "uvicorn[standard]>=0.30" python-multipart \
  numpy pillow "opencv-python-headless>=4.9" scipy requests \
  "transformers>=4.46" "huggingface_hub>=0.25" accelerate safetensors

# open3d is optional (better normals / KD-tree / plane removal). It may not have
# an aarch64 wheel; the analytic grasp backend falls back to scipy if absent.
pip install open3d || echo "[setup] open3d unavailable on this platform — using scipy fallback"

echo "[setup] done. Verify with:  source $VENV/bin/activate && python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
