"""Runtime configuration, driven entirely by environment variables.

Every knob has a sane default so the server runs with zero config on the DGX.
Override any of these by exporting the variable before launching uvicorn.
"""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def pick_device() -> str:
    """'cuda' when a GPU + torch CUDA build are present, else 'cpu'."""
    forced = os.environ.get("AP_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --- Network -----------------------------------------------------------------
HOST = _env("AP_HOST", "0.0.0.0")          # bind on all interfaces (LAN access)
PORT = int(_env("AP_PORT", "8000"))

# --- Devices / models --------------------------------------------------------
DEVICE = os.environ.get("AP_DEVICE")        # None => auto-detect per request

# Segmentation backends
GDINO_MODEL = _env("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
SAM_MODEL = _env("SAM_MODEL", "facebook/sam-vit-base")
SAM3_MODEL = _env("SAM3_MODEL", "facebook/sam3")

# Grasp backend (Contact-GraspNet checkpoint dir)
CGN_CKPT = _env("CGN_CKPT", "cgn_repo/checkpoints/contact_graspnet")

# --- Defaults / limits -------------------------------------------------------
DEFAULT_SEG_BACKEND = _env("AP_DEFAULT_SEG_BACKEND", "gsam")
DEFAULT_GRASP_BACKEND = _env("AP_DEFAULT_GRASP_BACKEND", "analytic")

# Cap the point count fed to grasp models (random downsample above this) to keep
# latency bounded. 0 disables the cap.
MAX_POINTS = int(_env("AP_MAX_POINTS", "80000"))

# Reject absurd payloads early (MB). RGB+depth at 640x480 is ~1-2 MB.
MAX_UPLOAD_MB = int(_env("AP_MAX_UPLOAD_MB", "64"))
