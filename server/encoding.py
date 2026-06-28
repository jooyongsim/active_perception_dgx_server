"""Wire (de)serialization helpers — decode uploaded bytes into arrays/images.

Everything crosses the LAN as raw uploaded bytes (never file paths), so the
Windows client and the DGX never have to agree on a filesystem layout.

Supported depth encodings (auto-detected by content / filename):
  - .npy : numpy array, uint16 (millimeters) or float32 (meters or mm) — raw units
  - .png : 16-bit single-channel PNG (millimeters), the RealSense default
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode_rgb(raw: bytes) -> Image.Image:
    """Bytes (PNG/JPG) -> PIL RGB image."""
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_depth(raw: bytes, filename: str | None = None) -> np.ndarray:
    """Bytes -> 2-D depth array in its native units (NOT scaled to meters).

    The caller multiplies by `depth_scale` to get meters. uint16/float kept as-is.
    """
    name = (filename or "").lower()
    if name.endswith(".npy") or raw[:6] == b"\x93NUMPY":
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        # 16-bit PNG (or any image) — preserve bit depth with ANYDEPTH.
        import cv2

        arr = cv2.imdecode(np.frombuffer(raw, np.uint8),
                           cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("could not decode depth image bytes")
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {arr.shape}")
    return arr


def load_npy(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def mask_to_png_b64(mask: np.ndarray) -> str:
    """bool (H,W) -> base64 of a 1-channel 0/255 PNG."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def array_to_npy_b64(arr: np.ndarray) -> str:
    """ndarray -> base64 of its .npy serialization (for optional cloud return)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")
