"""Service A -- GroundedSAM (Grounding-DINO + SAM) on :8001.

    mamba activate lerobot
    uvicorn service_gsam:app --host 0.0.0.0 --port 8001

POST /detect_segment  (multipart: image=<png/jpg>, prompt="the mug")
  -> {width, height, detections:[{label, score, box, mask_png_b64}]}

Contract matches TOOLS_SERVER_WSL_SETUP.md section 4.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from gsam_model import GroundedSAM

app = FastAPI(title="GroundedSAM", version="1.0")
_model: GroundedSAM | None = None


def get_model() -> GroundedSAM:
    global _model
    if _model is None:
        _model = GroundedSAM()      # loads once, on first request
    return _model


def mask_to_png_b64(mask: np.ndarray) -> str:
    """bool (H,W) -> base64 PNG, 1-channel 0/255."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.on_event("startup")
def _warmup():
    # Eager-load so the first /detect_segment isn't penalized by model init.
    get_model()


@app.get("/health")
def health():
    m = get_model()
    return {"status": "ok", "device": m.device,
            "gdino": m.gdino.config._name_or_path,
            "sam": m.sam.config._name_or_path}


@app.post("/detect_segment")
async def detect_segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    model = get_model()
    dets = model.detect_segment(pil, prompt, box_threshold, text_threshold)
    return {
        "width": pil.width,
        "height": pil.height,
        "detections": [
            {"label": d.label, "score": round(d.score, 4),
             "box": [round(v, 2) for v in d.box],
             "mask_png_b64": mask_to_png_b64(d.mask)}
            for d in dets
        ],
    }
