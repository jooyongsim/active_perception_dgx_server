"""Service B -- grasp pose detection on :8002.

    mamba activate lerobot
    uvicorn service_grasp:app --host 0.0.0.0 --port 8002

POST /grasps  (multipart: cloud=<.npy (N,3) or (N,6) in meters>,
               optional segmentation=<.npy bool (N,)>, gripper_width_max, topk,
               backend="analytic"|"cgn")
  -> {frame:"input_cloud", backend, grasps:[{pose:4x4, width, score}]}

Contract matches TOOLS_SERVER_WSL_SETUP.md section 4. Two backends:
  analytic -- antipodal sampler (CPU, always available)
  cgn      -- Contact-GraspNet PyTorch port (needs install + checkpoint; eGPU)
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from grasp import analytic

app = FastAPI(title="GraspPoseDetection", version="1.0")
_cgn = None  # lazy Contact-GraspNet singleton


def get_cgn():
    global _cgn
    if _cgn is None:
        from grasp.contact_graspnet import ContactGraspNet
        _cgn = ContactGraspNet()
    return _cgn


def _load_npy(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


@app.get("/health")
def health():
    return {"status": "ok", "backends": ["analytic", "cgn"],
            "default": "analytic"}


@app.post("/grasps")
async def grasps(
    cloud: UploadFile = File(...),
    segmentation: UploadFile | None = File(None),
    gripper_width_max: float = Form(0.085),
    topk: int = Form(20),
    backend: str = Form("analytic"),
):
    pts = _load_npy(await cloud.read())
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        return {"error": f"cloud must be (N,3) or (N,6), got {pts.shape}"}
    xyz = pts[:, :3].astype(np.float64)

    seg = None
    if segmentation is not None:
        seg = _load_npy(await segmentation.read()).astype(bool).reshape(-1)

    if backend == "cgn":
        try:
            sampler = get_cgn()
            gs = sampler.sample_grasps(xyz, seg, gripper_width_max, topk)
        except RuntimeError as e:
            return {"error": str(e), "backend": "cgn"}
    else:
        gs = analytic.sample_grasps(xyz, seg, gripper_width_max, topk)

    return {
        "frame": "input_cloud",
        "backend": backend,
        "grasps": [
            {"pose": g.pose.tolist(),
             "width": round(g.width, 4),
             "score": round(g.score, 4)}
            for g in gs
        ],
    }
