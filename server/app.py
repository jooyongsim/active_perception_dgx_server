"""FastAPI app — the perception server that runs on the DGX Spark.

Run:
    source .venv/bin/activate
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    # or: python -m server.app

Endpoints
---------
GET  /health           backends + device + GPU status
POST /segment          rgb + prompt            -> detections (boxes + masks)
POST /grasps           cloud OR depth+K        -> 6-DoF grasps
POST /perceive         rgb + depth + K + prompt-> masks + grasps (one round-trip)

All clouds/grasps are in the **camera optical frame** (OpenCV: +x right, +y down,
+z forward, meters). The client transforms to world/robot with its own VIO pose
and hand-eye calibration.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import config, grasp, segmentation
from .encoding import (array_to_npy_b64, decode_depth, decode_rgb, load_npy,
                       mask_to_png_b64)
from .geometry import Intrinsics, cloud_from_depth, downsample, scene_cloud

app = FastAPI(title="Active-Perception Server", version="1.0")

# Permit browser-based debug tools on the LAN. Harmless for a trusted local net.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _warmup():
    # Eagerly load the default backends so the first real request isn't slow.
    # Failures are recorded (not fatal) — other backends still serve.
    segmentation.try_load(config.DEFAULT_SEG_BACKEND)
    grasp.try_load(config.DEFAULT_GRASP_BACKEND)


def _gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"cuda": True, "device_name": torch.cuda.get_device_name(0),
                    "torch": torch.__version__}
        return {"cuda": False, "torch": torch.__version__}
    except Exception as e:  # pragma: no cover
        return {"cuda": False, "error": str(e)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": config.pick_device(),
        "gpu": _gpu_info(),
        "segmentation": segmentation.status(),
        "grasp": grasp.status(),
        "defaults": {"seg": config.DEFAULT_SEG_BACKEND,
                     "grasp": config.DEFAULT_GRASP_BACKEND},
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _detection_json(d, with_mask: bool = True) -> dict:
    out = {"label": d.label, "score": round(d.score, 4),
           "box": [round(float(v), 2) for v in d.box]}
    if with_mask:
        out["mask_png_b64"] = mask_to_png_b64(d.mask)
    return out


def _grasps_json(grasps) -> list[dict]:
    return [{"pose": g.pose.tolist(), "width": round(g.width, 4),
             "score": round(g.score, 4)} for g in grasps]


def _err(msg: str, **extra) -> dict:
    return {"error": msg, **extra}


# --------------------------------------------------------------------------- #
# /segment — image + prompt -> detections (boxes + masks)
# --------------------------------------------------------------------------- #
@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    backend: str = Form(config.DEFAULT_SEG_BACKEND),
    box_threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
):
    if backend not in segmentation.AVAILABLE:
        return _err(f"unknown seg backend '{backend}'", available=segmentation.AVAILABLE)
    try:
        seg = segmentation.get_segmenter(backend)
    except Exception as e:  # noqa: BLE001
        return _err(f"backend '{backend}' unavailable: {e}")

    pil = decode_rgb(await image.read())
    dets = seg.detect_segment(pil, prompt, box_threshold, text_threshold)
    return {"width": pil.width, "height": pil.height, "backend": backend,
            "detections": [_detection_json(d) for d in dets]}


# --------------------------------------------------------------------------- #
# /grasps — cloud OR depth+intrinsics -> 6-DoF grasps
# --------------------------------------------------------------------------- #
@app.post("/grasps")
async def grasps(
    # Option 1: precomputed cloud (back-compatible with the reference contract)
    cloud: UploadFile | None = File(None),
    segmentation_mask: UploadFile | None = File(None),   # .npy bool (N,)
    # Option 2: let the server build the cloud from a depth frame
    depth: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),                # .png/.npy (H,W)
    fx: float = Form(0.0), fy: float = Form(0.0),
    cx: float = Form(0.0), cy: float = Form(0.0),
    depth_scale: float = Form(0.0),     # 0 => auto-infer from dtype
    # common
    gripper_width_max: float = Form(0.085),
    topk: int = Form(20),
    backend: str = Form(config.DEFAULT_GRASP_BACKEND),
):
    if backend not in grasp.AVAILABLE:
        return _err(f"unknown grasp backend '{backend}'", available=grasp.AVAILABLE)

    # Build the (xyz, seg) pair from whichever input was supplied.
    if cloud is not None:
        pts = load_npy(await cloud.read())
        if pts.ndim != 2 or pts.shape[1] not in (3, 6):
            return _err(f"cloud must be (N,3) or (N,6), got {pts.shape}")
        xyz = pts[:, :3].astype(np.float64)
        seg = None
        if segmentation_mask is not None:
            seg = load_npy(await segmentation_mask.read()).astype(bool).reshape(-1)
    elif depth is not None:
        if fx <= 0 or fy <= 0:
            return _err("depth path requires fx, fy, cx, cy intrinsics")
        depth_raw = decode_depth(await depth.read(), depth.filename)
        m = None
        if mask is not None:
            m = decode_depth(await mask.read(), mask.filename).astype(bool)
        xyz, _, seg = cloud_from_depth(depth_raw, Intrinsics(fx, fy, cx, cy),
                                       depth_scale, mask=m)
    else:
        return _err("provide either `cloud` (.npy) or `depth` + intrinsics")

    xyz, seg = downsample(xyz, seg, config.MAX_POINTS)

    try:
        sampler = grasp.get_grasp_backend(backend)
        gs = sampler.sample_grasps(xyz, seg, gripper_width_max, topk)
    except Exception as e:  # noqa: BLE001
        return _err(str(e), backend=backend)

    return {"frame": "camera_optical", "backend": backend,
            "num_points": int(len(xyz)), "grasps": _grasps_json(gs)}


# --------------------------------------------------------------------------- #
# /perceive — the headline endpoint: rgb + depth + K + prompt -> masks + grasps
# --------------------------------------------------------------------------- #
@app.post("/perceive")
async def perceive(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    fx: float = Form(...), fy: float = Form(...),
    cx: float = Form(...), cy: float = Form(...),
    depth_scale: float = Form(0.0),     # 0 => auto-infer from dtype
    prompt: str = Form(...),
    seg_backend: str = Form(config.DEFAULT_SEG_BACKEND),
    grasp_backend: str = Form(config.DEFAULT_GRASP_BACKEND),
    box_threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
    gripper_width_max: float = Form(0.085),
    topk: int = Form(20),
    max_detections: int = Form(5),
    return_masks: bool = Form(True),
    return_cloud: bool = Form(False),
):
    """Full pipeline in one round-trip.

    1. segment `rgb` with the chosen seg backend (text prompt)
    2. deproject `depth` -> metric camera-frame cloud
    3. per detection: mask the cloud, run the grasp backend
    Returns detections, each with its grasps (camera frame).
    """
    if seg_backend not in segmentation.AVAILABLE:
        return _err(f"unknown seg backend '{seg_backend}'", available=segmentation.AVAILABLE)
    if grasp_backend not in grasp.AVAILABLE:
        return _err(f"unknown grasp backend '{grasp_backend}'", available=grasp.AVAILABLE)

    # decode inputs
    rgb_bytes = await rgb.read()
    pil = decode_rgb(rgb_bytes)
    rgb_np = np.asarray(pil)
    depth_raw = decode_depth(await depth.read(), depth.filename)
    if depth_raw.shape[:2] != (pil.height, pil.width):
        return _err(f"rgb {(pil.height, pil.width)} and depth {depth_raw.shape[:2]} "
                    "size mismatch")
    intr = Intrinsics(fx, fy, cx, cy)

    # 1. segmentation
    try:
        seg = segmentation.get_segmenter(seg_backend)
    except Exception as e:  # noqa: BLE001
        return _err(f"seg backend '{seg_backend}' unavailable: {e}")
    dets = seg.detect_segment(pil, prompt, box_threshold, text_threshold)
    dets = sorted(dets, key=lambda d: d.score, reverse=True)[:max_detections]

    # 2/3. per-detection cloud + grasps
    try:
        sampler = grasp.get_grasp_backend(grasp_backend)
    except Exception as e:  # noqa: BLE001
        return _err(f"grasp backend '{grasp_backend}' unavailable: {e}")

    # Build the scene cloud ONCE; map each detection's mask onto it via `valid`.
    # Passing the object through `segmentation` (not a pre-extracted cloud) means
    # the grasp backend uses object points directly — no spurious plane removal —
    # while still seeing scene context (Contact-GraspNet uses it for local regions).
    xyz_full, valid = scene_cloud(depth_raw, intr, depth_scale)
    rng = np.random.default_rng(0)
    n = len(xyz_full)
    if config.MAX_POINTS and n > config.MAX_POINTS:
        idx = rng.choice(n, config.MAX_POINTS, replace=False)
    else:
        idx = np.arange(n)
    xyz_ds = xyz_full[idx]

    results = []
    for d in dets:
        seg_ds = d.mask[valid][idx]                  # (M,) bool, aligned to xyz_ds
        n_obj = int(seg_ds.sum())
        entry = _detection_json(d, with_mask=return_masks)
        entry["num_object_points"] = n_obj
        if n_obj >= 20:
            gs = sampler.sample_grasps(xyz_ds, seg_ds, gripper_width_max, topk)
            entry["grasps"] = _grasps_json(gs)
        else:
            entry["grasps"] = []
            entry["note"] = "too few object points for a stable grasp"
        if return_cloud:
            entry["object_cloud_npy_b64"] = array_to_npy_b64(
                xyz_ds[seg_ds].astype(np.float32))
        results.append(entry)

    return {
        "width": pil.width, "height": pil.height,
        "frame": "camera_optical",
        "seg_backend": seg_backend, "grasp_backend": grasp_backend,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "depth_scale": depth_scale},
        "detections": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
