"""Windows-side adaptor for the DGX active-perception server.

This is the ONLY new piece the Windows PC needs: your existing code already does
RealSense capture (RGB + depth), camera intrinsics, and pose/VIO analysis. Drop
this client in and call it with the numpy frames you already have.

    from perception_client import PerceptionClient

    client = PerceptionClient("http://192.168.45.150:8000")   # the DGX on the LAN
    result = client.perceive(
        rgb=color_image,            # (H,W,3) uint8
        depth=depth_image,          # (H,W)   uint16 (millimeters) or float meters
        intrinsics=(fx, fy, cx, cy),
        depth_scale=0.001,          # meters per depth unit (RealSense default)
        prompt="the mug",
    )
    for det in result["detections"]:
        mask = det["mask"]                 # (H,W) bool, decoded for you
        for g in det["grasps"]:
            T_cam = g["pose"]              # (4,4) grasp in CAMERA frame, meters

Frames: grasp poses come back in the CAMERA optical frame (OpenCV: +x right,
+y down, +z forward). Use `to_world(T_cam, T_cam_to_world)` with your VIO/hand-eye
transform to move them into world/robot coordinates.

Dependencies: requests, numpy. (opencv-python or pillow used if present for PNG
encode/decode; a pure-numpy path is used otherwise.)
"""

from __future__ import annotations

import base64
import io
from typing import Iterable, Sequence

import numpy as np
import requests

# --- optional image codecs (the PC almost certainly has one) -----------------
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


def _encode_png(arr: np.ndarray, bgr: bool = False) -> bytes:
    """(H,W,3) uint8 -> PNG bytes. `bgr=True` if the array is BGR (OpenCV)."""
    if _HAVE_CV2:
        img = arr if bgr else arr[:, :, ::-1]          # cv2 wants BGR
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("cv2 failed to encode PNG")
        return buf.tobytes()
    if _HAVE_PIL:
        img = arr[:, :, ::-1] if bgr else arr          # PIL wants RGB
        bio = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(bio, format="PNG")
        return bio.getvalue()
    raise RuntimeError("need opencv-python or pillow to encode RGB")


def _encode_npy(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()


def _decode_mask_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if _HAVE_CV2:
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        return m > 0
    if _HAVE_PIL:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 0
    raise RuntimeError("need opencv-python or pillow to decode mask")


class PerceptionClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # --- introspection -------------------------------------------------------
    def health(self) -> dict:
        r = requests.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- segmentation only ---------------------------------------------------
    def segment(self, rgb: np.ndarray, prompt: str, backend: str = "gsam",
                box_threshold: float = 0.30, text_threshold: float = 0.25,
                bgr: bool = False) -> dict:
        files = {"image": ("rgb.png", _encode_png(rgb, bgr), "image/png")}
        data = {"prompt": prompt, "backend": backend,
                "box_threshold": box_threshold, "text_threshold": text_threshold}
        js = self._post("/segment", files, data)
        for d in js.get("detections", []):
            d["mask"] = _decode_mask_b64(d.pop("mask_png_b64"))
        return js

    # --- grasps from a precomputed cloud ------------------------------------
    def grasps_from_cloud(self, cloud: np.ndarray,
                          segmentation: np.ndarray | None = None,
                          backend: str = "analytic",
                          gripper_width_max: float = 0.085,
                          topk: int = 20) -> dict:
        files = {"cloud": ("cloud.npy", _encode_npy(np.asarray(cloud, np.float32)),
                           "application/octet-stream")}
        if segmentation is not None:
            files["segmentation_mask"] = (
                "seg.npy", _encode_npy(np.asarray(segmentation, bool)),
                "application/octet-stream")
        data = {"backend": backend, "gripper_width_max": gripper_width_max,
                "topk": topk}
        return self._grasp_post(files, data)

    # --- grasps from a depth frame (server builds the cloud) -----------------
    def grasps_from_depth(self, depth: np.ndarray,
                          intrinsics: Sequence[float],
                          mask: np.ndarray | None = None,
                          depth_scale: float = 0.0,    # 0 => server auto-infers
                          backend: str = "analytic",
                          gripper_width_max: float = 0.085,
                          topk: int = 20) -> dict:
        fx, fy, cx, cy = intrinsics
        files = {"depth": ("depth.npy", _encode_npy(depth),
                           "application/octet-stream")}
        if mask is not None:
            files["mask"] = ("mask.npy", _encode_npy(np.asarray(mask, bool)),
                             "application/octet-stream")
        data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": depth_scale,
                "backend": backend, "gripper_width_max": gripper_width_max,
                "topk": topk}
        return self._grasp_post(files, data)

    # --- full pipeline: rgb + depth + prompt -> masks + grasps ---------------
    def perceive(self, rgb: np.ndarray, depth: np.ndarray,
                 intrinsics: Sequence[float], prompt: str,
                 depth_scale: float = 0.0,    # 0 => server auto-infers from dtype
                 seg_backend: str = "gsam", grasp_backend: str = "analytic",
                 box_threshold: float = 0.30, text_threshold: float = 0.25,
                 gripper_width_max: float = 0.085, topk: int = 20,
                 max_detections: int = 5, return_masks: bool = True,
                 return_cloud: bool = False, bgr: bool = False) -> dict:
        fx, fy, cx, cy = intrinsics
        files = {
            "rgb": ("rgb.png", _encode_png(rgb, bgr), "image/png"),
            "depth": ("depth.npy", _encode_npy(depth), "application/octet-stream"),
        }
        data = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": depth_scale,
            "prompt": prompt, "seg_backend": seg_backend,
            "grasp_backend": grasp_backend, "box_threshold": box_threshold,
            "text_threshold": text_threshold, "gripper_width_max": gripper_width_max,
            "topk": topk, "max_detections": max_detections,
            "return_masks": return_masks, "return_cloud": return_cloud,
        }
        js = self._post("/perceive", files, data)
        for d in js.get("detections", []):
            if "mask_png_b64" in d:
                d["mask"] = _decode_mask_b64(d.pop("mask_png_b64"))
            d["grasps"] = [_grasp_to_np(g) for g in d.get("grasps", [])]
            if "object_cloud_npy_b64" in d:
                d["object_cloud"] = np.load(
                    io.BytesIO(base64.b64decode(d.pop("object_cloud_npy_b64"))))
        return js

    # --- internals -----------------------------------------------------------
    def _post(self, path: str, files: dict, data: dict) -> dict:
        r = requests.post(f"{self.base}{path}", files=files, data=data,
                          timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and js.get("error"):
            raise RuntimeError(f"server error on {path}: {js['error']}")
        return js

    def _grasp_post(self, files: dict, data: dict) -> dict:
        js = self._post("/grasps", files, data)
        js["grasps"] = [_grasp_to_np(g) for g in js.get("grasps", [])]
        return js


def _grasp_to_np(g: dict) -> dict:
    return {"pose": np.asarray(g["pose"], np.float64),
            "width": float(g["width"]), "score": float(g["score"])}


# --- frame utility -----------------------------------------------------------
def to_world(pose_cam: np.ndarray, T_cam_to_world: np.ndarray) -> np.ndarray:
    """Move a 4x4 grasp pose from the camera frame into the world/robot frame.

    `T_cam_to_world` is your (VIO pose) @ (hand-eye) transform — exactly the
    camera_to_world matrix your PC pipeline already computes.
    """
    return np.asarray(T_cam_to_world, np.float64) @ np.asarray(pose_cam, np.float64)


def best_grasp(detection: dict) -> dict | None:
    """Highest-scoring grasp of a perceive() detection, or None."""
    gs = detection.get("grasps") or []
    return max(gs, key=lambda g: g["score"]) if gs else None
