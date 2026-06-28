"""Depth + intrinsics -> metric point cloud, and mask-driven cloud extraction.

All clouds are in the **camera optical frame** (OpenCV convention):
    +x right, +y down, +z forward (out of the lens), units = meters.
The client transforms grasps into world/robot frame with its own VIO pose + the
hand-eye calibration (the DGX never needs those extrinsics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)


def effective_depth_scale(depth_raw: np.ndarray, depth_scale: float) -> float:
    """Resolve depth_scale, auto-inferring when <= 0.

    meters = depth_raw * scale. Two common encodings:
      - integer (uint16) raw RealSense depth in millimeters -> 0.001
      - float depth already converted to meters               -> 1.0
    Pass an explicit positive `depth_scale` (e.g. the RealSense SDK value) to
    override; <= 0 means "infer from dtype".
    """
    if depth_scale and depth_scale > 0:
        return float(depth_scale)
    return 1.0 if np.issubdtype(depth_raw.dtype, np.floating) else 0.001


def deproject(depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """(H,W) depth in meters -> (H,W,3) XYZ in the camera frame.

    Pixels with depth <= 0 keep z=0 (filtered out downstream by the valid mask).
    """
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    z = depth_m
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=-1)


def cloud_from_depth(
    depth_raw: np.ndarray,
    intr: Intrinsics,
    depth_scale: float,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    z_min: float = 0.05,
    z_max: float = 5.0,
):
    """Build a metric point cloud from a depth frame.

    Returns (points, colors, seg) where:
      points : (N,3) float64 meters, only valid (finite, in-range) pixels
      colors : (N,3) uint8 or None (if rgb given)
      seg    : (N,) bool or None — True where `mask` is set (object pixels)

    When `mask` is given, points outside it are still returned (so grasp
    backends keep table/scene context) but flagged False in `seg`.
    """
    scale = effective_depth_scale(depth_raw, depth_scale)
    depth_m = depth_raw.astype(np.float64) * scale
    xyz = deproject(depth_m, intr)

    valid = np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max)
    pts = xyz[valid]

    colors = None
    if rgb is not None:
        if rgb.shape[:2] != depth_raw.shape[:2]:
            raise ValueError(
                f"rgb {rgb.shape[:2]} and depth {depth_raw.shape[:2]} size mismatch")
        colors = rgb[valid].astype(np.uint8)

    seg = None
    if mask is not None:
        if mask.shape != depth_raw.shape:
            raise ValueError(
                f"mask {mask.shape} and depth {depth_raw.shape} size mismatch")
        seg = mask.astype(bool)[valid]

    return pts, colors, seg


def scene_cloud(depth_raw: np.ndarray, intr: Intrinsics, depth_scale: float,
                z_min: float = 0.05, z_max: float = 5.0):
    """Build the full-frame cloud + the (H,W) valid-pixel map.

    Returns (xyz_full (N,3) meters, valid (H,W) bool). Index any image-space mask
    with `valid` to get a per-point bool aligned to `xyz_full` — that's how a
    detection's segmentation is mapped onto the cloud.
    """
    scale = effective_depth_scale(depth_raw, depth_scale)
    depth_m = depth_raw.astype(np.float64) * scale
    xyz = deproject(depth_m, intr)
    valid = np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max)
    return xyz[valid], valid


def object_cloud_from_depth(
    depth_raw: np.ndarray,
    intr: Intrinsics,
    depth_scale: float,
    mask: np.ndarray,
    **kw,
) -> np.ndarray:
    """Convenience: just the object's (N,3) points (mask True & valid depth)."""
    pts, _, seg = cloud_from_depth(depth_raw, intr, depth_scale, mask=mask, **kw)
    return pts[seg] if seg is not None else pts


def downsample(points: np.ndarray, seg: np.ndarray | None, max_points: int,
               seed: int = 0):
    """Random-subsample to <= max_points (0 disables). Keeps seg aligned."""
    if max_points <= 0 or len(points) <= max_points:
        return points, seg
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], (seg[idx] if seg is not None else None)
