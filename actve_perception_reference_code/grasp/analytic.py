"""Analytic 6-DoF parallel-jaw grasp sampler (numpy + open3d, CPU-only).

Antipodal sampling: estimate surface normals, then for sampled contact points
find an opposing contact across the object within the gripper width whose normal
is roughly anti-parallel (force-closure condition). Each accepted pair becomes a
6-DoF grasp pose, scored by antipodal alignment quality.

Grasp pose convention (4x4, in the INPUT cloud's frame, meters):
    columns of R = [binormal (jaw-closing axis), hand axis, approach axis(+z)]
    translation   = grasp center (midpoint of the two contacts)
The client applies any approach standoff and transforms to the robot frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d
except Exception:  # open3d optional; normals fall back to a PCA estimate
    o3d = None


@dataclass
class Grasp:
    pose: np.ndarray      # (4,4)
    width: float          # meters between contacts
    score: float          # [0,1]


def _estimate_normals(pts: np.ndarray, radius: float = 0.02) -> np.ndarray:
    if o3d is not None:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        pc.orient_normals_towards_camera_location(np.zeros(3))  # toward camera origin
        return np.asarray(pc.normals)
    # Fallback: single global normal from PCA (coarse).
    c = pts - pts.mean(0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    n = vh[-1]
    return np.broadcast_to(n, pts.shape).copy()


def remove_plane(pts: np.ndarray, dist: float = 0.01) -> np.ndarray:
    """RANSAC-remove the dominant plane (table) to isolate the object."""
    if o3d is None or len(pts) < 100:
        return pts
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    _, inliers = pc.segment_plane(distance_threshold=dist,
                                  ransac_n=3, num_iterations=200)
    mask = np.ones(len(pts), bool)
    mask[inliers] = False
    return pts[mask] if mask.sum() > 50 else pts


def sample_grasps(cloud: np.ndarray,
                  segmentation: np.ndarray | None = None,
                  gripper_width_max: float = 0.085,
                  topk: int = 20,
                  n_samples: int = 2000,
                  seed: int = 0) -> list[Grasp]:
    """Sample antipodal 6-DoF grasps on the object points of `cloud` (N,3).

    If `segmentation` (bool, len N) is given, only its True points are the
    object; otherwise the dominant plane is RANSAC-removed first.
    """
    cloud = np.asarray(cloud, np.float64)
    if segmentation is not None:
        obj = cloud[np.asarray(segmentation, bool)]
    else:
        obj = remove_plane(cloud)
    if len(obj) < 20:
        return []

    normals = _estimate_normals(obj)
    rng = np.random.default_rng(seed)
    n = len(obj)
    half_w = gripper_width_max / 2.0

    # Build a KD-tree for opposing-contact lookups.
    if o3d is not None:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(obj)
        kdt = o3d.geometry.KDTreeFlann(pc)
    else:
        kdt = None

    grasps: list[Grasp] = []
    idxs = rng.choice(n, size=min(n_samples, n), replace=False)
    for i in idxs:
        p1, n1 = obj[i], normals[i]
        nn = np.linalg.norm(n1)
        if nn < 1e-6:
            continue
        n1 = n1 / nn

        # Look for the opposing contact along the -n1 ray within gripper width.
        if kdt is not None:
            _, nbr, _ = kdt.search_radius_vector_3d(p1, gripper_width_max)
            nbr = np.asarray(nbr)
            if len(nbr) < 2:
                continue
            cand = obj[nbr]
            cand_n = normals[nbr]
        else:
            cand, cand_n = obj, normals

        d = cand - p1
        dist = np.linalg.norm(d, axis=1)
        valid = (dist > 0.005) & (dist <= gripper_width_max)
        if not valid.any():
            continue
        # opposing point should lie along -n1 (cos of angle close to 1)
        proj = (d[valid] @ (-n1)) / np.maximum(dist[valid], 1e-9)
        sub = np.where(valid)[0][proj > 0.8]
        if len(sub) == 0:
            continue
        # pick the farthest aligned point -> widest stable antipodal pair
        j = sub[np.argmax(dist[sub])]
        p2, n2 = cand[j], cand_n[j]
        n2n = np.linalg.norm(n2)
        if n2n < 1e-6:
            continue
        n2 = n2 / n2n

        # Antipodal force-closure quality: normals should oppose.
        antipodal = float(-(n1 @ n2))          # 1.0 = perfectly opposed
        if antipodal < 0.3:
            continue

        center = (p1 + p2) / 2.0
        width = float(np.linalg.norm(p2 - p1))
        binormal = (p2 - p1) / max(width, 1e-9)         # jaw-closing axis (x)
        approach0 = np.array([0.0, 0.0, 1.0])           # camera looks along +z
        approach = approach0 - (approach0 @ binormal) * binormal
        an = np.linalg.norm(approach)
        if an < 1e-6:
            continue
        approach = approach / an                        # z
        hand = np.cross(approach, binormal)             # y
        R = np.stack([binormal, hand, approach], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center

        # Score: antipodal alignment, prefer narrower (more stable) grasps.
        score = antipodal * (1.0 - 0.5 * width / gripper_width_max)
        grasps.append(Grasp(T, width, float(np.clip(score, 0, 1))))

    grasps.sort(key=lambda g: g.score, reverse=True)
    # Non-maximum suppression by grasp center (5 mm) to avoid near-duplicates.
    kept: list[Grasp] = []
    for g in grasps:
        if all(np.linalg.norm(g.pose[:3, 3] - k.pose[:3, 3]) > 0.005
               for k in kept):
            kept.append(g)
        if len(kept) >= topk:
            break
    return kept
