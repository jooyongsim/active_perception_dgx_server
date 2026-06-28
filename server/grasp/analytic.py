"""Analytic 6-DoF parallel-jaw grasp sampler (numpy + scipy, CPU-only).

Antipodal sampling: estimate surface normals, then for sampled contact points
find an opposing contact across the object within the gripper width whose normal
is roughly anti-parallel (force-closure). Each accepted pair becomes a 6-DoF
grasp, scored by antipodal alignment quality.

Grasp pose convention (4x4, in the INPUT cloud's camera frame, meters):
    columns of R = [binormal (jaw-closing axis), hand axis, approach axis(+z)]
    translation   = grasp center (midpoint of the two contacts)
The client applies any approach standoff and transforms to the robot frame.

Works with no open3d (none on aarch64): uses scipy.spatial.cKDTree for neighbor
queries and local-PCA normals. Falls back to a global PCA normal if scipy is
also missing.
"""

from __future__ import annotations

import numpy as np

from .base import Grasp, GraspBackend

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _estimate_normals(pts: np.ndarray, k: int = 30) -> np.ndarray:
    """Per-point normals via local PCA, oriented toward the camera origin."""
    if cKDTree is None or len(pts) < k:
        c = pts - pts.mean(0)
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        return np.broadcast_to(vh[-1], pts.shape).copy()

    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=min(k, len(pts)))
    normals = np.empty_like(pts)
    for i in range(len(pts)):
        neigh = pts[nbr[i]]
        cov = np.cov((neigh - neigh.mean(0)).T)
        w, v = np.linalg.eigh(cov)
        n = v[:, 0]                              # smallest-eigenvalue direction
        if n @ (-pts[i]) < 0:                    # orient toward camera (origin)
            n = -n
        normals[i] = n
    return normals


def _remove_plane(pts: np.ndarray, dist: float = 0.01,
                  iters: int = 200, seed: int = 0) -> np.ndarray:
    """RANSAC-remove the dominant plane (table) to isolate the object."""
    if len(pts) < 100:
        return pts
    rng = np.random.default_rng(seed)
    best_in = None
    n = len(pts)
    for _ in range(iters):
        s = pts[rng.choice(n, 3, replace=False)]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        nv = np.cross(v1, v2)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        d = np.abs((pts - s[0]) @ nv)
        inl = d < dist
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None:
        return pts
    obj = pts[~best_in]
    return obj if len(obj) > 50 else pts


def sample_grasps(cloud: np.ndarray,
                  segmentation: np.ndarray | None = None,
                  gripper_width_max: float = 0.085,
                  topk: int = 20,
                  n_samples: int = 2000,
                  seed: int = 0) -> list[Grasp]:
    cloud = np.asarray(cloud, np.float64)
    if segmentation is not None:
        obj = cloud[np.asarray(segmentation, bool)]
    else:
        obj = _remove_plane(cloud)
    if len(obj) < 20:
        return []

    normals = _estimate_normals(obj)
    rng = np.random.default_rng(seed)
    n = len(obj)
    tree = cKDTree(obj) if cKDTree is not None else None

    grasps: list[Grasp] = []
    idxs = rng.choice(n, size=min(n_samples, n), replace=False)
    for i in idxs:
        p1, n1 = obj[i], normals[i]
        nn = np.linalg.norm(n1)
        if nn < 1e-6:
            continue
        n1 = n1 / nn

        if tree is not None:
            nbr = np.asarray(tree.query_ball_point(p1, gripper_width_max))
            if len(nbr) < 2:
                continue
            cand, cand_n = obj[nbr], normals[nbr]
        else:
            cand, cand_n = obj, normals

        d = cand - p1
        dist = np.linalg.norm(d, axis=1)
        valid = (dist > 0.005) & (dist <= gripper_width_max)
        if not valid.any():
            continue
        proj = (d[valid] @ (-n1)) / np.maximum(dist[valid], 1e-9)
        sub = np.where(valid)[0][proj > 0.8]
        if len(sub) == 0:
            continue
        j = sub[np.argmax(dist[sub])]            # widest aligned antipodal pair
        p2, n2 = cand[j], cand_n[j]
        n2n = np.linalg.norm(n2)
        if n2n < 1e-6:
            continue
        n2 = n2 / n2n

        antipodal = float(-(n1 @ n2))            # 1.0 = perfectly opposed
        if antipodal < 0.3:
            continue

        center = (p1 + p2) / 2.0
        width = float(np.linalg.norm(p2 - p1))
        binormal = (p2 - p1) / max(width, 1e-9)  # jaw-closing axis (x)
        approach0 = np.array([0.0, 0.0, 1.0])     # camera looks along +z
        approach = approach0 - (approach0 @ binormal) * binormal
        an = np.linalg.norm(approach)
        if an < 1e-6:
            continue
        approach = approach / an                  # z
        hand = np.cross(approach, binormal)       # y
        R = np.stack([binormal, hand, approach], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center

        score = antipodal * (1.0 - 0.5 * width / gripper_width_max)
        grasps.append(Grasp(T, width, float(np.clip(score, 0, 1))))

    grasps.sort(key=lambda g: g.score, reverse=True)
    # Non-max suppression by grasp center (5 mm) to drop near-duplicates.
    kept: list[Grasp] = []
    for g in grasps:
        if all(np.linalg.norm(g.pose[:3, 3] - k.pose[:3, 3]) > 0.005 for k in kept):
            kept.append(g)
        if len(kept) >= topk:
            break
    return kept


class AnalyticGrasp(GraspBackend):
    name = "analytic"
    device = "cpu"

    def sample_grasps(self, cloud, segmentation=None,
                      gripper_width_max=0.085, topk=20) -> list[Grasp]:
        return sample_grasps(cloud, segmentation, gripper_width_max, topk)
