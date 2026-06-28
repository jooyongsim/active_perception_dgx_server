"""Headless 3D visualization of grasps over a point cloud -> PNG.

Uses matplotlib's Agg backend so it works over SSH/WSL with no display.
Draws the cloud (object points highlighted) and a parallel-jaw gripper marker
for each grasp, colored by score.
"""

from __future__ import annotations

import numpy as np

# Local gripper schematic (meters), in the grasp frame:
#   x = binormal (jaw-closing), y = hand, z = approach (+z toward object).
# Fingertips sit at the contact plane (z=0); the palm/wrist extends back (-z).
_FINGER_LEN = 0.04
_WRIST_LEN = 0.03


def _gripper_lines(pose: np.ndarray, width: float) -> list[np.ndarray]:
    R, t = pose[:3, :3], pose[:3, 3]
    hw = max(width, 0.005) / 2.0
    fl, wl = _FINGER_LEN, _WRIST_LEN
    pts = {
        "Lt": (-hw, 0, 0.0), "Rt": (hw, 0, 0.0),
        "Lb": (-hw, 0, -fl), "Rb": (hw, 0, -fl),
        "mid": (0, 0, -fl), "wrist": (0, 0, -fl - wl),
    }
    w = {k: t + R @ np.array(v) for k, v in pts.items()}
    return [np.stack([w["Lt"], w["Lb"]]),     # left finger
            np.stack([w["Rt"], w["Rb"]]),     # right finger
            np.stack([w["Lb"], w["Rb"]]),     # base bar
            np.stack([w["mid"], w["wrist"]])]  # wrist


def save_grasp_figure(cloud: np.ndarray, grasps, path: str,
                      segmentation: np.ndarray | None = None,
                      title: str = "grasps", max_pts: int = 6000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    cloud = np.asarray(cloud)
    if len(cloud) > max_pts:
        idx = np.random.default_rng(0).choice(len(cloud), max_pts, replace=False)
        cloud, seg = cloud[idx], (None if segmentation is None
                                  else np.asarray(segmentation)[idx])
    else:
        seg = segmentation

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if seg is not None:
        ax.scatter(*cloud[~seg].T, s=1, c="lightgray", alpha=0.4)
        ax.scatter(*cloud[seg].T, s=2, c="steelblue", alpha=0.6)
    else:
        ax.scatter(*cloud.T, s=1, c="lightgray", alpha=0.4)

    scores = [g.score for g in grasps] or [0]
    smin, smax = min(scores), max(scores)
    for g in grasps:
        c = cm.viridis((g.score - smin) / (smax - smin + 1e-9))
        for seg_line in _gripper_lines(g.pose, g.width):
            ax.plot(*seg_line.T, c=c, linewidth=2)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"{title} — {len(grasps)} grasps (color = score)")
    ax.view_init(elev=-70, azim=-90)   # look roughly down the camera +z axis
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
