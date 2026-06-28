"""Contact-GraspNet backend wrapper (PyTorch port).

Wraps `contact_graspnet_pytorch` (https://github.com/elchun/contact_graspnet_pytorch)
behind the same interface as the analytic sampler so Service B can switch backends.

The port ships its checkpoint in-repo (`checkpoints/contact_graspnet/checkpoints/
model.pt`) and uses a pure-PyTorch PointNet2 (no custom CUDA op to build), so it
runs on CPU and accelerates on GPU automatically. Loads net + weights once. If the
package is not importable, `ContactGraspNet()` raises a clear, actionable error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Grasp:
    pose: np.ndarray      # (4,4) in the input cloud's frame, meters
    width: float
    score: float


class ContactGraspNet:
    """Lazy loader for the Contact-GraspNet PyTorch port."""

    def __init__(self, ckpt_dir: str | None = None):
        try:
            from contact_graspnet_pytorch import config_utils
            from contact_graspnet_pytorch.checkpoints import CheckpointIO
            from contact_graspnet_pytorch.contact_grasp_estimator import (
                GraspEstimator)
        except Exception as e:
            raise RuntimeError(
                "contact_graspnet_pytorch is not installed. Install with:\n"
                "  git clone https://github.com/elchun/contact_graspnet_pytorch\n"
                "  cd contact_graspnet_pytorch && pip install -e .\n"
                "(the checkpoint ships in the repo). "
                f"Import error: {e}") from e

        ckpt_dir = ckpt_dir or os.environ.get(
            "CGN_CKPT", "cgn_repo/checkpoints/contact_graspnet")
        cfg = config_utils.load_config(ckpt_dir, batch_size=1)
        self.estimator = GraspEstimator(cfg)
        self.device = self.estimator.device
        cpio = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"),
                            model=self.estimator.model)
        # The repo's checkpoint predates torch>=2.6's weights_only=True default
        # and stores numpy scalars. We trust this in-repo checkpoint, so force a
        # full load just for this call (scoped monkeypatch, restored after).
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        try:
            cpio.load("model.pt")
        finally:
            torch.load = _orig_load
        self.estimator.model.eval()

    @torch.no_grad()
    def sample_grasps(self, cloud: np.ndarray,
                      segmentation: np.ndarray | None = None,
                      gripper_width_max: float = 0.085,
                      topk: int = 20, **_) -> list[Grasp]:
        cloud = np.asarray(cloud, np.float32)
        if segmentation is not None:
            pc_seg = {1: cloud[np.asarray(segmentation, bool)]}
            local, filt = True, True
        else:
            pc_seg, local, filt = {}, False, False

        pred, scores, _, widths = self.estimator.predict_scene_grasps(
            cloud, pc_segments=pc_seg, local_regions=local, filter_grasps=filt)

        out: list[Grasp] = []
        for key in pred:
            poses = np.asarray(pred[key]).reshape(-1, 4, 4)
            sc = np.asarray(scores[key]).reshape(-1)
            wd = np.asarray(widths[key]).reshape(-1)
            for T, s, w in zip(poses, sc, wd):
                out.append(Grasp(T.astype(np.float64), float(w), float(s)))
        out.sort(key=lambda g: g.score, reverse=True)
        return out[:topk]
