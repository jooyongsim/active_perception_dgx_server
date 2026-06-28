"""Contact-GraspNet backend (PyTorch port) — optional, learned 6-DoF grasps.

Wraps `contact_graspnet_pytorch` (https://github.com/elchun/contact_graspnet_pytorch)
behind the GraspBackend interface. The port ships its checkpoint in-repo and uses
a pure-PyTorch PointNet2 (no custom CUDA op to build), so it runs on the GB10 GPU.
If the package is not installed, constructing this backend raises an actionable
RuntimeError and `/grasps?backend=cgn` returns that error — `analytic` still works.

Install on the DGX (once):
    git clone https://github.com/elchun/contact_graspnet_pytorch cgn_repo
    pip install -e cgn_repo --no-deps --no-build-isolation && pip install pyrender
    export CGN_CKPT=cgn_repo/checkpoints/contact_graspnet
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .. import config
from .base import Grasp, GraspBackend


class ContactGraspNet(GraspBackend):
    name = "cgn"

    def __init__(self, ckpt_dir: str | None = None):
        try:
            from contact_graspnet_pytorch import config_utils
            from contact_graspnet_pytorch.checkpoints import CheckpointIO
            from contact_graspnet_pytorch.contact_grasp_estimator import (
                GraspEstimator)
        except Exception as e:
            raise RuntimeError(
                "contact_graspnet_pytorch is not installed. See the install note "
                f"at the top of grasp/contact_graspnet.py. Import error: {e}") from e

        ckpt_dir = ckpt_dir or config.CGN_CKPT
        cfg = config_utils.load_config(ckpt_dir, batch_size=1)
        self.estimator = GraspEstimator(cfg)
        self.device = str(self.estimator.device)
        cpio = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"),
                            model=self.estimator.model)
        # The repo's checkpoint predates torch>=2.6's weights_only=True default and
        # stores numpy scalars; force a full load for this trusted in-repo file.
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            cpio.load("model.pt")
        finally:
            torch.load = _orig
        self.estimator.model.eval()

    @torch.no_grad()
    def sample_grasps(self, cloud, segmentation=None,
                      gripper_width_max=0.085, topk=20) -> list[Grasp]:
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
