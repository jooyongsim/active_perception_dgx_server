"""Run the Contact-GraspNet backend on examples/cloud.npy (uses GPU if present).

  python examples/demo_cgn.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from grasp.contact_graspnet import ContactGraspNet
from grasp.viz import save_grasp_figure


def main():
    cloud = np.load(os.path.join(HERE, "cloud.npy"))
    seg_path = os.path.join(HERE, "cloud_seg.npy")
    seg = np.load(seg_path) if os.path.exists(seg_path) else None

    cgn = ContactGraspNet()
    print(f"device={cgn.device}  cloud={cloud.shape}")
    grasps = cgn.sample_grasps(cloud, seg, gripper_width_max=0.085, topk=5)
    print(f"\n{len(grasps)} grasps (top 5):")
    for i, g in enumerate(grasps):
        t = g.pose[:3, 3]
        print(f"  [{i}] score={g.score:.3f}  width={g.width*1000:5.1f}mm  "
              f"center=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})m")
    if grasps:
        print("\nbest pose (4x4):")
        print(np.array2string(grasps[0].pose, precision=3, suppress_small=True))
        out = os.path.join(HERE, "..", "out", "cgn_grasps.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_grasp_figure(cloud, grasps, out, seg, title="Contact-GraspNet")
        print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
