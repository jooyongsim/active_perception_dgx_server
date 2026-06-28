"""End-to-end smoke test against a running server, using the bundled dataset.

    python scripts/smoke_test.py [--server http://localhost:8000] [--prompt "..."]

Checks: /health, /segment, /grasps (depth path), and /perceive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "client"))
from perception_client import PerceptionClient, best_grasp  # noqa: E402

DATASET = os.path.join(HERE, "..", "realsense_D435i_dataset", "dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--prompt", default="the object")
    ap.add_argument("--frame", default="frame_000000")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(DATASET, "meta.json")))
    K = meta["K"]
    intr = (K["fx"], K["fy"], K["cx"], K["cy"])
    # This dataset stores depth as float32 meters, so let the server auto-infer
    # the scale (pass 0). For raw RealSense uint16 depth pass the SDK value (~0.001).
    scale = 0.0
    rgb = np.asarray(Image.open(
        os.path.join(DATASET, "rgb", f"{args.frame}.png")).convert("RGB"))
    depth = np.load(os.path.join(DATASET, "depth", f"{args.frame}.npy"))
    print(f"frame {args.frame}: rgb {rgb.shape} depth {depth.dtype}{depth.shape} "
          f"scale={scale}")

    c = PerceptionClient(args.server, timeout=600)

    h = c.health()
    print("\n[health]", json.dumps(h, indent=2)[:600])

    print("\n[segment]")
    seg = c.segment(rgb, args.prompt)
    print(f"  {len(seg['detections'])} detections on {seg['width']}x{seg['height']}")
    for d in seg["detections"]:
        print(f"    {d['label']!r} score={d['score']} box={d['box']} "
              f"mask_px={int(d['mask'].sum())}")

    print("\n[grasps from depth, whole frame]")
    g = c.grasps_from_depth(depth, intr, depth_scale=scale, topk=5)
    print(f"  num_points={g['num_points']} grasps={len(g['grasps'])}")
    for gr in g["grasps"][:3]:
        print(f"    score={gr['score']} width={gr['width']} "
              f"center={np.round(gr['pose'][:3,3],3).tolist()}")

    print("\n[perceive]")
    r = c.perceive(rgb, depth, intr, args.prompt, depth_scale=scale, topk=5)
    print(f"  {len(r['detections'])} detections [{r['seg_backend']}/{r['grasp_backend']}]")
    for d in r["detections"]:
        b = best_grasp(d)
        ctr = np.round(b["pose"][:3, 3], 3).tolist() if b else None
        print(f"    {d['label']!r} score={d['score']} obj_pts={d['num_object_points']} "
              f"grasps={len(d['grasps'])} best_center_cam={ctr}")

    print("\nOK")


if __name__ == "__main__":
    main()
