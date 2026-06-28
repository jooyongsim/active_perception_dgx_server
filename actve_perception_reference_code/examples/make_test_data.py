"""Generate test inputs for both services.

  examples/test.png   -- a real RGB photo (COCO cats) for Grounding-DINO + SAM
  examples/cloud.npy  -- a synthetic (N,3) point cloud in METERS for grasping
                         (a small box sitting on a table plane, camera frame)

Run: python examples/make_test_data.py
"""

from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats


def make_image():
    out = os.path.join(HERE, "test.png")
    try:
        with urllib.request.urlopen(IMG_URL, timeout=20) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        print(f"downloaded real test image {img.size}")
    except Exception as e:  # offline fallback: synthetic red cube on gray bg
        print(f"download failed ({e}); generating synthetic image")
        arr = np.full((480, 640, 3), 200, np.uint8)
        arr[200:340, 250:430] = (200, 40, 40)
        img = Image.fromarray(arr)
    img.save(out)
    print("wrote", out)


def make_cloud():
    """Camera-frame cloud (meters): a tabletop plane + a box on it.

    Camera looks down the +Z axis; +X right, +Y down (OpenCV convention).
    """
    rng = np.random.default_rng(0)

    # Table plane ~0.6 m in front of the camera, 0.4 x 0.4 m patch.
    n_plane = 8000
    px = rng.uniform(-0.20, 0.20, n_plane)
    py = rng.uniform(-0.20, 0.20, n_plane)
    pz = np.full(n_plane, 0.60) + rng.normal(0, 0.001, n_plane)
    plane = np.stack([px, py, pz], 1)

    # A box 6x6x10 cm sitting on the plane (its top faces the camera).
    n_box = 4000
    bx = rng.uniform(-0.03, 0.03, n_box)
    by = rng.uniform(-0.03, 0.03, n_box)
    bz = rng.uniform(0.50, 0.60, n_box)        # nearer the camera than plane
    box = np.stack([bx, by, bz], 1)

    cloud = np.concatenate([plane, box], 0).astype(np.float32)
    seg = np.concatenate([np.zeros(n_plane, bool),
                          np.ones(n_box, bool)])   # True = object

    np.save(os.path.join(HERE, "cloud.npy"), cloud)
    np.save(os.path.join(HERE, "cloud_seg.npy"), seg)
    print(f"wrote cloud.npy {cloud.shape} (object pts: {seg.sum()})")


if __name__ == "__main__":
    make_image()
    make_cloud()
