"""Run Grounding-DINO + SAM on examples/test.png and save a mask overlay.

  python examples/demo_gsam.py "a cat"

Downloads the models on first run (~1 GB). CPU-only is fine for this smoke test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gsam_model import GroundedSAM


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a cat"
    img = Image.open(os.path.join(HERE, "test.png")).convert("RGB")

    model = GroundedSAM()
    print(f"device={model.device}  prompt={prompt!r}")
    dets = model.detect_segment(img, prompt)
    print(f"{len(dets)} detections:")

    overlay = np.array(img).astype(np.float32)
    color = np.array([255, 0, 0], np.float32)
    for d in dets:
        print(f"  label={d.label!r}  score={d.score:.3f}  "
              f"box={[round(v,1) for v in d.box]}  mask_px={int(d.mask.sum())}")
        overlay[d.mask] = 0.5 * overlay[d.mask] + 0.5 * color

    out = os.path.join(HERE, "..", "out", "gsam_overlay.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(overlay.astype(np.uint8)).save(out)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
