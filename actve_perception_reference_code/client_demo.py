"""End-to-end client: hit Service A (:8001) and Service B (:8002) over HTTP.

Mirrors how the Windows-side pipeline calls the WSL services.

  # start the servers first (separate terminals):
  uvicorn service_gsam:app  --host 0.0.0.0 --port 8001
  uvicorn service_grasp:app --host 0.0.0.0 --port 8002

  python client_demo.py "a cat"
"""

from __future__ import annotations

import base64
import os
import sys

import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
A = "http://localhost:8001"
B = "http://localhost:8002"


def call_gsam(prompt: str):
    img_path = os.path.join(HERE, "examples", "test.png")
    r = requests.post(f"{A}/detect_segment",
                      files={"image": open(img_path, "rb")},
                      data={"prompt": prompt}, timeout=300)
    r.raise_for_status()
    js = r.json()
    print(f"[A] {len(js['detections'])} detections on {js['width']}x{js['height']}")
    for d in js["detections"]:
        b64 = d["mask_png_b64"]
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), 0) > 0
        print(f"    {d['label']!r} score={d['score']} box={d['box']} "
              f"mask_px={int(mask.sum())}")


def call_grasp(backend: str = "analytic"):
    cloud_path = os.path.join(HERE, "examples", "cloud.npy")
    seg_path = os.path.join(HERE, "examples", "cloud_seg.npy")
    files = {"cloud": open(cloud_path, "rb")}
    if os.path.exists(seg_path):
        files["segmentation"] = open(seg_path, "rb")
    r = requests.post(f"{B}/grasps", files=files,
                      data={"backend": backend, "topk": 5}, timeout=300)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        print(f"[B] backend={backend} error: {js['error'][:120]}")
        return
    print(f"[B] backend={js['backend']} frame={js['frame']} "
          f"-> {len(js['grasps'])} grasps")
    for g in js["grasps"][:5]:
        t = [round(g["pose"][i][3], 3) for i in range(3)]
        print(f"    score={g['score']} width={g['width']} center={t}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a cat"
    call_gsam(prompt)
    call_grasp("analytic")
