"""Shared helpers for the example runs — data loading + dependency-light viz.

Uses only PIL + numpy (no matplotlib / open3d), so it runs anywhere the server
runs. Grasps are visualized by projecting their 6-DoF pose onto the RGB image
with the camera intrinsics: a line across the two jaw tips + a short approach
stub, colored by score.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "realsense_D435i_dataset", "dataset")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)


def load_frame(frame: str = "frame_000010"):
    meta = json.load(open(os.path.join(DATASET, "meta.json")))
    K = meta["K"]
    intr = (K["fx"], K["fy"], K["cx"], K["cy"])
    rgb = np.asarray(Image.open(
        os.path.join(DATASET, "rgb", f"{frame}.png")).convert("RGB"))
    depth = np.load(os.path.join(DATASET, "depth", f"{frame}.npy"))
    return rgb, depth, intr, meta


def _font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(rgb: np.ndarray, dets, title: str) -> Image.Image:
    img = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(img)
    for det in dets:
        x0, y0, x1, y1 = det.box
        d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)
        d.text((x0 + 3, max(0, y0 - 16)), f"{det.label} {det.score:.2f}",
               fill=(255, 255, 0), font=_font())
    _banner(d, title)
    return img


def overlay_masks(rgb: np.ndarray, dets, title: str,
                  color=(60, 220, 60)) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    for det in dets:
        m = det.mask
        out[m] = 0.45 * out[m] + 0.55 * np.array(color, np.float32)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for det in dets:
        x0, y0, x1, y1 = det.box
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
        d.text((x0 + 3, max(0, y0 - 16)), f"{det.label} {det.score:.2f}",
               fill=(255, 255, 0), font=_font())
    _banner(d, title)
    return img


def _project(pt_cam, intr):
    fx, fy, cx, cy = intr
    X, Y, Z = pt_cam
    if Z <= 1e-6:
        return None
    return (fx * X / Z + cx, fy * Y / Z + cy)


def _score_color(s, smax):
    t = 0.0 if smax <= 0 else max(0.0, min(1.0, s / smax))
    return (int(255 * (1 - t)), int(255 * t), 40)        # red(low) -> green(high)


def draw_grasps(rgb: np.ndarray, grasps, intr, title: str,
                finger_length: float = 0.0, finger_depth: float = 0.025,
                wrist_len: float = 0.03) -> Image.Image:
    """Project a parallel-jaw gripper for each grasp onto the image.

    `finger_length` advances the drawn jaw from the pose translation along the
    approach axis to the fingertips: pass 0 when the pose center is the contact
    midpoint (analytic), or the hand base->fingertip distance (~0.103 m for the
    Panda hand used by Contact-GraspNet, whose pose origin is the gripper base).
    """
    img = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(img)
    smax = max((g.score for g in grasps), default=1.0)
    for g in grasps:
        T = np.asarray(g.pose)
        base = T[:3, 3]
        binormal, approach = T[:3, 0], T[:3, 2]
        jaw_c = base + finger_length * approach          # fingertip plane center
        half = 0.5 * g.width * binormal
        tip_l, tip_r = jaw_c - half, jaw_c + half        # the two fingertips
        root_l = tip_l - finger_depth * approach         # finger roots (toward wrist)
        root_r = tip_r - finger_depth * approach
        wrist = jaw_c - (finger_depth + wrist_len) * approach
        col = _score_color(g.score, smax)
        segs = [(tip_l, tip_r), (tip_l, root_l), (tip_r, root_r),
                (root_l, root_r)]                        # gripper "U"
        for p, q in segs:
            pp, qq = _project(p, intr), _project(q, intr)
            if pp and qq:
                d.line([pp, qq], fill=col, width=3)
        a, b = _project(jaw_c, intr), _project(wrist, intr)
        if a and b:
            d.line([a, b], fill=(80, 160, 255), width=2)  # approach/wrist
    _banner(d, f"{title}  ({len(grasps)} grasps)")
    return img


def status_card(title: str, lines, size=(640, 480)) -> Image.Image:
    img = Image.new("RGB", size, (28, 28, 34))
    d = ImageDraw.Draw(img)
    d.text((16, 14), title, fill=(255, 210, 80), font=_font(20))
    y = 60
    for ln in lines:
        d.text((16, y), ln, fill=(220, 220, 220), font=_font(15))
        y += 24
    return img


def _banner(draw, text):
    draw.rectangle([0, 0, 640, 22], fill=(0, 0, 0))
    draw.text((6, 3), text, fill=(255, 255, 255), font=_font(15))


def save(img: Image.Image, name: str) -> str:
    path = os.path.join(OUT, name)
    img.save(path)
    return path
