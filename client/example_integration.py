"""Sample integration for the Windows PC — shows the seams, not a full app.

Your PC already has:  RealSense capture (pyrealsense2), camera intrinsics, and
pose/VIO analysis. This file marks the three places you wire those in. Everything
to do with the DGX (HTTP, encoding, mask/grasp decoding) is handled by
`perception_client.PerceptionClient`.

Run a quick offline check against the bundled dataset (no RealSense needed):
    python example_integration.py --server http://192.168.45.150:8000 \
        --dataset ../realsense_D435i_dataset/dataset --frame frame_000000 \
        --prompt "the object"
"""

from __future__ import annotations

import argparse

import numpy as np

from perception_client import PerceptionClient, best_grasp, to_world


# --------------------------------------------------------------------------- #
# Live path — fill these in with your existing PC code.
# --------------------------------------------------------------------------- #
def run_live(server: str, prompt: str):
    client = PerceptionClient(server)
    print("server health:", client.health()["status"])

    # ---- SEAM 1: your RealSense capture (already written on the PC) ----------
    #   color = np.asanyarray(color_frame.get_data())   # (H,W,3) uint8
    #   depth = np.asanyarray(depth_frame.get_data())   # (H,W)  uint16 (mm)
    #   depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    #   intr = color_frame.profile.as_video_stream_profile().intrinsics
    #   intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    color, depth, intrinsics, depth_scale = capture_rgbd()    # <- your function

    # ---- One call does segmentation + cloud + grasps on the DGX --------------
    result = client.perceive(
        rgb=color, depth=depth, intrinsics=intrinsics, depth_scale=depth_scale,
        prompt=prompt, seg_backend="gsam", grasp_backend="analytic",
        bgr=is_bgr(),            # set True if your color frame is BGR (OpenCV)
    )

    # ---- SEAM 2: your pose/VIO analysis (already written on the PC) ----------
    #   T_cam_to_world = pose_from_vio() @ hand_eye_calibration()   # (4,4)
    T_cam_to_world = current_camera_to_world()                # <- your function

    for det in result["detections"]:
        g = best_grasp(det)
        if g is None:
            print(f"  {det['label']}: no grasp")
            continue
        T_world = to_world(g["pose"], T_cam_to_world)
        print(f"  {det['label']} score={det['score']:.2f} "
              f"grasp@world t={np.round(T_world[:3, 3], 3)} width={g['width']:.3f}")

        # ---- SEAM 3: hand off T_world to your robot motion code --------------
        #   execute_grasp(T_world, width=g["width"])


# --- stubs you replace with the PC's real functions --------------------------
def capture_rgbd():
    raise NotImplementedError("wire in your RealSense capture here")


def current_camera_to_world():
    return np.eye(4)


def is_bgr():
    return False


# --------------------------------------------------------------------------- #
# Offline path — exercises the same client against the bundled dataset.
# --------------------------------------------------------------------------- #
def run_dataset(server: str, dataset: str, frame: str, prompt: str):
    import json
    import os

    from PIL import Image

    meta = json.load(open(os.path.join(dataset, "meta.json")))
    K = meta["K"]
    intrinsics = (K["fx"], K["fy"], K["cx"], K["cy"])
    depth_scale = meta["depth_scale_m"]

    rgb = np.asarray(Image.open(os.path.join(dataset, "rgb", f"{frame}.png")).convert("RGB"))
    depth = np.load(os.path.join(dataset, "depth", f"{frame}.npy"))

    client = PerceptionClient(server)
    print("health:", client.health())
    result = client.perceive(rgb=rgb, depth=depth, intrinsics=intrinsics,
                             depth_scale=depth_scale, prompt=prompt)
    print(f"{len(result['detections'])} detection(s) on "
          f"{result['width']}x{result['height']} [{result['seg_backend']}/"
          f"{result['grasp_backend']}]")
    for det in result["detections"]:
        g = best_grasp(det)
        t = np.round(g["pose"][:3, 3], 3).tolist() if g else None
        print(f"  {det['label']!r} score={det['score']} "
              f"pts={det['num_object_points']} grasps={len(det['grasps'])} "
              f"best_center_cam={t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://192.168.45.150:8000")
    ap.add_argument("--prompt", default="the object")
    ap.add_argument("--dataset", help="run offline against this dataset dir")
    ap.add_argument("--frame", default="frame_000000")
    args = ap.parse_args()

    if args.dataset:
        run_dataset(args.server, args.dataset, args.frame, args.prompt)
    else:
        run_live(args.server, args.prompt)
