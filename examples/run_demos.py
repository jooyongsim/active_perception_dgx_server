"""Example runs of every model on the bundled RealSense data.

Runs IN-PROCESS against the server modules (no HTTP needed) and writes annotated
overlays to ../out/. Backends that need access/install (SAM 3, Contact-GraspNet)
are reported with instructions instead of crashing.

    python examples/run_demos.py                       # default frame + prompt
    python examples/run_demos.py --frame frame_000010 --prompt "the computer mouse"

Outputs (../out/):
    01_grounding_dino_boxes.png   Grounding-DINO text-prompted detection
    02_sam_masks.png              SAM masks from those boxes  (= "gsam")
    03_sam3_instances.png         SAM 3 concept segmentation  (or status card)
    04_grasp_analytic.png         analytic 6-DoF grasps on the object
    05_grasp_cgn.png              Contact-GraspNet grasps     (or status card)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))            # import `server.*`

import demo_common as dc                              # noqa: E402
from server import grasp as G                         # noqa: E402
from server import segmentation as S                  # noqa: E402
from server.geometry import Intrinsics, scene_cloud   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="frame_000010")
    ap.add_argument("--prompt", default="the computer mouse")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    rgb, depth, intr_t, meta = dc.load_frame(args.frame)
    from PIL import Image
    pil = Image.fromarray(rgb)
    intr = Intrinsics(*intr_t)
    print(f"frame {args.frame}  rgb{rgb.shape}  depth {depth.dtype}{depth.shape}  "
          f"prompt={args.prompt!r}")
    summary = []

    # --- 1 & 2: Grounding-DINO + SAM (the 'gsam' backend) --------------------
    print("\n[1/2] Grounding-DINO + SAM ...")
    gsam = S.get_segmenter("gsam")                     # loads once
    # Grounding-DINO boxes (detection only)
    boxes = gsam._detect(pil, args.prompt, 0.30, 0.25)
    from server.segmentation.base import Detection
    box_dets = [Detection(b["label"], b["score"], b["box"],
                          np.zeros((rgb.shape[0], rgb.shape[1]), bool)) for b in boxes]
    p1 = dc.save(dc.draw_boxes(rgb, box_dets, "1) Grounding-DINO detection"),
                 "01_grounding_dino_boxes.png")
    print(f"   Grounding-DINO: {len(box_dets)} box(es) -> {p1}")
    summary.append(("Grounding-DINO", f"{len(box_dets)} boxes", p1))

    # SAM masks from those boxes
    dets = gsam.detect_segment(pil, args.prompt, 0.30, 0.25)
    p2 = dc.save(dc.overlay_masks(rgb, dets, "2) SAM masks (box-prompted)"),
                 "02_sam_masks.png")
    px = int(sum(int(d.mask.sum()) for d in dets))
    print(f"   SAM: {len(dets)} mask(s), {px} px -> {p2}")
    summary.append(("SAM", f"{len(dets)} masks, {px}px", p2))

    # --- 3: SAM 3 (gated) ----------------------------------------------------
    print("\n[3] SAM 3 ...")
    try:
        sam3 = S.get_segmenter("sam3")
        dets3 = sam3.detect_segment(pil, args.prompt, 0.30, 0.25)
        p3 = dc.save(dc.overlay_masks(rgb, dets3, "3) SAM 3 concept segmentation",
                                      color=(80, 120, 255)), "03_sam3_instances.png")
        print(f"   SAM 3: {len(dets3)} instance(s) -> {p3}")
        summary.append(("SAM 3", f"{len(dets3)} instances", p3))
    except Exception as e:
        p3 = dc.save(dc.status_card("3) SAM 3 — unavailable", [
            "Weights 'facebook/sam3' are GATED (manual approval).", "",
            "Enable on the DGX:",
            "  1. request access at huggingface.co/facebook/sam3",
            "     and WAIT for Meta to approve your account",
            "     (a token alone -> 403 'not in authorized list')",
            "  2. huggingface-cli login   (or export HF_TOKEN=...)",
            "  3. re-run this demo", "",
            f"{type(e).__name__}: {str(e)[:72]}",
        ]), "03_sam3_instances.png")
        print(f"   SAM 3 unavailable -> status card {p3}")
        summary.append(("SAM 3", "GATED (needs HF access)", p3))

    # --- object cloud (from the gsam mask) for the grasp demos ---------------
    xyz_full, valid = scene_cloud(depth, intr, 0.0)
    obj_mask = dets[0].mask if dets else np.zeros(rgb.shape[:2], bool)
    seg = obj_mask[valid]
    print(f"\n   object cloud: {int(seg.sum())} pts (of {len(xyz_full)} scene pts)")

    # --- 4: analytic grasps --------------------------------------------------
    print("[4] Analytic grasp sampler ...")
    ga = G.get_grasp_backend("analytic")
    grasps_a = ga.sample_grasps(xyz_full, seg, 0.085, args.topk)
    p4 = dc.save(dc.draw_grasps(rgb, grasps_a, intr_t,
                                "4) Analytic 6-DoF grasps", finger_length=0.0),
                 "04_grasp_analytic.png")
    print(f"   analytic: {len(grasps_a)} grasps -> {p4}")
    if grasps_a:
        print(f"     best score={grasps_a[0].score:.3f} width={grasps_a[0].width:.3f} "
              f"center={np.round(grasps_a[0].pose[:3,3],3).tolist()}")
    summary.append(("Grasp/analytic", f"{len(grasps_a)} grasps", p4))

    # --- 5: Contact-GraspNet (optional install) ------------------------------
    print("[5] Contact-GraspNet ...")
    try:
        gc = G.get_grasp_backend("cgn")
        grasps_c = gc.sample_grasps(xyz_full, seg, 0.085, args.topk)
        # CGN pose origin is the gripper BASE; advance to the Panda fingertips.
        p5 = dc.save(dc.draw_grasps(rgb, grasps_c, intr_t,
                                    "5) Contact-GraspNet grasps", finger_length=0.1034),
                     "05_grasp_cgn.png")
        print(f"   CGN: {len(grasps_c)} grasps -> {p5}")
        summary.append(("Grasp/CGN", f"{len(grasps_c)} grasps", p5))
    except Exception as e:
        p5 = dc.save(dc.status_card("5) Contact-GraspNet — not installed", [
            "Learned 6-DoF grasps (better on smooth objects).", "",
            "Install on the DGX:",
            "  git clone https://github.com/elchun/contact_graspnet_pytorch cgn_repo",
            "  pip install -e cgn_repo --no-deps --no-build-isolation",
            "  export CGN_CKPT=cgn_repo/checkpoints/contact_graspnet", "",
            f"{type(e).__name__}: {str(e)[:80]}",
        ]), "05_grasp_cgn.png")
        print(f"   CGN unavailable -> status card {p5}")
        summary.append(("Grasp/CGN", "not installed", p5))

    print("\n=== summary ===")
    for name, result, path in summary:
        print(f"  {name:16s} {result:28s} {os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
