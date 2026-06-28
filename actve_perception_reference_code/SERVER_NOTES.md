# Server notes — Grounded-SAM + grasp pose detection (WSL2 Ubuntu 22.04)

Implementation of Services A & B from `TOOLS_SERVER_WSL_SETUP.md`, adapted to
this machine. Two FastAPI services the Windows-side pipeline calls over localhost.

## Environment (decided with the user)

- **Single conda env: `lerobot`** (reused in-place, *not* cloned). Both services
  run in it. Python 3.12.12.
- This works because we run Grounding-DINO + SAM through **HuggingFace
  `transformers`** (already in `lerobot`) instead of the `Grounded-Segment-Anything`
  repo — so there is **no GroundingDINO CUDA-op to compile** (no `nvcc` here) and no
  TF/PyTorch clash. The analytic grasp backend is pure numpy/open3d. Only the
  optional Contact-GraspNet backend has heavier, isolatable deps.

### Installed versions
| package | version |
|---|---|
| torch | 2.10.0+cu128 |
| transformers | 5.5.4 |
| open3d | 0.19.0 |
| fastapi / uvicorn | 0.138.0 / 0.49.0 |
| trimesh | 4.12.2 |
| numpy | 1.26.4 |
| opencv-python | 4.13.0.92 |

Added to `lerobot` via: `pip install fastapi "uvicorn[standard]" python-multipart open3d trimesh`

### GPU status — eGPU now ACTIVE
An **NVIDIA RTX A6000 (49 GB), driver 596.72 / CUDA 13.2** is attached and
`torch.cuda.is_available()` is **True**. All three modules run on `cuda:0`
(Service A `/health` reports `"device":"cuda"`; the CGN demo prints
`device=cuda:0`). Everything is wired device-agnostic (`cuda if available else
cpu`), so it also still runs CPU-only if the GPU is detached — no code change.

## Models / weights

- Grounding-DINO: `IDEA-Research/grounding-dino-tiny` (HF, auto-downloaded to
  `~/.cache/huggingface`). Override with `GDINO_MODEL`.
- SAM: `facebook/sam-vit-base` — this is **SAM v1** (the original Segment
  Anything). Override with `SAM_MODEL` (e.g. `facebook/sam-vit-huge` for quality
  on GPU). To use **SAM 2** instead, see "SAM v1 vs SAM 2" below.
- Contact-GraspNet: `elchun/contact_graspnet_pytorch`, cloned to **`cgn_repo/`**
  (renamed from the default to avoid a namespace-package collision with the
  installed `contact_graspnet_pytorch` package). The checkpoint **ships in the
  repo** at `cgn_repo/checkpoints/contact_graspnet/checkpoints/model.pt` — no
  Google-Drive download. Installed with `pip install -e . --no-deps
  --no-build-isolation` plus `pip install pyrender`. Override checkpoint dir with
  `CGN_CKPT`. The wrapper forces a `weights_only=False` load for this trusted
  in-repo checkpoint (it predates torch>=2.6's `weights_only=True` default).

### SAM v1 vs SAM 2
Current install is **SAM v1** via `facebook/sam-vit-base` (`transformers.SamModel`).
SAM 2 (`facebook/sam2-hiera-*`, `transformers.Sam2Model`) is a separate, heavier
model with video support; switching means changing the SAM class + processor in
`gsam_model.py`. v1 is the right fit here (single-image, box-prompted mask) and is
what the contract needs.

## Files

| file | purpose |
|---|---|
| `gsam_model.py` | `GroundedSAM` class: GDINO detect → SAM segment, via transformers |
| `service_gsam.py` | Service A FastAPI, `POST /detect_segment` on :8001 |
| `grasp/analytic.py` | antipodal 6-DoF grasp sampler (numpy/open3d, CPU) |
| `grasp/contact_graspnet.py` | Contact-GraspNet backend wrapper (optional) |
| `service_grasp.py` | Service B FastAPI, `POST /grasps` on :8002 (backend=analytic\|cgn) |
| `examples/make_test_data.py` | makes `test.png` + `cloud.npy`/`cloud_seg.npy` |
| `examples/demo_gsam.py` | offline GSAM example → `out/gsam_overlay.png` |
| `examples/demo_grasp.py` | offline analytic-grasp example |
| `examples/demo_cgn.py` | offline Contact-GraspNet example |
| `client_demo.py` | HTTP client hitting both services |
| `cgn_repo/` | cloned Contact-GraspNet PyTorch port (renamed clone) |

## Run

```bash
conda activate lerobot
# terminal 1 — Service A
uvicorn service_gsam:app  --host 0.0.0.0 --port 8001
# terminal 2 — Service B
uvicorn service_grasp:app --host 0.0.0.0 --port 8002

# smoke tests
python examples/make_test_data.py
curl -F image=@examples/test.png -F prompt="a cat" http://localhost:8001/detect_segment | head -c 300
curl -F cloud=@examples/cloud.npy -F segmentation=@examples/cloud_seg.npy -F backend=analytic \
     http://localhost:8002/grasps | head -c 400
python client_demo.py "a cat"
```

## How to test each module

```bash
conda activate lerobot
cd ~/claude/active_perception
python examples/make_test_data.py        # makes test.png + cloud.npy/cloud_seg.npy
```

**Module 1 — Grounding-DINO + SAM (offline):**
```bash
python examples/demo_gsam.py "a cat"     # -> prints detections, writes out/gsam_overlay.png
```
Expect 2 cat detections on the COCO test image and a red mask overlay covering them.

**Module 2 — analytic grasp sampler (offline):**
```bash
python examples/demo_grasp.py            # prints grasps + writes out/analytic_grasps.png
```
Expect grasp centers clustered on the box (z≈0.50–0.60 m), high scores (~0.9).

**Module 3 — Contact-GraspNet (offline, GPU):**
```bash
python examples/demo_cgn.py              # device=cuda:0 + writes out/cgn_grasps.png
```
Expect `device=cuda:0` and ~70+ grasps for object 1 (scores lower — the synthetic
box is out-of-distribution for CGN; use a realistic depth cloud for good scores).

### Visualization outputs (`out/`)
- `gsam_overlay.png` — input image with SAM masks overlaid (Module 1).
- `analytic_grasps.png`, `cgn_grasps.png` — headless matplotlib 3D render of the
  point cloud (object highlighted) + parallel-jaw gripper markers colored by score
  (`grasp/viz.py`, Agg backend — no display needed). For an interactive view, the
  CGN repo also ships `visualization_utils_o3d.visualize_grasps` (needs a display).

**Services over HTTP (matches the Windows client path):**
```bash
uvicorn service_gsam:app  --host 0.0.0.0 --port 8001 &   # device=cuda at /health
uvicorn service_grasp:app --host 0.0.0.0 --port 8002 &
curl -s localhost:8001/health ; curl -s localhost:8002/health

# Service A
curl -F image=@examples/test.png -F prompt="a cat" localhost:8001/detect_segment | head -c 200
# Service B, analytic backend
curl -F cloud=@examples/cloud.npy -F segmentation=@examples/cloud_seg.npy \
     -F backend=analytic localhost:8002/grasps
# Service B, Contact-GraspNet backend
curl -F cloud=@examples/cloud.npy -F segmentation=@examples/cloud_seg.npy \
     -F backend=cgn localhost:8002/grasps | head -c 300

# full client (decodes masks with cv2 + calls both)
python client_demo.py "a cat"
```

## API contracts (unchanged from the brief, §4)

- `POST :8001/detect_segment` — multipart `image`, `prompt`, optional
  `box_threshold`/`text_threshold` → `{width,height,detections:[{label,score,box,mask_png_b64}]}`.
- `POST :8002/grasps` — multipart `cloud` (.npy (N,3)/(N,6), meters), optional
  `segmentation` (.npy bool), `gripper_width_max`, `topk`, `backend` →
  `{frame:"input_cloud",backend,grasps:[{pose:4x4,width,score}]}`.

Grasp `pose` columns = `[binormal(jaw-closing), hand, approach(+z)]`, translation =
grasp center in meters, in the input cloud's frame. Client applies standoff +
hand-eye transform for execution.
