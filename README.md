# Active-Perception Server (DGX Spark) + Windows client adaptor

Text-promptable segmentation + 6-DoF grasp detection, served over the LAN from
the **DGX Spark** to a **Windows PC** that does the RealSense capture and pose
estimation.

```
 Windows PC (client)                         DGX Spark  192.168.45.150  (server)
 ─────────────────────                       ──────────────────────────────────
 RealSense D435i capture (RGB+depth)         FastAPI :8000  (GB10 GPU, CUDA 13)
 camera intrinsics, VIO/pose      HTTP/LAN     /segment   rgb+prompt -> masks
 robot motion / grasp execution  <─────────>   /grasps    cloud|depth -> grasps
                                                /perceive  rgb+depth+prompt -> both
   perception_client.py  ──────────────────▶  server/  (seg + grasp backends)
```

This replaces the old WSL setup where the GPU box was *inside* the Windows host
and reachable at `localhost`. Now the DGX is a **separate machine on the wifi
LAN**: the server binds `0.0.0.0` and the client targets the DGX's IP.

---

## What changed vs. the reference (`actve_perception_reference_code/`)

| | Reference (WSL) | This (DGX Spark) |
|---|---|---|
| Transport | `localhost` (WSL auto-forward) | **LAN**: client → `http://192.168.45.150:8000` |
| Arch / CUDA | x86 conda `lerobot` | **aarch64 / GB10 / CUDA 13**, venv, torch `+cu130` |
| Depth → cloud | client computes `.npy` cloud | **server** deprojects depth+K → cloud (`/perceive`) |
| Segmentation | Grounding-DINO + SAM only | **pluggable**: `gsam` (default) **+ `sam3`** |
| Grasp | analytic / CGN, two services | analytic (+CGN), **one service**, pluggable registry |
| open3d | available | **not on aarch64** → scipy KD-tree/normals fallback |
| Process model | two ports (8001/8002) | **one port** 8000 (pure-torch stacks coexist) |

---

## Server — run on the DGX

```bash
cd ~/claude/01_active_perception_server
bash scripts/setup_env.sh            # one-time: venv + torch(cu130) + deps
bash scripts/run_server.sh           # serves on 0.0.0.0:8000 (prints the LAN URL)
# health from anywhere on the LAN:
curl http://192.168.45.150:8000/health
```

First request lazy-loads the default backends (≈few seconds; cached after).
Verified on this box: `torch 2.12.1+cu130`, `cuda avail True`, `NVIDIA GB10`.

### Endpoints

| method | path | input | output |
|---|---|---|---|
| GET | `/health` | — | device, GPU, per-backend load status |
| POST | `/segment` | `image`, `prompt`, `backend`, thresholds | `{detections:[{label,score,box,mask_png_b64}]}` |
| POST | `/grasps` | `cloud`(.npy) **or** `depth`+`fx,fy,cx,cy`(+`mask`) | `{grasps:[{pose 4x4,width,score}]}` |
| POST | `/perceive` | `rgb`+`depth`+`fx,fy,cx,cy`+`prompt` | detections, each with masks **and** grasps |

`/perceive` is the headline call: one round-trip from an RGB-D frame to grasps.

### Frames & units
All clouds and grasp poses are in the **camera optical frame** (OpenCV: +x
right, +y down, +z forward), **meters**. Grasp pose columns =
`[binormal (jaw-closing), hand, approach(+z)]`, translation = grasp center.
The client transforms to world/robot with its own VIO pose × hand-eye.

### Depth scale (important)
`meters = depth × depth_scale`. Pass `depth_scale=0` to **auto-infer**:
integer depth (raw RealSense uint16) → `0.001` (mm); float depth → `1.0`
(already meters). The bundled dataset stores **float32 meters**, so its
`meta.json: depth_scale_m=0.001` is the *original* sensor unit — pass `0` (auto)
or `1.0`, not `0.001`, for that data. For live RealSense uint16 frames, pass the
SDK's `get_depth_scale()` (≈0.001) to be exact.

---

## Client — copy to the Windows PC

Copy `client/perception_client.py` (and optionally `client/example_integration.py`)
to the PC. Only `numpy` + `requests` are required (`opencv-python` or `pillow`
used for PNG if present).

```python
from perception_client import PerceptionClient, best_grasp, to_world

client = PerceptionClient("http://192.168.45.150:8000")

# color: (H,W,3) uint8   depth: (H,W) uint16 mm (or float meters)
result = client.perceive(
    rgb=color, depth=depth, intrinsics=(fx, fy, cx, cy),
    depth_scale=0.0,                 # 0 = auto; or pass the RealSense SDK scale
    prompt="the computer mouse",
    seg_backend="gsam",              # or "sam3"
    grasp_backend="analytic",        # or "cgn"
    bgr=False,                       # True if your color frame is BGR (OpenCV)
)
for det in result["detections"]:
    mask = det["mask"]               # (H,W) bool, decoded for you
    g = best_grasp(det)              # highest-scoring grasp, or None
    if g:
        T_world = to_world(g["pose"], T_cam_to_world)   # your VIO×hand-eye (4,4)
        # execute_grasp(T_world, width=g["width"])
```

`example_integration.py` marks the three seams to wire into your existing PC
code (capture, pose, robot execution) and includes an **offline mode** that runs
the same client against the bundled dataset — no RealSense needed:

```bash
python example_integration.py --server http://192.168.45.150:8000 \
    --dataset ../realsense_D435i_dataset/dataset --frame frame_000010 \
    --prompt "the computer mouse"
```

---

## Example runs on the bundled data

```bash
python examples/run_demos.py --frame frame_000010 --prompt "the computer mouse"
```
Writes annotated overlays to `out/`:

| file | model | status |
|---|---|---|
| `01_grounding_dino_boxes.png` | Grounding-DINO detection | ✅ runs |
| `02_sam_masks.png` | SAM box-prompted masks (`gsam`) | ✅ runs |
| `03_sam3_instances.png` | SAM 3 concept segmentation | ⚠️ needs gated HF approval |
| `04_grasp_analytic.png` | analytic 6-DoF grasps | ✅ runs |
| `05_grasp_cgn.png` | Contact-GraspNet grasps | ✅ runs (installed) |

A backend that isn't enabled writes an instructional status card instead of
crashing. SAM 3 is the only one still pending (manual gated approval — see below).
Grasps render as projected parallel-jaw grippers colored by score.

## Remote access (Tailscale + VSCode/SSH)

See **[TAILSCALE_SETUP.md](TAILSCALE_SETUP.md)** for the full plan: install
Tailscale on the DGX (aarch64), reach `http://spark-46e5:8000` from any network,
SSH + VSCode Remote-SSH from your laptop, and run the server as a systemd service
(`scripts/perception-server.service`).

## Optional backends

**SAM 3** (`seg_backend="sam3"`) — single-model text/concept segmentation, wired
to the native transformers SAM 3 API (`Sam3Model`/`Sam3Processor`, present in
transformers ≥5.12). Weights (`facebook/sam3`) are **gated with manual approval**:
1. Request access at https://huggingface.co/facebook/sam3 and **wait for Meta to
   approve your account** (a valid token alone is not enough — an un-approved
   account gets `403 not in the authorized list`).
2. `huggingface-cli login` (or `export HF_TOKEN=...`) on the DGX.
Until approved, `/health` shows `sam3: loaded:false` and requests return an
actionable error; `gsam` keeps working.

**Contact-GraspNet** (`grasp_backend="cgn"`) — learned grasps, **installed and
verified** on this DGX (97 grasps on the mouse cloud; better than the analytic
baseline on smooth objects). Install steps, if recreating:
```bash
git clone https://github.com/elchun/contact_graspnet_pytorch cgn_repo
pip install -e cgn_repo --no-deps --no-build-isolation
pip install trimesh pyrender            # runtime deps (checkpoint ships in-repo)
export CGN_CKPT=cgn_repo/checkpoints/contact_graspnet PYOPENGL_PLATFORM=egl
```
The wrapper (`server/grasp/contact_graspnet.py`) includes a small numpy-2 compat
shim (`np.in1d` etc.) so the pre-numpy-2 repo runs unmodified. `run_server.sh`
and the systemd unit set `CGN_CKPT`/`PYOPENGL_PLATFORM` automatically when
`cgn_repo/` is present.

---

## Smoke test (verified)

```bash
bash scripts/run_server.sh &                 # on the DGX
python scripts/smoke_test.py --prompt "the computer mouse" --frame frame_000010
```
Expected: `gsam` segments the mouse (score ≈0.90), depth deprojects to a metric
cloud, and `/perceive` returns grasp pose(s) centered on the mouse at z≈0.2 m in
the camera frame.

---

## Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `AP_HOST` / `AP_PORT` | `0.0.0.0` / `8000` | bind address |
| `AP_DEVICE` | auto | force `cuda` / `cpu` |
| `AP_DEFAULT_SEG_BACKEND` | `gsam` | `gsam` \| `sam3` |
| `AP_DEFAULT_GRASP_BACKEND` | `analytic` | `analytic` \| `cgn` |
| `AP_MAX_POINTS` | `80000` | cloud downsample cap (0 = off) |
| `GDINO_MODEL` / `SAM_MODEL` | tiny / vit-base | swap for quality on the GPU |
| `SAM3_MODEL` | `facebook/sam3` | SAM 3 weights id or local path |
| `CGN_CKPT` | `cgn_repo/...` | Contact-GraspNet checkpoint dir |

For quality on the GB10, try `SAM_MODEL=facebook/sam-vit-huge` and
`GDINO_MODEL=IDEA-Research/grounding-dino-base`.

---

## Layout

```
server/
  app.py                 FastAPI: /health /segment /grasps /perceive
  config.py              env-driven config + device pick
  geometry.py            depth+K -> metric cloud, scene/object masking, downsample
  encoding.py            decode rgb/depth (png/npy), mask<->b64
  segmentation/          registry + base; gsam.py, sam3.py
  grasp/                 registry + base; analytic.py (scipy), contact_graspnet.py
client/
  perception_client.py   the Windows adaptor (copy to PC)
  example_integration.py seams for capture/pose/robot + offline dataset demo
  requirements_client.txt
scripts/
  setup_env.sh  run_server.sh  smoke_test.py
```
