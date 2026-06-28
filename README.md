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

## Optional backends

**SAM 3** (`seg_backend="sam3"`) — single-model text/concept segmentation.
Weights (`facebook/sam3`) are gated: on the DGX run `huggingface-cli login` and
accept the license, then it loads on first use. Until then `/health` shows
`sam3: loaded:false` and requests for it return an actionable error; `gsam`
keeps working. The transformers-API seam is `server/segmentation/sam3.py`.

**Contact-GraspNet** (`grasp_backend="cgn"`) — learned grasps, better than the
analytic baseline on curved/low-feature objects (like the mouse). Install:
```bash
git clone https://github.com/elchun/contact_graspnet_pytorch cgn_repo
pip install -e cgn_repo --no-deps --no-build-isolation && pip install pyrender
export CGN_CKPT=cgn_repo/checkpoints/contact_graspnet
```

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
