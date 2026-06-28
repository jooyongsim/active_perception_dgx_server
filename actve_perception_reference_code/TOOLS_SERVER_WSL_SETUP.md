# Tools server (WSL2 Ubuntu) — SAM + Grounding-DINO + Contact-GraspNet

**Save this as `CLAUDE.md` (auto-read by Claude Code) or `BRIEF.md` in the WSL project root,
then run `claude` there.** It tells a fresh Claude Code session in WSL2 Ubuntu what to build:
two GPU inference services that the Windows-side capture/geometry pipeline calls over
localhost. Commands are a **starting point — check each repo's current README for the exact
weights/versions**, they drift.

> Note for Claude: this is GPU/Linux setup. Many steps download multi-GB weights and compile
> CUDA ops; do them one service at a time and **smoke-test each before moving on**. Use the
> repos' own conda env / Dockerfile when offered — they are Linux-native and the smoothest path.

---

## 0. Why this split

Heavy deep models (SAM, Grounding-DINO, Contact-GraspNet) are Linux/CUDA-native and painful on
native Windows. They run here in **WSL2 Ubuntu with the GPU**. The **capture + geometry +
grasp orchestration stay on Windows** (pure numpy/OpenCV — already built in `grasp_pipeline/`
and `core/`). The two sides talk over **HTTP on localhost** (WSL2 forwards ports to Windows).

```
 Windows side (client)                         WSL2 Ubuntu side (this server)
 RealSense capture (pyrealsense2)              ┌─ Service A: GroundedSAM  :8001
 multi-view geometry / fusion (numpy)   HTTP   │    Grounding-DINO + SAM  (PyTorch, GPU)
 grasp_pipeline / core loop          <──────>  │    POST /detect_segment  (image+prompt -> mask)
                                               └─ Service B: GraspNet     :8002
                                                    Contact-GraspNet      (PyTorch/TF, GPU)
                                                    POST /grasps          (cloud -> 6-DoF grasps)
```

Build **two independent FastAPI microservices, each in its own conda env** (their dependency
pins clash — keep them isolated). Don't try to import both stacks in one process.

---

## 1. Verify the environment first

```bash
# in WSL2 Ubuntu
nvidia-smi                      # must show the GPU (NVIDIA driver is installed on WINDOWS, not WSL)
nproc && free -h && df -h ~     # cores / RAM / disk for weights
uname -a                        # confirm Ubuntu/WSL2
```
If `nvidia-smi` fails: install the **NVIDIA driver on Windows** (with WSL CUDA support) and the
**CUDA toolkit inside WSL** — never install a GPU driver inside WSL.

Install miniforge (mamba) for fast, isolated envs:
```bash
wget -O ~/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/mf.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && source ~/.bashrc
```

Work in the WSL filesystem (`~/...`), **not** `/mnt/c/...` — the Windows bridge is slow for
heavy I/O. Project root suggestion: `~/active_perception_server/`.

---

## 2. Service A — GroundedSAM (Grounding-DINO + SAM) on :8001

Text-promptable detection + segmentation: "the mug" → box (Grounding-DINO) → mask (SAM).

```bash
mamba create -n gsam python=3.10 -y && mamba activate gsam
# PyTorch matching your CUDA (example: CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy fastapi "uvicorn[standard]" python-multipart pillow

# Grounding-DINO + SAM (use the combined repo, which handles the CUDA op build):
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export CUDA_HOME=/usr/local/cuda                       # needed to build GroundingDINO's CUDA op
pip install -e segment_anything
pip install -e GroundingDINO
# weights:
mkdir -p weights && cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth   # SAM ViT-H
cd ../..
```
Then **Claude: write `service_gsam.py`** — a FastAPI app that loads Grounding-DINO + SAM once at
startup and serves the contract in §4. (SAM2 is an option for higher quality; same idea.)

Smoke test the models load and segment a test image before wiring HTTP.

---

## 3. Service B — Contact-GraspNet on :8002

6-DoF parallel-jaw grasps from a point cloud (or depth+K+segmentation). Prefer the **PyTorch
port** on WSL (the original is TensorFlow + custom CUDA ops, heavier to build):

```bash
mamba create -n cgn python=3.10 -y && mamba activate cgn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy open3d trimesh fastapi "uvicorn[standard]" python-multipart

git clone https://github.com/elchun/contact_graspnet_pytorch.git   # or NVlabs/contact_graspnet (TF)
cd contact_graspnet_pytorch
pip install -e .
# download the pretrained checkpoint per the repo README (Google-Drive link) into checkpoints/
cd ..
```
If you must use the **original** `NVlabs/contact_graspnet`: create its provided conda env
(`conda env create -f contact_graspnet_env.yml`), it pins TF + CUDA; it compiles PointNet++
`tf_ops` with nvcc — works on Linux, expect a build step.

Then **Claude: write `service_cgn.py`** — load the checkpoint once, serve §4. CGN outputs grasps
in the **input cloud's frame** (camera by default) with metric translation + width + score.

---

## 4. API contracts (match the existing numpy pipeline)

### Service A — `POST :8001/detect_segment`
- **Request** (multipart): `image` = PNG/JPG file; `prompt` = text (e.g. `"the mug"`);
  optional `box_threshold`, `text_threshold`.
- **Response** (JSON):
```json
{ "width": 640, "height": 480,
  "detections": [
    {"label": "mug", "score": 0.71, "box": [x0,y0,x1,y1],
     "mask_png_b64": "<base64 PNG, 1-channel 0/255 mask>"}
  ] }
```
The Windows client decodes `mask_png_b64` → boolean mask, exactly what `segment.py` expects.

### Service B — `POST :8002/grasps`
- **Request** (multipart or JSON): `cloud` = `.npy` of shape (N,3) [or (N,6) with color] in
  **meters**, in the camera/world frame you want grasps in; optional `segmentation` mask cloud,
  `gripper_width_max` (default 0.085), `topk`.
- **Response** (JSON):
```json
{ "frame": "input_cloud",
  "grasps": [ {"pose": [[..4x4..]], "width": 0.043, "score": 0.92}, ... ] }
```
`pose` is a 4×4 transform; `translation` is metric (meters) in the input frame. The client
transforms to world/robot via the VO/VIO pose (and hand-eye calibration for execution).

Keep payloads as **uploaded bytes** (image/npy), not file paths — Windows and WSL paths differ;
bytes over localhost avoid path translation.

---

## 5. Run + smoke test

```bash
# terminal 1
mamba activate gsam && uvicorn service_gsam:app --host 0.0.0.0 --port 8001
# terminal 2
mamba activate cgn  && uvicorn service_cgn:app  --host 0.0.0.0 --port 8002

# from WSL or from Windows (localhost is forwarded by WSL2):
curl -F image=@test.png -F prompt="the mug" http://localhost:8001/detect_segment | head -c 300
curl -F cloud=@cloud.npy http://localhost:8002/grasps | head -c 300
```
Confirm: A returns a mask that overlays the object; B returns grasps with metric widths.

---

## 6. Windows client integration (no architecture change)

The existing `grasp_pipeline/` already isolates these as swappable steps:
- `segment.py::segment_sam(img, prompt)` → POST the image to `:8001/detect_segment`, decode the
  returned mask. (Replaces the classical default; the rest of fuse→sample→rank is unchanged.)
- Optional: replace the analytic `grasp.py` with a call to `:8002/grasps` on the fused
  `out/cloud.ply` (converted to `.npy`), then rank/visualize the returned grasps.

Minimal client call:
```python
import requests, cv2, numpy as np, base64
r = requests.post("http://localhost:8001/detect_segment",
                  files={"image": open("frame.png","rb")}, data={"prompt": "the mug"})
det = r.json()["detections"][0]
mask = cv2.imdecode(np.frombuffer(base64.b64decode(det["mask_png_b64"]), np.uint8), 0) > 0
```

---

## 7. Gotchas (carry-over + WSL-specific)

- **One conda env per model** — TF vs PyTorch and CUDA pins clash; never share a process.
- **`CUDA_HOME` must be set** before `pip install -e GroundingDINO` (it builds a CUDA op).
- **GPU is on Windows, toolkit in WSL** — `nvidia-smi` should work in WSL; if not, fix the
  Windows driver first.
- **Avoid `/mnt/c` for weights/data** — copy into `~/` (ext4) for speed.
- **Port forwarding** — bind services to `0.0.0.0`; WSL2 exposes them to Windows at
  `localhost:<port>` automatically.
- **Weights** — large Google-Drive/FB links; download per the repos' READMEs (they move).
- **First call is slow** (model warm-up / lazy CUDA init); keep the services running.
- **Frames vs body frames** — Contact-GraspNet returns grasps in the **input cloud's frame**;
  feed the cloud in the frame you want, and transform with VO/VIO + hand-eye for the robot.

---

## 8. Build order for Claude

1. Verify GPU (`nvidia-smi`) and set up miniforge.
2. Service A env + GroundedSAM install + weights → load-and-segment smoke test → wrap in
   `service_gsam.py` (`/detect_segment`) → curl test.
3. Service B env + Contact-GraspNet install + checkpoint → grasp-on-test-cloud smoke test →
   wrap in `service_cgn.py` (`/grasps`) → curl test.
4. Provide a tiny `client_demo.py` that segments a sample frame (A) and gets grasps for a
   sample cloud (B), to confirm the Windows side can reach both.
5. Document the exact installed versions/weights in a `SERVER_NOTES.md` for reproducibility.

**Done when:** both services answer over `localhost`, A returns object masks and B returns
metric 6-DoF grasps, callable from the Windows pipeline.
