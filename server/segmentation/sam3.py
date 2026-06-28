"""SAM 3 backend — text/concept-promptable segmentation in one model.

SAM 3 unifies detection + segmentation from a text concept prompt ("the mug"),
so it replaces the Grounding-DINO -> SAM two-stage path with a single forward.

Status / gating
---------------
SAM 3 weights (`facebook/sam3`) are gated on the Hugging Face Hub. Before this
backend can load you must, once on the DGX:
    huggingface-cli login                       # accept the model license
    export SAM3_MODEL=facebook/sam3             # (already the default)
If the weights or the `transformers` SAM 3 classes are unavailable, constructing
this backend raises a clear RuntimeError; the registry then reports `sam3` as
unavailable in /health and `/segment?backend=sam3` returns an actionable error
instead of crashing — `gsam` keeps working.

Integration seam
----------------
The exact `transformers` SAM 3 processor/output field names are still settling
across releases. The inference adapter below tries the documented interface and
falls back across a couple of known output shapes. If your installed transformers
exposes a different SAM 3 API, this `_run` method is the single place to adjust.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import config
from .base import Detection, SegBackend


class SAM3(SegBackend):
    name = "sam3"

    def __init__(self, device: str | None = None, model_id: str | None = None):
        self.model_id = model_id or config.SAM3_MODEL
        self.device = device or config.pick_device()
        try:
            import transformers  # noqa: F401
            from transformers import AutoModel, AutoProcessor
        except Exception as e:  # transformers missing entirely
            raise RuntimeError(f"transformers unavailable for SAM 3: {e}") from e

        try:
            # SAM 3 ships dedicated classes; fall back to Auto* which resolve them.
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True).to(self.device).eval()
        except Exception as e:
            raise RuntimeError(
                f"could not load SAM 3 weights '{self.model_id}'. The model is "
                "gated: run `huggingface-cli login` and accept the license, or "
                f"set SAM3_MODEL to a local path. Underlying error: {e}") from e

    def detect_segment(self, image: Image.Image, prompt: str,
                       box_threshold: float = 0.30,
                       text_threshold: float = 0.25) -> list[Detection]:
        import torch

        with torch.no_grad():
            inputs = self.processor(images=image, text=prompt,
                                    return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        return self._postprocess(outputs, image, box_threshold)

    # --- the one method to adapt if the transformers SAM 3 API differs --------
    def _postprocess(self, outputs, image: Image.Image,
                     threshold: float) -> list[Detection]:
        h, w = image.size[1], image.size[0]
        post = getattr(self.processor, "post_process_instance_segmentation", None) \
            or getattr(self.processor, "post_process_grounded_segmentation", None)
        if post is None:
            raise RuntimeError(
                "this transformers build exposes no SAM 3 post-processor; "
                "adjust SAM3._postprocess for your version")
        results = post(outputs, threshold=threshold, target_sizes=[(h, w)])[0]

        masks = results.get("masks")
        scores = results.get("scores", [1.0] * (len(masks) if masks is not None else 0))
        boxes = results.get("boxes")
        labels = results.get("labels", ["object"] * len(scores))

        dets: list[Detection] = []
        for i in range(len(scores)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            mask = m.astype(bool)
            if boxes is not None:
                b = boxes[i]
                box = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
            else:
                ys, xs = np.where(mask)
                box = ([float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())] if xs.size else [0, 0, 0, 0])
            lbl = labels[i]
            dets.append(Detection(str(lbl), float(scores[i]), box, mask))
        return dets

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device, "model": self.model_id}
