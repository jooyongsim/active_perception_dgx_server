"""SAM 3 backend — text/concept-promptable detection + segmentation in one model.

SAM 3 takes a text concept ("the mug") and returns every matching instance's box
+ mask in a single forward, replacing the Grounding-DINO -> SAM two-stage path.
Wired against the native `transformers` SAM 3 API (Sam3Model / Sam3Processor),
verified present in transformers >= 5.12.

Gating
------
The weights (`facebook/sam3`) are GATED ("manual" approval) on the Hugging Face
Hub. Once, on the DGX:
    1. Request access at https://huggingface.co/facebook/sam3 (manual approval).
    2. `huggingface-cli login`  (paste a token from a granted account), or
       export HF_TOKEN=hf_xxx before launching the server.
Until then, constructing this backend raises a clear RuntimeError; the registry
reports `sam3` unavailable in /health and `/segment?backend=sam3` returns an
actionable error — `gsam` keeps working.
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
            from transformers import Sam3Model, Sam3Processor
        except Exception as e:
            raise RuntimeError(
                "this transformers build has no SAM 3 (need transformers>=5.12): "
                f"{e}") from e
        try:
            self.processor = Sam3Processor.from_pretrained(self.model_id)
            self.model = (Sam3Model.from_pretrained(self.model_id)
                          .to(self.device).eval())
        except Exception as e:
            raise RuntimeError(
                f"could not load SAM 3 weights '{self.model_id}'. The model is "
                "gated: request access at https://huggingface.co/facebook/sam3 "
                "and `huggingface-cli login` (or set HF_TOKEN). Underlying error: "
                f"{e}") from e

    def detect_segment(self, image: Image.Image, prompt: str,
                       box_threshold: float = 0.30,
                       text_threshold: float = 0.25) -> list[Detection]:
        import torch

        with torch.no_grad():
            inputs = self.processor(images=image, text=prompt,
                                    return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        # SAM 3 returns one dict with scores/boxes/masks; use box_threshold as the
        # instance score threshold (text_threshold is unused — SAM 3 has no text head).
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=box_threshold, mask_threshold=0.5,
            target_sizes=[(image.size[1], image.size[0])])[0]

        scores = results["scores"]
        boxes = results["boxes"]
        masks = results["masks"]
        dets: list[Detection] = []
        for i in range(len(scores)):
            m = masks[i]
            mask = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
            b = boxes[i]
            box = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
            dets.append(Detection(prompt.strip().rstrip("."),
                                  float(scores[i]), box, mask))
        return dets

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device, "model": self.model_id}
