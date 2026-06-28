"""Grounding-DINO + SAM backend via HuggingFace transformers.

    "the mug"  --Grounding-DINO-->  box(es)  --SAM-->  mask(s)

No CUDA-op compilation needed (everything runs through `transformers`), so it
loads on CPU and accelerates on the GB10 GPU automatically. This is the proven,
ungated default. Carried over from the reference `gsam_model.py`.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .. import config
from .base import Detection, SegBackend


class GroundedSAM(SegBackend):
    name = "gsam"

    def __init__(self, device: str | None = None,
                 gdino_id: str | None = None, sam_id: str | None = None):
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, SamModel, SamProcessor)

        gdino_id = gdino_id or config.GDINO_MODEL
        sam_id = sam_id or config.SAM_MODEL
        self.device = device or config.pick_device()

        self.gdino_processor = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(gdino_id).to(self.device).eval())
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()
        self._gdino_id, self._sam_id = gdino_id, sam_id

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        # Grounding-DINO wants lowercase, '.'-separated phrases ending in '.'.
        p = prompt.lower().strip()
        if not p.endswith("."):
            p += "."
        return p

    @torch.no_grad()
    def _detect(self, image: Image.Image, prompt: str,
                box_threshold: float, text_threshold: float) -> list[dict]:
        text = self._normalize_prompt(prompt)
        inputs = self.gdino_processor(images=image, text=text,
                                      return_tensors="pt").to(self.device)
        outputs = self.gdino(**inputs)
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],   # (height, width)
        )[0]
        dets = []
        for box, score, label in zip(results["boxes"], results["scores"],
                                     results["labels"]):
            dets.append({
                "label": label if isinstance(label, str) else str(label),
                "score": float(score),
                "box": [float(v) for v in box.tolist()],
            })
        return dets

    @torch.no_grad()
    def _segment_boxes(self, image: Image.Image,
                       boxes: list[list[float]]) -> list[np.ndarray]:
        if not boxes:
            return []
        inputs = self.sam_processor(image, input_boxes=[boxes],
                                    return_tensors="pt").to(self.device)
        outputs = self.sam(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]                                   # (num_boxes, multimask, H, W)
        iou = outputs.iou_scores.cpu()[0]      # (num_boxes, multimask)
        out = []
        for i in range(masks.shape[0]):
            best = int(torch.argmax(iou[i]))   # keep SAM's highest-IoU mask
            out.append(masks[i, best].numpy().astype(bool))
        return out

    def detect_segment(self, image: Image.Image, prompt: str,
                       box_threshold: float = 0.30,
                       text_threshold: float = 0.25) -> list[Detection]:
        dets = self._detect(image, prompt, box_threshold, text_threshold)
        masks = self._segment_boxes(image, [d["box"] for d in dets])
        return [Detection(d["label"], d["score"], d["box"], m)
                for d, m in zip(dets, masks)]

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device,
                "gdino": self._gdino_id, "sam": self._sam_id}
