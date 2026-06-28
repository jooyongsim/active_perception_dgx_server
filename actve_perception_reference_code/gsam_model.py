"""Grounding-DINO + SAM via HuggingFace transformers.

Text-promptable detection + segmentation:
    "the mug"  --Grounding-DINO-->  box(es)  --SAM-->  mask(s)

Device-agnostic: uses CUDA when available (e.g. an attached eGPU), else CPU.
No CUDA-op compilation is required -- everything runs through `transformers`,
so this works on a CPU-only WSL box and transparently accelerates on GPU later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# Default model ids. *-tiny / *-base are chosen so the stack is usable on CPU.
# Swap to grounding-dino-base / sam-vit-huge for quality once a GPU is attached.
GDINO_ID = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
SAM_ID = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]          # [x0, y0, x1, y1] in pixels
    mask: np.ndarray          # bool array (H, W)


class GroundedSAM:
    """Loads Grounding-DINO + SAM once, then segments on demand."""

    def __init__(self, device: str | None = None,
                 gdino_id: str = GDINO_ID, sam_id: str = SAM_ID):
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, SamModel, SamProcessor)

        self.device = device or pick_device()
        # float32 on CPU; fp16 only helps on GPU and SAM/ GDINO are fine in fp32.
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(gdino_id).to(self.device).eval())
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        # Grounding-DINO expects lowercase, '.'-separated phrases ending in '.'.
        p = prompt.lower().strip()
        if not p.endswith("."):
            p += "."
        return p

    @torch.no_grad()
    def detect(self, image: Image.Image, prompt: str,
               box_threshold: float = 0.30,
               text_threshold: float = 0.25) -> list[dict]:
        """Grounding-DINO: text prompt -> list of {label, score, box}."""
        text = self._normalize_prompt(prompt)
        inputs = self.gdino_processor(images=image, text=text,
                                      return_tensors="pt").to(self.device)
        outputs = self.gdino(**inputs)
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
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
    def segment_boxes(self, image: Image.Image,
                      boxes: list[list[float]]) -> list[np.ndarray]:
        """SAM: prompt with boxes -> one bool mask (H, W) per box."""
        if not boxes:
            return []
        # transformers SAM expects input_boxes shaped (batch, num_boxes, 4).
        inputs = self.sam_processor(image, input_boxes=[boxes],
                                    return_tensors="pt").to(self.device)
        outputs = self.sam(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (num_boxes, num_multimask, H, W) bool tensor
        iou = outputs.iou_scores.cpu()[0]      # (num_boxes, num_multimask)
        out = []
        for i in range(masks.shape[0]):
            best = int(torch.argmax(iou[i]))   # keep SAM's highest-IoU mask
            out.append(masks[i, best].numpy().astype(bool))
        return out

    def detect_segment(self, image: Image.Image, prompt: str,
                       box_threshold: float = 0.30,
                       text_threshold: float = 0.25) -> list[Detection]:
        dets = self.detect(image, prompt, box_threshold, text_threshold)
        masks = self.segment_boxes(image, [d["box"] for d in dets])
        return [Detection(d["label"], d["score"], d["box"], m)
                for d, m in zip(dets, masks)]
