"""Common interface shared by every segmentation backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]          # [x0, y0, x1, y1] in pixels
    mask: np.ndarray          # bool (H, W)


class SegBackend(ABC):
    """Text-promptable detection + segmentation. Loads weights once in __init__."""

    name: str = "base"
    device: str = "cpu"

    @abstractmethod
    def detect_segment(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        ...

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device}
