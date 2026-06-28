"""Shared grasp datatype + backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Grasp:
    pose: np.ndarray      # (4,4) in the INPUT cloud's frame (camera), meters
    width: float          # gripper opening at the grasp, meters
    score: float          # [0,1]


class GraspBackend(ABC):
    name: str = "base"
    device: str = "cpu"

    @abstractmethod
    def sample_grasps(self, cloud: np.ndarray,
                      segmentation: np.ndarray | None = None,
                      gripper_width_max: float = 0.085,
                      topk: int = 20) -> list[Grasp]:
        ...

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device}
