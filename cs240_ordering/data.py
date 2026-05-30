from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ViewSample:
    """A posed multi-view image used by the ordering experiments."""

    name: str
    image: np.ndarray
    camera_center: np.ndarray
    viewing_direction: np.ndarray
    image_path: Optional[Path] = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        center = np.asarray(self.camera_center, dtype=np.float64)
        direction = np.asarray(self.viewing_direction, dtype=np.float64)

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image must be an H x W x C RGB/RGBA array")
        if center.shape != (3,):
            raise ValueError("camera_center must be a length-3 vector")
        if direction.shape != (3,):
            raise ValueError("viewing_direction must be a length-3 vector")

        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("viewing_direction must be non-zero")

        object.__setattr__(self, "image", image)
        object.__setattr__(self, "camera_center", center)
        object.__setattr__(self, "viewing_direction", direction / norm)
