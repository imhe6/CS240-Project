from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import ViewSample


def load_blender_views(
    source: str | Path,
    split: str = "train",
    limit: int | None = None,
    white_background: bool = False,
) -> list[ViewSample]:
    """Load NeRF Synthetic / Blender-format posed views.

    Expected layout:
        scene/
          transforms_train.json
          train/r_0.png
          train/r_1.png

    The same format is used by the SequenceMatters Blender scripts.
    """

    source = Path(source)
    transforms_path = source / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing Blender transform file: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    frames = metadata.get("frames", [])
    if limit is not None:
        frames = frames[:limit]

    samples: list[ViewSample] = []
    for index, frame in enumerate(frames):
        image_path = _resolve_frame_path(source, frame["file_path"])
        image = _load_rgb_image(image_path, white_background=white_background)

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Frame {index} has invalid transform_matrix shape {c2w.shape}")

        center = c2w[:3, 3]
        viewing_direction = -c2w[:3, 2]
        samples.append(
            ViewSample(
                name=image_path.name,
                image=image,
                camera_center=center,
                viewing_direction=viewing_direction,
                image_path=image_path,
            )
        )

    if not samples:
        raise ValueError(f"No frames loaded from {transforms_path}")
    return samples


def _resolve_frame_path(source: Path, frame_path: str) -> Path:
    raw = Path(frame_path)
    candidates = []
    if raw.is_absolute():
        candidates.extend([raw, raw.with_suffix(".png")])
    else:
        candidates.extend([source / raw, (source / raw).with_suffix(".png")])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for frame path {frame_path!r} under {source}")


def _load_rgb_image(path: Path, white_background: bool) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    background = np.ones(3, dtype=np.float32) if white_background else np.zeros(3, dtype=np.float32)
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4] + background * (1.0 - rgba[:, :, 3:4])
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
