from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .data import ViewSample


def generate_synthetic_views(n: int = 12, seed: int = 240, image_size: int = 128) -> list[ViewSample]:
    """Create a small posed image set with one intentionally awkward outlier."""

    if n < 4:
        raise ValueError("n must be at least 4")

    rng = np.random.default_rng(seed)
    samples = []
    for idx in range(n):
        angle = 2.0 * np.pi * idx / n
        radius = 2.0 + float(rng.normal(0.0, 0.04))
        center = np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.25 * np.sin(2.0 * angle) + float(rng.normal(0.0, 0.02)),
            ],
            dtype=np.float64,
        )

        if idx == n - 1:
            center += np.array([0.95, -0.65, 0.55])

        direction = -center + rng.normal(0.0, 0.03, size=3)
        image = _render_synthetic_image(idx, n, angle, image_size, rng)
        samples.append(
            ViewSample(
                name=f"view_{idx:03d}.png",
                image=image,
                camera_center=center,
                viewing_direction=direction,
            )
        )
    return samples


def greedy_counterexample_matrix() -> np.ndarray:
    """A tiny metric-like matrix where nearest-neighbor greedy is not optimal."""

    return np.array(
        [
            [0.0, 2.0, 2.0, 9.0, 9.0],
            [2.0, 0.0, 3.0, 4.0, 9.0],
            [2.0, 3.0, 0.0, 9.0, 4.0],
            [9.0, 4.0, 9.0, 0.0, 1.0],
            [9.0, 9.0, 4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def expected_counterexample_optimum() -> tuple[list[int], float]:
    return [3, 4, 2, 0, 1], 9.0


def _render_synthetic_image(
    idx: int,
    n: int,
    angle: float,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y, x = np.mgrid[0:image_size, 0:image_size]
    x_norm = x / max(1, image_size - 1)
    y_norm = y / max(1, image_size - 1)

    base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base[..., 0] = np.clip(95 + 80 * np.cos(angle) + 50 * x_norm, 0, 255).astype(np.uint8)
    base[..., 1] = np.clip(105 + 75 * np.sin(angle) + 45 * y_norm, 0, 255).astype(np.uint8)
    base[..., 2] = np.clip(145 + 55 * np.sin(angle + x_norm * np.pi), 0, 255).astype(np.uint8)

    center = (
        int(image_size * (0.50 + 0.24 * np.cos(angle))),
        int(image_size * (0.50 + 0.24 * np.sin(angle))),
    )
    color = (
        int(150 + 80 * np.cos(angle + 0.7)),
        int(150 + 80 * np.sin(angle + 1.4)),
        int(120 + 80 * np.cos(angle - 0.5)),
    )
    cv2.circle(base, center, image_size // 7, color, thickness=-1)

    line_angle = angle + np.pi / 4.0
    start = (
        int(image_size * (0.5 - 0.36 * np.cos(line_angle))),
        int(image_size * (0.5 - 0.36 * np.sin(line_angle))),
    )
    end = (
        int(image_size * (0.5 + 0.36 * np.cos(line_angle))),
        int(image_size * (0.5 + 0.36 * np.sin(line_angle))),
    )
    cv2.line(base, start, end, (245, 245, 245), thickness=3)
    cv2.putText(
        base,
        str(idx),
        (image_size // 10, image_size - image_size // 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    if idx == n - 1:
        cv2.rectangle(base, (8, 8), (image_size // 2, image_size // 3), (255, 245, 40), thickness=-1)

    noise = rng.normal(0.0, 3.0, size=base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
