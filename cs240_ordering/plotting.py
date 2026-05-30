from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .data import ViewSample


def save_adjacency_plot(cost: np.ndarray, order: Sequence[int], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(cost[order[i], order[i + 1]]) for i in range(len(order) - 1)]
    plt.figure(figsize=(7, 3.5))
    plt.plot(range(len(values)), values, marker="o")
    plt.title(title)
    plt.xlabel("adjacent pair")
    plt.ylabel("edge cost")
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_sequence_strip(samples: Sequence[ViewSample], order: Sequence[int], path: Path, thumb_size: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not order:
        return

    label_height = 24
    canvas = Image.new("RGB", (thumb_size * len(order), thumb_size + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    for position, sample_idx in enumerate(order):
        sample = samples[sample_idx]
        image = Image.fromarray(sample.image[:, :, :3]).resize((thumb_size, thumb_size))
        x = position * thumb_size
        canvas.paste(image, (x, 0))
        draw.text((x + 4, thumb_size + 4), sample.name.replace(".png", ""), fill=(0, 0, 0))

    canvas.save(path)
