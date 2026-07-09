from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(bool) * 255).astype(np.uint8)).save(path)


def make_overlay(image: Image.Image, point_mask: np.ndarray, meniscus_mask: np.ndarray) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).copy()
    point = point_mask.astype(bool)
    meniscus = meniscus_mask.astype(bool)
    arr[meniscus] = (0.45 * arr[meniscus] + 0.55 * np.array([0, 220, 80])).astype(np.uint8)
    arr[point] = (255, 50, 50)
    return Image.fromarray(arr)


def draw_tmh_line(overlay: Image.Image, x_ref: float, y_upper: float, y_lower: float) -> Image.Image:
    out = overlay.copy()
    if np.isfinite(x_ref) and np.isfinite(y_upper) and np.isfinite(y_lower):
        draw = ImageDraw.Draw(out)
        x = int(round(x_ref))
        draw.line((x, int(round(y_upper)), x, int(round(y_lower))), fill=(255, 230, 0), width=2)
    return out

