from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Component:
    area: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    pixels_y: np.ndarray
    pixels_x: np.ndarray

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[Component]:
    """4-connected components for binary masks without requiring OpenCV/scipy."""
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []
    ys_all, xs_all = np.where(mask)

    for start_y, start_x in zip(ys_all, xs_all):
        if seen[start_y, start_x]:
            continue
        q: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        ys: list[int] = []
        xs: list[int] = []
        while q:
            y, x = q.popleft()
            ys.append(y)
            xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))

        if len(xs) >= min_area:
            yy = np.asarray(ys, dtype=np.int32)
            xx = np.asarray(xs, dtype=np.int32)
            comps.append(
                Component(
                    area=len(xs),
                    x1=int(xx.min()),
                    y1=int(yy.min()),
                    x2=int(xx.max()) + 1,
                    y2=int(yy.max()) + 1,
                    cx=float(xx.mean()),
                    cy=float(yy.mean()),
                    pixels_y=yy,
                    pixels_x=xx,
                )
            )
    return comps


def split_point_meniscus(label_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split dataset label into point and lower tear meniscus masks.

    The public labels usually contain two objects: upper central pupillary
    reference area and lower tear meniscus. A few Colour1 samples include tiny
    stray components, so classification is score based instead of blindly using
    the largest/smallest component.
    """
    h, w = label_mask.shape
    comps = connected_components(label_mask, min_area=1)
    non_tiny = [c for c in comps if c.area >= 20]
    candidates = non_tiny if non_tiny else comps

    point_mask = np.zeros((h, w), dtype=bool)
    meniscus_mask = np.zeros((h, w), dtype=bool)
    info = {
        "component_count": len(comps),
        "used_component_count": len(candidates),
        "point_area": 0,
        "meniscus_area": 0,
        "point_cx": np.nan,
        "point_cy": np.nan,
        "meniscus_cx": np.nan,
        "meniscus_cy": np.nan,
        "warning": "",
    }

    if not candidates:
        info["warning"] = "empty_label"
        return point_mask, meniscus_mask, info

    def point_score(c: Component) -> float:
        upper = max(0.0, (0.62 * h - c.cy) / h)
        center = 1.0 - min(1.0, abs(c.cx - 0.5 * w) / (0.5 * w))
        area_penalty = min(1.0, c.area / max(1.0, 0.02 * h * w))
        squareish = 1.0 - min(1.0, abs(c.aspect - 1.0))
        return 3.0 * upper + 1.0 * center + 1.0 * squareish - 0.7 * area_penalty

    def meniscus_score(c: Component) -> float:
        lower = max(0.0, (c.cy - 0.42 * h) / h)
        horizontal = min(3.0, c.aspect) / 3.0
        area = min(1.0, c.area / max(1.0, 0.015 * h * w))
        width = min(1.0, c.width / max(1.0, 0.35 * w))
        return 3.0 * lower + 1.4 * horizontal + area + width

    point_comp = max(candidates, key=point_score)
    meniscus_candidates = [c for c in candidates if c is not point_comp]
    if not meniscus_candidates:
        meniscus_candidates = candidates
        info["warning"] = "single_component"
    meniscus_comp = max(meniscus_candidates, key=meniscus_score)

    point_mask[point_comp.pixels_y, point_comp.pixels_x] = True
    meniscus_mask[meniscus_comp.pixels_y, meniscus_comp.pixels_x] = True

    info.update(
        {
            "point_area": point_comp.area,
            "meniscus_area": meniscus_comp.area,
            "point_cx": point_comp.cx,
            "point_cy": point_comp.cy,
            "meniscus_cx": meniscus_comp.cx,
            "meniscus_cy": meniscus_comp.cy,
            "point_bbox": f"{point_comp.x1},{point_comp.y1},{point_comp.x2},{point_comp.y2}",
            "meniscus_bbox": f"{meniscus_comp.x1},{meniscus_comp.y1},{meniscus_comp.x2},{meniscus_comp.y2}",
        }
    )
    if point_comp.cy > 0.7 * h or meniscus_comp.cy < 0.35 * h:
        info["warning"] = (info["warning"] + ";suspicious_geometry").strip(";")
    return point_mask, meniscus_mask, info


def gaussian_heatmap(shape: tuple[int, int], cx: float, cy: float, sigma: float = 6.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    if not np.isfinite(cx) or not np.isfinite(cy):
        return np.zeros((h, w), dtype=np.float32)
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return heat.astype(np.float32)

