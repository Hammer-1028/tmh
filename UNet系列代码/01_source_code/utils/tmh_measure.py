from __future__ import annotations

import math

import numpy as np


def tmh_pixel(mask: np.ndarray, x_ref: float, window: int = 5) -> dict[str, float]:
    mask = mask.astype(bool)
    h, w = mask.shape
    if not math.isfinite(x_ref):
        return {"tmh_pixel": math.nan, "y_upper": math.nan, "y_lower": math.nan, "valid_columns": 0}
    xc = int(round(x_ref))
    xs = range(max(0, xc - window), min(w, xc + window + 1))
    heights = []
    uppers = []
    lowers = []
    for x in xs:
        ys = np.where(mask[:, x])[0]
        if len(ys) == 0:
            continue
        y0 = int(ys.min())
        y1 = int(ys.max())
        heights.append(y1 - y0 + 1)
        uppers.append(y0)
        lowers.append(y1)
    if not heights:
        return {"tmh_pixel": math.nan, "y_upper": math.nan, "y_lower": math.nan, "valid_columns": 0}
    return {
        "tmh_pixel": float(np.median(heights)),
        "y_upper": float(np.median(uppers)),
        "y_lower": float(np.median(lowers)),
        "valid_columns": int(len(heights)),
    }

