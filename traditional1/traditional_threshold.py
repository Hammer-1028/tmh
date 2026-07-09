import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def list_first_n_stems(processed_subset_dir, stage, n):
    stage_dir = Path(processed_subset_dir) / stage
    images = sorted(p for p in stage_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return [p.stem for p in images[:n]]


def load_stage_image(processed_subset_dir, stage, stem, fallback_stage="images_gamma"):
    stage_dir = Path(processed_subset_dir) / stage
    path = stage_dir / f"{stem}.png"
    if not path.exists():
        path = Path(processed_subset_dir) / fallback_stage / f"{stem}.png"
    return to_gray(read_image_unicode(path)), path


def moving_average(values, window=9):
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")


def robust_polyfit(xs, ys):
    if len(xs) < 30:
        return ys
    try:
        xc = xs - np.mean(xs)
        coef = np.polyfit(xc, ys, 2)
        fit = np.polyval(coef, xc)
        keep = np.abs(ys - fit) <= 10
        if np.count_nonzero(keep) >= 20:
            coef = np.polyfit(xc[keep], ys[keep], 2)
            return np.polyval(coef, xc)
    except np.linalg.LinAlgError:
        pass
    return ys


def detect_point_from_inner_dark_disk(point_image, point_radius=6):
    h, w = point_image.shape[:2]
    x1, x2 = int(0.30 * w), int(0.70 * w)
    y1, y2 = int(0.25 * h), int(0.68 * h)
    roi = point_image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    low_thr = np.percentile(blur, 8)
    very_dark = (blur <= low_thr).astype(np.uint8) * 255
    very_dark = cv2.morphologyEx(
        very_dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    very_dark = cv2.morphologyEx(
        very_dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(very_dark, 8)
    best = None
    for label_id in range(1, num_labels):
        bx, by, bw, bh, area = stats[label_id]
        ccx, ccy = centroids[label_id]
        aspect = bw / max(bh, 1)
        if area < 12 or area > 0.05 * roi.size:
            continue
        if bw < 4 or bh < 4 or bw > 0.20 * roi_w or bh > 0.20 * roi_h:
            continue
        if aspect < 0.5 or aspect > 2.0:
            continue
        dist = np.hypot(ccx - roi_w / 2, ccy - roi_h / 2)
        score = area - 70 * dist - 3 * abs(bw - bh)
        if best is None or score > best["score"]:
            best = {"cx": ccx, "cy": ccy, "score": score}
    if best is not None:
        cx = int(round(best["cx"] + x1))
        cy = int(round(best["cy"] + y1))
        pred_point = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(pred_point, (cx, cy), point_radius, 255, -1)
        return pred_point, cx, cy, "dark_percentile_component"

    _, dark_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    best = None
    for label_id in range(1, num_labels):
        bx, by, bw, bh, area = stats[label_id]
        cx, cy = centroids[label_id]
        aspect = bw / max(bh, 1)
        if area < 20 or area > 0.08 * roi.size:
            continue
        if bw < 4 or bh < 4 or bw > 0.25 * roi_w or bh > 0.25 * roi_h:
            continue
        if aspect < 0.5 or aspect > 2.0:
            continue
        if bx < 5 or by < 5 or bx + bw > roi_w - 5 or by + bh > roi_h - 5:
            continue
        dist = np.hypot(cx - roi_w / 2, cy - roi_h / 2)
        score = area - 50 * dist - 2 * abs(bw - bh)
        if best is None or score > best["score"]:
            best = {"cx": cx, "cy": cy, "score": score}

    method_used = "inner_dark_disk"
    if best is not None:
        cx = int(round(best["cx"] + x1))
        cy = int(round(best["cy"] + y1))
    else:
        _, bright_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_mask[:5, :] = 0
        bright_mask[-5:, :] = 0
        bright_mask[:, :5] = 0
        bright_mask[:, -5:] = 0
        ratio = np.count_nonzero(bright_mask) / bright_mask.size
        if 0.02 <= ratio <= 0.60:
            ys, xs = np.where(bright_mask > 0)
            cx_roi = np.median(xs)
            cy_roi = np.median(ys)
            # Keep the fallback from drifting to eyelids or lashes.
            max_dx = 0.18 * roi_w
            max_dy = 0.18 * roi_h
            cx_roi = np.clip(cx_roi, roi_w / 2 - max_dx, roi_w / 2 + max_dx)
            cy_roi = np.clip(cy_roi, roi_h / 2 - max_dy, roi_h / 2 + max_dy)
            cx = int(round(cx_roi + x1))
            cy = int(round(cy_roi + y1))
            method_used = "bright_ring_fallback"
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            method_used = "fallback_center"

    pred_point = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(pred_point, (cx, cy), point_radius, 255, -1)
    return pred_point, cx, cy, method_used


def threshold_dark_roi(roi, method):
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    if method == "otsu":
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    elif method == "adaptive_gaussian":
        mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return mask


def choose_main_dark_component(mask, x_ref_roi):
    roi_h, roi_w = mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        cx, cy = centroids[label_id]
        if area <= 0.02 * mask.size or area >= 0.85 * mask.size:
            continue
        if w <= 0.25 * roi_w or h <= 0.10 * roi_h:
            continue
        covers = x <= x_ref_roi <= x + w
        dist = abs(cx - x_ref_roi) / roi_w
        score = area + w - 20000 * dist + (10000 if covers else 0)
        candidates.append((score, label_id, cx))
    if not candidates:
        return None
    _, label_id, _ = max(candidates, key=lambda t: t[0])
    component = np.zeros_like(mask)
    component[labels == label_id] = 255
    return component


def split_by_x(points, x_ref_roi):
    if len(points) == 0:
        return None
    points = sorted(points, key=lambda p: p[0])
    segments = [[points[0]]]
    for prev, cur in zip(points[:-1], points[1:]):
        if cur[0] - prev[0] > 3:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    def score(seg):
        xs = [p[0] for p in seg]
        contains = min(xs) <= x_ref_roi <= max(xs)
        dist = 0 if contains else min(abs(x - x_ref_roi) for x in xs)
        return (1 if contains else 0, len(seg), -dist)

    best = max(segments, key=score)
    return best if len(best) >= 20 else None


def estimate_lower_boundary(roi, x, y_upper):
    roi_h = roi.shape[0]
    y_start = int(y_upper) + 2
    y_end = min(y_start + 45, roi_h)
    if y_end - y_start < 8:
        return None
    profile = roi[y_start:y_end, x].astype(np.uint8)
    # A local 1-D Otsu threshold. If the profile is almost flat, fall back to mean.
    if profile.max() - profile.min() >= 8:
        thr, _ = cv2.threshold(profile.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr = float(profile.mean())
    bright = profile > thr
    for i in range(0, len(bright) - 2):
        if bright[i] and bright[i + 1] and bright[i + 2]:
            y_lower = y_start + i
            height = y_lower - y_upper
            if 4 <= height <= 35:
                return int(y_lower)
            return None
    # Conservative fallback: use a local contrast-derived thickness, not a full region.
    fallback_height = int(np.clip(np.std(profile) * 0.8 + 8, 6, 22))
    y_lower = min(int(y_upper) + fallback_height, roi_h - 1)
    return y_lower if 4 <= y_lower - y_upper <= 35 else None


def detect_meniscus_line_band(image, point_cx):
    h, w = image.shape[:2]
    x1, x2 = int(0.04 * w), int(0.96 * w)
    y1, y2 = int(0.52 * h), int(0.92 * h)
    roi = image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    x_ref_roi = int(np.clip(point_cx - x1, 0, roi_w - 1))

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.max(blackhat, tophat)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, line_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask, 8)
    best = None
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        cx, cy = centroids[label_id]
        aspect = bw / max(bh, 1)
        touches_edge = x <= 2 or x + bw >= roi_w - 2 or y <= 2 or y + bh >= roi_h - 2
        if touches_edge:
            continue
        if area < 18 or area > 0.08 * line_mask.size:
            continue
        if bw < 35 or bh > 34 or aspect < 2.0:
            continue
        if cy < 0.30 * roi_h or cy > 0.92 * roi_h:
            continue
        x_dist = 0 if x <= x_ref_roi <= x + bw else min(abs(x - x_ref_roi), abs(x + bw - x_ref_roi))
        score = 3.0 * bw + 1.2 * cy - 2.5 * bh - 0.02 * area - 1.5 * x_dist
        if best is None or score > best["score"]:
            best = {
                "label_id": label_id,
                "score": float(score),
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
            }

    if best is None:
        return None, {"failed": True, "method": "failed_no_line_component"}

    selected = np.zeros_like(line_mask)
    selected[labels == best["label_id"]] = 255

    # Convert the thresholded line into a narrow meniscus band, not a one-pixel curve.
    band_h = int(np.clip(best["h"] + 8, 9, 18))
    band_w = 7
    band = cv2.dilate(
        selected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_w, band_h)),
        iterations=1,
    )
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3)))

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[y1:y2, x1:x2] = band
    ys, xs = np.where(pred > 0)
    if len(xs) == 0:
        return pred, {"failed": True, "method": "failed_empty_line_band"}
    return pred, {
        "failed": False,
        "method": "line_structure_band",
        "area": int(np.count_nonzero(pred)),
        "bbox_x": int(xs.min()),
        "bbox_y": int(ys.min()),
        "bbox_w": int(xs.max() - xs.min() + 1),
        "bbox_h": int(ys.max() - ys.min() + 1),
    }


def detect_meniscus_transition_band(image, point_cx):
    h, w = image.shape[:2]
    x1, x2 = int(0.08 * w), int(0.92 * w)
    y1, y2 = int(0.54 * h), int(0.90 * h)
    roi = image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    x_ref_roi = int(np.clip(point_cx - x1, 0, roi_w - 1))
    blur = cv2.GaussianBlur(roi, (5, 5), 0).astype(np.float32)

    points = []
    y_min = int(0.40 * roi_h)
    y_max = int(0.96 * roi_h)
    for x in range(0, roi_w):
        profile = blur[:, x]
        smooth = moving_average(profile, 7)
        diff = np.diff(smooth)
        search = diff[y_min:y_max]
        if len(search) < 5:
            continue
        local_thr = max(3.0, float(np.percentile(search, 82)))
        candidates = np.where(search >= local_thr)[0]
        if len(candidates) == 0:
            continue
        # The tear-meniscus/lower-lid transition is usually the lower significant
        # dark-to-bright transition, while Placido ring transitions are higher.
        y = int(candidates[-1] + y_min)
        if y >= roi_h - 8:
            continue
        points.append((x, y))

    if len(points) < 25:
        return None, {"failed": True, "method": "failed_few_transition_points"}

    points = sorted(points, key=lambda p: p[0])
    segments = [[points[0]]]
    for prev, cur in zip(points[:-1], points[1:]):
        if cur[0] - prev[0] > 3 or abs(cur[1] - prev[1]) > 20:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    def segment_score(seg):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        contains_ref = min(xs) <= x_ref_roi <= max(xs)
        dist = 0 if contains_ref else min(abs(x - x_ref_roi) for x in xs)
        # Prefer long, lower, smooth-enough segments around the reference x.
        return (1 if contains_ref else 0, len(seg), float(np.median(ys)), -dist)

    segment = max(segments, key=segment_score)
    if len(segment) < 20:
        return None, {"failed": True, "method": "failed_short_transition_segment"}

    xs = np.array([p[0] for p in segment], dtype=np.float32)
    ys = np.array([p[1] for p in segment], dtype=np.float32)
    ys = moving_average(ys, 11)
    ys = robust_polyfit(xs, ys)

    band = np.zeros_like(roi, dtype=np.uint8)
    for x, yc in zip(xs.astype(np.int32), ys.astype(np.int32)):
        # A narrow band around the detected transition. The lower side is
        # slightly thicker because tear meniscus lies below the corneal edge.
        top = int(np.clip(yc - 3, 0, roi_h - 1))
        bottom = int(np.clip(yc + 13, 0, roi_h - 1))
        if bottom > top:
            band[top : bottom + 1, x] = 255
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5)))
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[y1:y2, x1:x2] = band
    ys_global, xs_global = np.where(pred > 0)
    if len(xs_global) == 0:
        return pred, {"failed": True, "method": "failed_empty_transition_band"}

    bbox_w = int(xs_global.max() - xs_global.min() + 1)
    bbox_h = int(ys_global.max() - ys_global.min() + 1)
    if bbox_w < 45 or bbox_h > 70 or int(ys_global.min()) < int(0.60 * h):
        return np.zeros((h, w), dtype=np.uint8), {"failed": True, "method": "failed_rejected_transition_band"}

    return pred, {
        "failed": False,
        "method": "vertical_transition_band",
        "area": int(np.count_nonzero(pred)),
        "bbox_x": int(xs_global.min()),
        "bbox_y": int(ys_global.min()),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
    }


def detect_meniscus_band(meniscus_image, point_cx, method):
    transition_pred, transition_info = detect_meniscus_transition_band(meniscus_image, point_cx)
    if transition_pred is not None and not transition_info.get("failed", True):
        return transition_pred, transition_info

    line_pred, line_info = detect_meniscus_line_band(meniscus_image, point_cx)
    if line_pred is not None and not line_info.get("failed", True):
        return line_pred, line_info

    h, w = meniscus_image.shape[:2]
    x1, x2 = int(0.05 * w), int(0.95 * w)
    y1, y2 = int(0.50 * h), int(0.94 * h)
    roi = meniscus_image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    x_ref_roi = int(np.clip(point_cx - x1, 0, roi_w - 1))

    dark = threshold_dark_roi(roi, method)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 7)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    component = choose_main_dark_component(dark, x_ref_roi)
    if component is None:
        return np.zeros((h, w), dtype=np.uint8), {"failed": True, "method": "failed_no_dark_component"}

    raw = []
    for x in range(roi_w):
        ys = np.where(component[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        y_upper = int(ys.max())
        if y_upper >= roi_h - 38:
            continue
        if x < int(0.06 * roi_w) or x > int(0.94 * roi_w):
            continue
        y_lower = estimate_lower_boundary(roi, x, y_upper)
        if y_lower is None:
            continue
        raw.append((x, y_upper, y_lower))

    segment = split_by_x(raw, x_ref_roi)
    if segment is None:
        return np.zeros((h, w), dtype=np.uint8), {"failed": True, "method": "failed_few_band_points"}

    xs = np.array([p[0] for p in segment], dtype=np.float32)
    upper = np.array([p[1] for p in segment], dtype=np.float32)
    lower = np.array([p[2] for p in segment], dtype=np.float32)
    upper = moving_average(upper, 11)
    lower = moving_average(lower, 11)
    upper = robust_polyfit(xs, upper)
    lower = robust_polyfit(xs, lower)

    valid = (lower - upper >= 4) & (lower - upper <= 35)
    if np.count_nonzero(valid) < 20:
        return np.zeros((h, w), dtype=np.uint8), {"failed": True, "method": "failed_invalid_thickness"}

    xs = xs[valid].astype(np.int32)
    upper = upper[valid].astype(np.int32)
    lower = lower[valid].astype(np.int32)

    mask_roi = np.zeros_like(roi, dtype=np.uint8)
    for x, yu, yl in zip(xs, upper, lower):
        yu = int(np.clip(yu, 0, roi_h - 1))
        yl = int(np.clip(yl, 0, roi_h - 1))
        if yl > yu:
            mask_roi[yu : yl + 1, x] = 255
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)))

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[y1:y2, x1:x2] = mask_roi
    if np.count_nonzero(pred) == 0:
        return pred, {"failed": True, "method": "failed_empty_after_clean"}

    ys, xs_global = np.where(pred > 0)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs_global.max() - xs_global.min() + 1)
    if bbox_h > 55 or bbox_w < 25:
        return np.zeros((h, w), dtype=np.uint8), {"failed": True, "method": "failed_rejected_dark_band"}

    return pred, {
        "failed": False,
        "method": "dark_component_band_fallback",
        "area": int(np.count_nonzero(pred)),
        "bbox_x": int(xs_global.min()),
        "bbox_y": int(ys.min()),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
    }


def compute_tmh_from_meniscus(pred_meniscus, point_cx):
    h, w = pred_meniscus.shape[:2]
    x1 = max(0, int(point_cx) - 5)
    x2 = min(w - 1, int(point_cx) + 5)
    heights, uppers, lowers = [], [], []
    for x in range(x1, x2 + 1):
        ys = np.where(pred_meniscus[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        uppers.append(int(ys.min()))
        lowers.append(int(ys.max()))
        heights.append(int(ys.max() - ys.min() + 1))
    if not heights:
        return {"tmh_pixel": np.nan, "tmh_x_ref": int(point_cx), "tmh_y_upper": np.nan, "tmh_y_lower": np.nan}
    return {
        "tmh_pixel": float(np.median(heights)),
        "tmh_x_ref": int(point_cx),
        "tmh_y_upper": float(np.median(uppers)),
        "tmh_y_lower": float(np.median(lowers)),
    }


def create_final_overlay(overlay_image, pred_point, pred_meniscus, tmh):
    base = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)
    color = base.copy()
    color[pred_meniscus > 0] = [0, 255, 0]
    blended = cv2.addWeighted(base, 0.65, color, 0.35, 0)
    blended[pred_point > 0] = [0, 0, 255]
    if not np.isnan(tmh["tmh_pixel"]):
        x = int(tmh["tmh_x_ref"])
        y1 = int(tmh["tmh_y_upper"])
        y2 = int(tmh["tmh_y_lower"])
        cv2.line(blended, (x, y1), (x, y2), (0, 255, 255), 2)
        cv2.putText(blended, f"TMH={tmh['tmh_pixel']:.1f}px", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return blended


def process_one_stem(stem, args):
    point_image, point_path = load_stage_image(args.processed_subset_dir, args.point_stage, stem)
    meniscus_image, meniscus_path = load_stage_image(args.processed_subset_dir, args.meniscus_stage, stem)
    overlay_image, _ = load_stage_image(args.processed_subset_dir, args.overlay_stage, stem)

    pred_point, cx, cy, point_method = detect_point_from_inner_dark_disk(point_image)
    pred_meniscus, meniscus_info = detect_meniscus_band(meniscus_image, cx, args.method)
    pred_all = cv2.bitwise_or(pred_point, pred_meniscus)
    tmh = compute_tmh_from_meniscus(pred_meniscus, cx)
    overlay = create_final_overlay(overlay_image, pred_point, pred_meniscus, tmh)

    out = Path(args.out_dir)
    write_image_unicode(out / "pred_point" / "Colour1" / f"{stem}.png", pred_point)
    write_image_unicode(out / "pred_meniscus" / "Colour1" / f"{stem}.png", pred_meniscus)
    write_image_unicode(out / "pred_all" / "Colour1" / f"{stem}.png", pred_all)
    write_image_unicode(out / "overlay_final" / "Colour1" / f"{stem}_overlay.png", overlay)

    return {
        "stem": stem,
        "input_point_path": str(point_path),
        "input_meniscus_path": str(meniscus_path),
        "method": args.method,
        "point_cx": cx,
        "point_cy": cy,
        "point_radius": 6,
        "point_method_used": point_method,
        "meniscus_failed": bool(meniscus_info.get("failed", True)),
        "meniscus_method_used": meniscus_info.get("method", "unknown"),
        "meniscus_area": int(meniscus_info.get("area", 0)),
        "meniscus_bbox_x": int(meniscus_info.get("bbox_x", 0)),
        "meniscus_bbox_y": int(meniscus_info.get("bbox_y", 0)),
        "meniscus_bbox_w": int(meniscus_info.get("bbox_w", 0)),
        "meniscus_bbox_h": int(meniscus_info.get("bbox_h", 0)),
        **tmh,
        "status": "success",
        "warning_message": "",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Clean A2 threshold method for meniscus band on first 10 Colour1 images.")
    parser.add_argument("--processed_subset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--point_stage", default="images_gamma")
    parser.add_argument("--meniscus_stage", default="images_bilateral")
    parser.add_argument("--overlay_stage", default="images_gamma")
    parser.add_argument("--method", choices=["otsu", "adaptive_mean", "adaptive_gaussian"], default="otsu")
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    for subdir in ["pred_point", "pred_meniscus", "pred_all", "overlay_final"]:
        ensure_dir(out / subdir / "Colour1")

    rows = []
    stems = list_first_n_stems(args.processed_subset_dir, args.point_stage, args.num_images)
    for stem in stems:
        try:
            rows.append(process_one_stem(stem, args))
        except Exception as exc:
            rows.append(
                {
                    "stem": stem,
                    "input_point_path": "",
                    "input_meniscus_path": "",
                    "method": args.method,
                    "point_cx": 0,
                    "point_cy": 0,
                    "point_radius": 6,
                    "point_method_used": "failed",
                    "meniscus_failed": True,
                    "meniscus_method_used": "failed",
                    "meniscus_area": 0,
                    "meniscus_bbox_x": 0,
                    "meniscus_bbox_y": 0,
                    "meniscus_bbox_w": 0,
                    "meniscus_bbox_h": 0,
                    "tmh_pixel": np.nan,
                    "tmh_x_ref": 0,
                    "tmh_y_upper": np.nan,
                    "tmh_y_lower": np.nan,
                    "status": "failed",
                    "warning_message": str(exc),
                }
            )

    summary = out / "threshold_summary_first10.csv"
    pd.DataFrame(rows).to_csv(summary, index=False, encoding="utf-8-sig")
    print("Traditional threshold meniscus-band processing finished.")
    print(f"Processed: {len(rows)}")
    print(f"Output dir: {out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
