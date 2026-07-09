from __future__ import annotations

import math

import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "f1": f1,
    }


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return math.nan, math.nan
    return float(xs.mean()), float(ys.mean())


def point_error_from_heatmap(pred_heatmap: np.ndarray, gt_point_mask: np.ndarray) -> float:
    if pred_heatmap.size == 0 or not np.isfinite(pred_heatmap).all():
        return math.nan
    y, x = np.unravel_index(int(np.argmax(pred_heatmap)), pred_heatmap.shape)
    gx, gy = mask_center(gt_point_mask)
    if not math.isfinite(gx) or not math.isfinite(gy):
        return math.nan
    return float(((x - gx) ** 2 + (y - gy) ** 2) ** 0.5)

