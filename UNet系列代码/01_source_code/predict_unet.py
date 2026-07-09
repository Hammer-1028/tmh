from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from dataset import TearMeniscusDataset
from models import UNetMultitask
from utils.label_split import connected_components
from utils.tmh_measure import tmh_pixel
from utils.visualization import draw_tmh_line, make_overlay, save_mask


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict point and meniscus masks with trained U-Net.")
    parser.add_argument("--processed_dir", type=Path, default=root / "processed")
    parser.add_argument("--splits_dir", type=Path, default=root / "splits")
    parser.add_argument("--checkpoint", type=Path, default=root / "results" / "unet_test" / "best_model.pth")
    parser.add_argument("--out_dir", type=Path, default=root / "results" / "unet_test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low_threshold", type=float, default=0.0, help="If >0, use low/high hysteresis for meniscus.")
    parser.add_argument("--point_method", choices=["argmax", "weighted"], default="argmax")
    parser.add_argument("--point_peak_ratio", type=float, default=0.65, help="For weighted point, keep pixels >= ratio * max heat.")
    parser.add_argument("--point_radius", type=int, default=6)
    parser.add_argument("--horizontal_closing", type=int, default=0, help="Horizontal binary closing width for meniscus mask before component filtering.")
    return parser


def postprocess_meniscus(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    comps = connected_components(mask, min_area=20)
    keep = []
    for c in comps:
        lower = c.cy > 0.38 * h
        horizontal = c.width >= max(12, c.height * 2)
        if lower and horizontal:
            keep.append(c)
    if not keep:
        keep = [c for c in comps if c.cy > 0.35 * h]
    if not keep:
        return mask.astype(bool)
    best = max(keep, key=lambda c: c.area + 10 * c.width)
    out = np.zeros_like(mask, dtype=bool)
    out[best.pixels_y, best.pixels_x] = True
    return out


def hysteresis_mask(prob: np.ndarray, high: float, low: float) -> np.ndarray:
    high_mask = prob > high
    low_mask = prob > low
    comps = connected_components(low_mask, min_area=20)
    out = np.zeros_like(low_mask, dtype=bool)
    for c in comps:
        comp_mask = np.zeros_like(low_mask, dtype=bool)
        comp_mask[c.pixels_y, c.pixels_x] = True
        if np.logical_and(comp_mask, high_mask).any():
            out[c.pixels_y, c.pixels_x] = True
    return out


def locate_point(heat: np.ndarray, method: str, peak_ratio: float) -> tuple[float, float]:
    h, w = heat.shape
    py, px = np.unravel_index(int(np.argmax(heat)), (h, w))
    if method == "argmax":
        return float(px), float(py)
    local_radius = 18
    y1 = max(0, py - local_radius)
    y2 = min(h, py + local_radius + 1)
    x1 = max(0, px - local_radius)
    x2 = min(w, px + local_radius + 1)
    crop = heat[y1:y2, x1:x2]
    keep = crop >= float(crop.max()) * peak_ratio
    weights = np.where(keep, crop, 0.0)
    if weights.sum() <= 0:
        return float(px), float(py)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    cx = float((xx * weights).sum() / weights.sum())
    cy = float((yy * weights).sum() / weights.sum())
    return cx, cy


def close_horizontal(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return mask
    return ndimage.binary_closing(mask, structure=np.ones((1, width), dtype=bool))


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = args.base_channels
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict) and args.base_channels == 16:
        base_channels = int(ckpt["args"].get("base_channels", base_channels))
    model = UNetMultitask(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = TearMeniscusDataset(args.processed_dir, args.splits_dir / f"{args.split}.txt", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pred_point_dir = args.out_dir / "pred_point"
    pred_meniscus_dir = args.out_dir / "pred_meniscus"
    pred_all_dir = args.out_dir / "pred_all"
    overlay_dir = args.out_dir / "overlay_final"
    for d in [pred_point_dir, pred_meniscus_dir, pred_all_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        point_heat = torch.sigmoid(outputs["point_logits"]).cpu().numpy()[:, 0]
        men_prob = torch.sigmoid(outputs["meniscus_logits"]).cpu().numpy()[:, 0]
        for i, rel_id in enumerate(batch["id"]):
            h, w = point_heat[i].shape
            px, py = locate_point(point_heat[i], args.point_method, args.point_peak_ratio)
            point_mask = np.zeros((h, w), dtype=bool)
            yy, xx = np.mgrid[0:h, 0:w]
            point_mask[(xx - px) ** 2 + (yy - py) ** 2 <= args.point_radius**2] = True
            if args.low_threshold > 0:
                raw_meniscus = hysteresis_mask(men_prob[i], high=args.threshold, low=args.low_threshold)
            else:
                raw_meniscus = men_prob[i] > args.threshold
            raw_meniscus = close_horizontal(raw_meniscus, args.horizontal_closing)
            meniscus_mask = postprocess_meniscus(raw_meniscus)
            all_mask = np.logical_or(point_mask, meniscus_mask)

            save_mask(pred_point_dir / f"{rel_id}.png", point_mask)
            save_mask(pred_meniscus_dir / f"{rel_id}.png", meniscus_mask)
            save_mask(pred_all_dir / f"{rel_id}.png", all_mask)
            image = Image.open(args.processed_dir / "images" / f"{rel_id}.png").convert("RGB")
            tmh = tmh_pixel(meniscus_mask, px, window=5)
            overlay = make_overlay(image, point_mask, meniscus_mask)
            overlay = draw_tmh_line(overlay, px, tmh["y_upper"], tmh["y_lower"])
            overlay.save(overlay_dir / f"{rel_id.replace('/', '_')}_overlay.png")
    print(f"predictions: {args.out_dir}")


if __name__ == "__main__":
    main()
