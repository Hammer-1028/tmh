from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from models import UNetBinary
from roi_dataset import MeniscusROIDataset
from utils.label_split import connected_components
from utils.tmh_measure import tmh_pixel
from utils.visualization import draw_tmh_line, make_overlay, save_mask


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict ROI refined meniscus and paste back to full image.")
    parser.add_argument("--roi_dir", type=Path, default=root / "processed_roi")
    parser.add_argument("--processed_dir", type=Path, default=root / "processed")
    parser.add_argument("--splits_dir", type=Path, default=root / "splits")
    parser.add_argument("--stage1_pred_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=root / "results" / "roi_unet" / "best_model.pth")
    parser.add_argument("--out_dir", type=Path, default=root / "results" / "roi_unet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low_threshold", type=float, default=0.0, help="If >0, use low/high hysteresis for ROI meniscus.")
    return parser


def load_roi_meta(path: Path) -> dict[str, tuple[int, int, int, int, int, int]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["rel_path"]] = tuple(int(row[k]) for k in ["x1", "y1", "x2", "y2", "roi_w", "roi_h"])
    return rows


def postprocess(mask: np.ndarray) -> np.ndarray:
    comps = connected_components(mask, min_area=20)
    if not comps:
        return mask.astype(bool)
    keep = [c for c in comps if c.width >= max(10, c.height * 2)]
    if not keep:
        keep = comps
    best = max(keep, key=lambda c: c.area + 5 * c.width)
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


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = args.base_channels
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict) and args.base_channels == 16:
        base_channels = int(ckpt["args"].get("base_channels", base_channels))
    model = UNetBinary(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = MeniscusROIDataset(args.roi_dir, args.splits_dir / f"{args.split}.txt", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    meta = load_roi_meta(args.roi_dir / "roi_meta.csv")
    for sub in ["pred_point", "pred_meniscus", "pred_all", "overlay_final"]:
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        logits = model(batch["image"].to(device))
        prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
        for i, rel_id in enumerate(batch["id"]):
            rel_png = f"{rel_id}.png"
            point_src = args.stage1_pred_dir / "pred_point" / rel_png
            point_dst = args.out_dir / "pred_point" / rel_png
            point_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(point_src, point_dst)
            point_mask = np.asarray(Image.open(point_dst).convert("L")) > 127
            full_image = Image.open(args.processed_dir / "images" / rel_png).convert("RGB")
            full_w, full_h = full_image.size
            x1, y1, x2, y2, _, _ = meta[rel_png]
            if args.low_threshold > 0:
                roi_raw = hysteresis_mask(prob[i], high=args.threshold, low=args.low_threshold)
            else:
                roi_raw = prob[i] > args.threshold
            roi_mask = postprocess(roi_raw)
            roi_full_size = Image.fromarray((roi_mask * 255).astype(np.uint8)).resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
            full_mask = np.zeros((full_h, full_w), dtype=bool)
            full_mask[y1:y2, x1:x2] = np.asarray(roi_full_size) > 127
            all_mask = np.logical_or(full_mask, point_mask)
            save_mask(args.out_dir / "pred_meniscus" / rel_png, full_mask)
            save_mask(args.out_dir / "pred_all" / rel_png, all_mask)
            ys, xs = np.where(point_mask)
            px = float(xs.mean()) if len(xs) else np.nan
            tmh = tmh_pixel(full_mask, px, window=5)
            overlay = draw_tmh_line(make_overlay(full_image, point_mask, full_mask), px, tmh["y_upper"], tmh["y_lower"])
            overlay.save(args.out_dir / "overlay_final" / f"{rel_id.replace('/', '_')}_overlay.png")
    print(f"roi predictions: {args.out_dir}")


if __name__ == "__main__":
    main()
