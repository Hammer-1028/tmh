from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from compare_with_original_labels import load_original_split_label
from dataset import TearMeniscusDataset
from models import UNetMultitask
from predict_unet import hysteresis_mask, locate_point, postprocess_meniscus
from utils.metrics import binary_metrics, mask_center
from utils.tmh_measure import tmh_pixel


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    project = here.parent
    parser = argparse.ArgumentParser(description="Fast validation sweep for U-Net post-processing.")
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--splits_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=project / "正式实验" / "03_后处理优化_进行中")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=16)
    return parser


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def make_point_mask(shape: tuple[int, int], px: float, py: float, radius: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - px) ** 2 + (yy - py) ** 2 <= radius**2)


def close_horizontal(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return mask
    structure = np.ones((1, width), dtype=bool)
    return ndimage.binary_closing(mask, structure=structure)


def parse_checkpoint_base_channels(ckpt: dict, fallback: int) -> int:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        return int(ckpt["args"].get("base_channels", fallback))
    return fallback


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = parse_checkpoint_base_channels(ckpt, args.base_channels)
    model = UNetMultitask(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = TearMeniscusDataset(args.processed_dir, args.splits_dir / f"{args.split}.txt", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    point_configs = []
    for method in ["argmax", "weighted"]:
        ratios = [0.65] if method == "argmax" else [0.45, 0.55, 0.65, 0.75]
        for ratio in ratios:
            for radius in [3, 4, 5, 6, 7, 8, 10]:
                point_configs.append((method, ratio, radius))

    meniscus_configs = []
    for high in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        meniscus_configs.append((high, 0.0, 0))
        for closing in [3, 5, 7, 9]:
            meniscus_configs.append((high, 0.0, closing))
    for high, low in [(0.90, 0.70), (0.93, 0.75), (0.95, 0.80), (0.97, 0.85)]:
        for closing in [0, 3, 5]:
            meniscus_configs.append((high, low, closing))

    rows: list[dict] = []
    accum: dict[tuple, dict[str, list[float]]] = {}
    for pc in point_configs:
        for mc in meniscus_configs:
            accum[(pc, mc)] = {
                "point_dice": [],
                "point_error": [],
                "meniscus_dice": [],
                "meniscus_iou": [],
                "meniscus_precision": [],
                "meniscus_recall": [],
                "tmh_mae": [],
            }

    for batch in loader:
        outputs = model(batch["image"].to(device))
        point_heat = torch.sigmoid(outputs["point_logits"]).cpu().numpy()[:, 0]
        men_prob = torch.sigmoid(outputs["meniscus_logits"]).cpu().numpy()[:, 0]
        for i, rel_id in enumerate(batch["id"]):
            rel_png = f"{rel_id}.png"
            h, w = point_heat[i].shape
            gt_point, gt_meniscus, _ = load_original_split_label(args.data_root, rel_id, (h, w))
            gx, gy = mask_center(gt_point)
            point_cache = {}
            for method, ratio, radius in point_configs:
                px, py = locate_point(point_heat[i], method, ratio)
                point_mask = make_point_mask((h, w), px, py, radius)
                point_metrics = binary_metrics(point_mask, gt_point)
                point_error = math.nan if not all(math.isfinite(v) for v in [px, py, gx, gy]) else float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
                point_cache[(method, ratio, radius)] = (px, point_metrics["dice"], point_error)

            men_cache = {}
            for high, low, closing in meniscus_configs:
                if low > 0:
                    raw = hysteresis_mask(men_prob[i], high=high, low=low)
                else:
                    raw = men_prob[i] > high
                raw = close_horizontal(raw, closing)
                meniscus_mask = postprocess_meniscus(raw)
                m = binary_metrics(meniscus_mask, gt_meniscus)
                men_cache[(high, low, closing)] = (meniscus_mask, m)

            for pc in point_configs:
                px, pdice, perr = point_cache[pc]
                for mc in meniscus_configs:
                    meniscus_mask, m = men_cache[mc]
                    pred_tmh = tmh_pixel(meniscus_mask, px, window=5)
                    gt_tmh = tmh_pixel(gt_meniscus, gx, window=5)
                    tmh_abs = (
                        abs(pred_tmh["tmh_pixel"] - gt_tmh["tmh_pixel"])
                        if math.isfinite(pred_tmh["tmh_pixel"]) and math.isfinite(gt_tmh["tmh_pixel"])
                        else math.nan
                    )
                    a = accum[(pc, mc)]
                    a["point_dice"].append(pdice)
                    a["point_error"].append(perr)
                    a["meniscus_dice"].append(m["dice"])
                    a["meniscus_iou"].append(m["iou"])
                    a["meniscus_precision"].append(m["precision"])
                    a["meniscus_recall"].append(m["recall"])
                    a["tmh_mae"].append(tmh_abs)

    for (pc, mc), a in accum.items():
        method, ratio, radius = pc
        high, low, closing = mc
        meniscus_dice = finite_mean(a["meniscus_dice"])
        tmh_mae = finite_mean(a["tmh_mae"])
        point_error = finite_mean(a["point_error"])
        score = meniscus_dice - 0.05 * tmh_mae - 0.002 * point_error
        rows.append(
            {
                "point_method": method,
                "point_peak_ratio": ratio,
                "point_radius": radius,
                "threshold": high,
                "low_threshold": low,
                "horizontal_closing": closing,
                "point_dice": finite_mean(a["point_dice"]),
                "point_error_px": point_error,
                "meniscus_dice": meniscus_dice,
                "meniscus_iou": finite_mean(a["meniscus_iou"]),
                "precision": finite_mean(a["meniscus_precision"]),
                "recall": finite_mean(a["meniscus_recall"]),
                "tmh_mae": tmh_mae,
                "score": score,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    csv_path = args.out_dir / f"postprocess_sweep_{args.split}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / f"postprocess_sweep_summary_{args.split}.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Postprocess Sweep: {args.split}\n\n")
        f.write("Score = meniscus_dice - 0.05 * tmh_mae - 0.002 * point_error_px\n\n")
        f.write("| rank | point | radius | high | low | close | Dice | IoU | P | R | point err | TMH MAE | score |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for idx, r in enumerate(rows[:30], start=1):
            f.write(
                f"| {idx} | {r['point_method']}:{r['point_peak_ratio']:.2f} | {r['point_radius']} | "
                f"{r['threshold']:.2f} | {r['low_threshold']:.2f} | {r['horizontal_closing']} | "
                f"{r['meniscus_dice']:.4f} | {r['meniscus_iou']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['point_error_px']:.2f} | {r['tmh_mae']:.2f} | {r['score']:.4f} |\n"
            )
    print(f"best: {rows[0]}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
