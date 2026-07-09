from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from utils.label_split import split_point_meniscus
from utils.metrics import binary_metrics, mask_center
from utils.tmh_measure import tmh_pixel


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    data_root = root.parent / "data"
    parser = argparse.ArgumentParser(description="Compare predictions directly with original data/*/Label files.")
    parser.add_argument("--data_root", type=Path, default=data_root)
    parser.add_argument("--splits_dir", type=Path, default=root / "splits")
    parser.add_argument("--pred_dir", type=Path, default=root / "results" / "unet_pilot_60")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_prefix", default="original_label")
    return parser


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def load_original_split_label(data_root: Path, rel_id: str, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset, stem = rel_id.split("/", 1)
    label_dir = data_root / subset / "Label"
    candidates = [label_dir / f"{stem}.PNG", label_dir / f"{stem}.png", label_dir / f"{stem}.jpg", label_dir / f"{stem}.jpeg"]
    label_path = next((p for p in candidates if p.exists()), None)
    if label_path is None:
        matches = list(label_dir.glob(stem + ".*"))
        if matches:
            label_path = matches[0]
    if label_path is None:
        raise FileNotFoundError(f"Missing original label for {rel_id}")
    h, w = size_hw
    label = Image.open(label_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    label_mask = np.asarray(label) > 127
    point, meniscus, _ = split_point_meniscus(label_mask)
    return point, meniscus, np.logical_or(point, meniscus)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def main() -> None:
    args = build_parser().parse_args()
    ids = [line.strip() for line in (args.splits_dir / f"{args.split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_rows: list[dict] = []
    tmh_rows: list[dict] = []

    for rel_id in ids:
        pred_point = load_mask(args.pred_dir / "pred_point" / f"{rel_id}.png")
        pred_meniscus = load_mask(args.pred_dir / "pred_meniscus" / f"{rel_id}.png")
        pred_all = load_mask(args.pred_dir / "pred_all" / f"{rel_id}.png")
        gt_point, gt_meniscus, gt_all = load_original_split_label(args.data_root, rel_id, pred_point.shape)

        pairs = [
            ("point", pred_point, gt_point),
            ("meniscus", pred_meniscus, gt_meniscus),
            ("all", pred_all, gt_all),
        ]
        for target, pred, gt in pairs:
            subset = rel_id.split("/", 1)[0]
            metric_rows.append({"id": rel_id, "subset": subset, "target": target, **binary_metrics(pred, gt)})

        px, py = mask_center(pred_point)
        gx, gy = mask_center(gt_point)
        pred_tmh = tmh_pixel(pred_meniscus, px, window=5)
        gt_tmh = tmh_pixel(gt_meniscus, gx, window=5)
        point_error = math.nan if not all(math.isfinite(v) for v in [px, py, gx, gy]) else float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
        tmh_abs_error = (
            abs(pred_tmh["tmh_pixel"] - gt_tmh["tmh_pixel"])
            if math.isfinite(pred_tmh["tmh_pixel"]) and math.isfinite(gt_tmh["tmh_pixel"])
            else math.nan
        )
        tmh_rows.append(
            {
                "id": rel_id,
                "subset": rel_id.split("/", 1)[0],
                "pred_point_x": px,
                "pred_point_y": py,
                "gt_point_x": gx,
                "gt_point_y": gy,
                "point_error_px": point_error,
                "pred_tmh_pixel": pred_tmh["tmh_pixel"],
                "gt_tmh_pixel": gt_tmh["tmh_pixel"],
                "tmh_abs_error_pixel": tmh_abs_error,
            }
        )

    metrics_path = args.pred_dir / f"{args.out_prefix}_metrics_{args.split}.csv"
    tmh_path = args.pred_dir / f"{args.out_prefix}_tmh_metrics_{args.split}.csv"
    summary_path = args.pred_dir / f"{args.out_prefix}_summary_{args.split}.md"
    write_csv(metrics_path, metric_rows)
    write_csv(tmh_path, tmh_rows)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Original Label Comparison: {args.split}\n\n")
        f.write("GT labels are read directly from `data/<subset>/Label`, resized to prediction size, then split into point and meniscus masks.\n\n")
        for target in ["point", "meniscus", "all"]:
            rows = [r for r in metric_rows if r["target"] == target]
            f.write(
                f"- {target}: Dice={finite_mean([r['dice'] for r in rows]):.4f}, "
                f"IoU={finite_mean([r['iou'] for r in rows]):.4f}, "
                f"Precision={finite_mean([r['precision'] for r in rows]):.4f}, "
                f"Recall={finite_mean([r['recall'] for r in rows]):.4f}, "
                f"F1={finite_mean([r['f1'] for r in rows]):.4f}\n"
            )
        f.write(f"- Point error mean: {finite_mean([r['point_error_px'] for r in tmh_rows]):.2f} px\n")
        valid_tmh = [r["tmh_abs_error_pixel"] for r in tmh_rows if math.isfinite(r["tmh_abs_error_pixel"])]
        f.write(f"- TMH MAE mean: {finite_mean(valid_tmh):.2f} px, valid pairs={len(valid_tmh)}/{len(tmh_rows)}\n")

        f.write("\n## By Subset\n\n")
        for subset in sorted({r["subset"] for r in metric_rows}):
            meniscus_rows = [r for r in metric_rows if r["subset"] == subset and r["target"] == "meniscus"]
            tmh_subset = [r for r in tmh_rows if r["subset"] == subset]
            valid_subset_tmh = [r["tmh_abs_error_pixel"] for r in tmh_subset if math.isfinite(r["tmh_abs_error_pixel"])]
            f.write(
                f"- {subset}: meniscus Dice={finite_mean([r['dice'] for r in meniscus_rows]):.4f}, "
                f"IoU={finite_mean([r['iou'] for r in meniscus_rows]):.4f}, "
                f"TMH MAE={finite_mean(valid_subset_tmh):.2f} px, n={len(tmh_subset)}\n"
            )

    print(f"metrics: {metrics_path}")
    print(f"tmh metrics: {tmh_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

