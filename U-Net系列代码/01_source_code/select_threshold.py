from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Select meniscus probability threshold on validation split.")
    parser.add_argument("--checkpoint", type=Path, default=root / "results" / "unet_pilot_60" / "best_model.pth")
    parser.add_argument("--work_dir", type=Path, default=root / "results" / "threshold_select")
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--score_tmh_weight", type=float, default=0.05)
    return parser


def read_metric_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_tmh_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    rows = []

    for threshold in args.thresholds:
        tag = f"t{threshold:.2f}".replace(".", "p")
        out_dir = args.work_dir / tag
        if out_dir.exists():
            shutil.rmtree(out_dir)
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "predict_unet.py"),
                "--split",
                args.split,
                "--checkpoint",
                str(args.checkpoint),
                "--out_dir",
                str(out_dir),
                "--threshold",
                f"{threshold:.4f}",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "compare_with_original_labels.py"),
                "--split",
                args.split,
                "--pred_dir",
                str(out_dir),
            ],
            check=True,
        )
        metric_rows = read_metric_rows(out_dir / f"original_label_metrics_{args.split}.csv")
        tmh_rows = read_tmh_rows(out_dir / f"original_label_tmh_metrics_{args.split}.csv")
        meniscus = [r for r in metric_rows if r["target"] == "meniscus"]
        dice = finite_mean([float(r["dice"]) for r in meniscus])
        iou = finite_mean([float(r["iou"]) for r in meniscus])
        precision = finite_mean([float(r["precision"]) for r in meniscus])
        recall = finite_mean([float(r["recall"]) for r in meniscus])
        tmh_mae = finite_mean([float(r["tmh_abs_error_pixel"]) for r in tmh_rows])
        score = dice - args.score_tmh_weight * tmh_mae
        rows.append(
            {
                "threshold": threshold,
                "meniscus_dice": dice,
                "meniscus_iou": iou,
                "precision": precision,
                "recall": recall,
                "tmh_mae": tmh_mae,
                "score": score,
            }
        )

    best = max(rows, key=lambda r: r["score"])
    with (args.work_dir / "threshold_sweep.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.work_dir / "best_threshold.txt").open("w", encoding="utf-8") as f:
        f.write(f"{best['threshold']:.4f}\n")
    with (args.work_dir / "threshold_selection_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Threshold Selection\n\n")
        f.write(f"Split: `{args.split}`\n\n")
        f.write(f"Score: `meniscus_dice - {args.score_tmh_weight} * tmh_mae`\n\n")
        f.write("| threshold | Dice | IoU | Precision | Recall | TMH MAE | Score |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['threshold']:.2f} | {row['meniscus_dice']:.4f} | {row['meniscus_iou']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['tmh_mae']:.2f} | {row['score']:.4f} |\n"
            )
        f.write(f"\nBest threshold: `{best['threshold']:.2f}`\n")
    print(f"best threshold: {best['threshold']:.4f}")
    print(f"summary: {args.work_dir / 'threshold_selection_summary.md'}")


if __name__ == "__main__":
    main()

