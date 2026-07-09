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
    parser = argparse.ArgumentParser(description="Select ROI meniscus threshold on validation split.")
    parser.add_argument("--checkpoint", type=Path, default=root / "results" / "stage2_roi_b24" / "best_model.pth")
    parser.add_argument("--stage1_pred_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=root / "results" / "stage2_roi_threshold")
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--low_thresholds", nargs="*", type=float, default=[])
    parser.add_argument("--score_tmh_weight", type=float, default=0.05)
    return parser


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def tag_for(high: float, low: float) -> str:
    high_tag = f"h{high:.2f}".replace(".", "p")
    if low > 0:
        return f"{high_tag}_l{low:.2f}".replace(".", "p")
    return high_tag


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    lows = [0.0] + [v for v in args.low_thresholds if v > 0]
    rows = []

    for threshold in args.thresholds:
        for low_threshold in lows:
            if low_threshold >= threshold:
                continue
            tag = tag_for(threshold, low_threshold)
            out_dir = args.work_dir / tag
            if out_dir.exists():
                shutil.rmtree(out_dir)
            cmd = [
                sys.executable,
                str(script_dir / "predict_roi.py"),
                "--split",
                args.split,
                "--stage1_pred_dir",
                str(args.stage1_pred_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--out_dir",
                str(out_dir),
                "--threshold",
                f"{threshold:.4f}",
            ]
            if low_threshold > 0:
                cmd.extend(["--low_threshold", f"{low_threshold:.4f}"])
            subprocess.run(cmd, check=True)
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
            metric_rows = read_rows(out_dir / f"original_label_metrics_{args.split}.csv")
            tmh_rows = read_rows(out_dir / f"original_label_tmh_metrics_{args.split}.csv")
            meniscus = [r for r in metric_rows if r["target"] == "meniscus"]
            point = [r for r in metric_rows if r["target"] == "point"]
            dice = finite_mean([float(r["dice"]) for r in meniscus])
            iou = finite_mean([float(r["iou"]) for r in meniscus])
            precision = finite_mean([float(r["precision"]) for r in meniscus])
            recall = finite_mean([float(r["recall"]) for r in meniscus])
            point_dice = finite_mean([float(r["dice"]) for r in point])
            tmh_mae = finite_mean([float(r["tmh_abs_error_pixel"]) for r in tmh_rows])
            point_error = finite_mean([float(r["point_error_px"]) for r in tmh_rows])
            score = dice - args.score_tmh_weight * tmh_mae
            rows.append(
                {
                    "threshold": threshold,
                    "low_threshold": low_threshold,
                    "meniscus_dice": dice,
                    "meniscus_iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "point_dice": point_dice,
                    "point_error_px": point_error,
                    "tmh_mae": tmh_mae,
                    "score": score,
                    "out_dir": str(out_dir),
                }
            )

    best = max(rows, key=lambda r: r["score"])
    fieldnames = list(rows[0].keys())
    with (args.work_dir / "threshold_sweep.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.work_dir / "best_threshold.txt").open("w", encoding="utf-8") as f:
        f.write(f"threshold={best['threshold']:.4f}\n")
        f.write(f"low_threshold={best['low_threshold']:.4f}\n")
        f.write(f"out_dir={best['out_dir']}\n")
    with (args.work_dir / "threshold_selection_summary.md").open("w", encoding="utf-8") as f:
        f.write("# ROI Threshold Selection\n\n")
        f.write(f"Split: `{args.split}`\n\n")
        f.write(f"Score: `meniscus_dice - {args.score_tmh_weight} * tmh_mae`\n\n")
        f.write("| high | low | Dice | IoU | Precision | Recall | Point Err | TMH MAE | Score |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda r: r["score"], reverse=True):
            f.write(
                f"| {row['threshold']:.2f} | {row['low_threshold']:.2f} | {row['meniscus_dice']:.4f} | "
                f"{row['meniscus_iou']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['point_error_px']:.2f} | {row['tmh_mae']:.2f} | {row['score']:.4f} |\n"
            )
        f.write(f"\nBest high threshold: `{best['threshold']:.2f}`\n")
        f.write(f"Best low threshold: `{best['low_threshold']:.2f}`\n")
        f.write(f"Best output: `{best['out_dir']}`\n")
    print(f"best threshold: {best['threshold']:.4f}, low: {best['low_threshold']:.4f}")
    print(f"summary: {args.work_dir / 'threshold_selection_summary.md'}")


if __name__ == "__main__":
    main()
