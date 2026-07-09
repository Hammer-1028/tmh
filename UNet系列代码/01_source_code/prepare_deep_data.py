from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from utils.label_split import gaussian_heatmap, split_point_meniscus
from utils.tmh_measure import tmh_pixel
from utils.visualization import make_overlay, save_mask


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare colour TMH data for multitask U-Net.")
    parser.add_argument("--data_root", type=Path, default=root / "data")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "processed")
    parser.add_argument("--splits_dir", type=Path, default=Path(__file__).resolve().parent / "splits")
    parser.add_argument("--subsets", nargs="+", default=["Colour1", "Colour2"])
    parser.add_argument("--image_size", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--limit_per_subset", type=int, default=0, help="0 means all images.")
    parser.add_argument("--qa_count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_sigma", type=float, default=8.0)
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.5, 0.2, 0.3])
    return parser


def sample_items(data_root: Path, subsets: list[str], limit_per_subset: int) -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = []
    for subset in subsets:
        original_dir = data_root / subset / "Original"
        label_dir = data_root / subset / "Label"
        image_paths = sorted(original_dir.glob("*.*"))
        if limit_per_subset > 0:
            image_paths = image_paths[:limit_per_subset]
        for img_path in image_paths:
            label_path = label_dir / img_path.name
            if not label_path.exists():
                stem = img_path.stem
                candidates = list(label_dir.glob(stem + ".*"))
                label_path = candidates[0] if candidates else label_path
            if label_path.exists():
                items.append((subset, img_path, label_path))
            else:
                print(f"[WARN] missing label for {subset}/{img_path.name}")
    return items


def write_splits(items: list[tuple[str, str]], splits_dir: Path, ratios: list[float], seed: int) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0] / sum(ratios)))
    n_val = int(round(n * ratios[1] / sum(ratios)))
    split_map = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    for name, rows in split_map.items():
        with (splits_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for subset, stem in rows:
                f.write(f"{subset}/{stem}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    width, height = args.image_size
    rng = random.Random(args.seed)

    dirs = {
        "images": args.out_dir / "images",
        "gt_point_mask": args.out_dir / "gt_point_mask",
        "gt_point_heatmap": args.out_dir / "gt_point_heatmap",
        "gt_meniscus": args.out_dir / "gt_meniscus",
        "gt_all": args.out_dir / "gt_all",
        "qa_overlays": args.out_dir / "qa_overlays",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    items = sample_items(args.data_root, args.subsets, args.limit_per_subset)
    qa_keys = set(rng.sample(range(len(items)), min(args.qa_count, len(items)))) if items else set()
    audit_rows = []
    split_ids: list[tuple[str, str]] = []

    for idx, (subset, img_path, label_path) in enumerate(items):
        stem = img_path.stem
        image = Image.open(img_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        label_gray = Image.open(label_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        label_mask = np.asarray(label_gray) > 127
        point_mask, meniscus_mask, info = split_point_meniscus(label_mask)
        all_mask = np.logical_or(point_mask, meniscus_mask)
        heat = gaussian_heatmap((height, width), info["point_cx"], info["point_cy"], sigma=args.point_sigma)

        prefix = Path(subset) / stem
        (dirs["images"] / subset).mkdir(parents=True, exist_ok=True)
        image.save(dirs["images"] / subset / f"{stem}.png")
        save_mask(dirs["gt_point_mask"] / subset / f"{stem}.png", point_mask)
        save_mask(dirs["gt_meniscus"] / subset / f"{stem}.png", meniscus_mask)
        save_mask(dirs["gt_all"] / subset / f"{stem}.png", all_mask)
        (dirs["gt_point_heatmap"] / subset).mkdir(parents=True, exist_ok=True)
        np.save(dirs["gt_point_heatmap"] / subset / f"{stem}.npy", heat)

        gt_tmh = tmh_pixel(meniscus_mask, info["point_cx"], window=5)
        if idx in qa_keys or info.get("warning"):
            overlay = make_overlay(image, point_mask, meniscus_mask)
            from utils.visualization import draw_tmh_line

            overlay = draw_tmh_line(overlay, info["point_cx"], gt_tmh["y_upper"], gt_tmh["y_lower"])
            (dirs["qa_overlays"] / subset).mkdir(parents=True, exist_ok=True)
            overlay.save(dirs["qa_overlays"] / subset / f"{stem}_gt_overlay.png")

        audit_rows.append(
            {
                "subset": subset,
                "stem": stem,
                "image_path": str(img_path),
                "label_path": str(label_path),
                **info,
                **{f"gt_{k}": v for k, v in gt_tmh.items()},
            }
        )
        split_ids.append((subset, stem))

    with (args.out_dir / "label_split_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = sorted({k for row in audit_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    write_splits(split_ids, args.splits_dir, args.split_ratio, args.seed)
    print(f"prepared {len(split_ids)} samples")
    print(f"processed: {args.out_dir}")
    print(f"splits: {args.splits_dir}")
    print(f"audit: {args.out_dir / 'label_split_audit.csv'}")
    print(f"qa overlays: {args.out_dir / 'qa_overlays'}")


if __name__ == "__main__":
    main()
