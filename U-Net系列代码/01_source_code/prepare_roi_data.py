from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare lower-eyelid ROI data for meniscus refinement.")
    parser.add_argument("--processed_dir", type=Path, default=root / "processed")
    parser.add_argument("--out_dir", type=Path, default=root / "processed_roi")
    parser.add_argument("--roi", nargs=4, type=float, default=[0.03, 0.45, 0.97, 0.95], metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--roi_size", nargs=2, type=int, default=[640, 256], metavar=("WIDTH", "HEIGHT"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = sorted((args.processed_dir / "images").glob("*/*.png"))
    out_img = args.out_dir / "images"
    out_mask = args.out_dir / "gt_meniscus"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    roi_w, roi_h = args.roi_size
    rows = []
    for img_path in image_paths:
        rel = img_path.relative_to(args.processed_dir / "images")
        mask_path = args.processed_dir / "gt_meniscus" / rel
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        w, h = image.size
        x1 = int(round(args.roi[0] * w))
        y1 = int(round(args.roi[1] * h))
        x2 = int(round(args.roi[2] * w))
        y2 = int(round(args.roi[3] * h))
        crop_box = (x1, y1, x2, y2)
        roi_image = image.crop(crop_box).resize((roi_w, roi_h), Image.Resampling.BILINEAR)
        roi_mask = mask.crop(crop_box).resize((roi_w, roi_h), Image.Resampling.NEAREST)
        (out_img / rel.parent).mkdir(parents=True, exist_ok=True)
        (out_mask / rel.parent).mkdir(parents=True, exist_ok=True)
        roi_image.save(out_img / rel)
        roi_mask.save(out_mask / rel)
        rows.append(f"{rel.as_posix()},{x1},{y1},{x2},{y2},{roi_w},{roi_h}")
    with (args.out_dir / "roi_meta.csv").open("w", encoding="utf-8") as f:
        f.write("rel_path,x1,y1,x2,y2,roi_w,roi_h\n")
        f.write("\n".join(rows))
        f.write("\n")
    print(f"prepared ROI samples: {len(rows)}")
    print(f"roi data: {args.out_dir}")


if __name__ == "__main__":
    main()

