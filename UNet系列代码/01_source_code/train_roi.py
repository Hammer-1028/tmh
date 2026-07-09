from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import UNetBinary
from roi_dataset import MeniscusROIDataset
from train_unet import dice_loss_from_logits, tversky_loss_from_logits


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train lower-ROI meniscus refinement U-Net.")
    parser.add_argument("--roi_dir", type=Path, default=root / "processed_roi")
    parser.add_argument("--splits_dir", type=Path, default=root / "splits")
    parser.add_argument("--out_dir", type=Path, default=root / "results" / "roi_unet")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--tversky_weight", type=float, default=0.4)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, tversky_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(1.0, 30.0).detach()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, target)
    tversky = tversky_loss_from_logits(logits, target, alpha=0.4, beta=0.6)
    loss = bce + dice + tversky_weight * tversky
    return loss, {"bce": float(bce.detach()), "dice_loss": float(dice.detach()), "tversky_loss": float(tversky.detach())}


@torch.no_grad()
def dice_metric(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = torch.sigmoid(logits) > threshold
    gt = target > 0.5
    tp = (pred & gt).sum(dim=(1, 2, 3)).float()
    fp = (pred & ~gt).sum(dim=(1, 2, 3)).float()
    fn = (~pred & gt).sum(dim=(1, 2, 3)).float()
    return float(((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean())


def run_epoch(model, loader, optimizer, device, train: bool, tversky_weight: float) -> dict[str, float]:
    model.train(train)
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss, parts = loss_fn(logits, target, tversky_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        row = {"loss": float(loss.detach()), "dice": dice_metric(logits, target), **parts}
        bs = image.shape[0]
        for k, v in row.items():
            if math.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_ds = MeniscusROIDataset(args.roi_dir, args.splits_dir / "train.txt", augment=not args.no_augment, seed=args.seed)
    val_ds = MeniscusROIDataset(args.roi_dir, args.splits_dir / "val.txt", augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = UNetBinary(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, True, args.tversky_weight)
        val_metrics = run_epoch(model, val_loader, optimizer, device, False, args.tversky_weight)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        score = row.get("val_dice", 0.0)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "best_score": best_score}, args.out_dir / "best_model.pth")
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row.get('train_loss', math.nan):.4f} "
            f"val_loss={row.get('val_loss', math.nan):.4f} "
            f"val_dice={row.get('val_dice', math.nan):.4f}"
        )
    with (args.out_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"best val dice: {best_score:.4f}")
    print(f"best model: {args.out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

