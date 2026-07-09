from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import TearMeniscusDataset
from models import UNetMultitask


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train multitask U-Net for TMH segmentation.")
    parser.add_argument("--processed_dir", type=Path, default=root / "processed")
    parser.add_argument("--splits_dir", type=Path, default=root / "splits")
    parser.add_argument("--out_dir", type=Path, default=root / "results" / "unet_test")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_samples", type=int, default=0, help="Use first N train samples for a quick sanity test.")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--point_score_weight", type=float, default=0.002)
    parser.add_argument("--point_loss_weight", type=float, default=1.0)
    parser.add_argument("--tversky_weight", type=float, default=0.4)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (prob * target).sum(dims)
    fp = (prob * (1.0 - target)).sum(dims)
    fn = ((1.0 - prob) * target).sum(dims)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()


def loss_fn(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    tversky_weight: float = 0.4,
    point_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    point_pred = torch.sigmoid(outputs["point_logits"])
    point_weight = 1.0 + 20.0 * batch["point_heatmap"]
    point_heat_loss = (point_weight * (point_pred - batch["point_heatmap"]) ** 2).mean()
    point_pos = batch["point_mask"].sum().clamp_min(1.0)
    point_neg = (1.0 - batch["point_mask"]).sum().clamp_min(1.0)
    point_pos_weight = (point_neg / point_pos).clamp(1.0, 50.0).detach()
    point_bce = F.binary_cross_entropy_with_logits(outputs["point_logits"], batch["point_mask"], pos_weight=point_pos_weight)
    point_dice = dice_loss_from_logits(outputs["point_logits"], batch["point_mask"])
    point_loss = 0.25 * point_heat_loss + 0.5 * point_bce + 0.5 * point_dice

    pos = batch["meniscus"].sum().clamp_min(1.0)
    neg = (1.0 - batch["meniscus"]).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(1.0, 30.0).detach()
    bce = F.binary_cross_entropy_with_logits(outputs["meniscus_logits"], batch["meniscus"], pos_weight=pos_weight)
    dice = dice_loss_from_logits(outputs["meniscus_logits"], batch["meniscus"])
    tversky = tversky_loss_from_logits(outputs["meniscus_logits"], batch["meniscus"], alpha=0.4, beta=0.6)
    meniscus_loss = bce + dice + tversky_weight * tversky
    loss = point_loss_weight * point_loss + meniscus_loss
    return loss, {
        "point_loss": float(point_loss.detach()),
        "point_heat_loss": float(point_heat_loss.detach()),
        "point_bce": float(point_bce.detach()),
        "point_dice_loss": float(point_dice.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "tversky_loss": float(tversky.detach()),
        "pos_weight": float(pos_weight.detach()),
    }


@torch.no_grad()
def batch_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    meniscus_prob = torch.sigmoid(outputs["meniscus_logits"])
    pred = meniscus_prob > 0.5
    gt = batch["meniscus"] > 0.5
    tp = (pred & gt).sum(dim=(1, 2, 3)).float()
    fp = (pred & ~gt).sum(dim=(1, 2, 3)).float()
    fn = (~pred & gt).sum(dim=(1, 2, 3)).float()
    dice = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()

    heat = torch.sigmoid(outputs["point_logits"]).detach().cpu().numpy()
    point_mask = batch["point_mask"].detach().cpu().numpy()
    errors = []
    for hp, pm in zip(heat[:, 0], point_mask[:, 0]):
        py, px = np.unravel_index(int(np.argmax(hp)), hp.shape)
        ys, xs = np.where(pm > 0.5)
        if len(xs):
            errors.append(float(((px - xs.mean()) ** 2 + (py - ys.mean()) ** 2) ** 0.5))
    point_error = float(np.mean(errors)) if errors else math.nan
    return {"meniscus_dice": dice, "point_error_px": point_error}


def run_epoch(model, loader, optimizer, device, train: bool, tversky_weight: float, point_loss_weight: float) -> dict[str, float]:
    model.train(train)
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        tensor_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "id"}
        with torch.set_grad_enabled(train):
            outputs = model(tensor_batch["image"])
            loss, parts = loss_fn(outputs, tensor_batch, tversky_weight=tversky_weight, point_loss_weight=point_loss_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metrics = batch_metrics(outputs, tensor_batch)
        row = {"loss": float(loss.detach()), **parts, **metrics}
        bs = tensor_batch["image"].shape[0]
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

    train_ds = TearMeniscusDataset(args.processed_dir, args.splits_dir / "train.txt", augment=not args.no_augment, seed=args.seed)
    val_ds = TearMeniscusDataset(args.processed_dir, args.splits_dir / "val.txt", augment=False, seed=args.seed)
    if args.overfit_samples > 0:
        n = min(args.overfit_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = Subset(TearMeniscusDataset(args.processed_dir, args.splits_dir / "train.txt", augment=False, seed=args.seed), list(range(n)))
        print(f"overfit mode: {n} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = UNetMultitask(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_path = args.out_dir / "history.csv"
    best_dice = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            tversky_weight=args.tversky_weight,
            point_loss_weight=args.point_loss_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            tversky_weight=args.tversky_weight,
            point_loss_weight=args.point_loss_weight,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        val_dice = row.get("val_meniscus_dice", 0.0)
        val_point_error = row.get("val_point_error_px", 999.0)
        val_score = val_dice - args.point_score_weight * val_point_error
        row["val_score"] = val_score
        if val_score > best_dice:
            best_dice = val_score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "best_dice": best_dice}, args.out_dir / "best_model.pth")
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row.get('train_loss', math.nan):.4f} "
            f"val_loss={row.get('val_loss', math.nan):.4f} "
            f"val_dice={row.get('val_meniscus_dice', math.nan):.4f} "
            f"val_point_err={row.get('val_point_error_px', math.nan):.2f} "
            f"val_score={row.get('val_score', math.nan):.4f}"
        )

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with history_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"best val score: {best_dice:.4f}")
    print(f"history: {history_path}")
    print(f"best model: {args.out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
