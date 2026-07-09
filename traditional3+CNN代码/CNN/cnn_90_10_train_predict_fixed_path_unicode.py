# -*- coding: utf-8 -*-
"""
CNN/FCN 双任务小样本训练 + 预测脚本
任务：
1) point_head：预测中心参考点 heatmap / point mask
2) meniscus_head：预测泪河 meniscus mask
3) 根据 pred_point + pred_meniscus 计算 TMH_pixel

模型说明：
- 不是 U-Net：没有 U-Net 的 encoder-decoder skip connection。
- 使用传统 CNN / FCN 风格：卷积编码器 + 上采样解码器 + 双输出头。

推荐数据结构一：简单版，适合你现在的 100 张小样本
D:/TM_CNN_100/
  original/
    xxx.png / xxx.jpg
  label/
    xxx.png / xxx.jpg

推荐运行：
由于本版本已经内置你的数据路径，直接运行：
python cnn_90_10_train_predict_fixed_path.py

你的默认数据路径：
D:\医学图像处理\大作业\Open DataSet\Open DataSet2\TM-CNN-100

也可以手动覆盖路径：
python cnn_90_10_train_predict_fixed_path.py --data_root "D:/TM_CNN_100"

数据结构二：原数据集版
D:/your_dataset/
  Colour1/Original/
  Colour1/Label/
  Colour2/Original/
  Colour2/Label/
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# =========================
# 1. 工具函数
# =========================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def find_case_insensitive_dir(parent: Path, candidates: List[str]) -> Optional[Path]:
    """在 parent 下按大小写不敏感方式寻找文件夹。"""
    if not parent.exists():
        return None
    child_map = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in candidates:
        if name.lower() in child_map:
            return child_map[name.lower()]
    return None


def normalize_stem(stem: str) -> str:
    """用于配对文件名：允许 label 文件名带 _label/_mask 等后缀。"""
    s = stem.lower()
    for suf in ["_label", "-label", "_mask", "-mask", "_gt", "-gt"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def build_pairs(data_root: Path, use_sets: List[str], layout: str = "auto") -> List[Tuple[Path, Path]]:
    """
    按文件 stem 匹配 original 和 label。
    layout=simple：
        data_root/original + data_root/label
    layout=dataset：
        data_root/Colour1/Original + data_root/Colour1/Label 等
    layout=auto：
        优先 simple，找不到再按 dataset 搜索。
    """
    pairs: List[Tuple[Path, Path]] = []

    def collect_from_dirs(img_dir: Path, lab_dir: Path) -> List[Tuple[Path, Path]]:
        imgs = list_images(img_dir)
        labs = list_images(lab_dir)
        lab_map: Dict[str, Path] = {normalize_stem(p.stem): p for p in labs}
        local_pairs: List[Tuple[Path, Path]] = []
        for img_path in imgs:
            key = normalize_stem(img_path.stem)
            if key in lab_map:
                local_pairs.append((img_path, lab_map[key]))
        return local_pairs

    # 1) 简单结构：data_root/original + data_root/label
    if layout in ["auto", "simple"]:
        img_dir = find_case_insensitive_dir(data_root, ["original", "Original", "images", "Images"])
        lab_dir = find_case_insensitive_dir(data_root, ["label", "Label", "labels", "Labels", "mask", "Mask", "masks", "Masks"])
        if img_dir is not None and lab_dir is not None:
            pairs.extend(collect_from_dirs(img_dir, lab_dir))
        if layout == "simple":
            return pairs
        if len(pairs) > 0:
            return pairs

    # 2) 原数据集结构：data_root/Colour1/Original + data_root/Colour1/Label
    if layout in ["auto", "dataset"]:
        for name in use_sets:
            subset_dir = data_root / name
            img_dir = find_case_insensitive_dir(subset_dir, ["Original", "original", "images", "Images"])
            lab_dir = find_case_insensitive_dir(subset_dir, ["Label", "label", "labels", "Labels", "mask", "Mask", "masks", "Masks"])
            if img_dir is None or lab_dir is None:
                continue
            pairs.extend(collect_from_dirs(img_dir, lab_dir))

    return pairs


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Windows 中文路径兼容读取。
    cv2.imread(str(path)) 在中文路径下可能返回 None，
    所以这里用 np.fromfile + cv2.imdecode。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在，请检查文件名或路径: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"文件为空或无法读取: {path}")

    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"OpenCV 无法解码该图片，可能文件损坏或格式不支持: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """
    Windows 中文路径兼容保存。
    """
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败，无法保存: {path}")
    encoded.tofile(str(path))


def read_rgb(path: Path) -> np.ndarray:
    img_bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_label_binary(path: Path, threshold: int = 20) -> np.ndarray:
    """
    读取 Label 并二值化。
    注意：如果 Label 是红/绿等彩色标注，灰度值可能低于 127，
    所以默认 threshold=20 更稳；如果你的 Label 是纯黑白，可以改成 127。
    """
    lab = imread_unicode(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    binary = (gray > threshold).astype(np.uint8)
    return binary


def connected_components(mask: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    h, w = mask.shape[:2]
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area <= 5:
            continue
        cx, cy = centroids[i]
        comp_mask = (labels == i).astype(np.uint8)
        aspect = bw / max(bh, 1)
        comps.append({
            "idx": i,
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area": int(area), "cx": float(cx), "cy": float(cy),
            "aspect": float(aspect), "mask": comp_mask,
            "norm_cy": float(cy / max(h - 1, 1)),
            "center_dist": float(abs(cx - w / 2) / max(w, 1)),
        })
    return comps


def split_point_and_meniscus(label_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从原始 Label 二值图中自动拆：
    - point_mask：上方/中部、面积较小、接近中心的连通域
    - meniscus_mask：下方、横向延展明显的连通域
    
    如果某张 Label 很特殊，拆分可能不完美；先看输出的 gt_qa_contact_sheet.png 做质检。
    """
    h, w = label_mask.shape[:2]
    comps = connected_components(label_mask)

    point_mask = np.zeros((h, w), dtype=np.uint8)
    meniscus_mask = np.zeros((h, w), dtype=np.uint8)

    if len(comps) == 0:
        return point_mask, meniscus_mask

    # meniscus：倾向选择下方、宽扁、面积较大的连通域
    meniscus_candidates = []
    for c in comps:
        lower_bonus = 1.0 + 2.0 * c["norm_cy"]
        wide_bonus = max(c["aspect"], 0.2)
        area_bonus = math.sqrt(max(c["area"], 1))
        # 如果在上半部，明显降低作为泪河的概率
        penalty = 0.35 if c["cy"] < 0.45 * h else 1.0
        score = area_bonus * lower_bonus * wide_bonus * penalty
        meniscus_candidates.append((score, c))

    meniscus_comp = max(meniscus_candidates, key=lambda x: x[0])[1]
    meniscus_mask = meniscus_comp["mask"].astype(np.uint8)

    # point：在剩余连通域里选更靠近中心、偏上/中部、面积相对较小者
    point_candidates = []
    for c in comps:
        if c["idx"] == meniscus_comp["idx"]:
            continue
        # point 通常不会特别靠下
        y_penalty = 1.0 if c["cy"] < 0.70 * h else 2.5
        # 越接近图像中心越好，面积不能太大
        compact_score = (c["center_dist"] * 4.0) + (c["area"] / max(h * w, 1)) * 20.0 + y_penalty
        point_candidates.append((compact_score, c))

    if len(point_candidates) > 0:
        point_comp = min(point_candidates, key=lambda x: x[0])[1]
        point_mask = point_comp["mask"].astype(np.uint8)
    else:
        # 如果只拆出一个区域，说明该 label 可能没有 point，保持空 mask
        point_mask = np.zeros((h, w), dtype=np.uint8)

    return point_mask, meniscus_mask


def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 6.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heat.astype(np.float32)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def dice_score_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def iou_score_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


def precision_recall_f1_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = float((pred & gt).sum())
    fp = float((pred & (1 - gt)).sum())
    fn = float(((1 - pred) & gt).sum())
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return precision, recall, f1


def point_from_heatmap(heat: np.ndarray) -> Tuple[int, int]:
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return int(x), int(y)


def calc_tmh_pixel(mask: np.ndarray, x_ref: int, half_window: int = 5) -> float:
    """
    在 x_ref ± half_window 的列范围内，统计每列泪河 mask 的上下边界高度，取 median。
    """
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    xs = range(max(0, x_ref - half_window), min(w, x_ref + half_window + 1))
    heights = []
    for x in xs:
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            heights.append(float(ys.max() - ys.min() + 1))
    if len(heights) == 0:
        return 0.0
    return float(np.median(heights))


# =========================
# 2. Dataset
# =========================

class TearMeniscusDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], img_size: int = 256,
                 label_threshold: int = 20, heat_sigma: float = 6.0):
        self.pairs = pairs
        self.img_size = img_size
        self.label_threshold = label_threshold
        self.heat_sigma = heat_sigma

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lab_path = self.pairs[idx]

        rgb = read_rgb(img_path)
        label_bin = read_label_binary(lab_path, threshold=self.label_threshold)
        point_mask, meniscus_mask = split_point_and_meniscus(label_bin)

        rgb_rs = resize_image(rgb, self.img_size)
        point_rs = resize_mask(point_mask, self.img_size)
        meniscus_rs = resize_mask(meniscus_mask, self.img_size)

        center = centroid_from_mask(point_rs)
        if center is None:
            point_heat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            cx, cy = center
            point_heat = gaussian_heatmap(self.img_size, self.img_size, cx, cy, sigma=self.heat_sigma)

        # 图像归一化到 [0,1]
        img = rgb_rs.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        sample = {
            "image": torch.from_numpy(img).float(),
            "point_mask": torch.from_numpy(point_rs[None, ...].astype(np.float32)),
            "point_heatmap": torch.from_numpy(point_heat[None, ...].astype(np.float32)),
            "meniscus_mask": torch.from_numpy(meniscus_rs[None, ...].astype(np.float32)),
            "name": img_path.stem,
            "img_path": str(img_path),
            "lab_path": str(lab_path),
            "rgb_vis": rgb_rs,
        }
        return sample


# =========================
# 3. 模型：传统 CNN / FCN 双任务网络
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNNMultitask(nn.Module):
    """
    FCN-like CNN：
    Encoder 用 stride=2 下采样；Decoder 用双线性上采样。
    注意：这里没有 U-Net skip connection。
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.stem = ConvBlock(3, base_ch)

        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),
            ConvBlock(base_ch * 4, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 8),
            nn.ReLU(inplace=True),
            ConvBlock(base_ch * 8, base_ch * 8),
        )

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = ConvBlock(base_ch * 2, base_ch)

        self.point_head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.meniscus_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)         # 256
        x = self.down1(x)        # 128
        x = self.down2(x)        # 64
        x = self.down3(x)        # 32
        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 64
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 128
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 256
        x = self.up1(x)

        point_logits = self.point_head(x)
        meniscus_logits = self.meniscus_head(x)
        return point_logits, meniscus_logits


# =========================
# 4. Loss
# =========================

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


def batch_pos_weight(target: torch.Tensor, max_weight: float = 80.0) -> torch.Tensor:
    pos = target.sum().clamp(min=1.0)
    neg = target.numel() - pos
    w = neg / pos
    w = torch.clamp(w, min=1.0, max=max_weight)
    return w.detach()


def multitask_loss(point_logits, meniscus_logits, point_mask, point_heatmap, meniscus_mask):
    dice_loss = DiceLoss()

    # point heatmap：加权 MSE，让中心高响应区域权重大一些
    point_prob = torch.sigmoid(point_logits)
    heat_weight = 1.0 + 20.0 * point_heatmap
    heat_mse = ((point_prob - point_heatmap) ** 2 * heat_weight).mean()

    # point mask：BCE + Dice，解决点太小导致的类别不平衡
    point_pw = batch_pos_weight(point_mask, max_weight=120.0).to(point_logits.device)
    point_bce = F.binary_cross_entropy_with_logits(point_logits, point_mask, pos_weight=point_pw)
    point_dice = dice_loss(point_logits, point_mask)
    point_loss = 0.25 * heat_mse + 0.5 * point_bce + 0.5 * point_dice

    # meniscus mask：BCE + Dice，解决泪河细小前景问题
    meniscus_pw = batch_pos_weight(meniscus_mask, max_weight=80.0).to(meniscus_logits.device)
    meniscus_bce = F.binary_cross_entropy_with_logits(meniscus_logits, meniscus_mask, pos_weight=meniscus_pw)
    meniscus_dice = dice_loss(meniscus_logits, meniscus_mask)
    meniscus_loss = meniscus_bce + meniscus_dice

    total = point_loss + meniscus_loss
    return total, {
        "total": float(total.detach().cpu()),
        "point": float(point_loss.detach().cpu()),
        "meniscus": float(meniscus_loss.detach().cpu()),
    }


# =========================
# 5. 训练、评价、预测可视化
# =========================

@dataclass
class EvalResult:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    point_error: float
    tmh_mae: float


def evaluate(model, loader, device, pred_threshold: float = 0.5, save_csv: Optional[Path] = None) -> EvalResult:
    model.eval()
    rows = []
    dices, ious, pres, recs, f1s, point_errors, tmh_errors = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            gt_point_mask = batch["point_mask"].cpu().numpy()
            gt_meniscus = batch["meniscus_mask"].cpu().numpy()

            point_logits, meniscus_logits = model(img)
            point_prob = torch.sigmoid(point_logits).cpu().numpy()
            meniscus_prob = torch.sigmoid(meniscus_logits).cpu().numpy()

            bs = img.size(0)
            for i in range(bs):
                name = batch["name"][i]
                pred_heat = point_prob[i, 0]
                pred_mask = (meniscus_prob[i, 0] > pred_threshold).astype(np.uint8)
                gt_pm = gt_point_mask[i, 0].astype(np.uint8)
                gt_mm = gt_meniscus[i, 0].astype(np.uint8)

                px, py = point_from_heatmap(pred_heat)
                gt_center = centroid_from_mask(gt_pm)
                if gt_center is None:
                    gx, gy = px, py
                    p_err = 0.0
                else:
                    gx, gy = gt_center
                    p_err = float(math.sqrt((px - gx) ** 2 + (py - gy) ** 2))

                pred_tmh = calc_tmh_pixel(pred_mask, px, half_window=5)
                gt_tmh = calc_tmh_pixel(gt_mm, int(round(gx)), half_window=5)
                tmh_err = abs(pred_tmh - gt_tmh)

                d = dice_score_np(pred_mask, gt_mm)
                j = iou_score_np(pred_mask, gt_mm)
                pr, rc, f1 = precision_recall_f1_np(pred_mask, gt_mm)

                dices.append(d)
                ious.append(j)
                pres.append(pr)
                recs.append(rc)
                f1s.append(f1)
                point_errors.append(p_err)
                tmh_errors.append(tmh_err)

                rows.append({
                    "name": name,
                    "meniscus_dice": d,
                    "meniscus_iou": j,
                    "precision": pr,
                    "recall": rc,
                    "f1": f1,
                    "point_error_px": p_err,
                    "pred_tmh_pixel": pred_tmh,
                    "gt_tmh_pixel": gt_tmh,
                    "tmh_abs_error_pixel": tmh_err,
                })

    if save_csv is not None:
        ensure_dir(save_csv.parent)
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    def mean_or_zero(x):
        return float(np.mean(x)) if len(x) else 0.0

    return EvalResult(
        dice=mean_or_zero(dices),
        iou=mean_or_zero(ious),
        precision=mean_or_zero(pres),
        recall=mean_or_zero(recs),
        f1=mean_or_zero(f1s),
        point_error=mean_or_zero(point_errors),
        tmh_mae=mean_or_zero(tmh_errors),
    )


def overlay_prediction(rgb: np.ndarray, pred_mask: np.ndarray, px: int, py: int,
                       tmh: float, title: str) -> np.ndarray:
    vis = rgb.copy()
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # 绿色：预测泪河区域
    green = np.zeros_like(vis)
    green[:, :, 1] = 255
    vis = np.where(pred_mask[..., None] > 0, (0.55 * vis + 0.45 * green).astype(np.uint8), vis)

    # 红点：预测中心点
    cv2.circle(vis, (int(px), int(py)), 5, (255, 0, 0), -1)

    # 黄色竖线：TMH 计算列
    ys = np.where(pred_mask[:, int(px)] > 0)[0] if 0 <= px < pred_mask.shape[1] else []
    if len(ys) > 0:
        cv2.line(vis, (int(px), int(ys.min())), (int(px), int(ys.max())), (255, 255, 0), 2)

    # 标题文字
    bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, title[:35], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, f"TMH={tmh:.1f}px", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_prediction_sheet(model, dataset: Dataset, indices: List[int], device, out_path: Path,
                          pred_threshold: float = 0.5, cols: int = 4) -> None:
    model.eval()
    tiles = []
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            point_logits, meniscus_logits = model(img)
            point_prob = torch.sigmoid(point_logits)[0, 0].cpu().numpy()
            meniscus_prob = torch.sigmoid(meniscus_logits)[0, 0].cpu().numpy()
            pred_mask = (meniscus_prob > pred_threshold).astype(np.uint8)
            px, py = point_from_heatmap(point_prob)
            tmh = calc_tmh_pixel(pred_mask, px, half_window=5)
            rgb = sample["rgb_vis"]
            tile = overlay_prediction(rgb, pred_mask, px, py, tmh, sample["name"])
            tiles.append(tile)

    if not tiles:
        return

    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = tile

    ensure_dir(out_path.parent)
    imwrite_unicode(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def make_gt_qa_sheet(dataset: Dataset, indices: List[int], out_path: Path, cols: int = 4) -> None:
    """生成 GT 标签拆分质检图：绿色泪河、红色 point、黄色 TMH 线。"""
    tiles = []
    for idx in indices:
        sample = dataset[idx]
        rgb = sample["rgb_vis"].copy()
        gt_point = sample["point_mask"][0].numpy().astype(np.uint8)
        gt_meniscus = sample["meniscus_mask"][0].numpy().astype(np.uint8)
        center = centroid_from_mask(gt_point)
        if center is None:
            cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
        else:
            cx, cy = int(round(center[0])), int(round(center[1]))
        gt_tmh = calc_tmh_pixel(gt_meniscus, cx, half_window=5)

        green = np.zeros_like(rgb)
        green[:, :, 1] = 255
        rgb = np.where(gt_meniscus[..., None] > 0, (0.55 * rgb + 0.45 * green).astype(np.uint8), rgb)
        cv2.circle(rgb, (cx, cy), 5, (255, 0, 0), -1)
        ys = np.where(gt_meniscus[:, cx] > 0)[0] if 0 <= cx < gt_meniscus.shape[1] else []
        if len(ys) > 0:
            cv2.line(rgb, (cx, int(ys.min())), (cx, int(ys.max())), (255, 255, 0), 2)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, sample["name"][:35], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"GT TMH={gt_tmh:.1f}px", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if not tiles:
        return

    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = tile

    ensure_dir(out_path.parent)
    imwrite_unicode(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    use_sets = [s.strip() for s in args.use_sets.split(",") if s.strip()]
    pairs = build_pairs(data_root, use_sets, layout=args.layout)
    if len(pairs) == 0:
        raise RuntimeError(
            "没有找到图像-标签配对。简单结构请检查：data_root/original 和 data_root/label；"
            "原数据集结构请检查：data_root/Colour1/Original 和 data_root/Colour1/Label。"
        )

    print(f"[INFO] matched pairs before sampling = {len(pairs)}")
    random.shuffle(pairs)
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    print(f"[INFO] matched pairs used = {len(pairs)}")
    dataset = TearMeniscusDataset(
        pairs=pairs,
        img_size=args.img_size,
        label_threshold=args.label_threshold,
        heat_sigma=args.heat_sigma,
    )

    # 小样本划分：默认按 train_ratio 进行 90/10 划分；也可以用 train_samples 手动指定
    n = len(dataset)
    if args.train_samples > 0:
        train_n = min(args.train_samples, max(1, n - 1))
    else:
        train_n = int(round(n * args.train_ratio))
        train_n = min(max(1, train_n), max(1, n - 1))

    train_idx = list(range(train_n))
    val_idx = list(range(train_n, n))
    if len(val_idx) == 0:
        val_idx = train_idx

    print(f"[INFO] split: train = {len(train_idx)}, test/val = {len(val_idx)}")

    # 保存划分清单，方便你写报告或复现实验
    split_csv = out_dir / "split_90_10.csv"
    with open(split_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "label_path"])
        for idx in train_idx:
            writer.writerow(["train", str(pairs[idx][0]), str(pairs[idx][1])])
        for idx in val_idx:
            writer.writerow(["test", str(pairs[idx][0]), str(pairs[idx][1])])
    print(f"[INFO] saved split list: {split_csv}")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # 先生成 GT 拆分质检图，确认 Label 拆得对不对
    qa_indices = list(range(min(args.qa_count, len(dataset))))
    make_gt_qa_sheet(dataset, qa_indices, out_dir / "gt_qa_contact_sheet.png", cols=4)
    print(f"[INFO] saved GT QA sheet: {out_dir / 'gt_qa_contact_sheet.png'}")

    model = SimpleCNNMultitask(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = -1.0
    best_path = out_dir / "best_cnn_multitask.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_list = []
        point_list = []
        meniscus_list = []

        for batch in train_loader:
            img = batch["image"].to(device)
            point_mask = batch["point_mask"].to(device)
            point_heatmap = batch["point_heatmap"].to(device)
            meniscus_mask = batch["meniscus_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            point_logits, meniscus_logits = model(img)
            loss, logs = multitask_loss(point_logits, meniscus_logits, point_mask, point_heatmap, meniscus_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_list.append(logs["total"])
            point_list.append(logs["point"])
            meniscus_list.append(logs["meniscus"])

        eval_res = evaluate(model, val_loader, device, pred_threshold=args.pred_threshold)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={np.mean(loss_list):.4f} "
            f"point={np.mean(point_list):.4f} "
            f"meniscus={np.mean(meniscus_list):.4f} | "
            f"val Dice={eval_res.dice:.4f} IoU={eval_res.iou:.4f} "
            f"PointErr={eval_res.point_error:.2f}px TMH_MAE={eval_res.tmh_mae:.2f}px"
        )

        if eval_res.dice > best_dice:
            best_dice = eval_res.dice
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_dice": best_dice,
            }, best_path)

    print(f"[INFO] best val Dice = {best_dice:.4f}")
    print(f"[INFO] saved model: {best_path}")

    # 加载最佳模型做预测图和 CSV 评价
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    eval_csv = out_dir / "metrics.csv"
    final_eval = evaluate(model, val_loader, device, pred_threshold=args.pred_threshold, save_csv=eval_csv)
    print(f"[INFO] saved metrics csv: {eval_csv}")
    print(
        "[FINAL] "
        f"Dice={final_eval.dice:.4f}, IoU={final_eval.iou:.4f}, "
        f"Precision={final_eval.precision:.4f}, Recall={final_eval.recall:.4f}, F1={final_eval.f1:.4f}, "
        f"PointErr={final_eval.point_error:.2f}px, TMH_MAE={final_eval.tmh_mae:.2f}px"
    )

    pred_indices = val_idx[:min(args.pred_count, len(val_idx))]
    if len(pred_indices) == 0:
        pred_indices = train_idx[:min(args.pred_count, len(train_idx))]
    make_prediction_sheet(
        model, dataset, pred_indices, device,
        out_dir / "prediction_contact_sheet.png",
        pred_threshold=args.pred_threshold,
        cols=4,
    )
    print(f"[INFO] saved prediction sheet: {out_dir / 'prediction_contact_sheet.png'}")


# =========================
# 6. 命令行参数
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\TM-CNN-100",
                        help="数据根目录。已默认设置为你的 TM-CNN-100 文件夹；简单结构：里面包含 original 和 label")
    parser.add_argument("--layout", type=str, default="simple", choices=["auto", "simple", "dataset"],
                        help="数据目录格式：simple 表示 data_root/original + data_root/label；dataset 表示 Colour1/Original + Colour1/Label；auto 自动判断")
    parser.add_argument("--use_sets", type=str, default="Colour1,Colour2",
                        help="layout=dataset 时使用哪些数据集，默认 Colour1,Colour2；layout=simple 时不用管")
    parser.add_argument("--out_dir", type=str, default="results/cnn_small",
                        help="输出结果文件夹")

    parser.add_argument("--max_samples", type=int, default=100,
                        help="最多使用多少张图像做小样本测试；-1 表示使用全部")
    parser.add_argument("--train_samples", type=int, default=0,
                        help="手动指定多少张用于训练；0 表示按 train_ratio 自动划分")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例，默认 0.9，即 100 张中 90 张训练、10 张测试")
    parser.add_argument("--img_size", type=int, default=256,
                        help="统一缩放尺寸，先用 256 比较省显存")
    parser.add_argument("--label_threshold", type=int, default=20,
                        help="Label 二值化阈值；彩色 label 建议 20，黑白 label 可改 127")
    parser.add_argument("--heat_sigma", type=float, default=6.0,
                        help="point Gaussian heatmap 的 sigma")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32,
                        help="CNN 基础通道数；显存不足可改 16")
    parser.add_argument("--pred_threshold", type=float, default=0.5,
                        help="meniscus 概率阈值")

    parser.add_argument("--qa_count", type=int, default=16,
                        help="保存多少张 GT 标签拆分质检图")
    parser.add_argument("--pred_count", type=int, default=16,
                        help="保存多少张预测可视化图")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows/PyCharm 建议先用 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.max_samples == -1:
        args.max_samples = 0
    train(args)
