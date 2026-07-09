# -*- coding: utf-8 -*-
"""
CNN/FCN 双任务泪河高度测量 + 三级分类完整脚本
=================================================

本脚本已按你的 Colour1 数据库结构改好，默认读取：
D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1
  ├─ images_gamma      # 推荐作为输入图像
  ├─ Label             # 原始 Label，包含圆点 + 泪河
  ├─ Label_processed   # 本脚本不依赖，可不用
  ├─ Original          # 若 images_gamma 不存在，会自动回退到 Original
  └─ results           # 输出目录

功能：
1. 自动配对 images_gamma 与 Label；
2. 按 70% / 15% / 15% 划分 train / val / test；
3. 使用 CNN/FCN 双任务网络同时预测：
   - point_head：圆点参考位置 heatmap；
   - meniscus_head：泪河区域 mask；
4. 自动从原始 Label 拆分圆点和泪河真值；
5. 输出 TMH 像素高度、TMH 毫米高度、三级分类结果；
6. 输出 Dice/IoU/Precision/Recall/F1、圆点定位误差、TMH MAE、分类准确率、Macro-F1；
7. 生成混淆矩阵、散点图、类别分布图、预测 overlay、样例拼图。

推荐直接运行：
python cnn_colour1_70_15_15_tmh_classification.py

如果想用显卡，确保安装的是 CUDA 版本 PyTorch，脚本会自动使用 cuda。

注意：
- 这里保留为 CNN/FCN 风格，不是 U-Net，没有 encoder-decoder skip connection。
- 如果你的 images_gamma/Label 高度是 480，而作者原始标尺是 1024 高度下 86 px/mm，
  默认换算为：px_per_mm = 86 * 当前Label高度 / 1024。
  因此 480 高度时为 40.31 px/mm，和你前面截图里的公式一致。
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# 0. 基础配置
# ============================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF"}
CLASS_NAMES = ["低泪河高度", "正常泪河高度", "较高泪河高度"]
CLASS_SHORT = ["low", "normal", "high"]

# Matplotlib 中文显示。Windows 上通常可用 Microsoft YaHei 或 SimHei。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """兼容 Windows 中文路径的 OpenCV 读取。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"文件为空或无法读取: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise RuntimeError(f"OpenCV 无法解码图片: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """兼容 Windows 中文路径的 OpenCV 保存。"""
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix if path.suffix else ".png"
    ok, enc = cv2.imencode(suffix, img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    enc.tofile(str(path))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix in IMG_EXTS])


def find_dir(parent: Path, names: Sequence[str]) -> Optional[Path]:
    """大小写不敏感寻找目录。"""
    if not parent.exists():
        return None
    children = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for n in names:
        if n.lower() in children:
            return children[n.lower()]
    return None


def normalize_stem(stem: str) -> str:
    """用于图像与 label 配对，去掉常见后缀。"""
    s = stem.lower()
    for suf in ["_label", "-label", "_mask", "-mask", "_gt", "-gt", "_seg", "-seg"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def read_rgb(path: Path) -> np.ndarray:
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_label_binary(path: Path, threshold: int = 20) -> np.ndarray:
    """
    读取原始 Label 并二值化。
    兼容红/绿/白等彩色 label；threshold=20 比 127 更稳。
    """
    lab = imread_unicode(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    return (gray > threshold).astype(np.uint8)


# ============================================================
# 1. 数据配对与划分
# ============================================================

def build_pairs(data_root: Path,
                image_dir_name: str = "images_gamma",
                label_dir_name: str = "Label") -> List[Tuple[Path, Path]]:
    """
    默认读取：Colour1/images_gamma + Colour1/Label。
    若 images_gamma 不存在，自动回退到 Original。
    """
    data_root = Path(data_root)
    img_dir = find_dir(data_root, [image_dir_name, "images_gamma", "Image", "Images", "image", "images", "Original", "original"])
    lab_dir = find_dir(data_root, [label_dir_name, "Label", "label", "Labels", "labels", "mask", "Mask"])

    if img_dir is None:
        raise FileNotFoundError(f"没有找到输入图像文件夹。请检查: {data_root} 下是否有 images_gamma 或 Original")
    if lab_dir is None:
        raise FileNotFoundError(f"没有找到 Label 文件夹。请检查: {data_root} 下是否有 Label")

    imgs = list_images(img_dir)
    labs = list_images(lab_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"输入图像文件夹为空: {img_dir}")
    if len(labs) == 0:
        raise RuntimeError(f"Label 文件夹为空: {lab_dir}")

    lab_map: Dict[str, Path] = {normalize_stem(p.stem): p for p in labs}
    pairs: List[Tuple[Path, Path]] = []
    missing = []
    for img in imgs:
        key = normalize_stem(img.stem)
        if key in lab_map:
            pairs.append((img, lab_map[key]))
        else:
            missing.append(img.name)

    if len(pairs) == 0:
        raise RuntimeError(
            f"没有配对成功的图像和 Label。\n图像目录: {img_dir}\nLabel目录: {lab_dir}\n"
            f"请确认两边文件名主体一致，例如 Color1_000001.png"
        )

    print(f"[数据] 图像目录: {img_dir}")
    print(f"[数据] Label目录: {lab_dir}")
    print(f"[数据] 成功配对: {len(pairs)} 张")
    if missing:
        print(f"[提醒] 有 {len(missing)} 张图像没有找到同名 Label，已跳过。前5个: {missing[:5]}")
    return pairs


def split_pairs_70_15_15(pairs: List[Tuple[Path, Path]], seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """按 70:15:15 随机划分。"""
    n = len(pairs)
    indices = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    # 确保 test 至少保留剩余样本
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def save_split_csv(path: Path, pairs: List[Tuple[Path, Path]], indices: List[int], split_name: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "name", "image_path", "label_path"])
        for idx in indices:
            img, lab = pairs[idx]
            writer.writerow([split_name, idx, img.stem, str(img), str(lab)])


# ============================================================
# 2. Label 拆分：point mask + meniscus mask
# ============================================================

def connected_components(mask: np.ndarray) -> List[Dict]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape[:2]
    comps: List[Dict] = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area <= 3:
            continue
        cx, cy = centroids[i]
        comp_mask = (labels == i).astype(np.uint8)
        comps.append({
            "idx": i,
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area": int(area), "cx": float(cx), "cy": float(cy),
            "aspect": float(bw / max(bh, 1)),
            "norm_cy": float(cy / max(h - 1, 1)),
            "center_dist": float(abs(cx - w / 2) / max(w, 1)),
            "mask": comp_mask,
        })
    return comps


def split_point_and_meniscus(label_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从原始 Label 二值图中拆出：
    - point_mask：上方/中部、面积较小、接近中心的圆点；
    - meniscus_mask：下方、横向延展明显的泪河区域。
    """
    h, w = label_mask.shape[:2]
    comps = connected_components(label_mask)
    point_mask = np.zeros((h, w), dtype=np.uint8)
    meniscus_mask = np.zeros((h, w), dtype=np.uint8)
    if not comps:
        return point_mask, meniscus_mask

    # 选择泪河：更靠下、更宽、更长、更大。
    meniscus_scores = []
    for c in comps:
        lower_score = 1.0 + 3.0 * c["norm_cy"]
        wide_score = max(c["aspect"], 0.1)
        area_score = math.sqrt(max(c["area"], 1))
        upper_penalty = 0.25 if c["cy"] < 0.42 * h else 1.0
        score = lower_score * wide_score * area_score * upper_penalty
        meniscus_scores.append((score, c))
    meniscus_comp = max(meniscus_scores, key=lambda x: x[0])[1]
    meniscus_mask = meniscus_comp["mask"].astype(np.uint8)

    # 选择圆点：剩余区域中更靠近图像中心、面积较小、不过度靠下。
    point_scores = []
    for c in comps:
        if c["idx"] == meniscus_comp["idx"]:
            continue
        area_ratio = c["area"] / max(h * w, 1)
        down_penalty = 2.0 if c["cy"] > 0.75 * h else 0.0
        score = c["center_dist"] * 6.0 + area_ratio * 50.0 + down_penalty + abs(c["cy"] / h - 0.40)
        point_scores.append((score, c))
    if point_scores:
        point_comp = min(point_scores, key=lambda x: x[0])[1]
        point_mask = point_comp["mask"].astype(np.uint8)

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


# ============================================================
# 3. Dataset
# ============================================================

class TearMeniscusDataset(Dataset):
    def __init__(self,
                 pairs: List[Tuple[Path, Path]],
                 img_size: int = 256,
                 label_threshold: int = 20,
                 heat_sigma: float = 6.0,
                 augment: bool = False):
        self.pairs = pairs
        self.img_size = img_size
        self.label_threshold = label_threshold
        self.heat_sigma = heat_sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        img_path, lab_path = self.pairs[idx]
        rgb_orig = read_rgb(img_path)
        label_bin = read_label_binary(lab_path, threshold=self.label_threshold)

        point_orig, meniscus_orig = split_point_and_meniscus(label_bin)
        orig_h, orig_w = label_bin.shape[:2]

        rgb = cv2.resize(rgb_orig, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        point = cv2.resize(point_orig, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        meniscus = cv2.resize(meniscus_orig, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            rgb, point, meniscus = self.apply_augment(rgb, point, meniscus)

        center = centroid_from_mask(point)
        if center is None:
            heat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            cx, cy = center
            heat = gaussian_heatmap(self.img_size, self.img_size, cx, cy, sigma=self.heat_sigma)

        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return {
            "image": torch.from_numpy(img).float(),
            "point_mask": torch.from_numpy(point[None, ...].astype(np.float32)),
            "point_heatmap": torch.from_numpy(heat[None, ...].astype(np.float32)),
            "meniscus_mask": torch.from_numpy(meniscus[None, ...].astype(np.float32)),
            "name": img_path.stem,
            "img_path": str(img_path),
            "lab_path": str(lab_path),
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
        }

    @staticmethod
    def apply_augment(rgb: np.ndarray, point: np.ndarray, meniscus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 左右翻转
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            point = np.ascontiguousarray(point[:, ::-1])
            meniscus = np.ascontiguousarray(meniscus[:, ::-1])

        # 亮度/对比度扰动
        if random.random() < 0.8:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 轻微高斯噪声
        if random.random() < 0.25:
            noise = np.random.normal(0, 3, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return rgb, point, meniscus


# ============================================================
# 4. CNN/FCN 双任务模型
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNNMultitask(nn.Module):
    """
    CNN/FCN 双任务网络：
    - encoder：stride=2 卷积下采样；
    - decoder：双线性上采样；
    - 无 U-Net skip connection。
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.stem = ConvBlock(3, base_ch)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2), nn.ReLU(inplace=True),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 4), nn.ReLU(inplace=True),
            ConvBlock(base_ch * 4, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 8), nn.ReLU(inplace=True),
            ConvBlock(base_ch * 8, base_ch * 8),
        )
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)
        self.up3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = ConvBlock(base_ch * 2, base_ch)
        self.point_head = nn.Conv2d(base_ch, 1, 1)
        self.meniscus_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        return self.point_head(x), self.meniscus_head(x)


# ============================================================
# 5. Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


def batch_pos_weight(target: torch.Tensor, max_weight: float) -> torch.Tensor:
    pos = target.sum().clamp(min=1.0)
    neg = target.numel() - pos
    return torch.clamp(neg / pos, min=1.0, max=max_weight).detach()


def multitask_loss(point_logits: torch.Tensor,
                   meniscus_logits: torch.Tensor,
                   point_mask: torch.Tensor,
                   point_heatmap: torch.Tensor,
                   meniscus_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    dice_loss = DiceLoss()

    point_prob = torch.sigmoid(point_logits)
    heat_weight = 1.0 + 20.0 * point_heatmap
    heat_mse = ((point_prob - point_heatmap) ** 2 * heat_weight).mean()

    point_pw = batch_pos_weight(point_mask, 120.0).to(point_logits.device)
    point_bce = F.binary_cross_entropy_with_logits(point_logits, point_mask, pos_weight=point_pw)
    point_dice = dice_loss(point_logits, point_mask)
    point_loss = 0.30 * heat_mse + 0.50 * point_bce + 0.50 * point_dice

    meniscus_pw = batch_pos_weight(meniscus_mask, 80.0).to(meniscus_logits.device)
    meniscus_bce = F.binary_cross_entropy_with_logits(meniscus_logits, meniscus_mask, pos_weight=meniscus_pw)
    meniscus_dice = dice_loss(meniscus_logits, meniscus_mask)
    meniscus_loss = meniscus_bce + meniscus_dice

    total = point_loss + meniscus_loss
    return total, {
        "total": float(total.detach().cpu()),
        "point": float(point_loss.detach().cpu()),
        "meniscus": float(meniscus_loss.detach().cpu()),
    }


# ============================================================
# 6. 后处理、测高与指标
# ============================================================

def dice_score_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return (2 * inter + eps) / (denom + eps)


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


def weighted_point_from_heatmap(heat: np.ndarray, radius: int = 6, peak_ratio: float = 0.55) -> Tuple[int, int]:
    """方案3类似的局部加权圆点定位，比单点 argmax 更稳。"""
    h, w = heat.shape[:2]
    y0, x0 = np.unravel_index(np.argmax(heat), heat.shape)
    x1, x2 = max(0, x0 - radius), min(w, x0 + radius + 1)
    y1, y2 = max(0, y0 - radius), min(h, y0 + radius + 1)
    patch = heat[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0 or patch.max() <= 0:
        return int(x0), int(y0)
    mask = patch >= patch.max() * peak_ratio
    weights = patch * mask
    if weights.sum() <= 1e-12:
        return int(x0), int(y0)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    cx = float((xx * weights).sum() / weights.sum())
    cy = float((yy * weights).sum() / weights.sum())
    return int(round(cx)), int(round(cy))


def postprocess_meniscus(mask: np.ndarray,
                         close_kernel: int = 11,
                         min_area: int = 20) -> np.ndarray:
    """保留下方宽扁主连通域，并做水平闭运算连接断裂泪河。"""
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    if mask.sum() == 0:
        return mask

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    comps = connected_components(mask)
    if not comps:
        return np.zeros_like(mask)

    scores = []
    for c in comps:
        if c["area"] < min_area:
            continue
        lower = 1.0 + 3.0 * c["norm_cy"]
        wide = max(c["aspect"], 0.2)
        area = math.sqrt(max(c["area"], 1))
        upper_penalty = 0.35 if c["cy"] < 0.45 * h else 1.0
        score = lower * wide * area * upper_penalty
        scores.append((score, c))

    if not scores:
        return np.zeros_like(mask)
    best = max(scores, key=lambda x: x[0])[1]
    return best["mask"].astype(np.uint8)


def calc_tmh_pixel(mask: np.ndarray, x_ref: int, half_window: int = 5) -> float:
    """在 x_ref 左右窗口内统计泪河垂直高度，取中位数。"""
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    x_ref = int(np.clip(x_ref, 0, w - 1))
    heights = []
    for x in range(max(0, x_ref - half_window), min(w, x_ref + half_window + 1)):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            heights.append(float(ys.max() - ys.min() + 1))
    if not heights:
        return 0.0
    return float(np.median(heights))


def px_per_mm_for_height(current_height: int, px_per_mm_original: float, author_original_height: float) -> float:
    """
    当前 Label 高度下的像素/毫米换算。
    例如：原始 1024 高度下 86 px/mm；当前高度 480，则 86*480/1024=40.31 px/mm。
    若你已经把预测/GT恢复到 1024 高度，则该函数返回 86。
    """
    if author_original_height and author_original_height > 0:
        return float(px_per_mm_original) * float(current_height) / float(author_original_height)
    return float(px_per_mm_original)


def classify_tmh_mm(tmh_mm: float) -> int:
    if tmh_mm <= 0.20:
        return 0
    if tmh_mm <= 0.27:
        return 1
    return 2


def class_mean_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 3) -> Dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = []
    for k in range(num_classes):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - cm[k, k])
        fn = float(cm[k, :].sum() - cm[k, k])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[k, :].sum())
        per_class.append({
            "class_id": k,
            "class_name": CLASS_NAMES[k],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    macro_f1 = float(np.mean([r["f1"] for r in per_class]))
    weighted_f1 = float(sum(r["f1"] * r["support"] for r in per_class) / max(cm.sum(), 1))
    return {"cm": cm, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class": per_class}


@dataclass
class EvalResult:
    split: str
    n: int
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    point_error_px: float
    tmh_mae_px: float
    tmh_mae_mm: float
    cls_acc: float
    macro_f1: float
    weighted_f1: float


# ============================================================
# 7. 可视化
# ============================================================

def overlay_prediction(rgb: np.ndarray,
                       pred_mask: np.ndarray,
                       px: int,
                       py: int,
                       pred_tmh_px: float,
                       gt_class: int,
                       pred_class: int) -> np.ndarray:
    """
    绿色泪河、红色圆点、黄色测高线。输入 rgb 为 RGB，输出 BGR 用于保存。

    重要修正：
    Colour1 中 images_gamma 和 Label 的分辨率可能不一致，例如：
        image: 480 x 640
        label: 1024 x 1360
    模型评价时 pred_mask/px/py 会恢复到 Label 坐标；但 overlay 要画在原图坐标上。
    因此这里会自动把 pred_mask、px、py 按比例缩放到 rgb 的尺寸，避免广播错误。
    """
    vis = rgb.copy()
    pred_mask = (pred_mask > 0).astype(np.uint8)

    img_h, img_w = vis.shape[:2]
    mask_h, mask_w = pred_mask.shape[:2]

    # 如果预测 mask 是 Label 尺寸，而原图是 images_gamma 尺寸，先转到原图尺寸。
    if (mask_h, mask_w) != (img_h, img_w):
        sx = img_w / max(mask_w, 1)
        sy = img_h / max(mask_h, 1)
        pred_mask_vis = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        px_vis = int(round(px * sx))
        py_vis = int(round(py * sy))
    else:
        pred_mask_vis = pred_mask
        px_vis = int(px)
        py_vis = int(py)

    green = np.zeros_like(vis)
    green[:, :, 1] = 255
    vis = np.where(pred_mask_vis[..., None] > 0,
                   (0.55 * vis + 0.45 * green).astype(np.uint8),
                   vis)

    h, w = pred_mask_vis.shape[:2]
    px_vis = int(np.clip(px_vis, 0, w - 1))
    py_vis = int(np.clip(py_vis, 0, h - 1))
    cv2.circle(vis, (px_vis, py_vis), max(3, int(round(w / 160))), (255, 0, 0), -1)  # RGB 红点

    ys = np.where(pred_mask_vis[:, px_vis] > 0)[0]
    if len(ys) > 0:
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.line(vis, (px_vis, y1), (px_vis, y2), (255, 255, 0), max(1, int(round(w / 300))))

    # OpenCV 默认不支持中文，这里只写简单英文/数字，避免乱码。
    cv2.putText(vis, f"GT:{CLASS_SHORT[gt_class]} Pred:{CLASS_SHORT[pred_class]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"TMH={pred_tmh_px:.1f}px", (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def save_contact_sheet(items: List[Tuple[str, np.ndarray]], out_path: Path, cols: int = 4, thumb: int = 240) -> None:
    if not items:
        return
    rows = int(math.ceil(len(items) / cols))
    sheet_h = rows * (thumb + 32)
    sheet_w = cols * thumb
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    for i, (name, bgr) in enumerate(items):
        r, c = divmod(i, cols)
        img = cv2.resize(bgr, (thumb, thumb), interpolation=cv2.INTER_AREA)
        y0 = r * (thumb + 32)
        x0 = c * thumb
        sheet[y0:y0 + thumb, x0:x0 + thumb] = img
        cv2.putText(sheet, name[:28], (x0 + 4, y0 + thumb + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    imwrite_unicode(out_path, sheet)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=160)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(3), CLASS_NAMES)

    for i in range(3):
        row_sum = cm[i].sum()
        for j in range(3):
            pct = 100.0 * cm[i, j] / row_sum if row_sum > 0 else 0.0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tmh_scatter(gt_mm: List[float], pred_mm: List[float], correct: List[bool], out_path: Path, title: str) -> None:
    gt = np.asarray(gt_mm, dtype=float)
    pred = np.asarray(pred_mm, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=160)
    if len(gt) > 0:
        ax.scatter(gt[corr], pred[corr], s=18, label="分类正确", alpha=0.8)
        ax.scatter(gt[~corr], pred[~corr], s=24, label="分类错误", alpha=0.9)
        mn = float(min(gt.min(), pred.min(), 0.0))
        mx = float(max(gt.max(), pred.max(), 0.30))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2, label="Pred=GT")
        ax.axvline(0.20, linestyle="--", linewidth=1.0)
        ax.axvline(0.27, linestyle="--", linewidth=1.0)
        ax.axhline(0.20, linestyle="--", linewidth=1.0)
        ax.axhline(0.27, linestyle="--", linewidth=1.0)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("GT TMH (mm)")
    ax.set_ylabel("Pred TMH (mm)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y_true: List[int], y_pred: List[int], out_path: Path, title: str) -> None:
    true_counts = [sum(1 for y in y_true if y == k) for k in range(3)]
    pred_counts = [sum(1 for y in y_pred if y == k) for k in range(3)]
    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)
    ax.bar(x - width / 2, true_counts, width, label="GT")
    ax.bar(x + width / 2, pred_counts, width, label="Pred")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("样本数")
    ax.legend()
    for i, v in enumerate(true_counts):
        ax.text(i - width / 2, v + 0.5, str(v), ha="center", fontsize=9)
    for i, v in enumerate(pred_counts):
        ax.text(i + width / 2, v + 0.5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_rows: List[Dict], out_path: Path) -> None:
    if not summary_rows:
        return
    labels = [r["split"] for r in summary_rows]
    metrics = ["dice", "tmh_mae_mm", "cls_acc", "macro_f1"]
    titles = ["泪河 Dice", "TMH MAE(mm)", "分类 Acc", "Macro-F1"]
    for metric, title in zip(metrics, titles):
        vals = [float(r[metric]) for r in summary_rows]
        fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=160)
        ax.bar(labels, vals)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path.parent / f"summary_{metric}.png", bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 8. 训练与评价
# ============================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    use_amp: bool = False) -> Dict[str, float]:
    model.train()
    totals = []
    point_losses = []
    meniscus_losses = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    non_blocking = device.type == "cuda"

    for batch in loader:
        img = batch["image"].to(device, non_blocking=non_blocking)
        point_mask = batch["point_mask"].to(device, non_blocking=non_blocking)
        point_heat = batch["point_heatmap"].to(device, non_blocking=non_blocking)
        meniscus_mask = batch["meniscus_mask"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                point_logits, meniscus_logits = model(img)
                loss, loss_items = multitask_loss(point_logits, meniscus_logits, point_mask, point_heat, meniscus_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            point_logits, meniscus_logits = model(img)
            loss, loss_items = multitask_loss(point_logits, meniscus_logits, point_mask, point_heat, meniscus_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        totals.append(loss_items["total"])
        point_losses.append(loss_items["point"])
        meniscus_losses.append(loss_items["meniscus"])

    return {
        "loss": float(np.mean(totals)) if totals else 0.0,
        "point_loss": float(np.mean(point_losses)) if point_losses else 0.0,
        "meniscus_loss": float(np.mean(meniscus_losses)) if meniscus_losses else 0.0,
    }


def evaluate_split(model: nn.Module,
                   dataset: TearMeniscusDataset,
                   indices: List[int],
                   split_name: str,
                   device: torch.device,
                   out_dir: Path,
                   args,
                   save_visuals: bool = True) -> Tuple[EvalResult, Dict]:
    """评价一个 split，并保存 CSV/图像。"""
    split_dir = out_dir / split_name
    ensure_dir(split_dir)
    overlay_dir = split_dir / "overlay_final"
    if save_visuals:
        ensure_dir(overlay_dir)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    rows = []
    dices, ious, pres, recs, f1s = [], [], [], [], []
    point_errors, tmh_errors_px, tmh_errors_mm = [], [], []
    y_true, y_pred = [], []
    gt_mm_all, pred_mm_all, correct_all = [], [], []
    contact_items: List[Tuple[str, np.ndarray]] = []

    with torch.no_grad():
        for batch in loader:
            img_tensor = batch["image"].to(device, non_blocking=(device.type == "cuda"))
            point_logits, meniscus_logits = model(img_tensor)
            point_prob = torch.sigmoid(point_logits).cpu().numpy()
            meniscus_prob = torch.sigmoid(meniscus_logits).cpu().numpy()

            bs = img_tensor.size(0)
            for i in range(bs):
                name = batch["name"][i]
                img_path = Path(batch["img_path"][i])
                lab_path = Path(batch["lab_path"][i])
                orig_h = int(batch["orig_h"][i])
                orig_w = int(batch["orig_w"][i])

                # 读原图和原始 Label，用原始/当前 Label 坐标评价
                rgb_orig = read_rgb(img_path)
                lab_bin = read_label_binary(lab_path, threshold=args.label_threshold)
                gt_point, gt_meniscus = split_point_and_meniscus(lab_bin)
                gt_center = centroid_from_mask(gt_point)

                # 网络输出在 img_size 坐标，先做后处理再恢复到 Label 坐标
                pred_heat_small = point_prob[i, 0]
                px_small, py_small = weighted_point_from_heatmap(pred_heat_small,
                                                                 radius=args.point_radius,
                                                                 peak_ratio=args.point_peak_ratio)
                pred_mask_small_raw = (meniscus_prob[i, 0] > args.pred_threshold).astype(np.uint8)
                pred_mask_small = postprocess_meniscus(pred_mask_small_raw,
                                                       close_kernel=args.close_kernel,
                                                       min_area=args.min_area)
                pred_mask = cv2.resize(pred_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                px = int(round(px_small * orig_w / args.img_size))
                py = int(round(py_small * orig_h / args.img_size))
                px = int(np.clip(px, 0, orig_w - 1))
                py = int(np.clip(py, 0, orig_h - 1))

                if gt_center is None:
                    gx, gy = px, py
                    point_err = 0.0
                else:
                    gx, gy = gt_center
                    point_err = float(math.sqrt((px - gx) ** 2 + (py - gy) ** 2))

                gt_tmh_px = calc_tmh_pixel(gt_meniscus, int(round(gx)), half_window=args.tmh_half_window)
                pred_tmh_px = calc_tmh_pixel(pred_mask, px, half_window=args.tmh_half_window)

                px_per_mm = px_per_mm_for_height(orig_h, args.px_per_mm_original, args.author_original_height)
                gt_tmh_mm = gt_tmh_px / px_per_mm if px_per_mm > 0 else 0.0
                pred_tmh_mm = pred_tmh_px / px_per_mm if px_per_mm > 0 else 0.0

                gt_cls = classify_tmh_mm(gt_tmh_mm)
                pred_cls = classify_tmh_mm(pred_tmh_mm)
                correct = (gt_cls == pred_cls)

                d = dice_score_np(pred_mask, gt_meniscus)
                j = iou_score_np(pred_mask, gt_meniscus)
                pr, rc, f1 = precision_recall_f1_np(pred_mask, gt_meniscus)
                tmh_err_px = abs(pred_tmh_px - gt_tmh_px)
                tmh_err_mm = abs(pred_tmh_mm - gt_tmh_mm)

                dices.append(d)
                ious.append(j)
                pres.append(pr)
                recs.append(rc)
                f1s.append(f1)
                point_errors.append(point_err)
                tmh_errors_px.append(tmh_err_px)
                tmh_errors_mm.append(tmh_err_mm)
                y_true.append(gt_cls)
                y_pred.append(pred_cls)
                gt_mm_all.append(gt_tmh_mm)
                pred_mm_all.append(pred_tmh_mm)
                correct_all.append(correct)

                rows.append({
                    "name": name,
                    "image_path": str(img_path),
                    "label_path": str(lab_path),
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                    "px_per_mm_used": px_per_mm,
                    "meniscus_dice": d,
                    "meniscus_iou": j,
                    "meniscus_precision": pr,
                    "meniscus_recall": rc,
                    "meniscus_f1": f1,
                    "gt_point_x": gx,
                    "gt_point_y": gy,
                    "pred_point_x": px,
                    "pred_point_y": py,
                    "point_error_px": point_err,
                    "gt_tmh_px": gt_tmh_px,
                    "pred_tmh_px": pred_tmh_px,
                    "tmh_abs_error_px": tmh_err_px,
                    "gt_tmh_mm": gt_tmh_mm,
                    "pred_tmh_mm": pred_tmh_mm,
                    "tmh_abs_error_mm": tmh_err_mm,
                    "gt_class_id": gt_cls,
                    "pred_class_id": pred_cls,
                    "gt_class": CLASS_NAMES[gt_cls],
                    "pred_class": CLASS_NAMES[pred_cls],
                    "class_correct": int(correct),
                })

                if save_visuals:
                    bgr_overlay = overlay_prediction(rgb_orig, pred_mask, px, py, pred_tmh_px, gt_cls, pred_cls)
                    imwrite_unicode(overlay_dir / f"{name}_overlay.png", bgr_overlay)
                    if len(contact_items) < args.num_visual_samples:
                        contact_items.append((name, bgr_overlay))

    # 逐图 CSV
    csv_path = split_dir / "逐图_px_mm_分类结果.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    cls_info = class_mean_metrics(y_true, y_pred, 3)
    cm = cls_info["cm"]

    def mean(x: Sequence[float]) -> float:
        return float(np.mean(x)) if len(x) else 0.0

    result = EvalResult(
        split=split_name,
        n=len(rows),
        dice=mean(dices),
        iou=mean(ious),
        precision=mean(pres),
        recall=mean(recs),
        f1=mean(f1s),
        point_error_px=mean(point_errors),
        tmh_mae_px=mean(tmh_errors_px),
        tmh_mae_mm=mean(tmh_errors_mm),
        cls_acc=cls_info["accuracy"],
        macro_f1=cls_info["macro_f1"],
        weighted_f1=cls_info["weighted_f1"],
    )

    # 保存分类报告 CSV
    with open(split_dir / "分类指标.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(cls_info["per_class"])

    # 保存 summary txt/json
    summary = {
        "split": split_name,
        "n": result.n,
        "meniscus_dice": result.dice,
        "meniscus_iou": result.iou,
        "meniscus_precision": result.precision,
        "meniscus_recall": result.recall,
        "meniscus_f1": result.f1,
        "point_error_px": result.point_error_px,
        "tmh_mae_px": result.tmh_mae_px,
        "tmh_mae_mm": result.tmh_mae_mm,
        "classification_accuracy": result.cls_acc,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "confusion_matrix": cm.tolist(),
        "class_report": cls_info["per_class"],
        "csv": str(csv_path),
    }
    with open(split_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(split_dir / "summary_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"{split_name} 结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"样本数: {result.n}\n")
        f.write(f"泪河 Dice / IoU: {result.dice:.4f} / {result.iou:.4f}\n")
        f.write(f"泪河 Precision / Recall / F1: {result.precision:.4f} / {result.recall:.4f} / {result.f1:.4f}\n")
        f.write(f"圆点平均误差: {result.point_error_px:.2f} px\n")
        f.write(f"TMH MAE: {result.tmh_mae_px:.2f} px = {result.tmh_mae_mm:.4f} mm\n")
        f.write(f"分类 Accuracy / Macro-F1 / Weighted-F1: {result.cls_acc:.4f} / {result.macro_f1:.4f} / {result.weighted_f1:.4f}\n")
        f.write("\n三级分类阈值：低泪河高度 TMH <= 0.20 mm；正常 0.20 < TMH <= 0.27 mm；较高 TMH > 0.27 mm。\n")
        f.write(f"换算：px_per_mm = {args.px_per_mm_original} * 当前Label高度 / {args.author_original_height}。\n")
        f.write("\n逐图结果 CSV：\n")
        f.write(str(csv_path) + "\n")

    # 图像
    plot_confusion_matrix(cm, split_dir / "三级分类混淆矩阵.png", f"CNN 三级分类混淆矩阵（{split_name}）")
    plot_tmh_scatter(gt_mm_all, pred_mm_all, correct_all, split_dir / "TMH_mm_预测_真实散点图.png", f"CNN TMH mm 预测-真实散点图（{split_name}）")
    plot_class_distribution(y_true, y_pred, split_dir / "真实_预测类别分布.png", f"CNN 真实/预测类别分布（{split_name}）")
    if save_visuals:
        save_contact_sheet(contact_items, split_dir / "预测可视化样例拼图.png", cols=4, thumb=240)

    return result, summary


def write_overall_summary(out_dir: Path, results: List[EvalResult], args) -> None:
    rows = []
    for r in results:
        rows.append({
            "split": r.split,
            "n": r.n,
            "dice": r.dice,
            "iou": r.iou,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "point_error_px": r.point_error_px,
            "tmh_mae_px": r.tmh_mae_px,
            "tmh_mae_mm": r.tmh_mae_mm,
            "cls_acc": r.cls_acc,
            "macro_f1": r.macro_f1,
            "weighted_f1": r.weighted_f1,
        })

    csv_path = out_dir / "总结果_train_val_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "总结果说明.txt", "w", encoding="utf-8") as f:
        f.write("CNN/FCN 双任务模型：泪河测高与三级分类总结果\n")
        f.write("=" * 70 + "\n")
        f.write(f"数据根目录: {args.data_root}\n")
        f.write("划分比例: train:val:test = 70:15:15\n")
        f.write(f"输入图像文件夹优先: {args.image_dir_name}\n")
        f.write(f"Label 文件夹: {args.label_dir_name}\n")
        f.write(f"img_size: {args.img_size}\n")
        f.write(f"换算公式: px_per_mm = {args.px_per_mm_original} * 当前Label高度 / {args.author_original_height}\n")
        f.write("分类标准: 低泪河高度 TMH <= 0.20 mm；正常 0.20 < TMH <= 0.27 mm；较高 TMH > 0.27 mm。\n\n")
        for r in results:
            f.write(f"[{r.split}] n={r.n}\n")
            f.write(f"  Dice/IoU={r.dice:.4f}/{r.iou:.4f}, PointErr={r.point_error_px:.2f}px, "
                    f"TMH_MAE={r.tmh_mae_px:.2f}px={r.tmh_mae_mm:.4f}mm, "
                    f"Acc={r.cls_acc:.4f}, Macro-F1={r.macro_f1:.4f}\n")
        f.write("\n主要查看 test 文件夹中的：\n")
        f.write("1) 逐图_px_mm_分类结果.csv\n")
        f.write("2) summary_metrics.txt\n")
        f.write("3) 三级分类混淆矩阵.png\n")
        f.write("4) TMH_mm_预测_真实散点图.png\n")
        f.write("5) 真实_预测类别分布.png\n")
        f.write("6) 预测可视化样例拼图.png\n")

    plot_metric_bars(rows, out_dir / "summary_metric_bars.png")


# ============================================================
# 9. 主流程
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CNN/FCN 泪河 TMH 测高 + 三级分类：Colour1 70:15:15")

    parser.add_argument("--data_root", type=str,
                        default=r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1",
                        help="Colour1 数据根目录")
    parser.add_argument("--image_dir_name", type=str, default="images_gamma",
                        help="输入图像文件夹名，默认 images_gamma；不存在时自动回退 Original")
    parser.add_argument("--label_dir_name", type=str, default="Label",
                        help="Label 文件夹名，默认 Label")
    parser.add_argument("--out_dir", type=str, default="",
                        help="输出目录；默认保存到 data_root/results/cnn_70_15_15_时间戳")

    parser.add_argument("--img_size", type=int, default=256,
                        help="CNN 输入尺寸。若显存够可改为 320 或 480；测高会恢复到 Label 尺寸后计算。")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_eval", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows 建议先用 0；确认没问题后可改 2 或 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练；仅 CUDA 有效")

    parser.add_argument("--label_threshold", type=int, default=20)
    parser.add_argument("--heat_sigma", type=float, default=6.0)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--point_radius", type=int, default=6)
    parser.add_argument("--point_peak_ratio", type=float, default=0.55)
    parser.add_argument("--close_kernel", type=int, default=11)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--tmh_half_window", type=int, default=5)

    # 毫米换算。默认与截图一致：1024 高度下 86 px/mm；若 Label 当前高度是 480，则 40.31 px/mm。
    parser.add_argument("--px_per_mm_original", type=float, default=86.0)
    parser.add_argument("--author_original_height", type=float, default=1024.0,
                        help="作者给 86 px/mm 时对应的原始图像高度。若你的 Label 已是原始 1024，就保持 1024；若不想按高度缩放，可设为 -1。")

    parser.add_argument("--num_visual_samples", type=int, default=12)
    parser.add_argument("--eval_train", action="store_true",
                        help="是否也完整评价 train。默认只评价 val/test，若想三者都输出可加此参数。")
    parser.add_argument("--eval_only", action="store_true",
                        help="只做正式评价，不重新训练。用于训练已经完成但评价/可视化中断的情况。")
    parser.add_argument("--resume_model", type=str, default="",
                        help="eval_only 时指定已保存的 best_cnn_multitask.pth；不填则默认使用 out_dir/best_cnn_multitask.pth。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = data_root / "results" / f"cnn_70_15_15_{now_str()}"
    ensure_dir(out_dir)

    with open(out_dir / "运行参数.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    pairs = build_pairs(data_root, args.image_dir_name, args.label_dir_name)
    train_idx, val_idx, test_idx = split_pairs_70_15_15(pairs, seed=args.seed)
    print(f"[划分] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    save_split_csv(out_dir / "split_train.csv", pairs, train_idx, "train")
    save_split_csv(out_dir / "split_val.csv", pairs, val_idx, "val")
    save_split_csv(out_dir / "split_test.csv", pairs, test_idx, "test")

    train_dataset = TearMeniscusDataset(pairs, img_size=args.img_size,
                                        label_threshold=args.label_threshold,
                                        heat_sigma=args.heat_sigma,
                                        augment=True)
    eval_dataset = TearMeniscusDataset(pairs, img_size=args.img_size,
                                       label_threshold=args.label_threshold,
                                       heat_sigma=args.heat_sigma,
                                       augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[设备] device={device}, amp={use_amp}")
    if device.type == "cuda":
        print(f"[显卡] {torch.cuda.get_device_name(0)}")
    else:
        print("[提示] 当前 PyTorch 没有检测到 CUDA 显卡，将使用 CPU；pin_memory 与 amp 会自动关闭。")

    train_loader = DataLoader(Subset(train_dataset, train_idx),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=False)

    model = SimpleCNNMultitask(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    best_score = -1e9
    best_path = Path(args.resume_model) if args.resume_model else (out_dir / "best_cnn_multitask.pth")
    latest_path = out_dir / "latest_cnn_multitask.pth"
    train_log_path = out_dir / "train_log.csv"

    if args.eval_only:
        # 只评价已有模型，不重新训练。适合训练已完成、但最终可视化/汇总阶段中断的情况。
        if not best_path.exists():
            raise FileNotFoundError(
                f"没有找到已有模型: {best_path}\n"
                f"请把 --out_dir 指向上次训练输出目录，或用 --resume_model 指定 best_cnn_multitask.pth 的完整路径。"
            )
        print(f"[eval_only] 跳过训练，直接载入已有模型: {best_path}")
    else:
        with open(train_log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "point_loss", "meniscus_loss", "val_dice", "val_tmh_mae_mm", "val_acc", "score", "lr"])

        for epoch in range(1, args.epochs + 1):
            loss_items = train_one_epoch(model, train_loader, optimizer, device, use_amp=use_amp)
            scheduler.step()

            # 每个 epoch 简评 val；只存少量图，避免训练阶段太慢。
            val_result, _ = evaluate_split(model, eval_dataset, val_idx, "val_epoch_tmp", device, out_dir / "_tmp_eval", args, save_visuals=False)
            score = val_result.dice + val_result.cls_acc - 0.5 * val_result.tmh_mae_mm
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:03d}/{args.epochs} | loss={loss_items['loss']:.4f} | "
                  f"val Dice={val_result.dice:.4f} | val TMH_MAE={val_result.tmh_mae_mm:.4f}mm | "
                  f"val Acc={val_result.cls_acc:.4f} | score={score:.4f}")

            with open(train_log_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_items["loss"], loss_items["point_loss"], loss_items["meniscus_loss"],
                                 val_result.dice, val_result.tmh_mae_mm, val_result.cls_acc, score, lr_now])

            # 每个 epoch 都保存 latest，避免后续评价/可视化阶段出错导致训练成果丢失。
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "score": score,
            }, latest_path)

            if score > best_score:
                best_score = score
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "score": best_score,
                }, best_path)
                print(f"  -> 保存最佳模型: {best_path}")

    # 载入最佳模型并正式评价；如果历史版本没有保存 best，但保存了 latest，则自动兜底。
    if not best_path.exists() and latest_path.exists():
        print(f"[提示] 未找到最佳模型，改用 latest 模型: {latest_path}")
        best_path = latest_path
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[最佳模型] epoch={ckpt.get('epoch')}, score={ckpt.get('score'):.4f}")

    final_results: List[EvalResult] = []
    if args.eval_train:
        train_res, _ = evaluate_split(model, eval_dataset, train_idx, "train", device, out_dir, args, save_visuals=True)
        final_results.append(train_res)
    val_res, _ = evaluate_split(model, eval_dataset, val_idx, "val", device, out_dir, args, save_visuals=True)
    test_res, _ = evaluate_split(model, eval_dataset, test_idx, "test", device, out_dir, args, save_visuals=True)
    final_results.extend([val_res, test_res])

    write_overall_summary(out_dir, final_results, args)

    print("\n================= 运行完成 =================")
    print(f"输出目录: {out_dir}")
    print(f"最佳模型: {best_path}")
    print("重点查看 test 文件夹：")
    print(f"  {out_dir / 'test' / 'summary_metrics.txt'}")
    print(f"  {out_dir / 'test' / '逐图_px_mm_分类结果.csv'}")
    print(f"  {out_dir / 'test' / '三级分类混淆矩阵.png'}")
    print(f"  {out_dir / 'test' / 'TMH_mm_预测_真实散点图.png'}")
    print(f"  {out_dir / 'test' / '预测可视化样例拼图.png'}")


if __name__ == "__main__":
    main()
