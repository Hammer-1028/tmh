# -*- coding: utf-8 -*-
"""
根据 original + label 生成“答案/GT 对比图”的独立脚本。

作用：
1) 不重新训练模型；
2) 读取 results/cnn_small/split_90_10.csv 中的 test 10 张；
3) 从 label 自动拆出：point_mask + meniscus_mask；
4) 生成 gt_test_answer_contact_sheet.png；
5) 如果已经有 prediction_contact_sheet.png，则额外生成 answer_vs_prediction_vertical.png。

直接运行：
python .\make_gt_answer_sheet_fixed_path.py
"""

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================
# 1. 这里已经写成你的路径
# =========================
DATA_ROOT = Path(r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\TM-CNN-100")
OUT_DIR = Path(r"results\cnn_small")
LABEL_THRESHOLD = 20      # 彩色 label 建议 20；纯黑白 label 可改 127
IMG_SIZE = 256            # 必须和你训练代码的 img_size 保持一致；你之前默认是 256
SEED = 42
MAX_SAMPLES = 100
TRAIN_RATIO = 0.9
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
# 2. 中文路径兼容读写
# =========================
def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"OpenCV 无法读取图片，可能图片损坏: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"OpenCV 无法编码保存图片: {path}")
    buf.tofile(str(path))


# =========================
# 3. 文件配对
# =========================
def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def normalize_stem(stem: str) -> str:
    s = stem.lower()
    for suf in ["_label", "-label", "_mask", "-mask", "_gt", "-gt"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def find_case_insensitive_dir(parent: Path, candidates: List[str]) -> Optional[Path]:
    if not parent.exists():
        return None
    child_map = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in candidates:
        if name.lower() in child_map:
            return child_map[name.lower()]
    return None


def build_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    img_dir = find_case_insensitive_dir(data_root, ["original", "Original", "images", "Images"])
    lab_dir = find_case_insensitive_dir(data_root, ["label", "Label", "labels", "Labels", "mask", "Mask", "masks", "Masks"])
    if img_dir is None or lab_dir is None:
        raise RuntimeError("没有找到 original 和 label 文件夹，请检查 TM-CNN-100 目录结构。")

    imgs = list_images(img_dir)
    labs = list_images(lab_dir)
    lab_map: Dict[str, Path] = {normalize_stem(p.stem): p for p in labs}

    pairs = []
    for img_path in imgs:
        key = normalize_stem(img_path.stem)
        if key in lab_map:
            pairs.append((img_path, lab_map[key]))
    return pairs


def get_test_pairs_from_split_or_random(all_pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    split_csv = OUT_DIR / "split_90_10.csv"
    if split_csv.exists():
        test_pairs = []
        with open(split_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").lower() == "test":
                    img_path = Path(row["image_path"])
                    lab_path = Path(row["label_path"])
                    if img_path.exists() and lab_path.exists():
                        test_pairs.append((img_path, lab_path))
        if len(test_pairs) > 0:
            print(f"[INFO] 使用 split_90_10.csv 中的 test 样本: {len(test_pairs)} 张")
            return test_pairs

    print("[WARN] 没找到可用 split_90_10.csv，改用 seed=42 重新划分 90/10。")
    pairs = list(all_pairs)
    random.seed(SEED)
    random.shuffle(pairs)
    if MAX_SAMPLES > 0:
        pairs = pairs[:MAX_SAMPLES]
    train_n = int(round(len(pairs) * TRAIN_RATIO))
    train_n = min(max(1, train_n), max(1, len(pairs) - 1))
    return pairs[train_n:]


# =========================
# 4. Label 拆分 + TMH 计算
# =========================
def read_rgb(path: Path) -> np.ndarray:
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_label_binary(path: Path, threshold: int = 20) -> np.ndarray:
    lab = imread_unicode(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    return (gray > threshold).astype(np.uint8)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


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
    h, w = label_mask.shape[:2]
    comps = connected_components(label_mask)

    point_mask = np.zeros((h, w), dtype=np.uint8)
    meniscus_mask = np.zeros((h, w), dtype=np.uint8)
    if len(comps) == 0:
        return point_mask, meniscus_mask

    meniscus_candidates = []
    for c in comps:
        lower_bonus = 1.0 + 2.0 * c["norm_cy"]
        wide_bonus = max(c["aspect"], 0.2)
        area_bonus = math.sqrt(max(c["area"], 1))
        penalty = 0.35 if c["cy"] < 0.45 * h else 1.0
        score = area_bonus * lower_bonus * wide_bonus * penalty
        meniscus_candidates.append((score, c))

    meniscus_comp = max(meniscus_candidates, key=lambda x: x[0])[1]
    meniscus_mask = meniscus_comp["mask"].astype(np.uint8)

    point_candidates = []
    for c in comps:
        if c["idx"] == meniscus_comp["idx"]:
            continue
        y_penalty = 1.0 if c["cy"] < 0.70 * h else 2.5
        compact_score = (c["center_dist"] * 4.0) + (c["area"] / max(h * w, 1)) * 20.0 + y_penalty
        point_candidates.append((compact_score, c))

    if len(point_candidates) > 0:
        point_comp = min(point_candidates, key=lambda x: x[0])[1]
        point_mask = point_comp["mask"].astype(np.uint8)

    return point_mask, meniscus_mask


def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def calc_tmh_pixel(mask: np.ndarray, x_ref: int, half_window: int = 5) -> float:
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
# 5. 生成答案图
# =========================
def make_gt_answer_sheet(test_pairs: List[Tuple[Path, Path]], out_path: Path, cols: int = 4) -> None:
    tiles = []
    for img_path, lab_path in test_pairs:
        rgb = resize_image(read_rgb(img_path), IMG_SIZE)
        label_bin = read_label_binary(lab_path, threshold=LABEL_THRESHOLD)
        point_mask, meniscus_mask = split_point_and_meniscus(label_bin)
        point_mask = resize_mask(point_mask, IMG_SIZE)
        meniscus_mask = resize_mask(meniscus_mask, IMG_SIZE)

        center = centroid_from_mask(point_mask)
        if center is None:
            cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
        else:
            cx, cy = int(round(center[0])), int(round(center[1]))
        gt_tmh = calc_tmh_pixel(meniscus_mask, cx, half_window=5)

        vis = rgb.copy()
        green = np.zeros_like(vis)
        green[:, :, 1] = 255
        vis = np.where(meniscus_mask[..., None] > 0, (0.55 * vis + 0.45 * green).astype(np.uint8), vis)
        cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
        ys = np.where(meniscus_mask[:, cx] > 0)[0] if 0 <= cx < meniscus_mask.shape[1] else []
        if len(ys) > 0:
            cv2.line(vis, (cx, int(ys.min())), (cx, int(ys.max())), (255, 255, 0), 2)

        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, img_path.stem[:35], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"GT TMH={gt_tmh:.1f}px", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if not tiles:
        raise RuntimeError("没有可用于生成答案图的 test 样本。")

    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = tile

    imwrite_unicode(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[INFO] 已保存答案/GT图: {out_path}")


def add_banner(bgr: np.ndarray, text: str) -> np.ndarray:
    banner_h = 42
    banner = np.zeros((banner_h, bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, bgr])


def make_vertical_comparison(gt_path: Path, pred_path: Path, out_path: Path) -> None:
    if not gt_path.exists() or not pred_path.exists():
        print("[WARN] 没找到预测图 prediction_contact_sheet.png，跳过上下对比大图。")
        return
    gt = imread_unicode(gt_path, cv2.IMREAD_COLOR)
    pred = imread_unicode(pred_path, cv2.IMREAD_COLOR)

    # 宽度不一致时，把预测图缩放到答案图同宽
    if pred.shape[1] != gt.shape[1]:
        new_h = int(round(pred.shape[0] * gt.shape[1] / pred.shape[1]))
        pred = cv2.resize(pred, (gt.shape[1], new_h), interpolation=cv2.INTER_AREA)

    gt_panel = add_banner(gt, "GROUND TRUTH ANSWER  (from Label)")
    pred_panel = add_banner(pred, "MODEL PREDICTION")
    comparison = np.vstack([gt_panel, pred_panel])
    imwrite_unicode(out_path, comparison)
    print(f"[INFO] 已保存上下对比图: {out_path}")


def main():
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")
    pairs = build_pairs(DATA_ROOT)
    print(f"[INFO] 找到 original-label 配对: {len(pairs)} 张")

    test_pairs = get_test_pairs_from_split_or_random(pairs)
    print("[INFO] 本次生成答案图的 test 文件：")
    for img_path, _ in test_pairs:
        print("   ", img_path.name)

    gt_path = OUT_DIR / "gt_test_answer_contact_sheet.png"
    pred_path = OUT_DIR / "prediction_contact_sheet.png"
    cmp_path = OUT_DIR / "answer_vs_prediction_vertical.png"

    make_gt_answer_sheet(test_pairs, gt_path, cols=4)
    make_vertical_comparison(gt_path, pred_path, cmp_path)


if __name__ == "__main__":
    main()
