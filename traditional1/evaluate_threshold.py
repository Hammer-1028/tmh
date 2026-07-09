from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


BASE = Path.cwd()
PROJECT_ROOT = BASE.parent
LABEL_DIR = PROJECT_ROOT / "data" / "Colour1" / "Label"
PRED_POINT_DIR = BASE / "pred_point" / "Colour1"
PRED_MENISCUS_DIR = BASE / "pred_meniscus" / "Colour1"
PRED_ALL_DIR = BASE / "pred_all" / "Colour1"
OUT_DIR = BASE / "evaluation_50"


def read_image_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image_unicode(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def binarize(mask):
    return (mask > 127).astype(np.uint8)


def resize_label(label):
    return cv2.resize(label, (640, 480), interpolation=cv2.INTER_NEAREST)


def split_label_components(label_binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_binary, 8)
    point = np.zeros_like(label_binary)
    curve = np.zeros_like(label_binary)

    comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        comps.append(
            {
                "id": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "cx": cx,
                "cy": cy,
                "aspect": w / max(h, 1),
            }
        )

    if not comps:
        return point, curve

    curve_candidates = [c for c in comps if c["w"] > 25 and c["aspect"] > 1.8]
    if curve_candidates:
        curve_comp = max(curve_candidates, key=lambda c: c["w"] + 0.5 * c["cy"])
        curve[labels == curve_comp["id"]] = 1

    remaining = [c for c in comps if np.count_nonzero(curve[labels == c["id"]]) == 0]
    if remaining:
        point_candidates = [
            c
            for c in remaining
            if c["area"] >= 10 and c["w"] <= 40 and c["h"] <= 40 and 0.4 <= c["aspect"] <= 2.5
        ]
        if point_candidates:
            point_comp = min(point_candidates, key=lambda c: abs(c["cx"] - 320) + abs(c["cy"] - 220))
            point[labels == point_comp["id"]] = 1

    return point, curve


def dilate(mask, ksize=(11, 11), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    return (cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0).astype(np.uint8)


def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
        "f1": f1,
    }


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
        "f1": f1,
    }


def tmh_from_mask(mask, x_ref):
    h, w = mask.shape
    if np.isnan(x_ref):
        return np.nan, np.nan, np.nan
    x1 = max(0, int(x_ref) - 5)
    x2 = min(w - 1, int(x_ref) + 5)
    heights, uppers, lowers = [], [], []
    for x in range(x1, x2 + 1):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        uppers.append(int(ys.min()))
        lowers.append(int(ys.max()))
        heights.append(int(ys.max() - ys.min() + 1))
    if not heights:
        return np.nan, np.nan, np.nan
    return float(np.median(heights)), float(np.median(uppers)), float(np.median(lowers))


def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(np.mean(xs)), float(np.mean(ys))


def plot_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["GT 0", "GT 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_bars(summary, out_path):
    metrics_cols = ["dice", "iou", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=160)
    x = np.arange(len(summary))
    width = 0.15
    for idx, col in enumerate(metrics_cols):
        ax.bar(x + (idx - 2) * width, summary[col], width, label=col)
    ax.set_xticks(x, labels=summary["target"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Traditional Threshold Metrics, first 50 images")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tmh(tmh_df, out_dir):
    valid = tmh_df.dropna(subset=["pred_tmh_pixel", "gt_tmh_pixel"]).copy()
    if valid.empty:
        return
    valid["error"] = valid["pred_tmh_pixel"] - valid["gt_tmh_pixel"]

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=160)
    ax.scatter(valid["gt_tmh_pixel"], valid["pred_tmh_pixel"], s=28, alpha=0.75)
    lo = min(valid["gt_tmh_pixel"].min(), valid["pred_tmh_pixel"].min())
    hi = max(valid["gt_tmh_pixel"].max(), valid["pred_tmh_pixel"].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("GT TMH proxy (px)")
    ax.set_ylabel("Pred TMH (px)")
    ax.set_title("TMH scatter")
    fig.tight_layout()
    fig.savefig(out_dir / "tmh_scatter.png")
    plt.close(fig)


def markdown_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.bar(valid["stem"], valid["error"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pred - GT (px)")
    ax.set_title("TMH error by sample")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "tmh_error_bar.png")
    plt.close(fig)


def make_overlay_eval(stem, background, pred_point, pred_meniscus, gt_point, gt_curve_tol):
    bg = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    color = bg.copy()
    color[gt_curve_tol > 0] = [255, 160, 0]
    color[pred_meniscus > 0] = [0, 255, 0]
    color[gt_point > 0] = [255, 0, 255]
    color[pred_point > 0] = [0, 0, 255]
    out = cv2.addWeighted(bg, 0.58, color, 0.42, 0)
    cv2.putText(out, stem, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted(PRED_ALL_DIR.glob("*.png"))[:50]]
    rows = []
    tmh_rows = []
    aggregate = {
        "point": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "meniscus": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "all": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    overlay_samples = []
    for stem in stems:
        pred_point = binarize(read_image_unicode(PRED_POINT_DIR / f"{stem}.png"))
        pred_meniscus = binarize(read_image_unicode(PRED_MENISCUS_DIR / f"{stem}.png"))
        pred_all = binarize(read_image_unicode(PRED_ALL_DIR / f"{stem}.png"))
        label_path = LABEL_DIR / f"{stem}.PNG"
        label = binarize(resize_label(read_image_unicode(label_path)))
        gt_point, gt_curve = split_label_components(label)

        gt_point_tol = dilate(gt_point, (15, 15), 1)
        gt_curve_tol = dilate(gt_curve, (13, 13), 1)
        gt_all_tol = ((gt_point_tol > 0) | (gt_curve_tol > 0)).astype(np.uint8)

        targets = {
            "point": (pred_point, gt_point_tol),
            "meniscus": (pred_meniscus, gt_curve_tol),
            "all": (pred_all, gt_all_tol),
        }
        for target, (pred, gt) in targets.items():
            m = metrics(pred, gt)
            for k in ["tp", "fp", "fn", "tn"]:
                aggregate[target][k] += m[k]
            rows.append({"stem": stem, "target": target, **m})

        pred_cx, pred_cy = centroid(pred_point)
        gt_cx, gt_cy = centroid(gt_point)
        pred_tmh, pred_upper, pred_lower = tmh_from_mask(pred_meniscus, pred_cx)
        gt_x_ref = gt_cx if not np.isnan(gt_cx) else pred_cx
        gt_tmh, gt_upper, gt_lower = tmh_from_mask(gt_curve_tol, gt_x_ref)
        tmh_rows.append(
            {
                "stem": stem,
                "pred_point_x": pred_cx,
                "pred_point_y": pred_cy,
                "gt_point_x": gt_cx,
                "gt_point_y": gt_cy,
                "point_error_px": float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy))
                if not np.isnan(pred_cx) and not np.isnan(gt_cx)
                else np.nan,
                "pred_tmh_pixel": pred_tmh,
                "pred_y_upper": pred_upper,
                "pred_y_lower": pred_lower,
                "gt_tmh_pixel": gt_tmh,
                "gt_y_upper": gt_upper,
                "gt_y_lower": gt_lower,
                "tmh_abs_error": abs(pred_tmh - gt_tmh)
                if not np.isnan(pred_tmh) and not np.isnan(gt_tmh)
                else np.nan,
            }
        )

        if len(overlay_samples) < 50:
            background_path = PROJECT_ROOT / "预处理" / "processed_original" / "Colour1" / "images_gamma" / f"{stem}.png"
            background = read_image_unicode(background_path)
            overlay_samples.append(make_overlay_eval(stem, background, pred_point, pred_meniscus, gt_point_tol, gt_curve_tol))

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(OUT_DIR / "metrics_per_sample_50.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for target, vals in aggregate.items():
        tp, fp, fn, tn = vals["tp"], vals["fp"], vals["fn"], vals["tn"]
        m = metrics_from_counts(tp, fp, fn, tn)
        summary_rows.append({"target": target, **m})
        cm = np.array([[tn, fp], [fn, tp]])
        plot_confusion_matrix(cm, f"{target} confusion matrix", OUT_DIR / f"confusion_matrix_{target}.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "metrics_summary_50.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars(summary, OUT_DIR / "metrics_bar_50.png")

    tmh_df = pd.DataFrame(tmh_rows)
    tmh_df.to_csv(OUT_DIR / "tmh_metrics_50.csv", index=False, encoding="utf-8-sig")
    plot_tmh(tmh_df, OUT_DIR)

    if overlay_samples:
        resized = [cv2.resize(img, (256, 192)) for img in overlay_samples]
        cols = 5
        rows_img = []
        for i in range(0, len(resized), cols):
            row = resized[i : i + cols]
            if len(row) < cols:
                row += [np.zeros_like(resized[0])] * (cols - len(row))
            rows_img.append(np.hstack(row))
        write_image_unicode(OUT_DIR / "evaluation_overlay_contact_50.png", np.vstack(rows_img))

    valid_tmh = tmh_df.dropna(subset=["pred_tmh_pixel", "gt_tmh_pixel"])
    doc = []
    doc.append("# 传统阈值法前50张评价结果说明\n")
    doc.append("## 评价对象\n")
    doc.append("- `pred_point`：圆点/中心参考点预测。")
    doc.append("- `pred_meniscus`：泪河区域预测。")
    doc.append("- `pred_all`：圆点与泪河区域的合并预测。")
    doc.append("\n## 评价方法\n")
    doc.append("预测阶段不读取 Label；评价阶段读取 `data/Colour1/Label` 作为人工标注参考。")
    doc.append("由于原始 Label 中泪河标注接近细弧线，而当前传统阈值法输出的是带厚度区域，因此评价时将人工泪河弧线做椭圆核膨胀，作为容差区域。")
    doc.append("\n## 总体指标\n")
    doc.append(markdown_table(summary[["target", "dice", "iou", "precision", "recall", "f1", "accuracy"]].round(4)))
    doc.append("\n## TMH 结果\n")
    doc.append(f"- 可同时获得预测 TMH 和参考 TMH 的样本数：{len(valid_tmh)} / {len(tmh_df)}")
    if not valid_tmh.empty:
        doc.append(f"- 预测 TMH 平均值：{valid_tmh['pred_tmh_pixel'].mean():.2f} px")
        doc.append(f"- 参考 TMH proxy 平均值：{valid_tmh['gt_tmh_pixel'].mean():.2f} px")
        doc.append(f"- TMH 平均绝对误差 MAE：{valid_tmh['tmh_abs_error'].mean():.2f} px")
        doc.append(f"- TMH 误差中位数：{valid_tmh['tmh_abs_error'].median():.2f} px")
    doc.append("\n## 结论\n")
    doc.append("传统阈值法能够在部分样本中提取下方泪河附近的带状区域，但对睫毛遮挡、Placido 环纹干扰、眼角暗区和局部纹理非常敏感。")
    doc.append("因此它适合作为课程项目中的传统基线方法和失败案例分析，不适合作为最终高精度方案。后续更适合继续比较 Canny/形态学、主动轮廓或 U-Net 类方法。")
    (OUT_DIR / "评价结果说明_前50张.md").write_text("\n".join(doc), encoding="utf-8")

    print(summary[["target", "dice", "iou", "precision", "recall", "f1", "accuracy"]].round(4).to_string(index=False))
    if not valid_tmh.empty:
        print(f"TMH valid: {len(valid_tmh)}/{len(tmh_df)}")
        print(f"TMH MAE: {valid_tmh['tmh_abs_error'].mean():.3f}px")


if __name__ == "__main__":
    main()
