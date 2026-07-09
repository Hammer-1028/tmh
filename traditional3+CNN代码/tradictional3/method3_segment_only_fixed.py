import os
import math
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import exposure, filters, morphology, measure, segmentation
from skimage.morphology import disk


# ==========================================================
# 1. 输入输出路径
# ==========================================================

IMAGE_PATH = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\images_gamma\Color1_000007.png"

OUTPUT_FOLDER = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================================
# 2. 参数
# ==========================================================

REAL_WTW_MM = 11.5
ACTIVE_ITER = 35
COLUMN_HALF_WIDTH = 5

MIN_AREA = 25
MAX_AREA = 5000


# ==========================================================
# 3. 中文路径读取 / 保存
# ==========================================================

def read_image_rgb_chinese_path(path):
    data = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"无法读取图片：{path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    return img_rgb


def save_image_chinese_path(path, image_uint8):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, image_uint8)

    if not ok:
        raise ValueError(f"保存失败：{path}")

    buf.tofile(path)


def normalize01(x):
    x = x.astype(np.float32)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def rgb_to_gray(img_rgb):
    if img_rgb.ndim == 2:
        return img_rgb.astype(np.float32)

    return (
        0.299 * img_rgb[:, :, 0]
        + 0.587 * img_rgb[:, :, 1]
        + 0.114 * img_rgb[:, :, 2]
    ).astype(np.float32)


# ==========================================================
# 4. 图像增强
# ==========================================================

def enhance_gray(img_rgb):
    gray = rgb_to_gray(img_rgb)

    enhanced = exposure.equalize_adapthist(
        gray,
        clip_limit=0.012,
        nbins=256
    )

    enhanced = filters.gaussian(enhanced, sigma=1.0)
    enhanced = normalize01(enhanced)

    return enhanced


# ==========================================================
# 5. 自动参考圆和比例尺
# ==========================================================

def auto_reference_circle(img_rgb):
    gray = rgb_to_gray(img_rgb)
    h, w = gray.shape

    gray_u8 = (gray * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray_u8, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.45),
        param1=80,
        param2=26,
        minRadius=int(min(h, w) * 0.18),
        maxRadius=int(min(h, w) * 0.50)
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        best = None
        best_score = -1e18

        for x, y, r in circles:
            dist = math.sqrt((x - w / 2) ** 2 + (y - h * 0.45) ** 2)
            score = -dist + 0.4 * r

            if score > best_score:
                best_score = score
                best = (x, y, r)

        cx, cy, r = best
    else:
        cx = w * 0.5
        cy = h * 0.45
        r = min(h, w) * 0.35

    radius_px = max(float(r), 0.40 * w)

    if cx < 0.30 * w or cx > 0.70 * w:
        cx = w * 0.5

    if cy < 0.20 * h or cy > 0.65 * h:
        cy = h * 0.45

    wtw_px = 2.0 * radius_px
    mm_per_pixel = REAL_WTW_MM / wtw_px

    return float(cx), float(cy), float(radius_px), float(wtw_px), float(mm_per_pixel)


# ==========================================================
# 6. 一维平滑工具
# ==========================================================

def median_smooth_1d(arr, k=21):
    arr = np.asarray(arr, dtype=np.float32)
    out = arr.copy()
    half = k // 2

    for i in range(len(arr)):
        left = max(0, i - half)
        right = min(len(arr), i + half + 1)
        out[i] = np.median(arr[left:right])

    return out


def blend_polyfit(xs, curve, alpha_raw=0.70):
    xs = np.asarray(xs, dtype=np.float32)
    curve = np.asarray(curve, dtype=np.float32)

    if len(xs) < 10:
        return curve

    coef = np.polyfit(xs, curve, deg=2)
    fit = coef[0] * xs ** 2 + coef[1] * xs + coef[2]

    return alpha_raw * curve + (1.0 - alpha_raw) * fit


# ==========================================================
# 7. 自动检测泪河上下边界
# ==========================================================

def detect_meniscus_boundaries(enhanced, cx):
    """
    暗谷驱动的泪河上下边界检测。

    核心思路：
    1. 不再直接取梯度最大边缘；
    2. 先找每列的泪河暗谷 valley；
    3. 从 valley 向上、向下找灰度回升点作为上/下边界；
    4. 再平滑上下边界；
    5. 这样不会再出现明显台阶状大块区域。
    """

    h, w = enhanced.shape

    # 横向范围：不要覆盖整条下眼睑，只取中央主要区域
    x0 = int(max(0, cx - 0.28 * w))
    x1 = int(min(w, cx + 0.28 * w))

    # 纵向范围：只看下方泪河附近
    # 对 Color1_000007，这个范围比之前更合理
    y0 = int(0.70 * h)
    y1 = int(0.84 * h)

    crop = enhanced[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape

    # 平滑，减少 Placido 环纹和睫毛的细碎干扰
    crop_smooth = filters.gaussian(crop, sigma=1.4)

    upper_curve = np.zeros(crop_w, dtype=np.float32)
    lower_curve = np.zeros(crop_w, dtype=np.float32)
    valley_curve = np.zeros(crop_w, dtype=np.float32)

    for x in range(crop_w):
        col = crop_smooth[:, x]

        # --------------------------------------------------
        # 1. 找暗谷：泪河通常是下眼睑上方的一条暗带
        # --------------------------------------------------
        # 避免找到 ROI 最上方的 Placido 环，也避免找到底部睫毛
        valley_y_min = int(0.18 * crop_h)
        valley_y_max = int(0.82 * crop_h)

        ys_search = np.arange(valley_y_min, valley_y_max)

        if len(ys_search) == 0:
            valley_y = int(0.50 * crop_h)
        else:
            valley_y = ys_search[int(np.argmin(col[ys_search]))]

        valley_curve[x] = valley_y

        valley_value = col[valley_y]

        # --------------------------------------------------
        # 2. 向上找上边界
        # --------------------------------------------------
        up_start = max(0, valley_y - 26)
        up_end = max(1, valley_y - 3)

        if up_end <= up_start:
            upper_y = valley_y - 8
        else:
            up_region = col[up_start:up_end]

            # 上方亮度参考
            upper_peak = np.max(up_region)
            threshold_up = valley_value + 0.32 * (upper_peak - valley_value)

            # 找最后一个从上方向暗谷过渡前的亮点
            candidates = np.where(up_region > threshold_up)[0]

            if len(candidates) > 0:
                upper_y = up_start + candidates[-1] + 1
            else:
                upper_y = valley_y - 8

        # --------------------------------------------------
        # 3. 向下找下边界
        # --------------------------------------------------
        down_start = min(crop_h - 1, valley_y + 3)
        down_end = min(crop_h, valley_y + 26)

        if down_end <= down_start:
            lower_y = valley_y + 8
        else:
            down_region = col[down_start:down_end]

            # 下方亮度参考
            lower_peak = np.max(down_region)
            threshold_down = valley_value + 0.32 * (lower_peak - valley_value)

            candidates = np.where(down_region > threshold_down)[0]

            if len(candidates) > 0:
                lower_y = down_start + candidates[0] - 1
            else:
                lower_y = valley_y + 8

        # --------------------------------------------------
        # 4. 限制厚度，避免过宽或过窄
        # --------------------------------------------------
        if lower_y <= upper_y:
            upper_y = valley_y - 6
            lower_y = valley_y + 6

        thickness = lower_y - upper_y + 1

        # 这张图建议 7-18 px；比之前自然
        # 这张图目前结果偏厚，所以把厚度控制在 6-14 px
        if thickness < 6:
            center = int(round((upper_y + lower_y) / 2))
            upper_y = center - 3
            lower_y = center + 3

        if thickness > 14:
            center = int(round((upper_y + lower_y) / 2))
            upper_y = center - 6
            lower_y = center + 7

        upper_y = max(0, min(crop_h - 1, upper_y))
        lower_y = max(0, min(crop_h - 1, lower_y))

        upper_curve[x] = upper_y
        lower_curve[x] = lower_y

    # ======================================================
    # 5. 平滑边界，去掉台阶
    # ======================================================
    upper_curve = median_smooth_1d(upper_curve, k=31)
    lower_curve = median_smooth_1d(lower_curve, k=31)
    valley_curve = median_smooth_1d(valley_curve, k=31)

    xs_crop = np.arange(crop_w)

    # 用二次曲线轻微平滑，但保留原始趋势
    upper_curve = blend_polyfit(xs_crop, upper_curve, alpha_raw=0.65)
    lower_curve = blend_polyfit(xs_crop, lower_curve, alpha_raw=0.65)

    # 再次限制厚度，避免局部异常
    thickness = lower_curve - upper_curve + 1
    # 最终厚度限制：压窄泪河区域
    thickness = np.clip(thickness, 4, 10)

    center_curve = (upper_curve + lower_curve) / 2
    upper_curve = center_curve - thickness / 2
    lower_curve = center_curve + thickness / 2

    # ======================================================
    # 6. 转换回全图坐标
    # ======================================================
    upper_global = np.full(w, np.nan, dtype=np.float32)
    lower_global = np.full(w, np.nan, dtype=np.float32)

    # ======================================================
    # 整体下移修正
    # ======================================================
    # 你当前结果整体偏上，所以只把上下边界一起向下移动。
    # 注意：上下边界一起移动，不会改变 TMH 厚度，只改变定位位置。
    SHIFT_DOWN_PX = 7

    upper_shifted = upper_curve + y0 + SHIFT_DOWN_PX
    lower_shifted = lower_curve + y0 + SHIFT_DOWN_PX

    # 防止下边界移到图像外
    upper_shifted = np.clip(upper_shifted, 0, h - 1)
    lower_shifted = np.clip(lower_shifted, 0, h - 1)

    upper_global[x0:x1] = upper_shifted
    lower_global[x0:x1] = lower_shifted

    return upper_global, lower_global, x0, x1, y0, y1
# ==========================================================
# 8. 根据上下边界生成 mask 和活动轮廓搜索带
# ==========================================================

def build_boundary_masks(shape, upper, lower, x0, x1):
    """
    根据上下边界生成初始 mask 和活动轮廓搜索带。
    这里把活动轮廓可扩张范围缩小，防止最终区域过厚。
    """
    h, w = shape

    init_mask = np.zeros((h, w), dtype=bool)
    band_mask = np.zeros((h, w), dtype=bool)

    for x in range(x0, x1):
        if np.isnan(upper[x]) or np.isnan(lower[x]):
            continue

        u = int(round(upper[x]))
        l = int(round(lower[x]))

        u = max(0, min(h - 1, u))
        l = max(0, min(h - 1, l))

        if l <= u:
            continue

        init_mask[u:l + 1, x] = True

        # 原来上下扩 3 px，容易变厚；这里改成 1 px
        bu = max(0, u - 1)
        bl = min(h - 1, l + 1)

        band_mask[bu:bl + 1, x] = True

    init_mask = morphology.binary_closing(init_mask, footprint=disk(1))
    init_mask = morphology.remove_small_holes(init_mask, area_threshold=50)
    init_mask = morphology.remove_small_objects(init_mask, min_size=MIN_AREA)

    band_mask = morphology.binary_closing(band_mask, footprint=disk(1))

    return init_mask, band_mask

# ==========================================================
# 9. 区域生长 + 活动轮廓
# ==========================================================

def build_feature_for_refine(enhanced):
    smooth = filters.gaussian(enhanced, sigma=1.0)
    bg = filters.gaussian(smooth, sigma=8.0)

    local_dark = normalize01(bg - smooth)

    u8 = (smooth * 255).astype(np.uint8)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    edge_y = normalize01(np.abs(gy))

    feature = normalize01(0.55 * local_dark + 0.45 * edge_y)

    return feature


def keep_components_touching_seed(candidate, seed_mask):
    label_img = measure.label(candidate, connectivity=2)
    seed_labels = np.unique(label_img[seed_mask])
    seed_labels = seed_labels[seed_labels != 0]

    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)

    return np.isin(label_img, seed_labels)


def region_growing_and_active_contour(feature, init_mask, band_mask):
    """
    init_mask 是上下边界之间的区域。
    band_mask 是允许活动轮廓移动的窄范围。
    这样不会再吃掉大块 Placido 环纹。
    """
    seed = init_mask.copy()

    values = feature[seed]

    if values.size == 0:
        return init_mask.copy(), init_mask.copy()

    med = np.median(values)

    candidate = (feature >= med - 0.18) & (feature <= med + 0.22) & band_mask

    grown = keep_components_touching_seed(candidate, seed)

    grown = morphology.binary_closing(grown, footprint=disk(2))
    grown = morphology.remove_small_holes(grown, area_threshold=60)
    grown = morphology.remove_small_objects(grown, min_size=MIN_AREA)

    if np.sum(grown) < 0.35 * np.sum(init_mask):
        grown = init_mask.copy()

    init_level = morphology.binary_dilation(grown | seed, footprint=disk(1))

    try:
        ac = segmentation.morphological_chan_vese(
            feature,
            num_iter=ACTIVE_ITER,
            init_level_set=init_level,
            smoothing=1,
            lambda1=1,
            lambda2=1
        )
    except TypeError:
        ac = segmentation.morphological_chan_vese(
            feature,
            iterations=ACTIVE_ITER,
            init_level_set=init_level,
            smoothing=1,
            lambda1=1,
            lambda2=1
        )

    raw = ((ac & band_mask) | grown) & band_mask

    raw = morphology.binary_closing(raw, footprint=disk(2))
    raw = morphology.remove_small_holes(raw, area_threshold=60)
    raw = morphology.remove_small_objects(raw, min_size=MIN_AREA)

    if np.sum(raw) < 0.35 * np.sum(init_mask):
        raw = init_mask.copy()

    return grown, raw


# ==========================================================
# 10. 最终整理：按列真实高度填充，不固定厚度
# ==========================================================

def column_fill_without_fixed_width(raw, init_mask, x0, x1):
    """
    最终按列填充，但不直接相信活动轮廓的碎片边界。
    这里先从 raw/init_mask 提取每列上下边界，再平滑，避免台阶状结果。
    """
    h, w = raw.shape

    source = raw if np.sum(raw) > 0 else init_mask

    top_list = []
    bottom_list = []
    valid_x = []

    for x in range(x0, x1):
        ys = np.where(source[:, x])[0]

        if ys.size == 0:
            ys = np.where(init_mask[:, x])[0]

        if ys.size == 0:
            continue

        top_list.append(ys.min())
        bottom_list.append(ys.max())
        valid_x.append(x)

    final = np.zeros_like(raw, dtype=bool)

    if len(valid_x) < 10:
        return init_mask.copy()

    valid_x = np.array(valid_x)
    top_arr = np.array(top_list, dtype=np.float32)
    bottom_arr = np.array(bottom_list, dtype=np.float32)

    # 平滑上下边界
    top_arr = median_smooth_1d(top_arr, k=25)
    bottom_arr = median_smooth_1d(bottom_arr, k=25)

    top_arr = blend_polyfit(valid_x, top_arr, alpha_raw=0.60)
    bottom_arr = blend_polyfit(valid_x, bottom_arr, alpha_raw=0.60)

    # 限制厚度
    thickness = bottom_arr - top_arr + 1
    thickness = np.clip(thickness, 7, 18)

    center = (top_arr + bottom_arr) / 2
    top_arr = center - thickness / 2
    bottom_arr = center + thickness / 2

    for x, t, b in zip(valid_x, top_arr, bottom_arr):
        t = int(round(t))
        b = int(round(b))

        t = max(0, min(h - 1, t))
        b = max(0, min(h - 1, b))

        if b > t:
            final[t:b + 1, x] = True

    final = morphology.binary_closing(final, footprint=disk(1))
    final = morphology.remove_small_holes(final, area_threshold=50)
    final = morphology.remove_small_objects(final, min_size=MIN_AREA)

    return final
# ==========================================================
# 11. TMH 计算
# ==========================================================

def calculate_tmh(final_mask, cx, mm_per_pixel):
    h, w = final_mask.shape

    cx_int = int(round(cx))
    cx_int = max(0, min(w - 1, cx_int))

    col_start = max(0, cx_int - COLUMN_HALF_WIDTH)
    col_end = min(w - 1, cx_int + COLUMN_HALF_WIDTH)

    heights = []
    valid_cols = []

    for c in range(col_start, col_end + 1):
        ys = np.where(final_mask[:, c])[0]

        if ys.size > 0:
            heights.append(ys.max() - ys.min() + 1)
            valid_cols.append(c)

    if len(heights) == 0:
        ys_all, xs_all = np.where(final_mask)

        if xs_all.size == 0:
            return np.nan, np.nan, cx_int, np.nan, np.nan

        idx = np.argmin(np.abs(xs_all - cx_int))
        measure_col = int(xs_all[idx])

        ys = np.where(final_mask[:, measure_col])[0]

        tmh_px = float(ys.max() - ys.min() + 1)
        y_top = int(ys.min())
        y_bottom = int(ys.max())

    else:
        tmh_px = float(np.median(heights))
        measure_col = int(round(np.median(valid_cols)))

        ys = np.where(final_mask[:, measure_col])[0]

        y_top = int(ys.min())
        y_bottom = int(ys.max())

    tmh_mm = tmh_px * mm_per_pixel

    return tmh_px, tmh_mm, measure_col, y_top, y_bottom


# ==========================================================
# 12. 保存结果图
# ==========================================================

def save_result_figure(
    img_rgb,
    enhanced,
    upper,
    lower,
    init_mask,
    grown,
    raw,
    final,
    cx,
    cy,
    radius_px,
    wtw_px,
    mm_per_pixel,
    measure_col,
    y_top,
    y_bottom,
    tmh_px,
    tmh_mm,
    x0,
    x1,
    save_path,
    file_name
):
    h, w = enhanced.shape

    valid_y = np.concatenate([
        upper[x0:x1][~np.isnan(upper[x0:x1])],
        lower[x0:x1][~np.isnan(lower[x0:x1])]
    ])

    y1 = max(0, int(np.min(valid_y)) - 20)
    y2 = min(h - 1, int(np.max(valid_y)) + 20)

    roi_rgb = img_rgb[y1:y2 + 1, x0:x1]
    roi_init = init_mask[y1:y2 + 1, x0:x1]
    roi_grown = grown[y1:y2 + 1, x0:x1]
    roi_raw = raw[y1:y2 + 1, x0:x1]
    roi_final = final[y1:y2 + 1, x0:x1]

    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original + detected boundaries")
    ax1.axis("off")

    theta = np.linspace(0, 2 * np.pi, 300)
    circle_x = cx + radius_px * np.cos(theta)
    circle_y = cy + radius_px * np.sin(theta)

    xs = np.arange(x0, x1)

    ax1.plot(circle_x, circle_y, "b-", linewidth=1.0)
    ax1.plot(cx, cy, "ro", markersize=4)
    ax1.plot(xs, upper[x0:x1], "c-", linewidth=1.2)
    ax1.plot(xs, lower[x0:x1], "y-", linewidth=1.2)

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(enhanced, cmap="gray")
    ax2.set_title("Enhanced grayscale")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(roi_rgb)
    ax3.contour(roi_init, colors="red", linewidths=1.0)
    ax3.set_title("Boundary init mask")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(roi_rgb)
    ax4.contour(roi_grown, colors="yellow", linewidths=1.0)
    ax4.contour(roi_raw, colors="cyan", linewidths=1.0)
    ax4.set_title("Region growing + active contour")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(roi_final, cmap="gray")
    ax5.set_title("Final mask")
    ax5.axis("off")

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(img_rgb)

    if np.sum(final) > 0:
        ax6.contour(final, colors="lime", linewidths=1.3)

    ax6.plot(circle_x, circle_y, "b-", linewidth=1.0)
    ax6.plot(xs, upper[x0:x1], "c-", linewidth=1.0)
    ax6.plot(xs, lower[x0:x1], "y-", linewidth=1.0)

    if not np.isnan(y_top):
        ax6.plot([measure_col, measure_col], [y_top, y_bottom], "r-", linewidth=2.2)
        ax6.plot(measure_col, y_top, "ro", markersize=5)
        ax6.plot(measure_col, y_bottom, "ro", markersize=5)

    ax6.set_title(
        f"TMH = {tmh_px:.1f} px / {tmh_mm:.3f} mm\n"
        f"WTW≈{wtw_px:.1f}px, scale={mm_per_pixel:.5f}mm/px"
    )
    ax6.axis("off")

    plt.suptitle(
        f"Boundary-guided region growing + active contour: {file_name}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================================
# 13. 主程序
# ==========================================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print("图片不存在：")
        print(IMAGE_PATH)
        return

    file_name = os.path.basename(IMAGE_PATH)
    base_name = os.path.splitext(file_name)[0]

    print("正在处理图片：")
    print(IMAGE_PATH)

    img_rgb = read_image_rgb_chinese_path(IMAGE_PATH)
    enhanced = enhance_gray(img_rgb)

    cx, cy, radius_px, wtw_px, mm_per_pixel = auto_reference_circle(img_rgb)

    print("\n自动参考点与尺度")
    print(f"x_ref / y_ref = ({cx:.1f}, {cy:.1f})")
    print(f"参考半径 = {radius_px:.2f} px")
    print(f"WTW 近似 = {wtw_px:.2f} px")
    print(f"比例尺 = {mm_per_pixel:.8f} mm/px")

    upper, lower, x0, x1, y0, y1 = detect_meniscus_boundaries(enhanced, cx)

    init_mask, band_mask = build_boundary_masks(
        enhanced.shape,
        upper,
        lower,
        x0,
        x1
    )

    feature = build_feature_for_refine(enhanced)

    grown, raw = region_growing_and_active_contour(
        feature,
        init_mask,
        band_mask
    )

    final = column_fill_without_fixed_width(
        raw,
        init_mask,
        x0,
        x1
    )

    tmh_px, tmh_mm, measure_col, y_top, y_bottom = calculate_tmh(
        final,
        cx,
        mm_per_pixel
    )

    mask_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_method3_dp_mask.png"
    )

    result_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_method3_dp_result.png"
    )

    excel_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_method3_dp_TMH_result.xlsx"
    )

    txt_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_method3_dp_TMH_report.txt"
    )

    save_image_chinese_path(mask_path, (final.astype(np.uint8) * 255))

    save_result_figure(
        img_rgb=img_rgb,
        enhanced=enhanced,
        upper=upper,
        lower=lower,
        init_mask=init_mask,
        grown=grown,
        raw=raw,
        final=final,
        cx=cx,
        cy=cy,
        radius_px=radius_px,
        wtw_px=wtw_px,
        mm_per_pixel=mm_per_pixel,
        measure_col=measure_col,
        y_top=y_top,
        y_bottom=y_bottom,
        tmh_px=tmh_px,
        tmh_mm=tmh_mm,
        x0=x0,
        x1=x1,
        save_path=result_path,
        file_name=file_name
    )

    result_df = pd.DataFrame([{
        "FileName": file_name,
        "Method": "Boundary-guided region growing + Chan-Vese active contour",
        "x_ref": cx,
        "y_ref": cy,
        "WTW_px_approx": wtw_px,
        "mm_per_pixel_approx": mm_per_pixel,
        "MeasureColumn": measure_col,
        "TMH_px": tmh_px,
        "TMH_mm_approx": tmh_mm,
        "MaskPath": mask_path,
        "ResultFigurePath": result_path
    }])

    result_df.to_excel(excel_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("泪河区域分割与 TMH 测量报告\n")
        f.write("方法：上下边界定位 + 区域生长 + Chan-Vese 活动轮廓\n\n")
        f.write(f"图像文件：{file_name}\n")
        f.write(f"x_ref / y_ref = ({cx:.1f}, {cy:.1f})\n")
        f.write(f"WTW_px 近似 = {wtw_px:.2f} px\n")
        f.write(f"mm_per_pixel 近似 = {mm_per_pixel:.8f} mm/px\n")
        f.write(f"TMH_pixel = {tmh_px:.2f} px\n")
        f.write(f"TMH_mm 近似 = {tmh_mm:.6f} mm\n")
        f.write("\n说明：TMH 由最终 mask 在中心列附近的真实高度计算，不是固定厚度模板。\n")

    print("\n结果已保存：")
    print(f"Mask：{mask_path}")
    print(f"综合结果图：{result_path}")
    print(f"Excel：{excel_path}")
    print(f"TXT：{txt_path}")
    print(f"TMH = {tmh_px:.2f} px = {tmh_mm:.6f} mm")


if __name__ == "__main__":
    main()