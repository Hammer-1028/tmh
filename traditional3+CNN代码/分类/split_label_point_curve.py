import os
import cv2
import csv
import numpy as np
from glob import glob


# =====================================================
# 1. 修改这里：Label 文件夹路径
# =====================================================

LABEL_DIR = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\Label"

SAVE_ROOT = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\Label_processed"

os.makedirs(SAVE_ROOT, exist_ok=True)


# =====================================================
# 2. 参数设置
# =====================================================

# 需要和你预处理后的图像大小一致
IMG_WIDTH = 640
IMG_HEIGHT = 480
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# 是否统一 resize 到 640×480
RESIZE_TO_TARGET = True

# 白点是否统一成固定半径圆形区域
# 建议 True，因为白点本质是“定位点”，统一半径更适合后续评价
USE_FIXED_POINT_RADIUS = True

# 白点半径，和你预测代码里的 POINT_RADIUS 保持一致即可
POINT_RADIUS = 10

# 是否加粗泪河弧线
# 如果后续预测曲线比较粗，可以设为 1；如果想保留原始 label，设为 0
CURVE_DILATE_ITER = 0


# =====================================================
# 3. 输出文件夹
# =====================================================

POINT_DIR = os.path.join(SAVE_ROOT, "gt_point")
POINT_RAW_DIR = os.path.join(SAVE_ROOT, "gt_point_raw")
CURVE_DIR = os.path.join(SAVE_ROOT, "gt_curve")
ALL_DIR = os.path.join(SAVE_ROOT, "gt_all")
DEBUG_DIR = os.path.join(SAVE_ROOT, "debug_overlay")

for d in [POINT_DIR, POINT_RAW_DIR, CURVE_DIR, ALL_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)


# =====================================================
# 4. 中文路径读写函数
# =====================================================

def cv_imread_chinese(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def cv_imwrite_chinese(path, img):
    try:
        ext = os.path.splitext(path)[1]
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            encoded_img.tofile(path)
            return True
        return False
    except Exception:
        return False


# =====================================================
# 5. 获取文件
# =====================================================

def get_label_files(folder):
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"]
    files = []

    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))

    # Windows 下 glob 可能大小写重复匹配，这里按绝对路径去重
    files = list(set(os.path.abspath(p) for p in files))

    # 排序，保证顺序稳定
    files = sorted(files)

    return files


# =====================================================
# 6. 连通域圆度计算
# =====================================================

def contour_circularity(component_mask):
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter <= 1e-6:
        return 0.0

    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    return float(circularity)


# =====================================================
# 7. 核心函数：拆分白点和泪河弧线
# =====================================================

def split_label(label_path):
    """
    输入：原始 label，黑底白色标注
    输出：
    point_mask：白点定位区域，固定半径圆
    point_raw：原始白点区域
    curve_mask：泪河弧线区域
    all_mask：白点 + 泪河弧线
    debug：彩色调试图，红色=泪河弧线，绿色=白点
    """

    label = cv_imread_chinese(label_path, cv2.IMREAD_GRAYSCALE)

    if label is None:
        raise ValueError(f"无法读取 label: {label_path}")

    if RESIZE_TO_TARGET:
        label = cv2.resize(label, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # 二值化
    binary = (label > 127).astype(np.uint8)
    h, w = binary.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    point_raw = np.zeros((h, w), dtype=np.uint8)
    point_mask = np.zeros((h, w), dtype=np.uint8)
    curve_mask = np.zeros((h, w), dtype=np.uint8)

    if num_labels <= 1:
        all_mask = np.zeros((h, w), dtype=np.uint8)
        debug = np.zeros((h, w, 3), dtype=np.uint8)
        return point_mask, point_raw, curve_mask, all_mask, debug, "empty", None, None

    # =================================================
    # A. 找泪河弧线：优先选择横向最长、宽高比最大的连通域
    # =================================================

    curve_idx = None
    best_curve_score = -1

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < 3:
            continue

        aspect = bw / (bh + 1e-6)

        # 泪河弧线特点：横向长、纵向薄
        curve_score = bw * 5.0 + aspect * 10.0 + area * 0.03

        if curve_score > best_curve_score:
            best_curve_score = curve_score
            curve_idx = i

    if curve_idx is not None:
        curve_mask[labels == curve_idx] = 255

    # 泪河弧线的位置信息
    if curve_idx is not None:
        curve_x, curve_y, curve_w, curve_h, curve_area = stats[curve_idx]
        curve_cx, curve_cy = centroids[curve_idx]
    else:
        curve_y, curve_cy = h, h

    # =================================================
    # B. 找白点：从“非曲线连通域”中选择最合理的点状区域
    #    注意：这里不要太严格，否则会漏掉很多白点
    # =================================================

    point_idx = None
    best_point_score = -1

    for i in range(1, num_labels):
        if i == curve_idx:
            continue

        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        # 白点可能很小，所以面积下限不要太高
        if area < 1 or area > 2000:
            continue

        aspect = bw / (bh + 1e-6)

        # 点状区域宽高比一般不会太极端
        if aspect < 0.2 or aspect > 5.0:
            continue

        component = np.zeros((h, w), dtype=np.uint8)
        component[labels == i] = 255

        circularity = contour_circularity(component)

        # 白点一般在泪河弧线上方；但不要强制，否则有些图会漏检
        if curve_idx is not None:
            vertical_relation = max(0.0, (curve_cy - cy) / h)
        else:
            vertical_relation = 0.0

        # 点状区域通常面积较小，宽度较小
        size_score = 1.0 / (abs(area - 30) + 10.0)

        # 越接近圆、越在曲线上方、面积越像小点，得分越高
        point_score = (
            circularity * 50.0
            + vertical_relation * 100.0
            + size_score * 200.0
            + area * 0.05
        )

        if point_score > best_point_score:
            best_point_score = point_score
            point_idx = i

    # =================================================
    # C. 如果上面仍然没找到白点，使用兜底策略：
    #    curve_mask 从 binary 中扣掉后，剩下的白色区域里选最大连通域
    # =================================================

    if point_idx is None:
        remain = binary.copy()
        remain[curve_mask > 0] = 0

        num2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(
            remain,
            connectivity=8
        )

        best_idx2 = None
        best_area2 = -1

        for j in range(1, num2):
            x, y, bw, bh, area = stats2[j]

            if area < 1 or area > 2000:
                continue

            aspect = bw / (bh + 1e-6)

            if aspect < 0.2 or aspect > 5.0:
                continue

            if area > best_area2:
                best_area2 = area
                best_idx2 = j

        if best_idx2 is not None:
            point_raw[labels2 == best_idx2] = 255

            cx, cy = centroids2[best_idx2]
            point_center = (float(cx), float(cy))

            if USE_FIXED_POINT_RADIUS:
                cv2.circle(
                    point_mask,
                    (int(round(cx)), int(round(cy))),
                    POINT_RADIUS,
                    255,
                    -1
                )
            else:
                point_mask = point_raw.copy()
        else:
            point_center = None

    else:
        point_raw[labels == point_idx] = 255

        cx, cy = centroids[point_idx]
        point_center = (float(cx), float(cy))

        if USE_FIXED_POINT_RADIUS:
            cv2.circle(
                point_mask,
                (int(round(cx)), int(round(cy))),
                POINT_RADIUS,
                255,
                -1
            )
        else:
            point_mask = point_raw.copy()

    # =================================================
    # D. 合并结果
    # =================================================

    if CURVE_DILATE_ITER > 0:
        kernel_curve = np.ones((3, 3), np.uint8)
        curve_mask = cv2.dilate(curve_mask, kernel_curve, iterations=CURVE_DILATE_ITER)

    all_mask = np.clip(point_mask + curve_mask, 0, 255).astype(np.uint8)

    # =================================================
    # E. debug 图
    # 红色：泪河弧线
    # 绿色：白点
    # =================================================

    debug = np.zeros((h, w, 3), dtype=np.uint8)
    debug[curve_mask > 0] = (0, 0, 255)
    debug[point_mask > 0] = (0, 255, 0)

    status = "ok"

    if point_center is None:
        status = "no_point"

    if curve_idx is None:
        status = "no_curve"

    curve_info = None
    if curve_idx is not None:
        x, y, bw, bh, area = stats[curve_idx]
        curve_info = {
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "area": int(area)
        }

    return point_mask, point_raw, curve_mask, all_mask, debug, status, point_center, curve_info


# =====================================================
# 8. 主程序：批量处理 Label
# =====================================================

if __name__ == "__main__":

    label_files = get_label_files(LABEL_DIR)

    print("Label 数量:", len(label_files))
    print("Label 输入路径:", LABEL_DIR)
    print("保存路径:", SAVE_ROOT)

    summary_path = os.path.join(SAVE_ROOT, "label_split_summary.csv")

    rows = []
    error_files = []

    for label_path in label_files:
        try:
            filename = os.path.basename(label_path)
            stem = os.path.splitext(filename)[0]

            point_mask, point_raw, curve_mask, all_mask, debug, status, point_center, curve_info = split_label(label_path)

            save_name = stem + ".png"

            cv_imwrite_chinese(os.path.join(POINT_DIR, save_name), point_mask)
            cv_imwrite_chinese(os.path.join(POINT_RAW_DIR, save_name), point_raw)
            cv_imwrite_chinese(os.path.join(CURVE_DIR, save_name), curve_mask)
            cv_imwrite_chinese(os.path.join(ALL_DIR, save_name), all_mask)
            cv_imwrite_chinese(os.path.join(DEBUG_DIR, save_name), debug)

            if point_center is not None:
                point_x, point_y = point_center
            else:
                point_x, point_y = "", ""

            if curve_info is not None:
                curve_x = curve_info["x"]
                curve_y = curve_info["y"]
                curve_w = curve_info["w"]
                curve_h = curve_info["h"]
                curve_area = curve_info["area"]
            else:
                curve_x = curve_y = curve_w = curve_h = curve_area = ""

            rows.append({
                "filename": filename,
                "status": status,
                "point_x": point_x,
                "point_y": point_y,
                "curve_x": curve_x,
                "curve_y": curve_y,
                "curve_w": curve_w,
                "curve_h": curve_h,
                "curve_area": curve_area
            })

        except Exception as e:
            print("处理失败:", label_path)
            print("错误原因:", e)
            error_files.append((label_path, str(e)))

    # 保存 summary
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "filename",
            "status",
            "point_x",
            "point_y",
            "curve_x",
            "curve_y",
            "curve_w",
            "curve_h",
            "curve_area"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nLabel 预处理完成！")
    print("成功数量:", len(rows))
    print("失败数量:", len(error_files))
    print("白点 mask:", POINT_DIR)
    print("白点原始区域:", POINT_RAW_DIR)
    print("泪河弧线 mask:", CURVE_DIR)
    print("合并 mask:", ALL_DIR)
    print("调试图:", DEBUG_DIR)
    print("统计表:", summary_path)