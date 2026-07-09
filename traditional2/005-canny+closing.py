import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================================================
# 0. 读取图像与预处理
# ==================================================
img_path = r"D:\_湿碎野\大三\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\Original\Color1_000006.PNG"

# 使用 imdecode 完美解决带中文路径的读取问题
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if img is None:
    raise Exception("图像读取失败，请检查路径")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

H, W = gray.shape

# 适度增强灰度图
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# ==================================================
# 1. 瞳孔中心提取分支 (绝对圆心定位)
# ==================================================
pupil_mask = np.zeros_like(gray)
cx, cy = int(W / 2), int(H / 2)
found_pupil = False
best_circularity = -1
pupil_bin_vis = np.zeros_like(gray)

for th in range(15, 65, 5):
    _, thresh = cv2.threshold(enhanced_gray, th, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < 10000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = float(bw) / bh

            if 0.7 < aspect_ratio < 1.4 and circularity > 0.65:
                if circularity > best_circularity:
                    best_circularity = circularity
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        found_pupil = True
                        pupil_bin_vis = thresh.copy()

    if found_pupil and best_circularity > 0.85:
        break

print(f"瞳孔定位报告 —— 坐标: ({cx}, {cy}), 完美圆形度: {best_circularity:.4f}")

# ==================================================
# 2. 虹膜整体半径获取 (物理标尺 radius)
# ==================================================
blur_iris = cv2.GaussianBlur(gray, (15, 15), 0)
_, iris_bin = cv2.threshold(blur_iris, 60, 255, cv2.THRESH_BINARY_INV)

kernel_iris = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
iris_bin = cv2.morphologyEx(iris_bin, cv2.MORPH_CLOSE, kernel_iris)
iris_bin = cv2.morphologyEx(iris_bin, cv2.MORPH_OPEN, kernel_iris)

contours_iris, _ = cv2.findContours(iris_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

radius = 300  # 兜底黑眼珠半径
max_area = 0
best_iris_c = None

for c in contours_iris:
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        best_iris_c = c

if best_iris_c is not None and max_area > 10000:
    (_, _), r = cv2.minEnclosingCircle(best_iris_c)
    radius = int(r)

print(f"虹膜解剖标尺 —— 半径: {radius} px")

# ==================================================
# Step3 构建泪河 ROI  💡
# ==================================================
# 上边缘提拉至 0.55，下边缘收缩至 0.85，打造极窄黄金截带
roi_top = int(cy + radius * 0.53)
roi_bottom = int(cy + radius * 0.90)

# 安全气囊：防越界报错
roi_top = max(0, min(roi_top, H - 10))
roi_bottom = max(roi_top + 5, min(roi_bottom, H))

roi_gray = gray[roi_top:roi_bottom, :]

# ==================================================
# Step4 Canny 与形态学搭桥闭运算
# ==================================================
roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
edges = cv2.Canny(roi_blur, 15, 50)

# 水平闭运算搭桥 (45x3 矩形核)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

# 轻微膨胀确保虚线接合
closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)

# ==================================================
# Step5 填充与连通域几何筛选
# ==================================================
filled = np.zeros_like(closed)
contours_fill, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours_fill:
    if cv2.contourArea(c) > 30:
        cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
tear_roi = np.zeros_like(filled)
best_label = -1
best_score = 0

for i in range(1, num_labels):
    x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
        i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]

    if area < 100: continue

    aspect = w / (h + 1)
    if aspect < 5: continue

    # 核心防线：必须横向穿过绝对精准的中心测量线 cx
    if not (x <= cx <= x + w): continue

    score = w * aspect
    if score > best_score:
        best_score = score
        best_label = i

if best_label > 0:
    tear_roi[labels == best_label] = 255

# ==================================================
# Step6 恢复全图与 TMH 测量计算
# ==================================================
tear_mask = np.zeros((H, W), dtype=np.uint8)
tear_mask[roi_top:roi_bottom, :] = tear_roi

tmh_px = 0
# 在中央测量轴左右各扩展 5 个像素寻找，防止 Canny 细线单像素断裂导致测量归零
search_zone = tear_mask[:, max(0, cx - 5):min(W, cx + 5)]
ys = np.where(search_zone > 0)[0]

if len(ys) > 0:
    tmh_px = np.max(ys) - np.min(ys) + 1

# 物理逻辑闭环：求得的 radius 是黑眼珠物理半径，乘2对应 11.5mm 临床直径公式
iris_diameter_px = radius * 2
tmh_mm = (11.5 * tmh_px / iris_diameter_px) if iris_diameter_px > 0 else 0

print("TMH像素 =", tmh_px)
print("TMH毫米 =", tmh_mm)

# ==================================================
# Step7 结果可视化
# ==================================================
overlay = rgb.copy()
overlay[tear_mask > 0] = [0, 255, 0]
result = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

# 红色瞳孔绝对中心点
cv2.circle(result, (cx, cy), 8, (255, 0, 0), -1)

# 绘制红色外接圆提示（展示计算物理分母的黑眼珠范围）
cv2.circle(result, (cx, cy), radius, (255, 0, 0), 1, lineType=cv2.LINE_AA)

# 垂直测量轴
cv2.line(result, (cx, 0), (cx, H), (255, 0, 0), 2)

if len(ys) > 0:
    cv2.line(result, (cx, np.min(ys)), (cx, np.max(ys)), (255, 255, 0), 4)

# ==================================================
# 显示
# ==================================================
plt.figure(figsize=(16, 10))

plt.subplot(231)
plt.imshow(rgb)
plt.title("原图")

# 展示抓取到的完美正圆瞳孔
plt.subplot(232)
plt.imshow(pupil_bin_vis, cmap="gray")
plt.title("绝对圆心定位")

plt.subplot(233)
plt.imshow(edges, cmap="gray")
plt.title("Canny (解剖级ROI内)")

plt.subplot(234)
plt.imshow(closed, cmap="gray")
plt.title("闭运算搭桥")

plt.subplot(235)
plt.imshow(tear_mask, cmap="gray")
plt.title("泪河Mask")

plt.subplot(236)
plt.imshow(result)
plt.title(f"TMH={tmh_mm:.3f} mm")

plt.tight_layout()
plt.show()