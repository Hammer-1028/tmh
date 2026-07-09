import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# =========================================================
# 1. 修改这里：你的数据集路径
# =========================================================

IMAGE_DIR = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\Original"
LABEL_DIR = r"D:\医学图像处理\大作业\Open DataSet\Open DataSet2\Colour1\Label"

SAVE_DIR = r"D:\医学图像处理\大作业\Open DataSet\processed"


# =========================================================
# 2. 参数设置
# =========================================================

IMG_WIDTH = 512
IMG_HEIGHT = 256
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

DILATE_KERNEL = 3   # 泪河线加粗程度，建议 3；如果线太细可以改成 5

os.makedirs(SAVE_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(SAVE_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, split, "masks"), exist_ok=True)


# =========================================================
# 3. 支持中文路径的图片读取与保存函数
# =========================================================

def cv_imread_chinese(path, flags=cv2.IMREAD_COLOR):
    """
    解决 OpenCV 在 Windows 下读取中文路径失败的问题
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
        return image
    except Exception:
        return None


def cv_imwrite_chinese(path, image):
    """
    解决 OpenCV 在 Windows 下保存中文路径失败的问题
    """
    try:
        ext = os.path.splitext(path)[1]
        success, encoded_img = cv2.imencode(ext, image)
        if success:
            encoded_img.tofile(path)
            return True
        return False
    except Exception:
        return False


# =========================================================
# 4. 原始图像预处理
# =========================================================

def preprocess_image(image_path, target_size=(512, 256)):
    """
    原始图像预处理：
    1. 读取彩色图像
    2. BGR 转 RGB
    3. resize
    4. 归一化到 [0, 1]
    """
    image = cv_imread_chinese(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"无法读取图像，可能文件损坏或路径异常: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0

    return image


# =========================================================
# 5. Label 预处理
# =========================================================

def process_label(label_path, target_size=(512, 256), dilate_kernel=3):
    """
    label预处理：
    1. 读取label
    2. resize
    3. 二值化
    4. 去掉白色圆点
    5. 保留下方泪河弧线
    6. 适当加粗泪河线条
    """
    label = cv_imread_chinese(label_path, cv2.IMREAD_GRAYSCALE)

    if label is None:
        raise ValueError(f"无法读取label，可能文件损坏或路径异常: {label_path}")

    label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)

    # 二值化：白色区域为 1，黑色背景为 0
    binary = (label > 127).astype(np.uint8)

    # 连通域分析，用来去掉上面的白色圆点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        clean_mask = binary
    else:
        best_idx = None
        best_width = 0

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # 泪河弧线一般是横向最长的连通区域
            # 白色圆点通常宽度小，所以会被排除
            if w > best_width:
                best_width = w
                best_idx = i

        clean_mask = np.zeros_like(binary)

        if best_idx is not None:
            clean_mask[labels == best_idx] = 1

    # 将泪河弧线适当加粗，否则线太细，U-Net训练会比较困难
    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    clean_mask = clean_mask.astype(np.uint8)

    return clean_mask


# =========================================================
# 6. 获取图片路径，支持 PNG / png / jpg 等格式
# =========================================================

def get_image_files(folder):
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files)


# =========================================================
# 7. 根据文件名匹配原图和 label
# =========================================================

def match_image_label(image_dir, label_dir):
    image_paths = get_image_files(image_dir)
    label_paths = get_image_files(label_dir)

    image_dict = {os.path.basename(p): p for p in image_paths}
    label_dict = {os.path.basename(p): p for p in label_paths}

    common_names = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))

    matched_images = [image_dict[name] for name in common_names]
    matched_labels = [label_dict[name] for name in common_names]

    missing_labels = sorted(list(set(image_dict.keys()) - set(label_dict.keys())))
    missing_images = sorted(list(set(label_dict.keys()) - set(image_dict.keys())))

    print("原始图像数量:", len(image_paths))
    print("Label数量:", len(label_paths))
    print("成功匹配数量:", len(common_names))

    if len(missing_labels) > 0:
        print("以下原图没有对应label:")
        for name in missing_labels[:20]:
            print(name)

    if len(missing_images) > 0:
        print("以下label没有对应原图:")
        for name in missing_images[:20]:
            print(name)

    return matched_images, matched_labels


# =========================================================
# 8. 保存预处理后的数据
# =========================================================

def save_data(image_list, label_list, split_name):
    error_files = []

    for img_path, label_path in tqdm(zip(image_list, label_list), total=len(image_list)):
        try:
            image = preprocess_image(img_path, TARGET_SIZE)
            mask = process_label(label_path, TARGET_SIZE, dilate_kernel=DILATE_KERNEL)

            filename = os.path.splitext(os.path.basename(img_path))[0]

            image_npy_path = os.path.join(SAVE_DIR, split_name, "images", filename + ".npy")
            mask_npy_path = os.path.join(SAVE_DIR, split_name, "masks", filename + ".npy")

            image_png_path = os.path.join(SAVE_DIR, split_name, "images", filename + ".png")
            mask_png_path = os.path.join(SAVE_DIR, split_name, "masks", filename + ".png")

            # 保存 npy，用于后续模型训练
            np.save(image_npy_path, image)
            np.save(mask_npy_path, mask)

            # 保存 png，用于人工检查和实验报告展示
            image_png = (image * 255).astype(np.uint8)
            image_png = cv2.cvtColor(image_png, cv2.COLOR_RGB2BGR)

            mask_png = (mask * 255).astype(np.uint8)

            cv_imwrite_chinese(image_png_path, image_png)
            cv_imwrite_chinese(mask_png_path, mask_png)

        except Exception as e:
            print("\n跳过异常文件:")
            print("原图:", img_path)
            print("Label:", label_path)
            print("错误原因:", e)

            error_files.append((img_path, label_path, str(e)))

    print(f"{split_name} 处理完成，异常文件数量: {len(error_files)}")

    return error_files


# =========================================================
# 9. 主程序
# =========================================================

if __name__ == "__main__":

    image_paths, label_paths = match_image_label(IMAGE_DIR, LABEL_DIR)

    assert len(image_paths) == len(label_paths), "匹配后的原图和label数量不一致！"
    assert len(image_paths) > 0, "没有找到任何匹配的图片和label，请检查路径！"

    # 划分训练集、验证集、测试集：70% / 15% / 15%
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        image_paths,
        label_paths,
        test_size=0.3,
        random_state=42
    )

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs,
        temp_labels,
        test_size=0.5,
        random_state=42
    )

    print("训练集数量:", len(train_imgs))
    print("验证集数量:", len(val_imgs))
    print("测试集数量:", len(test_imgs))

    all_errors = []

    all_errors.extend(save_data(train_imgs, train_labels, "train"))
    all_errors.extend(save_data(val_imgs, val_labels, "val"))
    all_errors.extend(save_data(test_imgs, test_labels, "test"))

    # 保存异常文件记录
    error_txt_path = os.path.join(SAVE_DIR, "error_files.txt")

    with open(error_txt_path, "w", encoding="utf-8") as f:
        for img_path, label_path, error_msg in all_errors:
            f.write("原图: " + img_path + "\n")
            f.write("Label: " + label_path + "\n")
            f.write("错误原因: " + error_msg + "\n")
            f.write("-" * 60 + "\n")

    print("全部预处理完成！")
    print("异常文件总数:", len(all_errors))
    print("异常文件记录保存到:", error_txt_path)
    print("预处理结果保存到:", SAVE_DIR)