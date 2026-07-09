# UNet 系列代码

## 1. 文件夹结构

```text
UNet系列/
  01_source_code/
    prepare_deep_data.py
    train_unet.py
    predict_unet.py
    eval_postprocess_unet.py
    compare_with_original_labels.py
    train_roi.py / predict_roi.py
    models/
    utils/
    splits/

  02_model_weights/
    scheme1_multitask_unet_best_model.pth
    scheme2_roi_unet_best_model.pth

  03_results_summary/
    三方案结果对比_当前结论.md
    scheme1_history.csv
    scheme2_roi_history.csv
    方案3_后处理强化指标/
    毫米换算与三级分类_三方案对比/

  04_postprocess_review/
    README_边界样本复核与后处理调参.md
    后处理参数搜索/
    边界样本清单/


  README.md
  requirements.txt
  FILE_MANIFEST.md
```

## 2. 最终采用的方案

最终以“方案3：一阶段多任务 U-Net + 后处理强化”为主方案。

整体流程为：

```text
原始眼表图像
  -> Label 自动拆分为参考点与泪河区域
  -> 多任务 U-Net 同时预测 point heatmap 和 meniscus mask
  -> 预测后处理：参考点定位、泪河 mask 连通性修正、测高窗口计算
  -> TMH_px
  -> 映射回原始 Label 尺度后按 86 px = 1 mm 换算
  -> TMH_mm
  -> 三级分类：低泪河 / 正常泪河 / 较高泪河
```

方案3没有重新训练新的网络权重，而是在一阶段多任务 U-Net 的基础上做更稳定的推理后处理。这样做的原因是：一阶段 U-Net 的泪河 Dice/IoU 已经较稳定，主要问题集中在参考点定位、局部断裂、测高窗口和边界样本分类。后处理强化主要解决这些几何和测量层面的误差。

## 3. 环境依赖

建议使用 Python 3.10 或 3.11。安装依赖：

```powershell
pip install -r requirements.txt
```

如果只运行 Streamlit Demo，也可以进入 `05_demo_app` 后使用：

```powershell
pip install -r requirements_demo.txt
streamlit run app.py
```

## 4. 数据放置方式

本提交包没有复制完整原始数据集。运行训练或评估时，需要把数据路径指向原项目中的数据目录，例如：


## 5. 主要代码说明

### 5.1 数据准备

`01_source_code/prepare_deep_data.py`

作用：

- 读取原始图像和 Label。
- 自动把 Label 拆分为参考点区域和泪河区域。
- 生成 point mask、point heatmap、meniscus mask。
- 生成 QA overlay，方便检查标签拆分是否正确。

示例：

```powershell
cd 01_source_code
python prepare_deep_data.py ^
  --data_root "D:\桌面文件\大三下\医学图像处理\大作业\data" ^
  --out_dir processed
```

### 5.2 一阶段多任务 U-Net 训练

`01_source_code/train_unet.py`

作用：

- 输入 RGB 图像。
- 使用多任务 U-Net。
- `point_head` 输出参考点 heatmap。
- `meniscus_head` 输出泪河区域 mask。
- 保存最佳模型权重和训练日志。

示例：

```powershell
python train_unet.py ^
  --processed_dir processed ^
  --splits_dir splits ^
  --out_dir results/unet_stage1 ^
  --epochs 50 ^
  --batch_size 2
```

已整理的模型权重位于：

```text
02_model_weights/scheme1_multitask_unet_best_model.pth
```

### 5.3 二阶段 ROI U-Net 对照

`01_source_code/train_roi.py`、`01_source_code/predict_roi.py`

作用：

- 将下眼睑区域裁剪为 ROI。
- 使用二阶段 U-Net 对泪河区域做精修。
- 本项目中该方案作为对照方案保存，验证集表现没有稳定超过方案1/方案3，因此未作为最终主方案。

已整理的模型权重位于：

```text
02_model_weights/scheme2_roi_unet_best_model.pth
```

### 5.4 U-Net 预测

`01_source_code/predict_unet.py`

作用：

- 加载训练好的多任务 U-Net 权重。
- 输出参考点、泪河 mask、合并 mask 和叠加图。
- 可用于验证集或测试集推理。

示例：

```powershell
python predict_unet.py ^
  --processed_dir processed ^
  --splits_dir splits ^
  --checkpoint "..\02_model_weights\scheme1_multitask_unet_best_model.pth" ^
  --out_dir results/scheme3_prediction ^
  --split test
```

### 5.5 后处理强化与参数搜索

`01_source_code/eval_postprocess_unet.py`

作用：

- 基于已保存的 U-Net 预测结果做推理后处理。
- 调整参考点半径、泪河连通性、水平闭运算和测高窗口。
- 重新计算 TMH_px。
- 与原始 Label 重新计算出的 GT_TMH_px 对比。

方案3的核心不是重新训练，而是把 U-Net 输出转为更稳定的测高结果。后处理阶段重点处理：

- 圆点定位偏移。
- 泪河 mask 局部断裂。
- 测高窗口附近无效列。
- 分类阈值附近样本对 1-2 px 误差敏感的问题。

边界样本复核与后处理调参记录位于：

```text
04_postprocess_review/
```

其中最佳搜索配置记录为：

```text
追加水平闭运算 close = 11
测高窗口 window = 15
```

### 5.6 原始 Label 尺度评估

`01_source_code/compare_with_original_labels.py`

作用：

- 将预测结果与原始 `data/Colour*/Label` 对齐。
- 计算 Dice、IoU、point error、TMH pixel error。
- 后续毫米换算和三级分类使用映射回原始 Label 尺度后的结果。

尺度换算原则：

```text
原始 Label 尺度下：86 px = 1 mm
TMH_mm = TMH_px_original_scale / 86
```

分类标准：

```text
TMH_mm <= 0.20        -> 低泪河高度
0.20 < TMH_mm <= 0.27 -> 正常泪河高度
TMH_mm > 0.27         -> 较高泪河高度
```

## 6. 当前主要结果

原始尺度校正版综合结果见：

```text
03_results_summary/毫米换算与三级分类_三方案对比/三方案_综合结论.md
```

核心结果摘录：

| 方案 | 评估集 | 泪河 Dice | 圆点误差(px) | TMH MAE(px) | TMH MAE(mm) | 分类 Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 方案1 一阶段 U-Net | test | 0.8771 | 11.27 | 1.72 | 0.0200 | 0.8661 | 0.8522 |
| 方案2 ROI U-Net | val | 0.8672 | 11.25 | 1.48 | 0.0172 | 0.8772 | 0.8610 |
| 方案3 U-Net + 后处理 | test | 0.8769 | 2.91 | 1.65 | 0.0192 | 0.8819 | 0.8700 |

边界样本复核进一步得到：

| 项目 | 当前方案3 | 后处理搜索最佳 |
|---|---:|---:|
| Accuracy | 0.8819 | 0.8839 |
| Macro F1 | 0.8700 | 0.8724 |
| 正常类 F1 | 0.7734 | 0.7782 |
| 边界样本 Accuracy | 0.5465 | 0.5698 |
| TMH MAE mm | 0.0192 | 0.0190 |

这部分主要说明边界样本和测高窗口对分类结果有影响，作为方案3的补充分析，而不是新的训练模型。

