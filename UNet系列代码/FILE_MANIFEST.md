# 文件清单

## 01_source_code

核心训练、预测、评估和后处理代码：

- `prepare_deep_data.py`：拆分 Label，生成训练数据。
- `dataset.py`：一阶段 U-Net 数据集。
- `train_unet.py`：一阶段多任务 U-Net 训练。
- `predict_unet.py`：一阶段 U-Net 推理。
- `eval_postprocess_unet.py`：后处理参数搜索与评估。
- `compare_with_original_labels.py`：与原始 Label 对齐评估。
- `prepare_roi_data.py`、`roi_dataset.py`、`train_roi.py`、`predict_roi.py`：二阶段 ROI U-Net 对照方案。
- `select_threshold.py`、`select_threshold_roi.py`：阈值选择脚本。
- `models/`：U-Net 网络结构。
- `utils/`：Label 拆分、指标、TMH 测量和可视化工具。
- `splits/`：train/val/test 划分。

## 02_model_weights

- `scheme1_multitask_unet_best_model.pth`：一阶段多任务 U-Net 权重，也是方案3后处理强化的基础权重。
- `scheme2_roi_unet_best_model.pth`：二阶段 ROI U-Net 对照权重。

## 03_results_summary

- 一阶段、二阶段训练日志。
- 方案3原始尺度指标。
- 毫米换算与三级分类结果。
- 三方案综合对比。

## 04_postprocess_review

- 边界样本清单。
- 后处理参数搜索结果。
- 最佳配置逐图结果。
- 边界样本复核说明。
