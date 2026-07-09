传统阈值方法：代码、方法与结果说明

1. 本阶段目标

本阶段实现传统图像处理方法一：Otsu  自适应阈值分割。

项目总目标是：

1. 从眼部图像中定位 Placido 同心圆中心  瞳孔中心参考点；
2. 从下眼睑附近提取泪河区域；
3. 基于泪河区域计算泪河高度 `TMH_pixel`；
4. 将传统阈值法结果与数据集人工 Label 做定量评价。

2. 预测脚本说明

主脚本：

```text
traditional_threshold.py
```
运行命令：

```powershell
python traditional_threshold_first10.py `
  --processed_subset_dir D桌面文件大三下医学图像处理大作业预处理processed_originalColour1 `
  --out_dir D桌面文件大三下医学图像处理大作业传统方法1 `
  --num_images 50 `
  --point_stage images_gamma `
  --meniscus_stage images_bilateral `
  --overlay_stage images_gamma `
  --method otsu
```

3. 输入图像选择

圆点检测使用：

```text
images_gamma
```

原因是 Gamma + CLAHE 后 Placido 同心圆更加清晰，适合估计圆心。

泪河区域检测使用：

```text
images_bilateral
```

原因是 bilateral 图像保留边缘且不过度增强纹理，相比 `images_gamma` 更不容易被睫毛和皮肤纹理干扰。

最终叠加图背景使用：

```text
images_gamma
```

用于更清楚地展示红色圆点和绿色泪河区域。

4. 方法设计

4.1 pred_point：中心参考点检测

目标是估计 Placido 同心圆中心。

主要步骤：

1. 在图像中心裁剪 ROI；
2. 优先寻找中心区域的内侧暗圆；
3. 使用低灰度百分位阈值提取暗结构；
4. 通过连通域筛选面积、宽高比、位置合理的暗圆；
5. 如果暗圆失败，则 fallback 到亮环中心估计；
6. 生成半径为 6 像素的圆点 mask。

输出：

```text
pred_pointColour1{stem}.png
```

 4.2 pred_meniscus：泪河区域检测

第三版以后，泪河不再输出单条曲线，而是输出带厚度的区域 mask。

当前采用三层策略：

1. vertical_transition_band
   在下眼睑 ROI 中按列扫描灰度变化，寻找暗到亮的纵向跃迁位置，再围绕跃迁边界生成带状区域。

2. line_structure_band
   对下方 ROI 做 top-hat  black-hat 增强，寻找横向细长结构，再膨胀为窄带区域。

3. dark_component_band_fallback
   当上述方法失败时，尝试从暗眼球区域的下边界推断泪河位置，但会严格拒绝过大的 U 形区域和侧边大块区域。

输出：

```text
pred_meniscusColour1{stem}.png
```

### 4.3 pred_all：合并结果

```text
pred_all = pred_point OR pred_meniscus
```

输出：

```text
pred_allColour1{stem}.png
```


5. TMH_pixel 计算方法

TMH_pixel计算逻辑：

1. 使用圆点的横坐标 `point_cx` 作为参考测量位置；
2. 在 `pred_meniscus` 中取 `point_cx ± 5` 像素范围；
3. 对每一列找到泪河 mask 的最上方和最下方像素；
4. 单列高度为：

```text
height_x = y_max - y_min + 1
```


