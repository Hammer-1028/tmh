# 方案3深度学习真实 Demo

这页面会加载训练好的 U-Net 权重，对上传图像重新执行推理、后处理、TMH 测量和三级分类。

## 运行方式

进入本文件夹后安装依赖：

```bash
pip install -r requirements_demo.txt
```

先检查环境：

```bash
python check_environment.py
```

启动界面：

```bash
streamlit run app.py --server.port 8510
```

或双击：

```text
run_demo.bat
```

## 默认模型权重

```text
D:\桌面文件\大三下\医学图像处理\大作业\深度学习_正式优化\正式实验\01_一阶段多任务UNet_已完成\best_model.pth
```

如果权重移动了，可以在 Streamlit 左侧栏修改路径。

## 输出内容

界面输出四部分：

1. 输入原图；
2. 泪河 mask；
3. 红点、绿色泪河、黄色 TMH 测量线叠加图；
4. 分类结果卡片。

同时显示 TMH_px、TMH_mm、px_per_mm 和三级分类结果。

## 分类标准

```text
TMH_mm <= 0.20: 低泪河高度
0.20 < TMH_mm <= 0.27: 正常泪河高度
TMH_mm > 0.27: 较高泪河高度
```

当前输入统一 resize 到 640 x 480，因此：

```text
px_per_mm = 86 x 480 / 1024 = 40.31
```
