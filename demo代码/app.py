from __future__ import annotations

import math
from pathlib import Path

import streamlit as st
from PIL import Image

from inference_core import (
    DEFAULT_MODEL_PATH,
    HORIZONTAL_CLOSING,
    INFER_SIZE,
    MENISCUS_THRESHOLD,
    POINT_METHOD,
    POINT_PEAK_RATIO,
    POINT_RADIUS,
    TMH_WINDOW,
    load_model,
    run_inference,
)


APP_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = APP_DIR / "examples"


def example_files() -> list[Path]:
    if not EXAMPLE_DIR.exists():
        return []
    return sorted([p for p in EXAMPLE_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str, device_name: str):
    return load_model(model_path, device_name)


def finite_text(value: float, suffix: str, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f} {suffix}"


def main() -> None:
    st.set_page_config(page_title="方案3深度学习Demo", layout="wide")

    st.title("方案3深度学习 Demo：U-Net + 后处理强化")
    st.caption("上传眼表图像后，实际加载训练好的 U-Net 权重，输出参考点、泪河区域、TMH_px、TMH_mm 和三级分类结果。")

    with st.sidebar:
        st.header("模型设置")
        model_path = st.text_input("模型权重路径", value=str(DEFAULT_MODEL_PATH))
        device_name = st.selectbox("运行设备", ["auto", "cpu", "cuda"], index=0)
        st.divider()
        st.markdown("**方案3后处理参数**")
        st.write(f"推理尺寸: {INFER_SIZE[0]} x {INFER_SIZE[1]}")
        st.write(f"参考点定位: {POINT_METHOD}, peak ratio = {POINT_PEAK_RATIO}")
        st.write(f"参考点半径: {POINT_RADIUS}")
        st.write(f"泪河阈值: {MENISCUS_THRESHOLD}")
        st.write(f"水平闭运算: {HORIZONTAL_CLOSING}")
        st.write(f"测高窗口: {TMH_WINDOW}")

    uploaded = st.file_uploader("上传眼表图像", type=["png", "jpg", "jpeg", "bmp"])
    examples = example_files()
    selected = None
    if examples:
        query_example = st.query_params.get("example", "")
        names = [p.name for p in examples]
        default_index = names.index(query_example) + 1 if query_example in names else 0
        choice = st.selectbox("或选择内置示例", ["不使用示例"] + names, index=default_index)
        if choice != "不使用示例":
            selected = EXAMPLE_DIR / choice

    if uploaded is None and selected is None:
        st.info("请上传图片，或选择一张内置示例。")
        return

    if uploaded is not None:
        source_name = uploaded.name
        original = Image.open(uploaded).convert("RGB")
    else:
        source_name = selected.name
        original = Image.open(selected).convert("RGB")

    try:
        model, device = cached_model(model_path, device_name)
    except Exception as exc:
        st.error(f"模型加载失败: {exc}")
        st.stop()

    with st.spinner("正在运行 U-Net 推理和方案3后处理..."):
        try:
            result = run_inference(model, device, original)
        except Exception as exc:
            st.error(f"推理失败: {exc}")
            st.stop()

    st.subheader(f"输入图像: {source_name}")
    st.caption(f"运行设备: {device}。输入会统一 resize 到 {INFER_SIZE[0]} x {INFER_SIZE[1]} 后送入模型。")

    c1, c2, c3, c4 = st.columns(4)
    c1.image(original, caption=f"原图 ({original.width} x {original.height})", width="stretch")
    c2.image(result.mask_image, caption="泪河 mask", width="stretch")
    c3.image(result.overlay_image, caption="红点 + 绿色泪河 + 黄色 TMH", width="stretch")
    c4.image(result.result_card, caption="分类结果卡片", width="stretch")

    st.subheader("数值结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TMH_px", finite_text(result.tmh_px, "px", 2))
    m2.metric("TMH_mm", finite_text(result.tmh_mm, "mm", 4))
    m3.metric("三级分类", result.class_name)
    m4.metric("px_per_mm", f"{result.px_per_mm:.2f}")

    with st.expander("中间结果与定位信息"):
        st.write(f"参考点坐标: x = {result.point_x:.2f}, y = {result.point_y:.2f}")
        st.write(f"有效测高列数: {result.tmh['valid_columns']}")
        st.write(f"上边界 y: {result.tmh['y_upper']:.2f}")
        st.write(f"下边界 y: {result.tmh['y_lower']:.2f}")
        st.write("毫米换算: px_per_mm = 86 x 480 / 1024 = 40.31")
        st.write("分类规则: TMH_mm <= 0.20 为低泪河高度；0.20 < TMH_mm <= 0.27 为正常泪河高度；TMH_mm > 0.27 为较高泪河高度。")


if __name__ == "__main__":
    main()
