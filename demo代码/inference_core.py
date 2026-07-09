from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


INFER_SIZE = (640, 480)
ORIGINAL_REFERENCE_HEIGHT = 1024
ORIGINAL_PX_PER_MM = 86.0

POINT_METHOD = "weighted"
POINT_PEAK_RATIO = 0.55
POINT_RADIUS = 6
MENISCUS_THRESHOLD = 0.90
LOW_THRESHOLD = 0.0
HORIZONTAL_CLOSING = 5
TMH_WINDOW = 15

DEFAULT_MODEL_PATH = Path(
    r"D:\桌面文件\大三下\医学图像处理\大作业\深度学习_正式优化\正式实验\01_一阶段多任务UNet_已完成\best_model.pth"
)


def norm_layer(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNetMultitask(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.inc = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 12)
        self.up1 = Up(c * 12, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.point_head = nn.Conv2d(c, 1, 1)
        self.meniscus_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return {"point_logits": self.point_head(x), "meniscus_logits": self.meniscus_head(x)}


@dataclass
class Component:
    pixels_y: np.ndarray
    pixels_x: np.ndarray
    area: int
    cy: float
    width: int
    height: int


@dataclass
class InferenceResult:
    processed_image: Image.Image
    point_heat: np.ndarray
    meniscus_prob: np.ndarray
    point_x: float
    point_y: float
    point_mask: np.ndarray
    raw_meniscus_mask: np.ndarray
    meniscus_mask: np.ndarray
    tmh: dict[str, float]
    tmh_px: float
    px_per_mm: float
    tmh_mm: float
    class_name: str
    class_color: str
    mask_image: Image.Image
    overlay_image: Image.Image
    result_card: Image.Image


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[Component]:
    labels, count = ndimage.label(mask.astype(bool))
    comps: list[Component] = []
    for idx in range(1, count + 1):
        ys, xs = np.where(labels == idx)
        if len(xs) < min_area:
            continue
        comps.append(
            Component(
                pixels_y=ys,
                pixels_x=xs,
                area=int(len(xs)),
                cy=float(ys.mean()),
                width=int(xs.max() - xs.min() + 1),
                height=int(ys.max() - ys.min() + 1),
            )
        )
    return comps


def locate_point(heat: np.ndarray, method: str = POINT_METHOD, peak_ratio: float = POINT_PEAK_RATIO) -> tuple[float, float]:
    h, w = heat.shape
    py, px = np.unravel_index(int(np.argmax(heat)), (h, w))
    if method == "argmax":
        return float(px), float(py)

    local_radius = 18
    y1 = max(0, py - local_radius)
    y2 = min(h, py + local_radius + 1)
    x1 = max(0, px - local_radius)
    x2 = min(w, px + local_radius + 1)
    crop = heat[y1:y2, x1:x2]
    keep = crop >= float(crop.max()) * peak_ratio
    weights = np.where(keep, crop, 0.0)
    if weights.sum() <= 0:
        return float(px), float(py)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    cx = float((xx * weights).sum() / weights.sum())
    cy = float((yy * weights).sum() / weights.sum())
    return cx, cy


def close_horizontal(mask: np.ndarray, width: int = HORIZONTAL_CLOSING) -> np.ndarray:
    if width <= 1:
        return mask.astype(bool)
    return ndimage.binary_closing(mask.astype(bool), structure=np.ones((1, width), dtype=bool))


def postprocess_meniscus(mask: np.ndarray) -> np.ndarray:
    h, _w = mask.shape
    comps = connected_components(mask, min_area=20)
    keep: list[Component] = []
    for comp in comps:
        lower = comp.cy > 0.38 * h
        horizontal = comp.width >= max(12, comp.height * 2)
        if lower and horizontal:
            keep.append(comp)
    if not keep:
        keep = [comp for comp in comps if comp.cy > 0.35 * h]
    if not keep:
        return mask.astype(bool)

    best = max(keep, key=lambda comp: comp.area + 10 * comp.width)
    out = np.zeros_like(mask, dtype=bool)
    out[best.pixels_y, best.pixels_x] = True
    return out


def tmh_pixel(mask: np.ndarray, x_ref: float, window: int = TMH_WINDOW) -> dict[str, float]:
    mask = mask.astype(bool)
    h, w = mask.shape
    if not math.isfinite(x_ref):
        return {"tmh_pixel": math.nan, "y_upper": math.nan, "y_lower": math.nan, "valid_columns": 0}
    xc = int(round(x_ref))
    xs = range(max(0, xc - window), min(w, xc + window + 1))
    heights: list[int] = []
    uppers: list[int] = []
    lowers: list[int] = []
    for x in xs:
        ys = np.where(mask[:, x])[0]
        if len(ys) == 0:
            continue
        uppers.append(int(ys.min()))
        lowers.append(int(ys.max()))
        heights.append(int(ys.max() - ys.min() + 1))
    if not heights:
        return {"tmh_pixel": math.nan, "y_upper": math.nan, "y_lower": math.nan, "valid_columns": 0}
    return {
        "tmh_pixel": float(np.median(heights)),
        "y_upper": float(np.median(uppers)),
        "y_lower": float(np.median(lowers)),
        "valid_columns": int(len(heights)),
    }


def px_per_mm_for_image(height: int) -> float:
    return ORIGINAL_PX_PER_MM * height / ORIGINAL_REFERENCE_HEIGHT


def classify_tmh(tmh_mm: float) -> tuple[str, str]:
    if not math.isfinite(tmh_mm):
        return "无法判断", "#64748B"
    if tmh_mm <= 0.20:
        return "低泪河高度", "#2563EB"
    if tmh_mm <= 0.27:
        return "正常泪河高度", "#059669"
    return "较高泪河高度", "#DC2626"


def make_point_mask(shape: tuple[int, int], px: float, py: float, radius: int = POINT_RADIUS) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - px) ** 2 + (yy - py) ** 2 <= radius**2)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_mask_image(mask: np.ndarray) -> Image.Image:
    gray = np.where(mask.astype(bool), 255, 0).astype(np.uint8)
    return Image.fromarray(gray, "L").convert("RGB")


def make_overlay(image: Image.Image, point_mask: np.ndarray, meniscus_mask: np.ndarray, px: float, py: float, tmh: dict[str, float]) -> Image.Image:
    base = image.convert("RGBA")
    arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    arr[meniscus_mask.astype(bool)] = [0, 190, 95, 155]
    arr[point_mask.astype(bool)] = [255, 45, 45, 230]
    overlay = Image.fromarray(arr, "RGBA")
    out = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(out)

    if math.isfinite(px) and math.isfinite(py):
        x = int(round(px))
        y = int(round(py))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 35, 35, 255), width=3)

    if math.isfinite(tmh["y_upper"]) and math.isfinite(tmh["y_lower"]) and math.isfinite(px):
        x = int(round(px))
        y1 = int(round(tmh["y_upper"]))
        y2 = int(round(tmh["y_lower"]))
        draw.line((x, y1, x, y2), fill=(255, 220, 0, 255), width=4)
        draw.line((x - 8, y1, x + 8, y1), fill=(255, 220, 0, 255), width=3)
        draw.line((x - 8, y2, x + 8, y2), fill=(255, 220, 0, 255), width=3)
    return out.convert("RGB")


def make_result_card(tmh_px: float, tmh_mm: float, px_per_mm: float, class_name: str, color: str) -> Image.Image:
    card = Image.new("RGB", (920, 520), "#FFFFFF")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((22, 22, 898, 498), radius=28, fill="#F8FAFC", outline="#D8E3F0", width=3)
    draw.rounded_rectangle((22, 22, 898, 110), radius=28, fill="#14315F")
    draw.rectangle((22, 72, 898, 110), fill="#14315F")

    title_font = get_font(42, bold=True)
    body_font = get_font(34, bold=True)
    small_font = get_font(25)
    draw.text((56, 45), "三级分类结果", font=title_font, fill="#FFFFFF")
    draw.rounded_rectangle((56, 145, 862, 242), radius=24, fill=color)
    label_w = draw.textlength(class_name, font=title_font)
    draw.text(((920 - label_w) / 2, 168), class_name, font=title_font, fill="#FFFFFF")

    tmh_px_text = "TMH_px: --" if not math.isfinite(tmh_px) else f"TMH_px: {tmh_px:.2f} px"
    tmh_mm_text = "TMH_mm: --" if not math.isfinite(tmh_mm) else f"TMH_mm: {tmh_mm:.4f} mm"
    draw.text((72, 292), tmh_px_text, font=body_font, fill="#0F172A")
    draw.text((72, 350), tmh_mm_text, font=body_font, fill="#0F172A")
    draw.text((72, 420), f"当前尺度换算: {px_per_mm:.2f} px/mm", font=small_font, fill="#52677C")
    draw.text((72, 462), "分类阈值: <=0.20 低, 0.20-0.27 正常, >0.27 较高", font=small_font, fill="#52677C")
    return card


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH, device_name: str = "auto") -> tuple[UNetMultitask, torch.device]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在: {path}")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_channels = 16
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        base_channels = int(ckpt["args"].get("base_channels", base_channels))
    model = UNetMultitask(base_channels=base_channels).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, device


@torch.no_grad()
def run_inference(model: UNetMultitask, device: torch.device, original: Image.Image) -> InferenceResult:
    image = original.convert("RGB").resize(INFER_SIZE, Image.Resampling.BILINEAR)
    x = np.asarray(image).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(x).to(device)

    outputs = model(tensor)
    point_heat = torch.sigmoid(outputs["point_logits"])[0, 0].detach().cpu().numpy()
    meniscus_prob = torch.sigmoid(outputs["meniscus_logits"])[0, 0].detach().cpu().numpy()

    px, py = locate_point(point_heat)
    point_mask = make_point_mask(point_heat.shape, px, py)
    raw_mask = meniscus_prob > MENISCUS_THRESHOLD
    raw_mask = close_horizontal(raw_mask, HORIZONTAL_CLOSING)
    meniscus_mask = postprocess_meniscus(raw_mask)

    tmh = tmh_pixel(meniscus_mask, px, TMH_WINDOW)
    tmh_px = float(tmh["tmh_pixel"])
    px_per_mm = px_per_mm_for_image(INFER_SIZE[1])
    tmh_mm = tmh_px / px_per_mm if math.isfinite(tmh_px) else math.nan
    class_name, color = classify_tmh(tmh_mm)
    mask_image = make_mask_image(meniscus_mask)
    overlay = make_overlay(image, point_mask, meniscus_mask, px, py, tmh)
    card = make_result_card(tmh_px, tmh_mm, px_per_mm, class_name, color)

    return InferenceResult(
        processed_image=image,
        point_heat=point_heat,
        meniscus_prob=meniscus_prob,
        point_x=px,
        point_y=py,
        point_mask=point_mask,
        raw_meniscus_mask=raw_mask,
        meniscus_mask=meniscus_mask,
        tmh=tmh,
        tmh_px=tmh_px,
        px_per_mm=px_per_mm,
        tmh_mm=tmh_mm,
        class_name=class_name,
        class_color=color,
        mask_image=mask_image,
        overlay_image=overlay,
        result_card=card,
    )
