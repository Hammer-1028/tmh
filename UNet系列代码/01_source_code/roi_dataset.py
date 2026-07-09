from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class MeniscusROIDataset(Dataset):
    def __init__(self, roi_dir: Path, split_file: Path, augment: bool = False, seed: int = 42) -> None:
        self.roi_dir = Path(roi_dir)
        self.ids = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _augment(self, image: Image.Image, mask: np.ndarray):
        if self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()
        if self.rng.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(float(self.rng.uniform(0.85, 1.15)))
            image = ImageEnhance.Contrast(image).enhance(float(self.rng.uniform(0.85, 1.15)))
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rel_id = self.ids[idx]
        image = Image.open(self.roi_dir / "images" / f"{rel_id}.png").convert("RGB")
        mask = (np.asarray(Image.open(self.roi_dir / "gt_meniscus" / f"{rel_id}.png").convert("L")) > 127).astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)
        x = np.asarray(image).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return {"id": rel_id, "image": torch.from_numpy(x), "mask": torch.from_numpy(mask[None, ...])}

