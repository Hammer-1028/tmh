from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class TearMeniscusDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        split_file: Path,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.ids = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, rel_id: str) -> Image.Image:
        return Image.open(self.processed_dir / "images" / f"{rel_id}.png").convert("RGB")

    def _load_mask(self, folder: str, rel_id: str) -> np.ndarray:
        return (np.asarray(Image.open(self.processed_dir / folder / f"{rel_id}.png").convert("L")) > 127).astype(np.float32)

    def _load_heatmap(self, rel_id: str) -> np.ndarray:
        return np.load(self.processed_dir / "gt_point_heatmap" / f"{rel_id}.npy").astype(np.float32)

    def _augment(self, image: Image.Image, point: np.ndarray, meniscus: np.ndarray, heatmap: np.ndarray):
        if self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            point = np.fliplr(point).copy()
            meniscus = np.fliplr(meniscus).copy()
            heatmap = np.fliplr(heatmap).copy()
        if self.rng.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(float(self.rng.uniform(0.85, 1.15)))
            image = ImageEnhance.Contrast(image).enhance(float(self.rng.uniform(0.85, 1.15)))
        return image, point, meniscus, heatmap

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rel_id = self.ids[idx]
        image = self._load_image(rel_id)
        point = self._load_mask("gt_point_mask", rel_id)
        meniscus = self._load_mask("gt_meniscus", rel_id)
        heatmap = self._load_heatmap(rel_id)
        if self.augment:
            image, point, meniscus, heatmap = self._augment(image, point, meniscus, heatmap)

        x = np.asarray(image).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return {
            "id": rel_id,
            "image": torch.from_numpy(x),
            "point_mask": torch.from_numpy(point[None, ...]),
            "point_heatmap": torch.from_numpy(heatmap[None, ...]),
            "meniscus": torch.from_numpy(meniscus[None, ...]),
        }

