"""對齊人臉影像 + BMI 標籤的 PyTorch Dataset。

每個 visit（ID）在 aligned/{ID}/ 下有 ~10 張臉。訓練：每次 __getitem__ 隨機抽 1 張
（跨 epoch 等於 10× 增強）；評估：攤平所有影像，由呼叫端對人平均。
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BMIFaceDataset(Dataset):
    """訓練用：每個 visit 隨機抽 1 張（無影像的 ID 自動丟棄）。"""

    def __init__(self, ids: List[str], bmi_values: np.ndarray, aligned_dir: Path, transform=None):
        self.ids = list(ids)
        self.bmi = bmi_values.astype(np.float32)
        self.aligned_dir = Path(aligned_dir)
        self.transform = transform

        self._image_paths = {}
        for sid in self.ids:
            pngs = sorted((self.aligned_dir / sid).glob("*.png"))
            if pngs:
                self._image_paths[sid] = pngs

        valid_mask = [sid in self._image_paths for sid in self.ids]
        if sum(valid_mask) < len(self.ids):
            n_miss = len(self.ids) - sum(valid_mask)
            logger.warning(f"{n_miss} IDs have no aligned images, dropping them")
            self.ids = [sid for sid, ok in zip(self.ids, valid_mask) if ok]
            self.bmi = self.bmi[valid_mask]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        paths = self._image_paths[self.ids[idx]]
        img_path = paths[torch.randint(len(paths), (1,)).item()]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.bmi[idx]


class BMIFaceEvalDataset(Dataset):
    """評估用：每個 visit 攤平所有影像（呼叫端按 sid 平均預測）。"""

    def __init__(self, ids: List[str], bmi_values: np.ndarray, aligned_dir: Path, transform=None):
        self.aligned_dir = Path(aligned_dir)
        self.transform = transform

        self._entries = []
        for i, sid in enumerate(ids):
            pngs = sorted((self.aligned_dir / sid).glob("*.png"))
            for p in pngs:
                self._entries.append((sid, p, bmi_values[i]))

        self._sid_to_indices = {}
        for idx, (sid, _, _) in enumerate(self._entries):
            self._sid_to_indices.setdefault(sid, []).append(idx)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        sid, img_path, bmi = self._entries[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, bmi, sid

    @property
    def subject_ids(self):
        return list(self._sid_to_indices.keys())
