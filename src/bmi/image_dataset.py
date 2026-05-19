"""
PyTorch Dataset for loading aligned face images paired with BMI labels.

Each visit (ID) has ~10 aligned face images under
    workspace/preprocess/aligned/{ID}/*.png

Training mode:  randomly pick 1 of 10 images per __getitem__ call
                (effectively 10× augmentation across epochs).
Eval mode:      return all images stacked; caller averages predictions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
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
    """Single-image-per-visit dataset for training (random pick from 10)."""

    def __init__(
        self,
        ids: List[str],
        bmi_values: np.ndarray,
        aligned_dir: Path,
        transform=None,
    ):
        self.ids = list(ids)
        self.bmi = bmi_values.astype(np.float32)
        self.aligned_dir = Path(aligned_dir)
        self.transform = transform

        self._image_paths = {}
        for sid in self.ids:
            subdir = self.aligned_dir / sid
            if subdir.is_dir():
                pngs = sorted(subdir.glob("*.png"))
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
        sid = self.ids[idx]
        paths = self._image_paths[sid]
        img_path = paths[torch.randint(len(paths), (1,)).item()]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.bmi[idx]


class BMIFaceEvalDataset(Dataset):
    """All-images-per-visit dataset for evaluation (caller averages)."""

    def __init__(
        self,
        ids: List[str],
        bmi_values: np.ndarray,
        aligned_dir: Path,
        transform=None,
    ):
        self.aligned_dir = Path(aligned_dir)
        self.transform = transform

        self._entries = []
        for i, sid in enumerate(ids):
            subdir = self.aligned_dir / sid
            if not subdir.is_dir():
                continue
            pngs = sorted(subdir.glob("*.png"))
            if not pngs:
                continue
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
