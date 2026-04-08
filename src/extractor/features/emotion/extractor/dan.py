"""
DAN (Distract Your Attention Network) Emotion 特徵提取器

使用 DAN 模型提取 7-class emotion probability
RAF-DB trained, ResNet18 backbone + multi-head attention

Reference:
  Wen et al., "Distract Your Attention: Multi-head Cross Attention
  Network for Facial Expression Recognition", 2023
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import (
    HARMONIZED_EMOTIONS,
    DAN_DIR,
    DAN_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


class DANExtractor(BaseAUExtractor):
    """
    DAN Emotion 提取器

    - Input: 224x224 RGB aligned face image
    - Output: 7-class emotion probability (softmax)
    - 無 AU 輸出
    """

    # RAF-DB label order
    RAFDB_INDEX = {
        0: "surprise", 1: "fear", 2: "disgust", 3: "happiness",
        4: "sadness", 5: "anger", 6: "neutral",
    }

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self._available = None

    @property
    def tool_name(self) -> str:
        return "dan"

    @property
    def au_columns(self) -> List[str]:
        return []

    @property
    def emotion_columns(self) -> List[str]:
        return list(HARMONIZED_EMOTIONS)

    @property
    def extra_columns(self) -> List[str]:
        return []

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        checkpoint = DAN_WEIGHTS_DIR / "rafdb_epoch21_acc0.8970_bacc0.8272.pth"
        if not checkpoint.exists():
            logger.warning(f"DAN 權重不存在: {checkpoint}")
            self._available = False
        else:
            self._available = True
        return self._available

    def _init_model(self):
        if self._model is not None:
            return

        dan_dir_str = str(DAN_DIR)
        if dan_dir_str not in sys.path:
            sys.path.insert(0, dan_dir_str)
        from networks.dan import DAN

        model = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint_path = DAN_WEIGHTS_DIR / "rafdb_epoch21_acc0.8970_bacc0.8272.pth"
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(self._device)
        model.eval()
        self._model = model
        logger.info(f"DAN 模型載入完成 (device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                out, _, _ = self._model(tensor)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            return {self.RAFDB_INDEX[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            logger.debug(f"  DAN 提取失敗: {e}")
            return None
