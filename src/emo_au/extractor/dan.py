"""DAN (Distract Your Attention Network) emotion 提取器:RAF-DB 訓練、ResNet18 + multi-head attention，輸出 7-class emotion 機率。

Reference: Wen et al., "Distract Your Attention: Multi-head Cross Attention Network
for Facial Expression Recognition", 2023.
"""

import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import logging

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import (
    HARMONIZED_EMOTIONS,
    DAN_DIR,
    DAN_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


class DANExtractor(EmoAUExtractor):
    """輸入 224×224 RGB aligned 臉，softmax 輸出 7-class emotion 機率（無 AU）。"""

    # RAF-DB label order
    RAFDB_INDEX = {
        0: "surprise", 1: "fear", 2: "disgust", 3: "happiness",
        4: "sadness", 5: "anger", 6: "neutral",
    }

    # RAF-DB checkpoint 檔名
    CHECKPOINT_NAME = "rafdb_epoch21_acc0.8970_bacc0.8272.pth"

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

    @property
    def model_name(self) -> str:
        return "dan"

    @property
    def output_columns(self) -> List[str]:
        # 只輸出 7 情緒（harmonized 名稱、無 AU）；extract() 回 name→prob dict。
        return list(HARMONIZED_EMOTIONS)

    def _probe(self) -> bool:
        return self._probe_weights(DAN_WEIGHTS_DIR / self.CHECKPOINT_NAME)

    def _load(self) -> None:
        """載入 DAN 模型（RAF-DB checkpoint）。"""
        dan_dir_str = str(DAN_DIR)
        if dan_dir_str not in sys.path:
            sys.path.insert(0, dan_dir_str)
        from networks.dan import DAN

        model = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint_path = DAN_WEIGHTS_DIR / self.CHECKPOINT_NAME
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(self._device)
        model.eval()
        self._model = model
        logger.info(f"DAN 模型載入完成 (device={self._device})")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out, _, _ = self._model(tensor)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
        return {name: float(probs[i]) for i, name in self.RAFDB_INDEX.items()}
