"""
EmoNeXt Emotion 特徵提取器

使用 EmoNeXt 模型（ConvNeXt backbone + STN + Self-Attention）
提取 7-class emotion probability

Reference:
  "EmoNeXt: an Efficient and Explainable Emotion Recognition Model
  using ConvNeXt"
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
    EMONEXT_DIR,
)

logger = logging.getLogger(__name__)

# EmoNeXt 7-class label order (RAF-DB convention)
EMONEXT_EMOTION_INDEX = {
    0: "surprise", 1: "fear", 2: "disgust", 3: "happiness",
    4: "sadness", 5: "anger", 6: "neutral",
}


class EmoNeXtExtractor(BaseAUExtractor):
    """
    EmoNeXt Emotion 提取器

    - Input: 224x224 RGB aligned face image
    - Output: 7-class emotion probability (softmax)
    - 無 AU 輸出
    - 需要 fine-tuned checkpoint（backbone 使用 Facebook ConvNeXt pretrained）
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[Path] = None,
        model_size: str = "tiny",
    ):
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._model_size = model_size
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
        return "emonext"

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
        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            self._available = True
        else:
            # 沒有 fine-tuned checkpoint 就不可用
            logger.warning(
                f"EmoNeXt checkpoint 不存在。請提供 fine-tuned checkpoint 路徑。"
                f"搜尋目錄: {EMONEXT_DIR}"
            )
            self._available = False
        return self._available

    def _init_model(self):
        if self._model is not None:
            return

        emonext_dir_str = str(EMONEXT_DIR)
        if emonext_dir_str not in sys.path:
            sys.path.insert(0, emonext_dir_str)
        from models import get_model

        model = get_model(num_classes=7, model_size=self._model_size, in_22k=False)

        # 載入 fine-tuned checkpoint（如果有）
        if self._checkpoint_path:
            checkpoint = torch.load(
                str(self._checkpoint_path), map_location="cpu", weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            # 處理 DataParallel 格式
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)

        model = model.to(self._device)
        model.eval()
        self._model = model
        logger.info(f"EmoNeXt 模型載入完成 (size={self._model_size}, device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                _, logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            return {EMONEXT_EMOTION_INDEX[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            logger.debug(f"  EmoNeXt 提取失敗: {e}")
            return None
