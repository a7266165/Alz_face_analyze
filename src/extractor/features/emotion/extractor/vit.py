"""
ViT (trpakov/vit-face-expression) Emotion 特徵提取器

使用 HuggingFace ViT fine-tuned 模型提取 7-class emotion probability

Reference:
  https://huggingface.co/trpakov/vit-face-expression
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import HARMONIZED_EMOTIONS

logger = logging.getLogger(__name__)

# trpakov ViT label order:
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
VIT_LABEL_MAP = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "neutral", 5: "sadness", 6: "surprise",
}


class ViTExtractor(BaseAUExtractor):
    """
    ViT Emotion 提取器

    - Input: RGB aligned face image
    - Output: 7-class emotion probability (softmax)
    - 無 AU 輸出
    - 需要 transformers pip package
    - 模型自動從 HuggingFace Hub 下載
    """

    def __init__(self, device: str = "cuda", model_name: str = "trpakov/vit-face-expression"):
        self._device = device
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._available = None

    @property
    def tool_name(self) -> str:
        return "vit"

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
        try:
            from transformers import ViTImageProcessor, ViTForImageClassification  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning(
                "transformers 套件未安裝。請執行: pip install transformers"
            )
            self._available = False
        return self._available

    def _init_model(self):
        if self._model is not None:
            return
        from transformers import ViTImageProcessor, ViTForImageClassification
        self._processor = ViTImageProcessor.from_pretrained(self._model_name)
        self._model = ViTForImageClassification.from_pretrained(
            self._model_name,
        ).to(self._device)
        self._model.eval()
        logger.info(f"ViT 模型載入完成 (model={self._model_name}, device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
            return {VIT_LABEL_MAP[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            logger.debug(f"  ViT 提取失敗: {e}")
            return None
