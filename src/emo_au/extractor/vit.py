"""ViT (trpakov/vit-face-expression) emotion 提取器:HuggingFace fine-tuned ViT，輸出 7-class emotion 機率。

Reference: https://huggingface.co/trpakov/vit-face-expression
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import logging
from PIL import Image

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import HARMONIZED_EMOTIONS

logger = logging.getLogger(__name__)

# trpakov ViT label order:
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
VIT_LABEL_MAP = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "neutral", 5: "sadness", 6: "surprise",
}


class ViTExtractor(EmoAUExtractor):
    """輸入 aligned 臉，softmax 輸出 7-class emotion 機率（無 AU）；需 transformers，模型自 HuggingFace Hub 下載。"""

    def __init__(self, device: str = "cuda", hub_id: str = "trpakov/vit-face-expression"):
        self._device = device
        self._hub_id = hub_id
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return "vit"

    @property
    def output_columns(self) -> List[str]:
        # 只輸出 7 情緒（harmonized 名稱、無 AU）；extract() 回 name→prob dict。
        return list(HARMONIZED_EMOTIONS)

    def _probe(self) -> bool:
        return self._probe_import("transformers", "pip install transformers")

    def _load(self) -> None:
        """從 HuggingFace Hub 載入 ViT processor + 模型（首次會自動下載）。"""
        from transformers import ViTImageProcessor, ViTForImageClassification
        self._processor = ViTImageProcessor.from_pretrained(self._hub_id)
        self._model = ViTForImageClassification.from_pretrained(
            self._hub_id,
        ).to(self._device)
        self._model.eval()
        logger.info(f"ViT 模型載入完成 (model={self._hub_id}, device={self._device})")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
        return {name: float(probs[idx]) for idx, name in VIT_LABEL_MAP.items()}
