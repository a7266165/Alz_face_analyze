"""
FER-former Emotion 特徵提取器

使用 FER-former 模型（IR-50 backbone + Vision Transformer）
提取 7-class emotion probability

Reference:
  "FER-former: Multi-modal Transformer for Facial Expression Recognition"
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
    FER_FORMER_DIR,
)

logger = logging.getLogger(__name__)

# FER-former 7-class label order (RAF-DB convention)
FER_FORMER_EMOTION_INDEX = {
    0: "surprise", 1: "fear", 2: "disgust", 3: "happiness",
    4: "sadness", 5: "anger", 6: "neutral",
}


class FERFormerExtractor(BaseAUExtractor):
    """
    FER-former Emotion 提取器

    - Input: 224x224 RGB aligned face image
    - Output: 7-class emotion probability (softmax)
    - 無 AU 輸出
    - 需要 fine-tuned checkpoint
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[Path] = None,
    ):
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._backbone = None
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
        return "fer_former"

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
            logger.warning(
                f"FER-former checkpoint 不存在。請提供 fine-tuned checkpoint 路徑。"
                f"搜尋目錄: {FER_FORMER_DIR}"
            )
            self._available = False
        return self._available

    def _init_model(self):
        if self._model is not None:
            return

        fer_former_dir_str = str(FER_FORMER_DIR)
        if fer_former_dir_str not in sys.path:
            sys.path.insert(0, fer_former_dir_str)

        from networks.model_irse import Backbone
        from f_vit import ViT

        # IR-50 backbone
        backbone = Backbone(input_size=(224, 224), num_layers=50, mode="ir_se")
        backbone = backbone.to(self._device)
        backbone.eval()
        self._backbone = backbone

        # ViT classification head
        vit = ViT(
            patch_size=16, in_c=512, num_classes=7,
            num_patches=9, embed_dim=128, depth=12, num_heads=12,
        )

        # 載入 checkpoint
        if self._checkpoint_path:
            checkpoint = torch.load(
                str(self._checkpoint_path), map_location="cpu", weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            vit.load_state_dict(new_state_dict, strict=False)

        vit = vit.to(self._device)
        vit.eval()
        self._model = vit
        logger.info(f"FER-former 模型載入完成 (device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # Backbone feature extraction
                features = self._backbone(tensor)
                # ViT classification
                logits, _ = self._model(features)

            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            return {FER_FORMER_EMOTION_INDEX[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            logger.debug(f"  FER-former 提取失敗: {e}")
            return None
