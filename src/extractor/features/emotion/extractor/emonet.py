"""
EmoNet Emotion 特徵提取器

使用 EmoNet 模型提取 8-class emotion probability + valence/arousal
支援 5-class 和 8-class 兩種模式

Reference:
  Toisoul et al., "Estimation of continuous valence and arousal levels
  from faces in naturalistic conditions", Nature Machine Intelligence, 2021
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch import nn
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import (
    EMONET_EMOTION_INDEX,
    HARMONIZED_EMOTIONS,
    EMONET_DIR,
    EMONET_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


class EmoNetExtractor(BaseAUExtractor):
    """
    EmoNet Emotion 提取器

    - Input: 256x256 RGB aligned face image
    - Output: 7-class emotion probability (softmax, dropping contempt)
              + valence [-1,1] + arousal [-1,1]
    """

    def __init__(self, device: str = "cuda", n_expression: int = 8):
        self._device = device
        self._n_expression = n_expression
        self._model = None
        self._available = None

    @property
    def tool_name(self) -> str:
        return "emonet"

    @property
    def au_columns(self) -> List[str]:
        return []

    @property
    def emotion_columns(self) -> List[str]:
        return list(HARMONIZED_EMOTIONS)

    @property
    def extra_columns(self) -> List[str]:
        return ["valence", "arousal"]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        weight_path = EMONET_WEIGHTS_DIR / f"emonet_{self._n_expression}.pth"
        if not weight_path.exists():
            logger.warning(f"EmoNet 權重不存在: {weight_path}")
            self._available = False
        else:
            self._available = True
        return self._available

    def _init_model(self):
        if self._model is not None:
            return

        emonet_dir_str = str(EMONET_DIR)
        if emonet_dir_str not in sys.path:
            sys.path.insert(0, emonet_dir_str)
        from emonet.models import EmoNet

        weight_path = EMONET_WEIGHTS_DIR / f"emonet_{self._n_expression}.pth"
        state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model = EmoNet(n_expression=self._n_expression).to(self._device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self._model = model
        logger.info(f"EmoNet 模型載入完成 (n_expression={self._n_expression}, device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256))

            # [0, 255] → [0, 1] tensor
            tensor = torch.Tensor(rgb).permute(2, 0, 1).to(self._device) / 255.0

            with torch.no_grad():
                output = self._model(tensor.unsqueeze(0))

            # Emotion probabilities
            expr_logits = output["expression"]
            probs = nn.functional.softmax(expr_logits, dim=1).squeeze(0).cpu().numpy()

            # Map to harmonized emotions (drop contempt if 8-class)
            result: Dict[str, float] = {}
            for idx, emo_name in EMONET_EMOTION_INDEX.items():
                if idx < len(probs) and emo_name in HARMONIZED_EMOTIONS:
                    result[emo_name] = float(probs[idx])

            # Valence and arousal
            result["valence"] = float(output["valence"].clamp(-1.0, 1.0).cpu().item())
            result["arousal"] = float(output["arousal"].clamp(-1.0, 1.0).cpu().item())

            return result
        except Exception as e:
            logger.debug(f"  EmoNet 提取失敗: {e}")
            return None
