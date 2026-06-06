"""EmoNet emotion 提取器:輸出 emotion 機率 + valence/arousal（[-1,1]），支援 5-class / 8-class。

Reference: Toisoul et al., "Estimation of continuous valence and arousal levels
from faces in naturalistic conditions", Nature Machine Intelligence, 2021.
"""

import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch import nn
import logging

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import (
    EMONET_EMOTION_INDEX,
    HARMONIZED_EMOTIONS,
    EMONET_DIR,
    EMONET_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


class EmoNetExtractor(EmoAUExtractor):
    """輸入 256×256 RGB aligned 臉，輸出 harmonized 情緒機率（8-class softmax 去 contempt）+ valence/arousal（[-1,1]）。"""

    # 落地序照 EMONET_EMOTION_INDEX 過濾序（非 harmonized 序）
    _EMOTION_NAMES = [e for e in EMONET_EMOTION_INDEX.values() if e in HARMONIZED_EMOTIONS]
    _NAME_TO_IDX = {n: i for i, n in EMONET_EMOTION_INDEX.items()}

    def __init__(self, device: str = "cuda", n_expression: int = 8):
        self._device = device
        self._n_expression = n_expression
        self._model = None

    @property
    def model_name(self) -> str:
        return "emonet"

    @property
    def output_columns(self) -> List[str]:
        # 落地序照 _EMOTION_NAMES（EMONET_EMOTION_INDEX 過濾序）+ valence/arousal
        return self._EMOTION_NAMES + ["valence", "arousal"]

    def _probe(self) -> bool:
        return self._probe_weights(EMONET_WEIGHTS_DIR / f"emonet_{self._n_expression}.pth")

    def _load(self) -> None:
        """載入 EmoNet 模型權重。"""
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

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
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
        for name in self._EMOTION_NAMES:
            idx = self._NAME_TO_IDX[name]
            if idx < len(probs):
                result[name] = float(probs[idx])

        # Valence and arousal
        result["valence"] = float(output["valence"].clamp(-1.0, 1.0).cpu().item())
        result["arousal"] = float(output["arousal"].clamp(-1.0, 1.0).cpu().item())

        return result
