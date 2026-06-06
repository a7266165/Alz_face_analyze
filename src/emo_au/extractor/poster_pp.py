"""POSTER++ (POSTER V2) emotion 提取器:pyramid_trans_expr2，輸出 7-class emotion 機率（無 AU）。

Reference: Mao et al., "POSTER++: A simpler and stronger facial expression
recognition network", Pattern Recognition 2025. https://github.com/Talented-Q/POSTER_V2
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import logging

from .base import EmoAUExtractor
from ._torch_patch import patched_torch_load
from src.emo_au.extractor.au_config import (
    POSTER_PP_EMOTION_INDEX,
    POSTER_PP_WEIGHTS_DIR,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

# POSTER_V2 repo 路徑（external/emotion/POSTER_V2）
_POSTER_V2_DIR = WEIGHTS_DIR / "POSTER_V2"


class PosterPPExtractor(EmoAUExtractor):
    """輸入 224×224 RGB aligned 臉，softmax 輸出 7-class emotion 機率（無 AU）。"""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._weights_dir = Path(weights_dir or POSTER_PP_WEIGHTS_DIR)
        self._device = device or "cuda"
        self._model = None
        self._transform = None

    @property
    def model_name(self) -> str:
        return "poster_pp"

    @property
    def output_columns(self) -> List[str]:
        return list(POSTER_PP_EMOTION_INDEX.values())

    def _probe(self) -> bool:
        return self._probe_weights(
            self._weights_dir / "poster_pp_rafdb.pth",
            self._weights_dir / "ir50.pth",
            self._weights_dir / "mobilefacenet_model_best.pth.tar",
        )

    def _load(self) -> None:
        """載入 POSTER++ 模型（torch.load 重導 hardcode 權重路徑 + pickle 訓練類別 shim）。"""
        # 把 POSTER_V2 repo 加入 sys.path 以便 import
        poster_v2_str = str(_POSTER_V2_DIR)
        if poster_v2_str not in sys.path:
            sys.path.insert(0, poster_v2_str)

        # PosterV2_7cls.py hardcode 了 ir50 / mobilefacenet 權重路徑，建構模型時重導。
        # weights_only=False 以 setdefault 帶入（第三方載入未指定，等同強制）。
        redirect = {
            "mobilefacenet_model_best.pth.tar": str(
                self._weights_dir / "mobilefacenet_model_best.pth.tar"
            ),
            "ir50.pth": str(self._weights_dir / "ir50.pth"),
        }
        with patched_torch_load(redirect,
                                map_location=lambda storage, loc: storage,
                                weights_only=False):
            # 避免重複 reload — 只在首次 import 時 patch
            if "models.PosterV2_7cls" in sys.modules:
                del sys.modules["models.PosterV2_7cls"]
            from models.PosterV2_7cls import pyramid_trans_expr2
            model = pyramid_trans_expr2(img_size=224, num_classes=7)

        # 載入 task checkpoint
        # checkpoint 包含 RecorderMeter1 等訓練用 class，
        # 需要先把 POSTER_V2 的 main module 注入到 __main__ 讓 pickle 能找到
        checkpoint_path = self._weights_dir / "poster_pp_rafdb.pth"
        logger.info(f"載入 POSTER++ checkpoint: {checkpoint_path}")

        # 建立假的 class 讓 pickle unpickle 不會失敗
        _main = sys.modules.get("__main__")
        _patched_attrs = []
        for cls_name in ("RecorderMeter", "RecorderMeter1"):
            if not hasattr(_main, cls_name):
                # 建立 dummy class
                dummy = type(cls_name, (), {
                    "__init__": lambda self, *a, **kw: None,
                    "__setstate__": lambda self, state: self.__dict__.update(state) if isinstance(state, dict) else None,
                })
                setattr(_main, cls_name, dummy)
                _patched_attrs.append(cls_name)

        try:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
        finally:
            # 清理 dummy classes
            for cls_name in _patched_attrs:
                delattr(_main, cls_name)

        # checkpoint 可能是 DataParallel 格式 (key 帶 "module." 前綴)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.removeprefix("module.")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self._device)
        model.eval()

        self._model = model

        # 推論用 transform（與 POSTER_V2 test 時一致）
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(f"POSTER++ 模型載入完成（device={self._device}）")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """numpy BGR 影像 → 7-class emotion probability。"""
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        # Forward
        with torch.no_grad():
            logits = self._model(tensor)

        # Softmax → probability
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Map index → emotion name
        result = {}
        for idx, emo_name in POSTER_PP_EMOTION_INDEX.items():
            result[emo_name] = float(probs[idx])

        return result
