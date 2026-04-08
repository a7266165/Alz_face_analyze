"""
POSTER++ (POSTER V2) Emotion 特徵提取器

使用 POSTER V2 模型（pyramid_trans_expr2）
提取 7-class emotion probability（無 AU 輸出）

Reference:
  Mao et al., "POSTER++: A simpler and stronger facial expression
  recognition network", Pattern Recognition 2025
  https://github.com/Talented-Q/POSTER_V2
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import (
    POSTER_PP_EMOTION_INDEX,
    POSTER_PP_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

# POSTER_V2 repo 路徑
_POSTER_V2_DIR = Path(__file__).resolve().parents[4] / "models" / "POSTER_V2"


class PosterPPExtractor(BaseAUExtractor):
    """
    POSTER++ (POSTER V2) Emotion 提取器

    - Input: 224x224 RGB aligned face image
    - Output: 7-class emotion probability (softmax)
    - 無 AU 輸出
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._weights_dir = Path(weights_dir or POSTER_PP_WEIGHTS_DIR)
        self._device = device or "cuda"
        self._model = None
        self._transform = None
        self._available = None

    @property
    def tool_name(self) -> str:
        return "poster_pp"

    @property
    def au_columns(self) -> List[str]:
        return []

    @property
    def emotion_columns(self) -> List[str]:
        return list(POSTER_PP_EMOTION_INDEX.values())

    @property
    def extra_columns(self) -> List[str]:
        return []

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            checkpoint_path = self._weights_dir / "poster_pp_rafdb.pth"
            ir50_path = self._weights_dir / "ir50.pth"
            mobilefacenet_path = self._weights_dir / "mobilefacenet_model_best.pth.tar"

            missing = []
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
            if not ir50_path.exists():
                missing.append(str(ir50_path))
            if not mobilefacenet_path.exists():
                missing.append(str(mobilefacenet_path))

            if missing:
                logger.warning(
                    f"POSTER++ 權重不存在:\n  " + "\n  ".join(missing)
                )
                self._available = False
            else:
                self._available = True
        except Exception as e:
            logger.warning(f"POSTER++ 可用性檢查失敗: {e}")
            self._available = False
        return self._available

    def _init_model(self):
        """延遲載入 POSTER++ 模型"""
        if self._model is not None:
            return

        # 把 POSTER_V2 repo 加入 sys.path 以便 import
        poster_v2_str = str(_POSTER_V2_DIR)
        if poster_v2_str not in sys.path:
            sys.path.insert(0, poster_v2_str)

        # POSTER_V2 的 PosterV2_7cls.py 中 hardcode 了 weight 路徑，
        # 我們 monkey-patch torch.load 來重定向路徑
        original_torch_load = torch.load
        weight_redirect = {
            "mobilefacenet_model_best.pth.tar": str(
                self._weights_dir / "mobilefacenet_model_best.pth.tar"
            ),
            "ir50.pth": str(self._weights_dir / "ir50.pth"),
        }

        def patched_load(f, *args, **kwargs):
            f_str = str(f)
            for key, redirect in weight_redirect.items():
                if key in f_str:
                    f = redirect
                    break
            kwargs.setdefault("map_location", lambda storage, loc: storage)
            kwargs["weights_only"] = False
            return original_torch_load(f, *args, **kwargs)

        try:
            torch.load = patched_load
            # 避免重複 reload — 只在首次 import 時 patch
            if "models.PosterV2_7cls" in sys.modules:
                del sys.modules["models.PosterV2_7cls"]
            from models.PosterV2_7cls import pyramid_trans_expr2
            model = pyramid_trans_expr2(img_size=224, num_classes=7)
        finally:
            torch.load = original_torch_load

        # 載入 task checkpoint
        # checkpoint 包含 RecorderMeter1 等訓練用 class，
        # 需要先把 POSTER_V2 的 main module 注入到 __main__ 讓 pickle 能找到
        checkpoint_path = self._weights_dir / "poster_pp_rafdb.pth"
        logger.info(f"載入 POSTER++ checkpoint: {checkpoint_path}")

        import sys as _sys
        import types
        # 建立假的 class 讓 pickle unpickle 不會失敗
        _main = _sys.modules.get("__main__")
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
            new_key = k.replace("module.", "") if k.startswith("module.") else k
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

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        對單一影像（numpy BGR）提取 7-class emotion probability

        1. BGR → RGB
        2. Resize 224x224 + ImageNet normalize
        3. Forward pass → 7-dim logits
        4. Softmax → probability
        """
        self._init_model()

        try:
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

        except Exception as e:
            logger.debug(f"  POSTER++ 提取失敗: {e}")
            return None

    def extract_subject(self, subject_dir: Path) -> Optional[pd.DataFrame]:
        """提取受試者所有影像的 emotion 特徵"""
        self._init_model()

        image_paths = sorted(
            [p for p in subject_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
            key=lambda p: p.name,
        )

        if not image_paths:
            logger.warning(f"  {subject_dir.name}: 沒有找到影像")
            return None

        results = []
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            frame_data = self.extract_frame(image)
            if frame_data is not None:
                frame_data["frame"] = img_path.stem
                results.append(frame_data)

        if not results:
            logger.warning(f"  {subject_dir.name}: 沒有成功提取任何幀")
            return None

        df = pd.DataFrame(results)
        cols = ["frame"] + [c for c in df.columns if c != "frame"]
        return df[cols]
