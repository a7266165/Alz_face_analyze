"""
TopoFR 特徵提取器

提取 512 維人臉特徵
"""

import sys
from typing import Optional
import numpy as np
import logging

from .base import BaseExtractor
from .feature_extractor import FeatureExtractor
from src.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)


@FeatureExtractor.register("topofr")
class TopoFRExtractor(BaseExtractor):
    """
    TopoFR 特徵提取器

    使用 TopoFR 模型提取 512 維人臉特徵
    """

    def __init__(self):
        self._available = False
        self._model = None
        self._device = None
        self._init_topofr()

    @property
    def model_name(self) -> str:
        return "topofr"

    @property
    def feature_dim(self) -> int:
        return 512

    def is_available(self) -> bool:
        return self._available

    def _init_topofr(self):
        """初始化 TopoFR"""
        try:
            import torch
            self._torch = torch
        except ImportError:
            logger.debug("PyTorch 未安裝")
            return

        try:
            # 嘗試載入 TopoFR
            topofr_path = EXTERNAL_DIR / "TopoFR"

            if not topofr_path.exists():
                logger.debug("TopoFR 路徑不存在")
                return

            sys.path.insert(0, str(topofr_path))
            from backbones import get_model

            # 尋找模型
            model_dir = topofr_path / "model"
            model_files = list(model_dir.glob("*TopoFR*.pt")) + \
                         list(model_dir.glob("*TopoFR*.pth"))

            if not model_files:
                logger.warning(f"TopoFR 模型檔案不存在於: {model_dir}")
                return

            model_path = model_files[0]

            # 判斷架構
            model_name = model_path.stem.upper()
            if "R100" in model_name:
                network = "r100"
            elif "R50" in model_name:
                network = "r50"
            elif "R200" in model_name:
                network = "r200"
            else:
                network = "r100"

            # 載入
            self._device = self._torch.device(
                'cuda' if self._torch.cuda.is_available() else 'cpu'
            )
            self._model = get_model(network, fp16=False)

            checkpoint = self._torch.load(model_path, map_location=self._device)
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get('state_dict', checkpoint)

            self._model.load_state_dict(checkpoint, strict=False)
            self._model.to(self._device).eval()

            self._available = True
            logger.debug(f"TopoFR 使用架構: {network}, 設備: {self._device}")

        except Exception as e:
            logger.warning(f"TopoFR 初始化失敗: {e}")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 TopoFR 特徵 (512維)

        Args:
            image: BGR 格式的影像

        Returns:
            512 維特徵向量
        """
        import cv2

        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = self._torch.from_numpy(img).unsqueeze(0).float()
        img = img.div(255).sub(0.5).div(0.5).to(self._device)

        with self._torch.no_grad():
            embedding = self._model(img)
            embedding = self._torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0].astype(np.float32)
