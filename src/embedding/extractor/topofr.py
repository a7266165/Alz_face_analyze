"""
TopoFR 特徵提取器（TopoFR 模型，輸出 512 維）。
"""

import sys
from typing import Optional
import numpy as np
import logging

from .base import EmbeddingExtractor
from src.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)


class TopoFRExtractor(EmbeddingExtractor):
    """
    TopoFR 512 維人臉特徵提取器
    """

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = None

    @property
    def model_name(self) -> str:
        return "topofr"

    @property
    def feature_dim(self) -> int:
        return 512

    def _topofr_dir(self):
        return EXTERNAL_DIR / "embedding" / "TopoFR"

    # 顯式 pin 權重（不再用 glob 取第一個）：glob 順序由檔案系統決定、脆弱，換機器/增刪檔
    # 都可能選到別的檢查點；且 backbone 與檔名不符時 load_state_dict(strict=False) 會靜默
    # 丟棄不符層、產生錯誤特徵。要換變體改這裡即可。
    _MODEL_FILE = "Glint360K_R100_TopoFR_9760.pt"

    def _find_model_file(self):
        """external/embedding/TopoFR/model/ 下顯式指定的權重檔（_MODEL_FILE）；無則 None。"""
        path = self._topofr_dir() / "model" / self._MODEL_FILE
        return path if path.exists() else None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.debug("PyTorch 未安裝")
            return False
        return self._topofr_dir().exists() and self._find_model_file() is not None

    def initialize(self) -> None:
        """載入 TopoFR backbone（依權重檔名判斷 r50/r100/r200 架構）。"""
        if self._model is not None:
            return
        import torch

        self._torch = torch

        topofr_path = self._topofr_dir()
        if str(topofr_path) not in sys.path:
            sys.path.insert(0, str(topofr_path))
        from backbones import get_model

        model_path = self._find_model_file()
        if model_path is None:
            raise FileNotFoundError(f"TopoFR 模型檔案不存在於: {topofr_path / 'model'}")

        # 由檔名判斷架構
        weight_stem = model_path.stem.upper()
        if "R100" in weight_stem:
            network = "r100"
        elif "R50" in weight_stem:
            network = "r50"
        elif "R200" in weight_stem:
            network = "r200"
        else:
            network = "r100"
            logger.warning(
                "無法從檔名推斷 TopoFR 架構，預設 r100（若權重非 r100，"
                "strict=False 會靜默丟棄不符層並產生錯誤特徵）: %s",
                model_path.name,
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(network, fp16=False)

        checkpoint = torch.load(model_path, map_location=self._device)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("state_dict", checkpoint)

        model.load_state_dict(checkpoint, strict=False)
        model.to(self._device).eval()
        self._model = model
        logger.debug(f"TopoFR 使用架構: {network}, 設備: {self._device}")

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
