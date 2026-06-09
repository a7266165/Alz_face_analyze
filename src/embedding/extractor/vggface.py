"""
VGGFace 特徵提取器（DeepFace VGG-Face，輸出 4096 維）。
"""

import os
from typing import Optional
import numpy as np
import logging

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class VGGFaceExtractor(EmbeddingExtractor):
    """
    VGGFace 4096 維人臉特徵提取器
    """

    def __init__(self):
        self._deepface = None

    @property
    def model_name(self) -> str:
        return "vggface"

    @property
    def feature_dim(self) -> int:
        return 4096

    def is_available(self) -> bool:
        try:
            from deepface import DeepFace  # noqa: F401

            return True
        except ImportError:
            logger.debug("deepface 未安裝")
            return False

    def initialize(self) -> None:
        """載入 DeepFace VGG-Face 模型，並以空圖預熱一次。"""
        if self._deepface is not None:
            return
        self._limit_tf_threads()
        from deepface import DeepFace

        # 預熱模型
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            test_img,
            model_name="VGG-Face",
            enforce_detection=False,
        )
        self._deepface = DeepFace

    @staticmethod
    def _limit_tf_threads() -> None:
        """依 EMB_CPU_CORES 限制 TF 的 CPU 執行緒。

        TF 不讀 OMP/TF_NUM_*THREADS，須在任何 op 前呼叫 threading API 才生效，故在 import
        deepface(→TF) 前設定；EMB_CPU_CORES 由 extract.py 的 setup_cpu_limit 匯出。
        """
        cores = int(os.environ.get("EMB_CPU_CORES", "0") or 0)
        if cores <= 0:
            return
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cores)
            tf.config.threading.set_inter_op_parallelism_threads(cores)
            logger.info(f"TF CPU 執行緒上限: intra/inter = {cores}")
        except RuntimeError as e:
            logger.warning(f"無法設定 TF 執行緒上限（TF runtime 已初始化）: {e}")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 VGGFace 特徵 (4096維)

        Args:
            image: BGR 格式的影像

        Returns:
            4096 維特徵向量
        """
        result = self._deepface.represent(
            image,
            model_name="VGG-Face",
            enforce_detection=False,
        )

        if result and len(result) > 0:
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding

        return None
