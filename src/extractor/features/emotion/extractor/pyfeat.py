"""
Py-Feat AU 特徵提取器

使用 py-feat (feat) 套件的 Detector API 提取 AU 與情緒特徵
AU 輸出為 probability [0, 1]，情緒輸出為 probability [0, 1]
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import PYFEAT_AU_MAP, PYFEAT_EMOTION_MAP

logger = logging.getLogger(__name__)


class PyFeatExtractor(BaseAUExtractor):
    """
    Py-Feat AU 特徵提取器

    使用 feat.Detector 進行人臉偵測與 AU/情緒分析
    輸出 20 AU probabilities + 7 emotion probabilities
    """

    def __init__(self, device: str = "cpu", **kwargs):
        self._detector = None
        self._available = None
        self._device = device

    @property
    def tool_name(self) -> str:
        return "pyfeat"

    @property
    def au_columns(self) -> List[str]:
        return list(PYFEAT_AU_MAP.keys())

    @property
    def emotion_columns(self) -> List[str]:
        return list(PYFEAT_EMOTION_MAP.keys())

    @property
    def extra_columns(self) -> List[str]:
        return []

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from feat import Detector  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("py-feat 未安裝，請執行: pip install py-feat")
            self._available = False
        return self._available

    def _get_detector(self):
        """懶載入 Detector

        Monkey-patch Fex.compute_identities 成 no-op —
        pyfeat 0.6 在 Windows 跑 identity embeddings (FaceNet) 會有 GIL
        violation fatal crash（>5000 張時必現）。我們只要 AU + emotion，
        不需要 identity 聚合。
        """
        if self._detector is None:
            from feat import Detector
            from feat.data import Fex
            if not getattr(Fex.compute_identities, "_patched", False):
                def _noop_compute_identities(self, *args, **kwargs):
                    # inplace=True 時原版返回 None；維持同語義
                    if kwargs.get("inplace", False):
                        return None
                    return self
                _noop_compute_identities._patched = True
                Fex.compute_identities = _noop_compute_identities
            logger.info("載入 Py-Feat Detector（compute_identities 已 no-op）...")
            self._detector = Detector(device=self._device)
            logger.info(f"Py-Feat Detector 載入完成 (device={self._device})")
        return self._detector

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        從單一影像提取 AU 和情緒特徵（需先存為暫存檔）

        Py-Feat detect_image 需要檔案路徑，無法直接接受 numpy array
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, image)

        try:
            return self._do_extract(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _extract_from_path(self, image_path: Path) -> Optional[Dict[str, float]]:
        """直接使用檔案路徑提取，避免不必要的暫存"""
        return self._do_extract(str(image_path))

    def _do_extract(self, image_path: str) -> Optional[Dict[str, float]]:
        """從檔案路徑提取 AU 和情緒特徵"""
        if not self.is_available():
            return None

        detector = self._get_detector()

        try:
            result = detector.detect_image(image_path)

            if result is None or len(result) == 0:
                return None

            row = result.iloc[0]
            features = {}

            for pf_col in PYFEAT_AU_MAP:
                if pf_col in result.columns:
                    features[pf_col] = float(row[pf_col])

            for pf_col in PYFEAT_EMOTION_MAP:
                if pf_col in result.columns:
                    features[pf_col] = float(row[pf_col])

            return features if features else None

        except Exception as e:
            logger.debug(f"Py-Feat 提取失敗: {e}")
            return None
