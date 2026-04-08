"""
Py-Feat AU 特徵提取器

使用 py-feat (feat) 套件的 Detector API 提取 AU 與情緒特徵
AU 輸出為 probability [0, 1]，情緒輸出為 probability [0, 1]
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import PYFEAT_AU_MAP, PYFEAT_EMOTION_MAP

logger = logging.getLogger(__name__)


class PyFeatExtractor(BaseAUExtractor):
    """
    Py-Feat AU 特徵提取器

    使用 feat.Detector 進行人臉偵測與 AU/情緒分析
    輸出 12 AU probabilities + 7 emotion probabilities
    """

    def __init__(self, device: str = "cpu", **kwargs):
        self._detector = None
        self._available = None
        self.device = device

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
        """懶載入 Detector"""
        if self._detector is None:
            from feat import Detector
            logger.info("載入 Py-Feat Detector...")
            self._detector = Detector(device=self.device)
            logger.info(f"Py-Feat Detector 載入完成 (device={self.device})")
        return self._detector

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        從單一影像提取 AU 和情緒特徵

        Args:
            image: BGR 格式影像

        Returns:
            特徵字典，失敗返回 None
        """
        if not self.is_available():
            return None

        detector = self._get_detector()

        try:
            # Py-Feat 接受 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect_image(image_rgb)

            if result is None or len(result) == 0:
                return None

            # 取第一張偵測到的臉
            row = result.iloc[0]
            features = {}

            # 提取 AU 欄位
            for pf_col in PYFEAT_AU_MAP:
                if pf_col in result.columns:
                    features[pf_col] = float(row[pf_col])

            # 提取情緒欄位
            for pf_col in PYFEAT_EMOTION_MAP:
                if pf_col in result.columns:
                    features[pf_col] = float(row[pf_col])

            return features if features else None

        except Exception as e:
            logger.debug(f"Py-Feat 提取失敗: {e}")
            return None

    def extract_subject(self, subject_dir: Path) -> Optional[pd.DataFrame]:
        """
        提取受試者所有影像（使用 detect_image 的批次路徑）
        """
        if not self.is_available():
            return None

        image_paths = sorted(
            [p for p in subject_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
            key=lambda p: p.name,
        )

        if not image_paths:
            logger.warning(f"  {subject_dir.name}: 沒有找到影像")
            return None

        detector = self._get_detector()

        results = []
        for img_path in image_paths:
            try:
                det_result = detector.detect_image(str(img_path))

                if det_result is None or len(det_result) == 0:
                    continue

                row = det_result.iloc[0]
                features = {"frame": img_path.stem}

                for pf_col in PYFEAT_AU_MAP:
                    if pf_col in det_result.columns:
                        features[pf_col] = float(row[pf_col])

                for pf_col in PYFEAT_EMOTION_MAP:
                    if pf_col in det_result.columns:
                        features[pf_col] = float(row[pf_col])

                results.append(features)

            except Exception as e:
                logger.debug(f"Py-Feat 處理 {img_path.name} 失敗: {e}")
                continue

        if not results:
            logger.warning(f"  {subject_dir.name}: 沒有成功提取任何幀")
            return None

        df = pd.DataFrame(results)
        cols = ["frame"] + [c for c in df.columns if c != "frame"]
        return df[cols]
