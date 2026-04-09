"""
LibreFace AU 特徵提取器

使用 libreface 套件的 get_au_intensities() API
提取 12 AU intensity + 情緒標籤
需要在獨立 conda 環境 (libreface_env) 中執行
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import LIBREFACE_AU_MAP, LIBREFACE_EMOTION_MAP, LIBREFACE_WEIGHTS_DIR

logger = logging.getLogger(__name__)

# LibreFace get_au_intensities() 回傳的 key → 統一 AU 名稱
LIBREFACE_INTENSITY_KEY_MAP: Dict[str, str] = {
    "au_1_intensity": "AU1",
    "au_2_intensity": "AU2",
    "au_4_intensity": "AU4",
    "au_5_intensity": "AU5",
    "au_6_intensity": "AU6",
    "au_9_intensity": "AU9",
    "au_12_intensity": "AU12",
    "au_15_intensity": "AU15",
    "au_17_intensity": "AU17",
    "au_20_intensity": "AU20",
    "au_25_intensity": "AU25",
    "au_26_intensity": "AU26",
}

# LibreFace detect_action_units() 回傳的 key → 統一 AU 名稱（二值）
LIBREFACE_DETECTION_KEY_MAP: Dict[str, str] = {
    "au_1": "AU1_det",
    "au_2": "AU2_det",
    "au_4": "AU4_det",
    "au_6": "AU6_det",
    "au_7": "AU7_det",
    "au_10": "AU10_det",
    "au_12": "AU12_det",
    "au_14": "AU14_det",
    "au_15": "AU15_det",
    "au_17": "AU17_det",
    "au_23": "AU23_det",
    "au_24": "AU24_det",
}

# 情緒標籤 → harmonized 名稱
LIBREFACE_EMOTION_LABEL_MAP: Dict[str, str] = {
    "Anger": "anger",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happiness",
    "Sadness": "sadness",
    "Surprise": "surprise",
    "Neutral": "neutral",
    "Contempt": "contempt",
}


class LibreFaceExtractor(BaseAUExtractor):
    """
    LibreFace AU 特徵提取器

    API:
    - libreface.get_au_intensities(path) → dict, 12 AU intensity [0, ~5]
    - libreface.detect_action_units(path) → dict, 12 AU binary (0/1)
    - libreface.get_facial_expression(path) → str, emotion label

    注意：情緒只回傳標籤字串，不提供 probability，
    因此 harmonized emotion 欄位用 one-hot 編碼
    """

    def __init__(self, device: str = "cpu"):
        self._available = None
        self._device = device

    @property
    def tool_name(self) -> str:
        return "libreface"

    @property
    def au_columns(self) -> List[str]:
        return list(LIBREFACE_INTENSITY_KEY_MAP.values())

    @property
    def emotion_columns(self) -> List[str]:
        return list(LIBREFACE_EMOTION_MAP.values())

    @property
    def extra_columns(self) -> List[str]:
        return list(LIBREFACE_DETECTION_KEY_MAP.values())

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import libreface  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("libreface 未安裝（需在 libreface_env 環境中執行）")
            self._available = False
        return self._available

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """從 numpy 影像提取（需先存為暫存檔）"""
        import tempfile
        import cv2

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
        """從檔案路徑提取所有特徵（單次 API 呼叫）"""
        try:
            import libreface

            attrs = libreface.get_facial_attributes_image(
                image_path, device=self._device,
                weights_download_dir=str(LIBREFACE_WEIGHTS_DIR),
            )
            result = {}

            # AU intensity（12 個，range [0, ~5]）
            intensities = attrs.get("au_intensities", {})
            for lf_key, au_name in LIBREFACE_INTENSITY_KEY_MAP.items():
                result[au_name] = float(intensities.get(lf_key, 0.0))

            # AU detection（12 個，binary 0/1）
            detections = attrs.get("detected_aus", {})
            for lf_key, det_name in LIBREFACE_DETECTION_KEY_MAP.items():
                result[det_name] = float(detections.get(lf_key, 0))

            # 情緒（只有標籤，轉 one-hot）
            emotion_label = attrs.get("facial_expression", "")
            harmonized_label = LIBREFACE_EMOTION_LABEL_MAP.get(emotion_label, "")
            for emo in LIBREFACE_EMOTION_MAP.values():
                result[emo] = 1.0 if emo == harmonized_label else 0.0

            return result

        except Exception as e:
            logger.debug(f"  LibreFace 提取失敗 {Path(image_path).name}: {e}")
            return None
