"""LibreFace AU 特徵提取器（單次 get_facial_attributes_image() 呼叫；需 libreface_env）。

輸出 12 AU intensity + 12 AU detection + 情緒 one-hot。
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import logging

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import LIBREFACE_EMOTION_MAP, LIBREFACE_WEIGHTS_DIR

logger = logging.getLogger(__name__)

# attrs["au_intensities"] 的 key → 統一 AU 名稱
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

# attrs["detected_aus"] 的 key → 統一 AU 名稱（二值）
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


class LibreFaceExtractor(EmoAUExtractor):
    """LibreFace AU 特徵提取器。

    單次 get_facial_attributes_image(path) 取得 au_intensities / detected_aus /
    facial_expression 三組結果。情緒只回標籤字串（無 probability），故用 one-hot 編碼。
    """

    def __init__(self, device: str = "cpu"):
        self._available = None
        self._device = device

    @property
    def model_name(self) -> str:
        return "libreface"

    @property
    def output_columns(self) -> List[str]:
        # 落地序照 _do_extract:AU intensity → AU detection → emotion one-hot
        return (list(LIBREFACE_INTENSITY_KEY_MAP.values())
                + list(LIBREFACE_DETECTION_KEY_MAP.values())
                + list(LIBREFACE_EMOTION_MAP.values()))

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

    def initialize(self) -> None:
        """no-op:LibreFace 由套件內部於每次呼叫時自行管理模型載入/快取，無可預載的 handle。

        為對齊三家族契約（建構 → is_available → initialize → extract）保留此方法。
        """

    def extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """從 numpy 影像提取。

        libreface 的 API 是 path-only，故在此將 numpy array 暫存為臨時檔案再餵入
        （temp 檔處理為私有實作細節，不洩漏到契約）。
        用 .png 無損暫存:與「直接讀原始對齊 PNG」像素一致，避免 JPEG 重壓縮改變輸出。
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, image)

        try:
            return self._do_extract(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

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
            emotion_cols = list(LIBREFACE_EMOTION_MAP.values())
            emotion_label = attrs.get("facial_expression", "")
            harmonized_label = LIBREFACE_EMOTION_LABEL_MAP.get(emotion_label, "")
            # 預測落在 harmonized 7 類外（如 Contempt）或無預測時，該列情緒「無法表示」→
            # 填 NaN 視為缺值；不要用 all-zero 偽裝成「確定都不是」而汙染下游均值。
            if harmonized_label in emotion_cols:
                for emo in emotion_cols:
                    result[emo] = 1.0 if emo == harmonized_label else 0.0
            else:
                if emotion_label:
                    logger.debug(f"  LibreFace 情緒標籤 {emotion_label!r} 無對應欄，該列填 NaN")
                for emo in emotion_cols:
                    result[emo] = float("nan")

            return result

        except Exception as e:
            logger.debug(f"  LibreFace 提取失敗 {Path(image_path).name}: {e}")
            return None
