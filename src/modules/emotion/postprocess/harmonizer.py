"""
AU 特徵 Harmonization

將三套工具的原始輸出統一到相同的量綱與欄位名稱
- 欄位名稱對映：各工具原始欄名 → 統一名稱
- 量綱轉換：OpenFace 3.0 已經是 [0,1]（sigmoid），不需再轉
- 輸出 harmonized 15 維特徵（8 AUs + 7 emotions）
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import logging

from src.modules.emotion.extractor.au_config import (
    HARMONIZED_AUS,
    HARMONIZED_EMOTIONS,
    HARMONIZED_FEATURES,
    OPENFACE_AU_MAP,
    OPENFACE_EMOTION_MAP,
    PYFEAT_AU_MAP,
    PYFEAT_EMOTION_MAP,
    LIBREFACE_AU_MAP,
    LIBREFACE_EMOTION_MAP,
    POSTER_PP_AU_MAP,
    POSTER_PP_EMOTION_MAP,
    FER_AU_MAP,
    FER_EMOTION_MAP,
    DAN_AU_MAP,
    DAN_EMOTION_MAP,
    HSEMOTION_AU_MAP,
    HSEMOTION_EMOTION_MAP,
    VIT_AU_MAP,
    VIT_EMOTION_MAP,
    AU_SCALE_INFO,
    AU_RAW_DIR,
    AU_HARMONIZED_DIR,
)

logger = logging.getLogger(__name__)


class AUHarmonizer:
    """
    AU 特徵 harmonization

    將各工具的原始 AU/情緒輸出統一到 [0, 1] 量綱，
    使用統一的欄位名稱
    """

    AU_MAPS = {
        "openface": OPENFACE_AU_MAP,
        "pyfeat": PYFEAT_AU_MAP,
        "libreface": LIBREFACE_AU_MAP,
        "poster_pp": POSTER_PP_AU_MAP,
        "fer": FER_AU_MAP,
        "dan": DAN_AU_MAP,
        "hsemotion": HSEMOTION_AU_MAP,
        "vit": VIT_AU_MAP,
    }

    EMOTION_MAPS = {
        "openface": OPENFACE_EMOTION_MAP,
        "pyfeat": PYFEAT_EMOTION_MAP,
        "libreface": LIBREFACE_EMOTION_MAP,
        "poster_pp": POSTER_PP_EMOTION_MAP,
        "fer": FER_EMOTION_MAP,
        "dan": DAN_EMOTION_MAP,
        "hsemotion": HSEMOTION_EMOTION_MAP,
        "vit": VIT_EMOTION_MAP,
    }

    def harmonize_subject(
        self, raw_df: pd.DataFrame, tool: str
    ) -> pd.DataFrame:
        """
        Harmonize 單一受試者的所有幀

        Args:
            raw_df: 原始 per-frame DataFrame（來自 raw/{tool}/{subject}.csv）
            tool: 工具名稱

        Returns:
            Harmonized DataFrame，只保留 19 統一特徵 + frame 欄
        """
        au_map = self.AU_MAPS.get(tool, {})
        emotion_map = self.EMOTION_MAPS.get(tool, {})
        scale_info = AU_SCALE_INFO.get(tool, {})

        result = pd.DataFrame()

        # 保留 frame 欄位
        if "frame" in raw_df.columns:
            result["frame"] = raw_df["frame"]

        # 對映並轉換 AU 欄位
        for raw_col, harmonized_name in au_map.items():
            if harmonized_name not in HARMONIZED_AUS:
                continue  # 跳過非共有 AU
            if raw_col in raw_df.columns:
                values = raw_df[raw_col].values.astype(float)
                # 量綱轉換到 [0, 1]
                if scale_info.get("type") == "intensity":
                    max_val = scale_info.get("max", 5.0)
                    values = np.clip(values / max_val, 0.0, 1.0)
                result[harmonized_name] = values
            else:
                result[harmonized_name] = np.nan

        # 對映情緒欄位
        for raw_col, harmonized_name in emotion_map.items():
            if harmonized_name not in HARMONIZED_EMOTIONS:
                continue
            if raw_col in raw_df.columns:
                values = raw_df[raw_col].values.astype(float)
                result[harmonized_name] = np.clip(values, 0.0, 1.0)
            else:
                result[harmonized_name] = np.nan

        # 確保所有統一欄位都存在（缺失的填 NaN）
        for feat in HARMONIZED_FEATURES:
            if feat not in result.columns:
                result[feat] = np.nan

        # 按統一順序排列
        cols = (["frame"] if "frame" in result.columns else []) + HARMONIZED_FEATURES
        return result[cols]

    def harmonize_all(
        self,
        tool: str,
        raw_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> int:
        """
        Harmonize 指定工具的所有受試者

        Args:
            tool: 工具名稱
            raw_dir: 原始檔案目錄（預設 AU_RAW_DIR/{tool}）
            output_dir: 輸出目錄（預設 AU_HARMONIZED_DIR/{tool}）

        Returns:
            處理成功的受試者數量
        """
        if raw_dir is None:
            raw_dir = AU_RAW_DIR / tool
        if output_dir is None:
            output_dir = AU_HARMONIZED_DIR / tool

        output_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            logger.error(f"原始檔案目錄不存在: {raw_dir}")
            return 0

        csv_files = sorted(raw_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"  {tool}: 沒有找到原始 CSV 檔案")
            return 0

        success_count = 0
        for csv_path in csv_files:
            subject_id = csv_path.stem
            output_file = output_dir / f"{subject_id}.csv"

            # checkpoint
            if output_file.exists():
                success_count += 1
                continue

            try:
                raw_df = pd.read_csv(csv_path)
                harmonized_df = self.harmonize_subject(raw_df, tool)
                harmonized_df.to_csv(output_file, index=False, encoding="utf-8-sig")
                success_count += 1
            except Exception as e:
                logger.error(f"  {subject_id} harmonization 失敗: {e}")

        logger.info(f"{tool}: {success_count}/{len(csv_files)} 成功 harmonize")
        return success_count
