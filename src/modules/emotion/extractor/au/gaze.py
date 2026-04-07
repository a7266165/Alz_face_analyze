"""
Gaze 特徵提取（OpenFace 3.0）

從 OpenFace raw output 提取 gaze yaw/pitch，
計算時序統計量作為 subject-level gaze 特徵
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.modules.emotion.extractor.au_config import (
    OPENFACE_GAZE_COLUMNS,
    AU_RAW_DIR,
    AU_AGGREGATED_DIR,
    TEMPORAL_STATS,
)
from src.modules.emotion.postprocess.aggregator import STAT_FUNCTIONS

logger = logging.getLogger(__name__)


class GazeFeatureExtractor:
    """
    Gaze 特徵提取器

    OpenFace 3.0 輸出 2D gaze（yaw, pitch），
    從 raw CSV 讀取並聚合到 subject-level
    """

    def extract_gaze_from_raw(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """從 OpenFace raw output 提取 gaze 欄位"""
        available_cols = [c for c in OPENFACE_GAZE_COLUMNS if c in raw_df.columns]
        if not available_cols:
            return None
        return raw_df[available_cols].copy()

    def aggregate_gaze(
        self,
        raw_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        min_frames: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        聚合所有受試者的 gaze 特徵到 subject-level

        每個 gaze 欄位計算 5 種時序統計量（mean, std, range, trend, entropy）
        """
        if raw_dir is None:
            raw_dir = AU_RAW_DIR / "openface"
        if output_dir is None:
            output_dir = AU_AGGREGATED_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "openface_gaze.csv"

        if not raw_dir.exists():
            logger.error(f"OpenFace raw 目錄不存在: {raw_dir}")
            return None

        csv_files = sorted(raw_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("沒有找到 OpenFace raw CSV")
            return None

        all_results = []
        for csv_path in csv_files:
            subject_id = csv_path.stem
            try:
                raw_df = pd.read_csv(csv_path)
                gaze_df = self.extract_gaze_from_raw(raw_df)

                if gaze_df is None or len(gaze_df) == 0:
                    continue

                result = {"subject_id": subject_id}
                n_frames = len(gaze_df)

                for col in gaze_df.columns:
                    values = gaze_df[col].dropna().values.astype(float)
                    if len(values) == 0:
                        for stat_name in TEMPORAL_STATS:
                            result[f"{col}_{stat_name}"] = 0.0
                        continue

                    for stat_name, stat_fn in STAT_FUNCTIONS.items():
                        if stat_name in ("trend", "entropy") and n_frames < min_frames:
                            result[f"{col}_{stat_name}"] = 0.0
                        else:
                            result[f"{col}_{stat_name}"] = stat_fn(values)

                all_results.append(result)

            except Exception as e:
                logger.error(f"  {subject_id} gaze 聚合失敗: {e}")

        if not all_results:
            logger.error("沒有成功聚合任何受試者的 gaze 特徵")
            return None

        result_df = pd.DataFrame(all_results)
        cols = ["subject_id"] + [c for c in result_df.columns if c != "subject_id"]
        result_df = result_df[cols]

        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(
            f"Gaze 特徵聚合完成: {len(result_df)} 受試者, "
            f"{len(result_df.columns) - 1} 特徵"
        )

        return result_df
