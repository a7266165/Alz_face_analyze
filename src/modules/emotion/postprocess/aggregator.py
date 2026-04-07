"""
時序統計聚合

將 per-frame 的 harmonized 特徵聚合為 subject-level 特徵向量
計算 mean, std, range, trend, entropy 五項統計量
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import logging
from scipy.stats import linregress

from src.modules.emotion.extractor.au_config import (
    HARMONIZED_FEATURES,
    TEMPORAL_STATS,
    AU_HARMONIZED_DIR,
    AU_AGGREGATED_DIR,
    AU_RAW_DIR,
)

logger = logging.getLogger(__name__)


def _safe_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """計算直方圖熵，處理邊界情況"""
    if len(values) < 2:
        return 0.0
    # 去除 NaN
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist + 1e-10  # 避免 log(0)
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def _safe_trend(values: np.ndarray) -> float:
    """計算線性回歸斜率，處理邊界情況"""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    try:
        slope, _, _, _, _ = linregress(np.arange(len(values)), values)
        return float(slope)
    except Exception:
        return 0.0


# 統計量計算函數（非時序資料，不含 trend）
STAT_FUNCTIONS: Dict[str, Callable] = {
    "mean": lambda x: float(np.nanmean(x)) if len(x) > 0 else 0.0,
    "std": lambda x: float(np.nanstd(x)) if len(x) > 1 else 0.0,
    "range": lambda x: float(np.nanmax(x) - np.nanmin(x)) if len(x) > 0 else 0.0,
    "entropy": _safe_entropy,
}


class TemporalAggregator:
    """
    時序統計聚合器

    將多幀特徵聚合為 subject-level 向量
    """

    def __init__(self, min_frames: int = 3):
        """
        Args:
            min_frames: 計算 trend/entropy 的最少幀數
        """
        self.min_frames = min_frames

    def aggregate_subject(
        self,
        frame_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        聚合單一受試者的多幀特徵

        Args:
            frame_df: per-frame DataFrame
            feature_columns: 要聚合的特徵欄位（預設用 HARMONIZED_FEATURES）

        Returns:
            聚合後的特徵字典 {feature_stat: value}
        """
        if feature_columns is None:
            feature_columns = [
                c for c in HARMONIZED_FEATURES if c in frame_df.columns
            ]

        result = {}
        n_frames = len(frame_df)

        for col in feature_columns:
            if col not in frame_df.columns:
                for stat_name in TEMPORAL_STATS:
                    result[f"{col}_{stat_name}"] = 0.0
                continue

            values = frame_df[col].dropna().values.astype(float)

            if len(values) == 0:
                for stat_name in TEMPORAL_STATS:
                    result[f"{col}_{stat_name}"] = 0.0
                continue

            for stat_name, stat_fn in STAT_FUNCTIONS.items():
                # trend/entropy 在幀數不足時設為 0
                if stat_name in ("trend", "entropy") and n_frames < self.min_frames:
                    result[f"{col}_{stat_name}"] = 0.0
                else:
                    result[f"{col}_{stat_name}"] = stat_fn(values)

        return result

    def aggregate_tool(
        self,
        tool: str,
        feature_set: str = "harmonized",
        harmonized_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Optional[pd.DataFrame]:
        """
        聚合指定工具所有受試者的特徵

        Args:
            tool: 工具名稱
            feature_set: "harmonized" 或 "extended"
            harmonized_dir: harmonized 檔案目錄
            raw_dir: raw 檔案目錄（extended 模式用）
            output_dir: 輸出目錄

        Returns:
            聚合後的 DataFrame (n_subjects, n_features * 5)
        """
        if output_dir is None:
            output_dir = AU_AGGREGATED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{tool}_{feature_set}.csv"

        # 決定輸入目錄和特徵欄位
        if feature_set == "harmonized":
            input_dir = harmonized_dir or (AU_HARMONIZED_DIR / tool)
            feature_columns = HARMONIZED_FEATURES
        else:
            input_dir = raw_dir or (AU_RAW_DIR / tool)
            feature_columns = None  # 使用所有非 frame 欄位

        if not input_dir.exists():
            logger.error(f"輸入目錄不存在: {input_dir}")
            return None

        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"  {tool}: 沒有找到 CSV 檔案")
            return None

        all_results = []
        for csv_path in csv_files:
            subject_id = csv_path.stem

            try:
                df = pd.read_csv(csv_path)

                # extended 模式：使用所有非 frame 欄位
                if feature_columns is None:
                    cols = [c for c in df.columns if c != "frame"]
                else:
                    cols = feature_columns

                agg = self.aggregate_subject(df, cols)
                agg["subject_id"] = subject_id
                all_results.append(agg)

            except Exception as e:
                logger.error(f"  {subject_id} 聚合失敗: {e}")

        if not all_results:
            logger.error(f"{tool}_{feature_set}: 沒有成功聚合任何受試者")
            return None

        result_df = pd.DataFrame(all_results)

        # 將 subject_id 移到第一欄
        cols = ["subject_id"] + [c for c in result_df.columns if c != "subject_id"]
        result_df = result_df[cols]

        # 儲存
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(
            f"{tool}_{feature_set}: 聚合完成，"
            f"{len(result_df)} 受試者，{len(result_df.columns) - 1} 特徵"
        )

        return result_df
