"""
時序統計聚合

將 per-frame 的 harmonized 特徵聚合為 subject-level 特徵向量
計算 mean, std, range, entropy 四項統計量
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import logging

from src.emo_au.extractor.au_config import (
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
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist + 1e-10  # 避免 log(0)
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


# name → 統計函數；統計量名稱集合的單一真相在 au_config.TEMPORAL_STATS（迴圈都依它排序）。
STAT_FUNCTIONS: Dict[str, Callable] = {
    "mean": lambda x: float(np.mean(x)),
    "std": lambda x: float(np.std(x)),
    "range": lambda x: float(np.max(x) - np.min(x)),
    "entropy": _safe_entropy,
}
assert set(STAT_FUNCTIONS) == set(TEMPORAL_STATS), \
    "STAT_FUNCTIONS 與 au_config.TEMPORAL_STATS 不一致"


def _zero_stats(col: str) -> Dict[str, float]:
    """該特徵無資料時的零填統計（每個 stat 補 0.0）。"""
    return {f"{col}_{s}": 0.0 for s in TEMPORAL_STATS}


class TemporalAggregator:
    """將 per-frame 特徵聚合為 subject-level 統計向量。"""

    def __init__(self, min_frames: int = 3):
        """
        Args:
            min_frames: 計算 entropy 的最少幀數
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

        for col in feature_columns:
            if col not in frame_df.columns:
                result.update(_zero_stats(col))
                continue

            values = frame_df[col].dropna().values.astype(float)

            if len(values) == 0:
                result.update(_zero_stats(col))
                continue

            n_valid = len(values)

            for stat_name in TEMPORAL_STATS:
                # entropy 在有效幀數不足時設為 0
                if stat_name == "entropy" and n_valid < self.min_frames:
                    result[f"{col}_{stat_name}"] = 0.0
                else:
                    result[f"{col}_{stat_name}"] = STAT_FUNCTIONS[stat_name](values)

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
            聚合後的 DataFrame (n_subjects, n_features * 4)
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
