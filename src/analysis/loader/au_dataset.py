"""
AU 特徵資料集載入器

載入 aggregated AU 特徵，結合 demographics，組裝成 Dataset
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.analysis.loader.base import Dataset
from src.config import DEMOGRAPHICS_DIR
from src.modules.emotion.extractor.au_config import AU_AGGREGATED_DIR

logger = logging.getLogger(__name__)


def _extract_base_id(subject_id: str) -> str:
    """
    從 subject_id 提取 base_id（去除 session 後綴）

    例如：P1002-1 → P1002, NAD100-1 → NAD100
    """
    match = re.match(r"^(.+?)-\d+$", subject_id)
    return match.group(1) if match else subject_id


def _infer_label(subject_id: str) -> Optional[int]:
    """
    從 subject_id 推斷標籤

    P 開頭 → 1（患者），NAD 開頭 → 0（健康）
    ACS 開頭 → None（排除）
    """
    if subject_id.startswith("P"):
        return 1
    elif subject_id.startswith("NAD"):
        return 0
    elif subject_id.startswith("ACS"):
        return None
    return None


class AUDatasetLoader:
    """
    AU 特徵資料集載入器

    載入 aggregated CSV 並組裝成 Dataset，供分類器使用
    """

    def __init__(
        self,
        aggregated_dir: Optional[Path] = None,
        demographics_dir: Optional[Path] = None,
        exclude_acs: bool = True,
    ):
        self.aggregated_dir = aggregated_dir or AU_AGGREGATED_DIR
        self.demographics_dir = demographics_dir or DEMOGRAPHICS_DIR
        self.exclude_acs = exclude_acs

    def load(
        self,
        tool: str,
        feature_set: str = "harmonized",
    ) -> Optional[Dataset]:
        """
        載入指定工具和特徵集的資料

        Args:
            tool: 工具名稱（openface / pyfeat / libreface）
            feature_set: 特徵集（harmonized / extended）

        Returns:
            Dataset 物件
        """
        csv_path = self.aggregated_dir / f"{tool}_{feature_set}.csv"

        if not csv_path.exists():
            logger.error(f"找不到聚合檔案: {csv_path}")
            return None

        logger.info(f"載入 {tool}_{feature_set} 特徵...")

        df = pd.read_csv(csv_path)

        if "subject_id" not in df.columns:
            logger.error("缺少 subject_id 欄位")
            return None

        # 推斷標籤並篩選
        subject_ids = []
        base_ids = []
        labels = []
        feature_rows = []

        feature_columns = [c for c in df.columns if c != "subject_id"]

        for _, row in df.iterrows():
            sid = row["subject_id"]

            if self.exclude_acs and sid.startswith("ACS"):
                continue

            label = _infer_label(sid)
            if label is None:
                continue

            subject_ids.append(sid)
            base_ids.append(_extract_base_id(sid))
            labels.append(label)
            feature_rows.append(row[feature_columns].values.astype(float))

        if not feature_rows:
            logger.error("沒有有效的受試者")
            return None

        X = np.array(feature_rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # 處理 NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"發現 {nan_count} 個 NaN 值，以 0 填補")
            X = np.nan_to_num(X, nan=0.0)

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        logger.info(
            f"載入完成: {len(subject_ids)} 受試者, "
            f"{X.shape[1]} 特徵, "
            f"患者={n_pos}, 健康={n_neg}"
        )

        return Dataset(
            X=X,
            y=y,
            subject_ids=subject_ids,
            base_ids=base_ids,
            metadata={
                "tool": tool,
                "feature_set": feature_set,
                "model": f"au_{tool}",
                "feature_type": feature_set,
                "cdr_threshold": 0,
                "n_features": X.shape[1],
                "feature_names": feature_columns,
            },
            data_format="averaged",
        )

    def load_all(
        self,
        tools: Optional[List[str]] = None,
        feature_sets: Optional[List[str]] = None,
    ) -> List[Dataset]:
        """
        載入所有工具 × 特徵集組合的資料

        Args:
            tools: 工具列表（預設 openface/pyfeat/libreface）
            feature_sets: 特徵集列表（預設 harmonized/extended）

        Returns:
            Dataset 列表
        """
        if tools is None:
            tools = ["openface", "pyfeat", "libreface"]
        if feature_sets is None:
            feature_sets = ["harmonized", "extended"]

        datasets = []
        for tool in tools:
            for fs in feature_sets:
                dataset = self.load(tool, fs)
                if dataset is not None:
                    datasets.append(dataset)

        logger.info(f"共載入 {len(datasets)} 個資料集")
        return datasets
