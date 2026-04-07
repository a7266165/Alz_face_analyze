"""
人口學資料載入模組

負責載入和處理人口學資料（年齡、性別、CDR等）
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DemographicsLoader:
    """人口學資料載入器"""

    def __init__(self, demographics_dir: Path):
        """
        初始化

        Args:
            demographics_dir: 人口學資料目錄
        """
        self.demographics_dir = Path(demographics_dir)
        self._cache = None

    def load(self) -> pd.DataFrame:
        """
        載入人口學資料

        Returns:
            包含 subject_id, Age, Sex, CDR 等欄位的 DataFrame
        """
        if self._cache is not None:
            return self._cache.copy()

        df = self._load_from_files()
        self._cache = df
        return df.copy()

    def _load_from_files(self) -> pd.DataFrame:
        """從檔案載入原始資料"""
        # 載入健康組
        healthy_file = self.demographics_dir / "healthy.csv"
        patient_file = self.demographics_dir / "patient.csv"

        dfs = []

        if healthy_file.exists():
            healthy_df = pd.read_csv(healthy_file)
            healthy_df['label'] = 0
            dfs.append(healthy_df)
            logger.debug(f"載入健康組: {len(healthy_df)} 筆")
        else:
            logger.warning(f"健康組檔案不存在: {healthy_file}")

        if patient_file.exists():
            patient_df = pd.read_csv(patient_file)
            patient_df['label'] = 1
            dfs.append(patient_df)
            logger.debug(f"載入病患組: {len(patient_df)} 筆")
        else:
            logger.warning(f"病患組檔案不存在: {patient_file}")

        if not dfs:
            logger.error("沒有載入任何人口學資料")
            return pd.DataFrame()

        # 合併
        df = pd.concat(dfs, ignore_index=True)

        # 標準化欄位名稱
        df = self._standardize_columns(df)

        logger.info(f"載入人口學資料: {len(df)} 筆")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化欄位名稱"""
        # 常見的欄位名稱對應
        column_mappings = {
            'id': 'subject_id',
            'ID': 'subject_id',
            'patient_id': 'subject_id',
            'age': 'Age',
            'AGE': 'Age',
            'sex': 'Sex',
            'SEX': 'Sex',
            'gender': 'Sex',
            'Gender': 'Sex',
            'cdr': 'CDR',
        }

        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        return df

    def clear_cache(self):
        """清除記憶體快取"""
        self._cache = None
