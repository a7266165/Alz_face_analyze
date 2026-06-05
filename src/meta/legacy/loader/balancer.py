"""
資料平衡模組

提供基於年齡的資料平衡功能
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataBalancer:
    """資料平衡器"""

    def __init__(
        self,
        n_bins: int = 5,
        random_seed: int = 42
    ):
        """
        初始化

        Args:
            n_bins: 年齡分箱數量
            random_seed: 隨機種子
        """
        self.n_bins = n_bins
        self.random_seed = random_seed

    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        執行資料平衡

        基於年齡分箱，平衡健康組和病患組的比例

        Args:
            df: 包含 'label', 'Age' 欄位的 DataFrame

        Returns:
            平衡後的 DataFrame
        """
        logger.info("執行資料平衡...")

        if len(df) == 0:
            logger.warning("沒有資料可供平衡")
            return df

        # 分離健康組和病患組
        health_group = df[df['label'] == 0].copy()
        patient_group = df[df['label'] == 1].copy()

        n_health = len(health_group)
        n_patient = len(patient_group)

        if n_health == 0 or n_patient == 0:
            logger.warning("健康組或病患組為空，跳過平衡")
            return df

        # 計算目標比例
        target_ratio = min(n_health, n_patient) / max(n_health, n_patient)
        majority_is_health = n_health > n_patient

        logger.debug(
            f"原始樣本數: 健康組 {n_health}, 病患組 {n_patient}, "
            f"目標比例 {target_ratio:.2f}"
        )

        # 計算年齡分箱
        all_ages = pd.concat([health_group['Age'], patient_group['Age']])
        age_bins = self._calculate_bins(all_ages)

        # 為每組添加年齡箱標籤
        health_group['age_bin'] = pd.cut(
            health_group['Age'], bins=age_bins, include_lowest=True
        )
        patient_group['age_bin'] = pd.cut(
            patient_group['Age'], bins=age_bins, include_lowest=True
        )

        # 執行平衡
        balanced_dfs = self._balance_bins(
            health_group, patient_group,
            target_ratio, majority_is_health
        )

        if not balanced_dfs:
            logger.warning("沒有任何箱可以配對")
            return pd.DataFrame()

        # 合併結果
        result = pd.concat(balanced_dfs, ignore_index=True)
        result = result.drop(columns=['age_bin'], errors='ignore')

        # 統計
        final_health = len(result[result['label'] == 0])
        final_patient = len(result[result['label'] == 1])

        logger.info(
            f"資料平衡完成: {len(df)} → {len(result)} 筆 "
            f"(健康組 {n_health}→{final_health}, 病患組 {n_patient}→{final_patient})"
        )

        return result

    def _calculate_bins(self, ages: pd.Series) -> np.ndarray:
        """計算年齡分箱邊界"""
        try:
            age_bins = pd.qcut(
                ages,
                q=self.n_bins,
                duplicates='drop',
                retbins=True
            )[1]
        except ValueError:
            n_bins_effective = min(self.n_bins, ages.nunique())
            age_bins = pd.qcut(
                ages,
                q=n_bins_effective,
                duplicates='drop',
                retbins=True
            )[1]
            logger.debug(f"年齡箱數自動調整為 {n_bins_effective}")

        return age_bins

    def _balance_bins(
        self,
        health_group: pd.DataFrame,
        patient_group: pd.DataFrame,
        target_ratio: float,
        majority_is_health: bool
    ) -> list:
        """對每個年齡箱進行平衡"""
        balanced_dfs = []
        rng = np.random.RandomState(self.random_seed)

        for bin_val in health_group['age_bin'].cat.categories:
            health_in_bin = health_group[health_group['age_bin'] == bin_val]
            patient_in_bin = patient_group[patient_group['age_bin'] == bin_val]

            n_health_bin = len(health_in_bin)
            n_patient_bin = len(patient_in_bin)

            # 如果某一組為空，保留另一組
            if n_health_bin == 0 or n_patient_bin == 0:
                if n_health_bin > 0:
                    balanced_dfs.append(health_in_bin)
                if n_patient_bin > 0:
                    balanced_dfs.append(patient_in_bin)
                continue

            # 根據多數組進行欠採樣
            if majority_is_health:
                target_health = int(n_patient_bin / target_ratio)
                target_health = min(target_health, n_health_bin)

                if n_health_bin > target_health:
                    health_sample = health_in_bin.sample(n=target_health, random_state=rng)
                else:
                    health_sample = health_in_bin

                balanced_dfs.append(health_sample)
                balanced_dfs.append(patient_in_bin)
            else:
                target_patient = int(n_health_bin / target_ratio)
                target_patient = min(target_patient, n_patient_bin)

                if n_patient_bin > target_patient:
                    patient_sample = patient_in_bin.sample(n=target_patient, random_state=rng)
                else:
                    patient_sample = patient_in_bin

                balanced_dfs.append(health_in_bin)
                balanced_dfs.append(patient_sample)

        return balanced_dfs
