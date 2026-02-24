"""
Meta Analysis 資料載入器

載入 14 個特徵：
  - age_error (real - predicted)
  - real_age
  - lr_score_original (原始臉 LR 分數)
  - lr_score_asymmetry (不對稱性 LR 分數)
  - 8 個表情 + Valence + Arousal
合併後返回 MetaDataset。
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.meta_analysis.data.dataset import MetaDataset

logger = logging.getLogger(__name__)


class MetaDataLoader:
    """
    Meta 分析資料載入器

    負責載入並合併：
    1. 兩組 Logistic regression 預測分數 (original + asymmetry)
    2. Emotion scores (8 表情 + Valence + Arousal)
    3. 人口學資料 (real_age) + 預測年齡 (age_error)
    4. 從 subject_id 推斷 labels
    """

    # Emotion 特徵欄位
    EMOTION_COLUMNS = [
        "Anger", "Contempt", "Disgust", "Fear", "Happiness",
        "Neutral", "Sadness", "Surprise", "Valence", "Arousal",
    ]

    # 完整 14 特徵欄位順序
    FEATURE_COLUMNS = [
        "age_error", "real_age",
        "lr_score_original", "lr_score_asymmetry",
        "Anger", "Contempt", "Disgust", "Fear", "Happiness",
        "Neutral", "Sadness", "Surprise", "Valence", "Arousal",
    ]

    def __init__(
        self,
        predictions_dir: Path,
        emotion_scores_file: Path,
        demographics_dir: Path,
        predicted_ages_file: Path,
        n_features: int,
        model: str,
        asymmetry_method: str = "absolute_relative_differences",
        cdr_threshold: float = 0,
    ):
        """
        初始化載入器

        Args:
            predictions_dir: pred_probability 目錄路徑
            emotion_scores_file: emotion_score.csv 檔案路徑
            demographics_dir: 人口學資料目錄 (含 ACS.csv, NAD.csv, P.csv)
            predicted_ages_file: predicted_ages.json 檔案路徑
            n_features: 指定的 n_features 層級
            model: 模型名稱 ("arcface", "topofr", etc.)
            asymmetry_method: 不對稱性方法名稱
            cdr_threshold: CDR 閾值 (預設 0)
        """
        self.predictions_dir = Path(predictions_dir)
        self.emotion_scores_file = Path(emotion_scores_file)
        self.demographics_dir = Path(demographics_dir)
        self.predicted_ages_file = Path(predicted_ages_file)
        self.n_features = n_features
        self.model = model
        self.asymmetry_method = asymmetry_method
        self.cdr_threshold = cdr_threshold

    def load(self) -> MetaDataset:
        """
        載入並合併所有資料

        Returns:
            MetaDataset 物件 (14 特徵)
        """
        # 1. 載入兩組 LR 預測分數
        lr_original_df = self._load_lr_predictions("original")
        logger.info(f"載入 LR 原始: {len(lr_original_df)} 筆")

        lr_asymmetry_df = self._load_lr_predictions(self.asymmetry_method)
        logger.info(f"載入 LR 不對稱: {len(lr_asymmetry_df)} 筆")

        # 2. 載入 emotion scores
        emotion_df = self._load_emotion_scores()
        logger.info(f"載入 emotion scores: {len(emotion_df)} 筆")

        # 3. 載入年齡相關資料
        age_df = self._load_age_data()
        logger.info(f"載入年齡資料: {len(age_df)} 筆")

        # 4. 合併資料
        merged_df = self._merge_all(lr_original_df, lr_asymmetry_df, emotion_df, age_df)
        logger.info(f"合併後: {len(merged_df)} 筆")

        # 5. 從 subject_id 推斷 labels
        merged_df = self._infer_labels(merged_df)

        # 6. 建立 MetaDataset
        return self._create_dataset(merged_df)

    def _load_lr_predictions(self, method: str) -> pd.DataFrame:
        """載入指定方法的 LR test 預測分數"""
        n_features_dir = self.predictions_dir / f"n_features_{self.n_features}"
        cdr_str = f"cdr{int(self.cdr_threshold)}"
        filename = f"{self.model}_{method}_{cdr_str}_test.csv"
        file_path = n_features_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"找不到預測檔案: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        # 標準化欄位名稱
        df = df.rename(columns={
            "個案編號": "subject_id",
            "預測分數": "lr_pred_score",
        })

        score_col = f"lr_score_{method}" if method == "original" else "lr_score_asymmetry"
        df = df.rename(columns={"lr_pred_score": score_col})

        return df[["subject_id", score_col, "fold"]]

    def _load_emotion_scores(self) -> pd.DataFrame:
        """載入 emotion scores"""
        if not self.emotion_scores_file.exists():
            raise FileNotFoundError(f"找不到 emotion 檔案: {self.emotion_scores_file}")

        df = pd.read_csv(self.emotion_scores_file, encoding="utf-8-sig")

        required_cols = ["subject_id"] + self.EMOTION_COLUMNS
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"emotion 檔案缺少欄位: {missing_cols}")

        return df[required_cols]

    def _load_age_data(self) -> pd.DataFrame:
        """
        載入人口學資料 + 預測年齡，計算 age_error

        Returns:
            DataFrame with columns: [subject_id, real_age, age_error]
        """
        # 載入人口學資料
        dfs = []
        for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
            csv_path = self.demographics_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dfs.append(df[["ID", "Age"]])

        if not dfs:
            raise FileNotFoundError(f"找不到人口學資料: {self.demographics_dir}")

        demo_df = pd.concat(dfs, ignore_index=True)
        demo_df = demo_df.rename(columns={"ID": "subject_id", "Age": "real_age"})
        demo_df["real_age"] = pd.to_numeric(demo_df["real_age"], errors="coerce")

        # 載入預測年齡
        if not self.predicted_ages_file.exists():
            raise FileNotFoundError(f"找不到預測年齡檔案: {self.predicted_ages_file}")

        with open(self.predicted_ages_file, "r") as f:
            predicted_ages = json.load(f)

        # 計算 age_error = real_age - predicted_age
        demo_df["predicted_age"] = demo_df["subject_id"].map(predicted_ages)
        demo_df = demo_df.dropna(subset=["real_age", "predicted_age"])
        demo_df["age_error"] = demo_df["real_age"] - demo_df["predicted_age"]

        return demo_df[["subject_id", "real_age", "age_error"]]

    def _merge_all(
        self,
        lr_original_df: pd.DataFrame,
        lr_asymmetry_df: pd.DataFrame,
        emotion_df: pd.DataFrame,
        age_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """合併所有資料來源"""
        # 先合併兩組 LR 分數
        merged = pd.merge(
            lr_original_df, lr_asymmetry_df,
            on=["subject_id", "fold"], how="inner",
        )

        # 合併 emotion scores
        merged = pd.merge(merged, emotion_df, on="subject_id", how="inner")

        # 合併年齡資料
        merged = pd.merge(merged, age_df, on="subject_id", how="inner")

        if len(merged) == 0:
            raise ValueError("合併後無資料，請檢查 subject_id 是否一致")

        # 報告合併結果
        n_orig = len(lr_original_df)
        n_asym = len(lr_asymmetry_df)
        n_merged = len(merged)
        if n_merged < min(n_orig, n_asym):
            logger.warning(
                f"合併損失: LR原始={n_orig}, LR不對稱={n_asym}, "
                f"合併後={n_merged}"
            )

        return merged

    def _infer_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """從 subject_id 推斷 labels"""
        def get_label(subject_id: str) -> int:
            if subject_id.startswith("P"):
                return 1
            elif subject_id.startswith("ACS") or subject_id.startswith("NAD"):
                return 0
            else:
                logger.warning(f"無法識別 subject_id 類型: {subject_id}")
                return -1

        df = df.copy()
        df["label"] = df["subject_id"].apply(get_label)

        invalid = df[df["label"] == -1]
        if len(invalid) > 0:
            logger.warning(f"移除 {len(invalid)} 筆無法識別的樣本")
            df = df[df["label"] != -1]

        return df

    def _create_dataset(self, df: pd.DataFrame) -> MetaDataset:
        """建立 MetaDataset (14 特徵)"""
        X = df[self.FEATURE_COLUMNS].values.astype(np.float32)
        y = df["label"].values.astype(np.int32)
        subject_ids = df["subject_id"].tolist()
        fold_assignments = df["fold"].values.astype(np.int32)
        base_ids = [self._extract_base_id(sid) for sid in subject_ids]

        metadata = {
            "n_features": self.n_features,
            "model": self.model,
            "asymmetry_method": self.asymmetry_method,
            "cdr_threshold": self.cdr_threshold,
            "predictions_dir": str(self.predictions_dir),
            "emotion_file": str(self.emotion_scores_file),
        }

        return MetaDataset(
            X=X,
            y=y,
            subject_ids=subject_ids,
            base_ids=base_ids,
            fold_assignments=fold_assignments,
            feature_names=self.FEATURE_COLUMNS,
            metadata=metadata,
        )

    @staticmethod
    def _extract_base_id(subject_id: str) -> str:
        """從 subject_id 提取 base_id (如 P1-2 -> P1)"""
        match = re.match(r"^([A-Za-z]+\d+)", subject_id)
        if match:
            return match.group(1)
        return subject_id

    @staticmethod
    def discover_n_features(predictions_dir: Path) -> List[int]:
        """自動發現所有可用的 n_features 層級"""
        predictions_dir = Path(predictions_dir)
        n_features_list = []

        for d in predictions_dir.iterdir():
            if d.is_dir() and d.name.startswith("n_features_"):
                try:
                    n = int(d.name.split("_")[-1])
                    n_features_list.append(n)
                except ValueError:
                    continue

        return sorted(n_features_list)
