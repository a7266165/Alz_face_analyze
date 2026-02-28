"""
Meta Analysis 資料載入器（折疊對齊版）

載入 14 個特徵：
  - age_error (real - predicted)
  - real_age
  - lr_score_original (原始臉 LR 分數)
  - lr_score_asymmetry (不對稱性 LR 分數)
  - 8 個表情 + Valence + Arousal

從 run_analyze 的 train/test CSV 建構 per-fold 資料，
使 meta-learner 的折疊與 base model 的 K-fold CV 對齊。
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.meta_analysis.data.dataset import FoldData, MetaDataset

logger = logging.getLogger(__name__)


class MetaDataLoader:
    """
    Meta 分析資料載入器（折疊對齊版）

    載入 base model 的 train/test 預測分數（含 fold 欄位），
    結合 emotion + age 特徵，建構 per-fold 資料集。
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
        載入並建構折疊對齊的 MetaDataset

        Returns:
            MetaDataset (per-fold 結構)
        """
        # 1. 載入兩組 LR 預測分數（train + test，含 fold 和 split）
        lr_original_df = self._load_lr_predictions("original")
        logger.info(f"載入 LR 原始: {len(lr_original_df)} 筆")

        lr_asymmetry_df = self._load_lr_predictions(self.asymmetry_method)
        logger.info(f"載入 LR 不對稱: {len(lr_asymmetry_df)} 筆")

        # 2. 載入 fold-independent 資料
        emotion_df = self._load_emotion_scores()
        logger.info(f"載入 emotion scores: {len(emotion_df)} 筆")

        age_df = self._load_age_data()
        logger.info(f"載入年齡資料: {len(age_df)} 筆")

        # 3. 合併 LR 分數（original + asymmetry）
        lr_merged = pd.merge(
            lr_original_df, lr_asymmetry_df,
            on=["subject_id", "fold", "split"], how="inner",
        )
        logger.info(f"合併 LR 分數後: {len(lr_merged)} 筆")

        # 4. 建構 per-fold 資料
        fold_data = self._build_fold_data(lr_merged, emotion_df, age_df)

        metadata = {
            "n_features": self.n_features,
            "model": self.model,
            "asymmetry_method": self.asymmetry_method,
            "cdr_threshold": self.cdr_threshold,
            "predictions_dir": str(self.predictions_dir),
            "emotion_file": str(self.emotion_scores_file),
        }

        dataset = MetaDataset(
            fold_data=fold_data,
            feature_names=self.FEATURE_COLUMNS,
            metadata=metadata,
        )

        logger.info(f"建構完成: {dataset}")
        return dataset

    def _load_lr_predictions(self, method: str) -> pd.DataFrame:
        """
        載入指定方法的 LR 預測分數（train + test）

        Returns:
            DataFrame with columns: [subject_id, lr_score_xxx, fold, split]
        """
        n_features_dir = self.predictions_dir / f"n_features_{self.n_features}"
        cdr_str = f"cdr{int(self.cdr_threshold)}"

        # 載入 test CSV
        test_filename = f"{self.model}_{method}_{cdr_str}_test.csv"
        test_path = n_features_dir / test_filename
        if not test_path.exists():
            raise FileNotFoundError(f"找不到測試預測檔案: {test_path}")

        test_df = pd.read_csv(test_path, encoding="utf-8-sig")
        test_df["split"] = "test"

        # 載入 train CSV
        train_filename = f"{self.model}_{method}_{cdr_str}_train.csv"
        train_path = n_features_dir / train_filename
        if not train_path.exists():
            raise FileNotFoundError(f"找不到訓練預測檔案: {train_path}")

        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        train_df["split"] = "train"

        # 合併 train + test
        combined = pd.concat([train_df, test_df], ignore_index=True)

        # 標準化欄位名稱
        combined = combined.rename(columns={
            "個案編號": "subject_id",
            "預測分數": "lr_pred_score",
        })

        score_col = f"lr_score_{method}" if method == "original" else "lr_score_asymmetry"
        combined = combined.rename(columns={"lr_pred_score": score_col})

        return combined[["subject_id", score_col, "fold", "split"]]

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
        """載入人口學資料 + 預測年齡，計算 age_error"""
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

        if not self.predicted_ages_file.exists():
            raise FileNotFoundError(f"找不到預測年齡檔案: {self.predicted_ages_file}")

        with open(self.predicted_ages_file, "r") as f:
            predicted_ages = json.load(f)

        demo_df["predicted_age"] = demo_df["subject_id"].map(predicted_ages)
        demo_df = demo_df.dropna(subset=["real_age", "predicted_age"])
        demo_df["age_error"] = demo_df["real_age"] - demo_df["predicted_age"]

        return demo_df[["subject_id", "real_age", "age_error"]]

    def _build_fold_data(
        self,
        lr_merged: pd.DataFrame,
        emotion_df: pd.DataFrame,
        age_df: pd.DataFrame,
    ) -> Dict[int, FoldData]:
        """
        從合併的 LR 資料建構 per-fold FoldData

        Args:
            lr_merged: 合併的 LR 分數（含 fold, split）
            emotion_df: emotion scores
            age_df: 年齡資料

        Returns:
            Dict[fold_idx, FoldData]
        """
        unique_folds = sorted(lr_merged["fold"].unique())
        logger.info(f"偵測到 {len(unique_folds)} 個折疊: {unique_folds}")

        fold_data = {}

        for fold_idx in unique_folds:
            fold_lr = lr_merged[lr_merged["fold"] == fold_idx]

            # 合併 emotion + age（fold-independent）
            fold_merged = pd.merge(fold_lr, emotion_df, on="subject_id", how="inner")
            fold_merged = pd.merge(fold_merged, age_df, on="subject_id", how="inner")

            # 推斷 labels
            fold_merged = self._infer_labels(fold_merged)

            # 分離 train/test
            train_df = fold_merged[fold_merged["split"] == "train"]
            test_df = fold_merged[fold_merged["split"] == "test"]

            if train_df.empty or test_df.empty:
                logger.warning(f"Fold {fold_idx}: train={len(train_df)}, test={len(test_df)}，跳過")
                continue

            # 建構 FoldData
            X_train = train_df[self.FEATURE_COLUMNS].values.astype(np.float32)
            y_train = train_df["label"].values.astype(np.int32)
            X_test = test_df[self.FEATURE_COLUMNS].values.astype(np.float32)
            y_test = test_df["label"].values.astype(np.int32)

            train_sids = train_df["subject_id"].tolist()
            test_sids = test_df["subject_id"].tolist()
            train_bids = [self._extract_base_id(s) for s in train_sids]
            test_bids = [self._extract_base_id(s) for s in test_sids]

            fold_data[fold_idx] = FoldData(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_subject_ids=train_sids,
                test_subject_ids=test_sids,
                train_base_ids=train_bids,
                test_base_ids=test_bids,
            )

            logger.info(
                f"  Fold {fold_idx}: train={len(train_df)}, test={len(test_df)}, "
                f"train_class={dict(zip(*np.unique(y_train, return_counts=True)))}, "
                f"test_class={dict(zip(*np.unique(y_test, return_counts=True)))}"
            )

        return fold_data

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
