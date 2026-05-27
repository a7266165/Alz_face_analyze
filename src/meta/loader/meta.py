"""
Meta Analysis 資料載入器（折疊對齊版 v2）

載入 14 個特徵（最大，依 emotion_method 動態調整）：
  - lr_score_original, lr_score_asymmetry (OOF scores)
  - real_age, age_error (demographics + age prediction)
  - 7~10 emotion features (依 method: emonet=10, openface=8, 其他=7)

從 embedding classification pipeline 的 forward_oof_scores_subject.csv 讀取
OOF scores，搭配 EmoNet CSV 或 per-visit .npy 讀取 emotion features。
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR,
    cohort_name,
    cohort_spec_from_name,
)
from src.meta.loader.dataset import FoldData, MetaDataset

logger = logging.getLogger(__name__)

FIXED_COLUMNS = ["lr_score_original", "lr_score_asymmetry", "real_age", "age_error"]

EMOTION_NAMES = frozenset({
    "anger", "contempt", "disgust", "fear", "happiness",
    "sadness", "surprise", "neutral",
})

EMONET_COLUMNS = [
    "anger", "contempt", "disgust", "fear", "happiness",
    "neutral", "sadness", "surprise", "valence", "arousal",
]


class MetaDataLoader:
    """
    Meta 分析資料載入器（折疊對齊版 v2）

    讀取 OOF scores + emotion + age，建構 per-fold 資料集。
    """

    def __init__(
        self,
        emb_model: str,
        # embedding pipeline path params
        cohort_mode: str,
        bg_mode: str,
        photo_mode: str,
        reducer: str,
        base_classifier: str,
        base_classifier_param: str,
        direction: str,
        eval_method: str,
        match_level: str,
        eval_unit: str,
        match_strategy: str,
        partition: str,
        # emotion
        emotion_method: str,
        asymmetry_variant: str = "absolute_relative_differences",
        emotion_features_dir: Optional[Path] = None,
        emotion_schema_file: Optional[Path] = None,
        emonet_csv: Optional[Path] = None,
        # age
        demographics_dir: Optional[Path] = None,
        predicted_ages_file: Optional[Path] = None,
        hc_source_mode: str = "ACS",
    ):
        self.emb_model = emb_model
        self.cohort_mode = cohort_mode
        self.hc_source_mode = hc_source_mode
        self.bg_mode = bg_mode
        self.photo_mode = photo_mode
        self.reducer = reducer
        self.base_classifier = base_classifier
        self.base_classifier_param = base_classifier_param
        self.direction = direction
        self.eval_method = eval_method
        self.match_level = match_level
        self.eval_unit = eval_unit
        self.match_strategy = match_strategy
        self.partition = partition
        self.emotion_method = emotion_method
        self.asymmetry_variant = asymmetry_variant
        self.emotion_features_dir = Path(emotion_features_dir) if emotion_features_dir else None
        self.emotion_schema_file = Path(emotion_schema_file) if emotion_schema_file else None
        self.emonet_csv = Path(emonet_csv) if emonet_csv else None
        self.demographics_dir = Path(demographics_dir) if demographics_dir else None
        self.predicted_ages_file = Path(predicted_ages_file) if predicted_ages_file else None

        self.emotion_columns = self._resolve_emotion_columns()
        self.feature_columns = list(FIXED_COLUMNS) + self.emotion_columns

    def _resolve_emotion_columns(self) -> List[str]:
        if self.emotion_method is None:
            return []

        if self.emotion_method == "emonet":
            return list(EMONET_COLUMNS)

        if self.emotion_schema_file is None:
            raise ValueError("emotion_schema_file 必須指定（非 emonet 模式）")
        schema = json.load(open(self.emotion_schema_file, encoding="utf-8"))
        all_cols = schema["methods"][self.emotion_method]["columns"]
        return [c for c in all_cols if c in EMOTION_NAMES]

    def load(self) -> MetaDataset:
        lr_original_df = self._load_oof_scores("original")
        logger.info(f"載入 OOF original: {len(lr_original_df)} 筆")
        merged = lr_original_df

        try:
            lr_asymmetry_df = self._load_oof_scores(self.asymmetry_variant)
            logger.info(f"載入 OOF asymmetry: {len(lr_asymmetry_df)} 筆")
            merged = pd.merge(
                merged, lr_asymmetry_df,
                on=["subject_id", "base_id", "fold", "y_true"], how="inner",
            )
            logger.info(f"合併 OOF 後: {len(merged)} 筆")
        except FileNotFoundError as e:
            logger.warning(f"跳過 asymmetry OOF: {e}")
            if "lr_score_asymmetry" in self.feature_columns:
                self.feature_columns = [c for c in self.feature_columns if c != "lr_score_asymmetry"]

        age_df = self._load_age_data()
        logger.info(f"載入 age: {len(age_df)} 筆")
        merged = pd.merge(merged, age_df, on="subject_id", how="inner")

        if self.emotion_columns:
            emotion_df = self._load_emotion_scores(merged["subject_id"].tolist())
            logger.info(f"載入 emotion: {len(emotion_df)} 筆 ({len(self.emotion_columns)} 欄)")
            merged = pd.merge(merged, emotion_df, on="subject_id", how="inner")
        else:
            logger.info("跳過 emotion（未指定 emotion_method）")

        merged = self._infer_labels(merged)
        logger.info(f"全部合併後: {len(merged)} 筆, {len(self.feature_columns)} 特徵")

        fold_data = self._build_fold_data(merged)

        metadata = {
            "emb_model": self.emb_model,
            "reducer": self.reducer,
            "emotion_method": self.emotion_method,
            "cohort_mode": self.cohort_mode,
            "bg_mode": self.bg_mode,
            "partition": self.partition,
            "base_classifier": self.base_classifier,
        }

        dataset = MetaDataset(
            fold_data=fold_data,
            feature_names=self.feature_columns,
            metadata=metadata,
        )
        logger.info(f"建構完成: {dataset}")
        return dataset

    def _load_oof_scores(self, feature_type: str) -> pd.DataFrame:
        spec = cohort_spec_from_name(cohort_name(self.cohort_mode))
        base = (
            EMBEDDING_CLASSIFICATION_DIR
            / spec.visit_dir / spec.cdr_mmse_dir
            / self.bg_mode / self.emb_model / feature_type
            / self.photo_mode / self.reducer
        )
        oof_dir = (
            base
            / self.base_classifier / self.base_classifier_param
            / self.direction / self.eval_method / self.match_level
            / self.eval_unit / self.match_strategy / self.partition
        )
        oof_file = oof_dir / "oof_scores_subject.csv"
        if not oof_file.exists():
            legacy = oof_dir / "forward_oof_scores_subject.csv"
            if legacy.exists():
                oof_file = legacy
            else:
                raise FileNotFoundError(f"找不到 OOF 檔案: {oof_file}")

        df = pd.read_csv(oof_file)
        score_col = (
            "lr_score_original" if feature_type == "original"
            else "lr_score_asymmetry"
        )
        df = df.rename(columns={"y_score": score_col, "ID": "subject_id"})
        return df[["subject_id", "base_id", score_col, "y_true", "fold"]]

    def _load_emotion_scores(self, subject_ids: List[str]) -> pd.DataFrame:
        if self.emotion_method == "emonet":
            return self._load_emonet_csv(subject_ids)
        return self._load_emotion_npy(subject_ids)

    def _load_emonet_csv(self, subject_ids: List[str]) -> pd.DataFrame:
        if self.emonet_csv is None or not self.emonet_csv.exists():
            raise FileNotFoundError(f"找不到 EmoNet CSV: {self.emonet_csv}")

        df = pd.read_csv(self.emonet_csv, encoding="utf-8-sig")
        col_map = {
            "Anger": "anger", "Contempt": "contempt", "Disgust": "disgust",
            "Fear": "fear", "Happiness": "happiness", "Neutral": "neutral",
            "Sadness": "sadness", "Surprise": "surprise",
            "Valence": "valence", "Arousal": "arousal",
        }
        df = df.rename(columns=col_map)
        df = df[df["subject_id"].isin(subject_ids)]
        return df[["subject_id"] + self.emotion_columns]

    def _load_emotion_npy(self, subject_ids: List[str]) -> pd.DataFrame:
        if self.emotion_schema_file is None:
            raise ValueError("emotion_schema_file 必須指定")
        if self.emotion_features_dir is None:
            raise ValueError("emotion_features_dir 必須指定")

        schema = json.load(open(self.emotion_schema_file, encoding="utf-8"))
        all_columns = schema["methods"][self.emotion_method]["columns"]
        col_indices = [all_columns.index(c) for c in self.emotion_columns]

        method_dir = self.emotion_features_dir / self.emotion_method
        rows = []
        for sid in subject_ids:
            npy_path = method_dir / f"{sid}.npy"
            if not npy_path.exists():
                continue
            arr = np.load(npy_path)[:, col_indices]
            mean_vec = arr.mean(axis=0)
            rows.append({"subject_id": sid, **dict(zip(self.emotion_columns, mean_vec))})

        return pd.DataFrame(rows)

    def _load_age_data(self) -> pd.DataFrame:
        from src.cohort import load_combined_demographics

        hc_source_mode = getattr(self, "hc_source_mode", "ACS")
        demo = load_combined_demographics(hc_source_mode)
        demo = demo.rename(columns={"ID": "subject_id", "Age": "real_age"})
        demo["real_age"] = pd.to_numeric(demo["real_age"], errors="coerce")

        if not self.predicted_ages_file.exists():
            raise FileNotFoundError(f"找不到預測年齡檔案: {self.predicted_ages_file}")

        with open(self.predicted_ages_file, "r") as f:
            predicted_ages = json.load(f)

        demo["predicted_age"] = demo["subject_id"].map(predicted_ages)
        demo = demo.dropna(subset=["real_age", "predicted_age"])
        demo["age_error"] = demo["real_age"] - demo["predicted_age"]

        return demo[["subject_id", "real_age", "age_error"]]

    def _build_fold_data(self, merged_df: pd.DataFrame) -> Dict[int, FoldData]:
        unique_folds = sorted(merged_df["fold"].unique())
        logger.info(f"偵測到 {len(unique_folds)} 個折疊: {unique_folds}")

        fold_data = {}
        for fold_idx in unique_folds:
            test_mask = merged_df["fold"] == fold_idx
            train_df = merged_df[~test_mask]
            test_df = merged_df[test_mask]

            if train_df.empty or test_df.empty:
                logger.warning(f"Fold {fold_idx}: train={len(train_df)}, test={len(test_df)}，跳過")
                continue

            X_train = train_df[self.feature_columns].values.astype(np.float32)
            y_train = train_df["label"].values.astype(np.int32)
            X_test = test_df[self.feature_columns].values.astype(np.float32)
            y_test = test_df["label"].values.astype(np.int32)

            train_sids = train_df["subject_id"].tolist()
            test_sids = test_df["subject_id"].tolist()
            train_bids = [self._extract_base_id(s) for s in train_sids]
            test_bids = [self._extract_base_id(s) for s in test_sids]

            fold_data[fold_idx] = FoldData(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                train_subject_ids=train_sids, test_subject_ids=test_sids,
                train_base_ids=train_bids, test_base_ids=test_bids,
            )

            logger.info(
                f"  Fold {fold_idx}: train={len(train_df)}, test={len(test_df)}, "
                f"train_class={dict(zip(*np.unique(y_train, return_counts=True)))}, "
                f"test_class={dict(zip(*np.unique(y_test, return_counts=True)))}"
            )

        return fold_data

    def _infer_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = df["y_true"].astype(int)
        invalid = df[df["label"] == -1]
        if len(invalid) > 0:
            logger.warning(f"移除 {len(invalid)} 筆無法識別的樣本")
            df = df[df["label"] != -1]
        return df

    @staticmethod
    def _extract_base_id(subject_id: str) -> str:
        match = re.match(r"^([A-Za-z]+\d+)", subject_id)
        if match:
            return match.group(1)
        return subject_id
