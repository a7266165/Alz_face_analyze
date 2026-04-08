"""
年齡預測校正模組

提供三種校正方法的核心邏輯：
1. CalibrationModel — 10-Fold StratifiedKFold 誤差校正
2. BootstrapCorrector — Bootstrap 隨機抽樣校正
3. MeanCorrector — 每整數歲平均誤差線性迴歸校正

共用工具函式：
- load_predicted_ages: 載入預測年齡 JSON
- load_demographics_for_calibration: 載入 ACS/NAD/P 人口學資料
- match_ages: 配對預測年齡與真實年齡
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共用資料載入
# ---------------------------------------------------------------------------


def load_predicted_ages(path: Path) -> dict:
    """載入預測年齡 JSON 檔案。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demographics_for_calibration(
    demo_dir: Path,
    predicted_ages_file: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    載入人口學資料與預測年齡，回傳 (df_acs, df_nad, df_p)。

    每個 DataFrame 包含欄位：
    ID, subject, group, real_age, predicted_age, error, age_int
    """
    predicted_ages = json.loads(predicted_ages_file.read_text(encoding="utf-8"))

    dfs = {}
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        group = csv_name.replace(".csv", "")
        df = pd.read_csv(demo_dir / csv_name, encoding="utf-8-sig")
        df["group"] = group
        df["subject"] = df["ID"].apply(lambda x: x.rsplit("-", 1)[0])
        df["predicted_age"] = df["ID"].map(predicted_ages)
        df = df.dropna(subset=["predicted_age", "Age"])
        df["real_age"] = df["Age"]
        df["error"] = df["real_age"] - df["predicted_age"]
        df["age_int"] = df["real_age"].apply(lambda x: int(np.floor(x)))
        dfs[group] = df[
            ["ID", "subject", "group", "real_age", "predicted_age", "error", "age_int"]
        ].copy()

    logger.info(
        f"載入完成: ACS={len(dfs['ACS'])}, NAD={len(dfs['NAD'])}, P={len(dfs['P'])}"
    )
    return dfs["ACS"], dfs["NAD"], dfs["P"]


def match_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    """配對預測年齡與真實年齡。"""
    records = []
    for subject_id, pred_age in predicted_ages.items():
        row = demo[demo["ID"] == subject_id]
        if not row.empty:
            real_age = row["Age"].values[0]
            group = row["group"].values[0]
            subject = subject_id.rsplit("-", 1)[0]
            records.append({
                "ID": subject_id,
                "subject": subject,
                "real_age": real_age,
                "predicted_age": pred_age,
                "group": group,
                "error": real_age - pred_age,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 年齡分層工具
# ---------------------------------------------------------------------------


def get_age_stratum(age: float) -> str:
    """將年齡分類到年齡層。"""
    if age < 65:
        return "<65"
    elif age < 75:
        return "65-75"
    elif age < 85:
        return "75-85"
    else:
        return ">=85"


AGE_BINS = [
    ("<65", lambda a: a < 65),
    ("65-75", lambda a: (a >= 65) & (a < 75)),
    ("75-85", lambda a: (a >= 75) & (a < 85)),
    (">=85", lambda a: a >= 85),
]


# ---------------------------------------------------------------------------
# CalibrationModel — 10-Fold 校正
# ---------------------------------------------------------------------------


class CalibrationModel:
    """
    使用健康族群 (ACS + NAD) 進行 10-fold 校正。

    按年齡分層抽樣 (StratifiedKFold)，
    每個 fold 訓練 error ~ predicted_age 線性迴歸，
    Patient 族群取 10 個 fold 的平均校正值。
    """

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit_and_transform(self, df_matched: pd.DataFrame) -> pd.DataFrame:
        """執行 10-fold 校正，回傳包含 calibrated_age 的 DataFrame。"""
        df_healthy = df_matched[df_matched["group"].isin(["ACS", "NAD"])].copy()
        df_patient = df_matched[df_matched["group"] == "P"].copy()

        logger.info(f"健康族群: {len(df_healthy)} 人次")
        logger.info(f"Patient 族群: {len(df_patient)} 人次")

        subject_ages = df_healthy.groupby("subject")["real_age"].mean()
        healthy_subjects = subject_ages.index.values
        subject_strata = np.array([
            get_age_stratum(subject_ages[s]) for s in healthy_subjects
        ])

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.random_state,
        )

        healthy_calibrated = []
        patient_fold_calibrated: Dict[str, List[float]] = {
            pid: [] for pid in df_patient["ID"]
        }

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(healthy_subjects, subject_strata)
        ):
            train_subjects = healthy_subjects[train_idx]
            val_subjects = healthy_subjects[val_idx]

            df_train = df_healthy[df_healthy["subject"].isin(train_subjects)]
            df_val = df_healthy[df_healthy["subject"].isin(val_subjects)]

            # 擬合 error ~ predicted_age
            x_train = df_train["predicted_age"].values
            y_train = df_train["error"].values
            a, b = np.polyfit(x_train, y_train, 1)

            # 校正 validation
            for _, row in df_val.iterrows():
                pred = row["predicted_age"]
                calibrated = pred + (a * pred + b)
                healthy_calibrated.append({
                    "ID": row["ID"],
                    "subject": row["subject"],
                    "group": row["group"],
                    "real_age": row["real_age"],
                    "predicted_age": pred,
                    "calibrated_age": calibrated,
                    "error_before": row["error"],
                    "error_after": row["real_age"] - calibrated,
                    "fold": fold_idx + 1,
                })

            # 校正 Patient
            for _, row in df_patient.iterrows():
                pred = row["predicted_age"]
                calibrated = pred + (a * pred + b)
                patient_fold_calibrated[row["ID"]].append(calibrated)

        df_healthy_result = pd.DataFrame(healthy_calibrated)

        patient_results = []
        for _, row in df_patient.iterrows():
            calibrated_mean = np.mean(patient_fold_calibrated[row["ID"]])
            patient_results.append({
                "ID": row["ID"],
                "subject": row["subject"],
                "group": row["group"],
                "real_age": row["real_age"],
                "predicted_age": row["predicted_age"],
                "calibrated_age": calibrated_mean,
                "error_before": row["error"],
                "error_after": row["real_age"] - calibrated_mean,
                "fold": "avg",
            })

        df_patient_result = pd.DataFrame(patient_results)
        return pd.concat([df_healthy_result, df_patient_result], ignore_index=True)


# ---------------------------------------------------------------------------
# BootstrapCorrector
# ---------------------------------------------------------------------------


class BootstrapCorrector:
    """
    Bootstrap 校正：從 NAD (age>=60) 每個整數歲隨機抽 1 人建模。

    重複 n_iter 次取平均係數。
    NAD 剔除每次被抽到的訓練樣本後才校正。
    """

    def __init__(self, n_iter: int = 1000, seed: int = 42, min_age: int = 60):
        self.n_iter = n_iter
        self.seed = seed
        self.min_age = min_age

    @staticmethod
    def sample_one_per_age(
        df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """每個整數歲隨機抽 1 位 subject 的 1 筆紀錄。"""
        rows = []
        for age_int, group_df in df.groupby("age_int"):
            subjects = group_df["subject"].unique()
            chosen_subject = rng.choice(subjects)
            sub_rows = group_df[group_df["subject"] == chosen_subject]
            rows.append(sub_rows.iloc[[rng.integers(len(sub_rows))]])
        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def fit_error_model(df_train: pd.DataFrame) -> Tuple[float, float]:
        """擬合 error = a * real_age + b。"""
        a, b = np.polyfit(
            df_train["real_age"].values, df_train["error"].values, 1
        )
        return float(a), float(b)

    def run(
        self,
        df_acs: pd.DataFrame,
        df_nad: pd.DataFrame,
        df_p: pd.DataFrame,
    ) -> dict:
        """
        執行 bootstrap 校正。

        Returns:
            dict with keys: coefs (DataFrame), acs/nad/p_corrections (dict of lists),
            age_counts (Series)
        """
        df_nad_model = df_nad[df_nad["real_age"] >= self.min_age].copy()
        age_counts = df_nad_model.groupby("age_int").size().sort_index()

        all_coefs = []
        nad_corrections: Dict[str, List[float]] = defaultdict(list)
        acs_corrections: Dict[str, List[float]] = defaultdict(list)
        p_corrections: Dict[str, List[float]] = defaultdict(list)

        for i in range(self.n_iter):
            rng = np.random.default_rng(self.seed + i)
            df_sample = self.sample_one_per_age(df_nad_model, rng)
            a, b = self.fit_error_model(df_sample)
            all_coefs.append({"iter": i + 1, "a": a, "b": b})

            trained_subjects = set(df_sample["subject"].values)

            for _, row in df_acs.iterrows():
                corrected = row["predicted_age"] + (a * row["real_age"] + b)
                acs_corrections[row["ID"]].append(corrected)

            for _, row in df_p.iterrows():
                corrected = row["predicted_age"] + (a * row["real_age"] + b)
                p_corrections[row["ID"]].append(corrected)

            for _, row in df_nad.iterrows():
                if row["subject"] not in trained_subjects:
                    corrected = row["predicted_age"] + (a * row["real_age"] + b)
                    nad_corrections[row["ID"]].append(corrected)

        coefs_df = pd.DataFrame(all_coefs)
        logger.info(
            f"平均係數: a={coefs_df['a'].mean():.4f}, b={coefs_df['b'].mean():.4f}"
        )

        return {
            "coefs": coefs_df,
            "acs_corrections": dict(acs_corrections),
            "nad_corrections": dict(nad_corrections),
            "p_corrections": dict(p_corrections),
            "age_counts": age_counts,
        }

    @staticmethod
    def build_results(
        df_acs: pd.DataFrame,
        df_nad: pd.DataFrame,
        df_p: pd.DataFrame,
        bootstrap_results: dict,
    ) -> pd.DataFrame:
        """平均各 ID 的 bootstrap 校正結果。"""
        rows = []
        for df_grp, group, corr_dict in [
            (df_acs, "ACS", bootstrap_results["acs_corrections"]),
            (df_nad, "NAD", bootstrap_results["nad_corrections"]),
            (df_p, "P", bootstrap_results["p_corrections"]),
        ]:
            for _, row in df_grp.iterrows():
                corrections = corr_dict.get(row["ID"], [])
                if not corrections:
                    continue
                corrected = np.mean(corrections)
                rows.append({
                    "ID": row["ID"],
                    "subject": row["subject"],
                    "group": group,
                    "real_age": row["real_age"],
                    "predicted_age": row["predicted_age"],
                    "corrected_predicted_age": corrected,
                    "error_before": row["error"],
                    "error_after": row["real_age"] - corrected,
                    "age_int": row["age_int"],
                    "n_iters": len(corrections),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MeanCorrector
# ---------------------------------------------------------------------------


class MeanCorrector:
    """
    Mean Correction：每個整數歲平均 error → 線性迴歸 → 全體校正。

    使用 NAD (age>=60) 作為建模資料。
    """

    def __init__(self, min_age: int = 60):
        self.min_age = min_age
        self.a: float = 0.0
        self.b: float = 0.0
        self.per_age_stats: pd.DataFrame = pd.DataFrame()

    def fit(self, df_nad: pd.DataFrame) -> Tuple[float, float]:
        """擬合模型，回傳 (a, b)。"""
        df_model = df_nad[df_nad["real_age"] >= self.min_age].copy()
        self.per_age_stats = (
            df_model.groupby("age_int")["error"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_error", "std": "std_error", "count": "n"})
            .sort_index()
            .reset_index()
        )
        self.a, self.b = np.polyfit(
            self.per_age_stats["age_int"].values,
            self.per_age_stats["mean_error"].values,
            1,
        )
        self.a, self.b = float(self.a), float(self.b)
        logger.info(f"線性迴歸: error = {self.a:.4f} × age + {self.b:.2f}")
        return self.a, self.b

    def transform(
        self,
        df_acs: pd.DataFrame,
        df_nad: pd.DataFrame,
        df_p: pd.DataFrame,
    ) -> pd.DataFrame:
        """套用校正到全體。"""
        rows = []
        for df_grp in [df_acs, df_nad, df_p]:
            for _, row in df_grp.iterrows():
                fitted_error = self.a * row["real_age"] + self.b
                corrected = row["predicted_age"] + fitted_error
                rows.append({
                    "ID": row["ID"],
                    "group": row["group"],
                    "real_age": row["real_age"],
                    "predicted_age": row["predicted_age"],
                    "corrected_predicted_age": corrected,
                    "error_before": row["error"],
                    "error_after": row["real_age"] - corrected,
                    "age_int": row["age_int"],
                })
        return pd.DataFrame(rows)
