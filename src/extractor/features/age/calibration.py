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

統一輸出格式：
- CSV 欄位: ID, subject, group, real_age, predicted_age, corrected_age,
             error_before, error_after, age_int
- 共用圖表: scatter_before_after, error_distribution_before_after,
            residual_by_age_combined
- 共用統計: summary_stats.csv
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# 共用配色
COLORS = {"ACS": "#4CAF50", "NAD": "#2196F3", "P": "#F44336"}


# ---------------------------------------------------------------------------
# 共用資料載入
# ---------------------------------------------------------------------------


def load_predicted_ages(path: Path) -> dict:
    """載入預測年齡 JSON 檔案。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_first_visit_with_npy_fallback(
    df_visits: pd.DataFrame, predicted_ages: dict,
) -> pd.DataFrame:
    """Per subject: pick the earliest visit that has a predicted_age (= has
    .npy + age extraction). Falls back to the earliest visit if none qualify.
    Mirrors `_pick_first_visit_with_features` in run_4arm_deep_dive.py.
    """
    picked = []
    for subj, g in df_visits.groupby("subject", as_index=False, sort=False):
        g = g.sort_values("ID")
        with_pred = g[g["ID"].astype(str).isin(predicted_ages)]
        if len(with_pred) > 0:
            picked.append(with_pred.iloc[0])
        else:
            picked.append(g.iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def load_demographics_for_calibration(
    demo_dir: Path,
    predicted_ages_file: Path,
    cohort_mode: str = "all",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    載入人口學資料與預測年齡，回傳 (df_acs, df_nad, df_p)。

    每個 DataFrame 包含欄位：
    ID, subject, group, real_age, predicted_age, error, age_int

    cohort_mode:
        "all" (default, backward-compat): every visit per subject. Drops only
            visits without predicted_age or real Age.
        "p_first_hc_all": P keeps only first-visit (with Global_CDR>=0.5 and
            .npy fallback); NAD/ACS keep ALL visits without strict HC filter.
    """
    if cohort_mode not in ("all", "p_first_hc_all"):
        raise ValueError(f"unknown cohort_mode: {cohort_mode}")

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

        if cohort_mode == "p_first_hc_all" and group == "P":
            # P side: filter to Global_CDR>=0.5 visits, then pick earliest
            # eligible visit per subject (with .npy fallback).
            cdr = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
            df_eligible = df[cdr >= 0.5].copy()
            df = _pick_first_visit_with_npy_fallback(df_eligible, predicted_ages)

        dfs[group] = df[
            ["ID", "subject", "group", "real_age", "predicted_age", "error", "age_int"]
        ].copy()

    logger.info(
        f"載入完成 (cohort_mode={cohort_mode}): "
        f"ACS={len(dfs['ACS'])}, NAD={len(dfs['NAD'])}, P={len(dfs['P'])}"
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
    使用健康族群 (ACS + NAD) 進行 K-fold 校正。

    按年齡分層抽樣 (StratifiedKFold)，
    每個 fold 訓練 error ~ predicted_age 線性迴歸。

    use_val_only=True  (Train 90% / Val 10%):
        健康組只用被分到 val set 的 1 個 fold 校正。
    use_val_only=False (Train 10% / Val 90%):
        健康組被分到 val set 的 9 個 fold 校正後取平均。
    P 組在兩種模式下都用全部 K 個 fold 平均。
    """

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit_and_transform(
        self, df_matched: pd.DataFrame, use_val_only: bool = True,
    ) -> pd.DataFrame:
        """執行 K-fold 校正，回傳包含 corrected_age 的 DataFrame。"""
        df_healthy = df_matched[df_matched["group"].isin(["ACS", "NAD"])].copy()
        df_patient = df_matched[df_matched["group"] == "P"].copy()

        mode = "Train 90%/Val 10%" if use_val_only else "Train 10%/Val 90%"
        logger.info(f"模式: {mode}")
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

        # 健康組和 P 組都用 dict 累積多 fold 結果
        healthy_fold_calibrated: Dict[str, List[float]] = defaultdict(list)
        patient_fold_calibrated: Dict[str, List[float]] = defaultdict(list)

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

            # 校正健康組 validation set
            for _, row in df_val.iterrows():
                pred = row["predicted_age"]
                calibrated = pred + (a * pred + b)
                healthy_fold_calibrated[row["ID"]].append(calibrated)

            # 校正 Patient（全部 fold 都套用）
            for _, row in df_patient.iterrows():
                pred = row["predicted_age"]
                calibrated = pred + (a * pred + b)
                patient_fold_calibrated[row["ID"]].append(calibrated)

        # 組裝結果
        def _build_rows(df_src, fold_dict):
            rows = []
            for _, row in df_src.iterrows():
                vals = fold_dict.get(row["ID"], [])
                if not vals:
                    continue
                corrected = np.mean(vals)
                rows.append({
                    "ID": row["ID"],
                    "subject": row["subject"],
                    "group": row["group"],
                    "real_age": row["real_age"],
                    "predicted_age": row["predicted_age"],
                    "corrected_age": corrected,
                    "error_before": row["error"],
                    "error_after": row["real_age"] - corrected,
                    "age_int": int(np.floor(row["real_age"])),
                })
            return rows

        healthy_rows = _build_rows(df_healthy, healthy_fold_calibrated)
        patient_rows = _build_rows(df_patient, patient_fold_calibrated)

        return pd.concat(
            [pd.DataFrame(healthy_rows), pd.DataFrame(patient_rows)],
            ignore_index=True,
        )


def run_multi_seed_calibration(
    df_matched: pd.DataFrame,
    n_splits: int = 10,
    n_seeds: int = 30,
    use_val_only: bool = True,
    base_seed: int = 42,
) -> pd.DataFrame:
    """多隨機種子校正：跑 n_seeds 次 StratifiedKFold，每個 ID 取平均。"""
    all_corrected: Dict[str, List[float]] = defaultdict(list)

    for i in range(n_seeds):
        seed = base_seed + i
        model = CalibrationModel(n_splits=n_splits, random_state=seed)
        result = model.fit_and_transform(df_matched, use_val_only=use_val_only)
        for _, row in result.iterrows():
            all_corrected[row["ID"]].append(row["corrected_age"])

        if (i + 1) % 10 == 0:
            logger.info(f"  Seed {i+1}/{n_seeds} done")

    # 取回 metadata（從原始 df_matched），計算跨種子平均
    rows = []
    for _, row in df_matched.iterrows():
        vals = all_corrected.get(row["ID"], [])
        if not vals:
            continue
        corrected = np.mean(vals)
        rows.append({
            "ID": row["ID"],
            "subject": row["subject"],
            "group": row["group"],
            "real_age": row["real_age"],
            "predicted_age": row["predicted_age"],
            "corrected_age": corrected,
            "error_before": row["error"],
            "error_after": row["real_age"] - corrected,
            "age_int": row.get("age_int", int(np.floor(row["real_age"]))),
        })

    logger.info(f"多種子校正完成: {n_seeds} seeds, {len(rows)} 筆")
    return pd.DataFrame(rows)


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
                    "corrected_age": corrected,
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
                    "subject": row.get("subject", row["ID"].rsplit("-", 1)[0]),
                    "group": row["group"],
                    "real_age": row["real_age"],
                    "predicted_age": row["predicted_age"],
                    "corrected_age": corrected,
                    "error_before": row["error"],
                    "error_after": row["real_age"] - corrected,
                    "age_int": row["age_int"],
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 共用輸出函式 — 統一三種方法的圖表與統計
# ---------------------------------------------------------------------------


def export_summary_stats(df_result: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """輸出校正前後統計表 CSV（族群 × 年齡分層）。"""

    def _row(sub: pd.DataFrame, group: str, age_group: str) -> dict:
        n = len(sub)
        return {
            "group": group,
            "age_group": age_group,
            "n": n,
            "real_age": f"{sub['real_age'].mean():.2f}±{sub['real_age'].std():.2f}",
            "predicted_age": f"{sub['predicted_age'].mean():.2f}±{sub['predicted_age'].std():.2f}",
            "corrected_age": f"{sub['corrected_age'].mean():.2f}±{sub['corrected_age'].std():.2f}",
            "error_before": f"{sub['error_before'].mean():.2f}±{sub['error_before'].std():.2f}",
            "error_after": f"{sub['error_after'].mean():.2f}±{sub['error_after'].std():.2f}",
            "MAE_before": f"{sub['error_before'].abs().mean():.2f}",
            "MAE_after": f"{sub['error_after'].abs().mean():.2f}",
        }

    rows = []
    for grp in ["ACS", "NAD", "ACS+NAD", "P", "All"]:
        if grp == "All":
            df_grp = df_result
        elif grp == "ACS+NAD":
            df_grp = df_result[df_result["group"].isin(["ACS", "NAD"])]
        else:
            df_grp = df_result[df_result["group"] == grp]

        if len(df_grp) == 0:
            continue
        rows.append(_row(df_grp, grp, "Total"))

        for label, mask_fn in AGE_BINS:
            sub = df_grp[mask_fn(df_grp["real_age"])]
            if len(sub) > 0:
                rows.append(_row(sub, grp, label))

    df_stats = pd.DataFrame(rows)
    csv_path = output_dir / "summary_stats.csv"
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"統計表已儲存: {csv_path}")
    return df_stats


def print_summary(df_result: pd.DataFrame, method_name: str = ""):
    """印出校正前後統計摘要。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{method_name} 校正前後統計比較")
    logger.info("=" * 60)
    for grp in ["ACS", "NAD", "P", "All"]:
        sub = df_result if grp == "All" else df_result[df_result["group"] == grp]
        n = len(sub)
        if n == 0:
            continue
        mae_b = sub["error_before"].abs().mean()
        mae_a = sub["error_after"].abs().mean()
        mean_b, std_b = sub["error_before"].mean(), sub["error_before"].std()
        mean_a, std_a = sub["error_after"].mean(), sub["error_after"].std()
        logger.info(
            f"{grp:>5s} (n={n}): "
            f"MAE {mae_b:.2f} → {mae_a:.2f}, "
            f"Mean {mean_b:.2f}±{std_b:.2f} → {mean_a:.2f}±{std_a:.2f}"
        )


def plot_scatter_before_after(
    df_result: pd.DataFrame, output_dir: Path, method_name: str = "",
):
    """2×3 散佈圖：校正前後 × ACS/NAD/P。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    groups = ["ACS", "NAD", "P"]

    for col_idx, grp in enumerate(groups):
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        color = COLORS[grp]

        for row_idx, (y_col, label, err_col) in enumerate([
            ("predicted_age", "Before Correction", "error_before"),
            ("corrected_age", "After Correction", "error_after"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                sub["real_age"], sub[y_col],
                c=color, alpha=0.4, s=20, edgecolors="white", linewidth=0.3,
            )
            vmin, vmax = 20, 100
            ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1,
                    label="y = x")

            x = sub["real_age"].values.astype(np.float64)
            y = sub[y_col].values.astype(np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) > 2:
                a, b = np.polyfit(x, y, 1)
                x_line = np.array([vmin, vmax])
                ax.plot(x_line, a * x_line + b, color="#FF9800", linewidth=2,
                        alpha=0.8, label=f"y = {a:.2f}x + {b:.2f}")

            corr = sub["real_age"].corr(sub[y_col])
            mae = sub[err_col].abs().mean()
            mean_err = sub[err_col].mean()

            ax.set_xlabel("Real Age", fontsize=11)
            ax.set_ylabel("Predicted Age" if row_idx == 0 else "Corrected Age",
                          fontsize=11)
            ax.set_title(
                f"{grp} - {label}\n"
                f"(n={len(sub)}, r={corr:.3f}, MAE={mae:.2f}, "
                f"Mean Err={mean_err:.2f})",
                fontsize=11,
            )
            ax.legend(fontsize=9)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    title = "Real Age vs Predicted Age: Before & After"
    if method_name:
        title += f" {method_name}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "scatter_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("scatter_before_after.png 已儲存")


def plot_error_distribution(
    df_result: pd.DataFrame, output_dir: Path, method_name: str = "",
):
    """2×3 直方圖：校正前後誤差分佈。"""
    groups = ["ACS", "NAD", "P"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col_idx, grp in enumerate(groups):
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        color = COLORS[grp]

        ax_before = axes[0, col_idx]
        ax_before.hist(sub["error_before"], bins=30, color=color, alpha=0.7,
                       edgecolor="white", linewidth=0.5)
        mae_b = sub["error_before"].abs().mean()
        mean_b = sub["error_before"].mean()
        ax_before.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax_before.set_title(
            f"{grp} Before (n={len(sub)})\nMean={mean_b:.2f}, MAE={mae_b:.2f}",
            fontsize=11,
        )
        ax_before.set_xlabel("Error (real − predicted)")
        ax_before.grid(True, alpha=0.3)

        ax_after = axes[1, col_idx]
        ax_after.hist(sub["error_after"], bins=30, color=color, alpha=0.7,
                      edgecolor="white", linewidth=0.5)
        mae_a = sub["error_after"].abs().mean()
        mean_a = sub["error_after"].mean()
        ax_after.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax_after.set_title(
            f"{grp} After\nMean={mean_a:.2f}, MAE={mae_a:.2f}",
            fontsize=11,
        )
        ax_after.set_xlabel("Corrected Error")
        ax_after.grid(True, alpha=0.3)

    title = "Age Error Distribution: Before vs After"
    if method_name:
        title += f" {method_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "error_distribution_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("error_distribution_before_after.png 已儲存")


def plot_residual_combined(
    df_result: pd.DataFrame, output_dir: Path, method_name: str = "",
):
    """三組疊合的校正後殘差折線圖。"""
    fig, ax = plt.subplots(figsize=(14, 5))

    for grp, color in COLORS.items():
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        st = sub.groupby("age_int")["error_after"].agg(["mean", "std", "count"])
        st = st[st["count"] >= 3].sort_index()
        if len(st) == 0:
            continue
        ages = st.index.values
        means = st["mean"].values
        stds = st["std"].fillna(0).values
        ax.plot(ages, means, color=color, linewidth=2,
                marker="o", markersize=4, label=f"{grp} (n={len(sub)})")
        ax.fill_between(ages, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (y)", fontsize=12)
    ax.set_ylabel("Residual ε = y − corrected", fontsize=12)
    title = "Correction Residual by True Age — ACS / NAD / P (mean ± std)"
    if method_name:
        title += f"\n{method_name}"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(output_dir / "residual_by_age_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("residual_by_age_combined.png 已儲存")


def save_and_plot_all(
    df_result: pd.DataFrame,
    output_dir: Path,
    method_name: str = "",
):
    """統一輸出入口：CSV → output_dir/data/，PNG → output_dir/plots/。"""
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # CSV → data/
    csv_path = data_dir / "corrected_ages.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"校正結果已儲存: {csv_path}")

    # 統計 CSV → data/
    print_summary(df_result, method_name)
    export_summary_stats(df_result, data_dir)

    # 共用圖表 → plots/
    plot_scatter_before_after(df_result, plots_dir, method_name)
    plot_error_distribution(df_result, plots_dir, method_name)
    plot_residual_combined(df_result, plots_dir, method_name)
