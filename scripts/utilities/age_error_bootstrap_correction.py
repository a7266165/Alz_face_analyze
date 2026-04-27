"""
scripts/utilities/age_error_bootstrap_correction.py
使用 NAD (age>=60) 進行 bootstrap 年齡誤差校正：
1. 篩選 NAD 年齡 >= 60 的受試者
2. 每個整數歲隨機抽 1 人，擬合 e = a*y + b（y = 真實年齡）
3. 重複 1000 次取 (a, b) 平均值
4. 校正公式：corrected = yp + (a*y + b)
5. 繪製校正前後結果圖 + 殘差折線圖

統一輸出至 bootstrap_correction/ 子目錄：
  corrected_ages.csv, summary_stats.csv,
  scatter_before_after.png, error_distribution_before_after.png,
  residual_by_age_combined.png,
  + bootstrap 專屬: bootstrap_coefficients.csv/.png
"""

import sys
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import (
    DEMOGRAPHICS_DIR,
    BOOTSTRAP_DIR,
    PREDICTED_AGES_FILE,
)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "calibration",
    PROJECT_ROOT / "src" / "extractor" / "features" / "age" / "calibration.py",
)
_cal = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cal)

load_demographics_for_calibration = _cal.load_demographics_for_calibration
save_and_plot_all = _cal.save_and_plot_all
COLORS = _cal.COLORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心演算法
# ---------------------------------------------------------------------------


def count_age_distribution(df: pd.DataFrame) -> pd.Series:
    return df.groupby("age_int").size().sort_index()


def filter_nad_60plus(df_nad: pd.DataFrame) -> pd.DataFrame:
    df = df_nad[df_nad["real_age"] >= 60].copy()
    logger.info(f"NAD age>=60: {len(df)} visits ({df['subject'].nunique()} subjects)")
    return df


def sample_one_per_age(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for age_int, group_df in df.groupby("age_int"):
        subjects = group_df["subject"].unique()
        chosen_subject = rng.choice(subjects)
        sub_rows = group_df[group_df["subject"] == chosen_subject]
        rows.append(sub_rows.iloc[[rng.integers(len(sub_rows))]])
    return pd.concat(rows, ignore_index=True)


def fit_error_model(df_train: pd.DataFrame) -> tuple[float, float]:
    a, b = np.polyfit(df_train["real_age"].values, df_train["error"].values, 1)
    return float(a), float(b)


def run_bootstrap(
    df_nad_model: pd.DataFrame,
    df_acs: pd.DataFrame,
    df_nad: pd.DataFrame,
    df_p: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    age_counts = count_age_distribution(df_nad_model)
    logger.info(
        f"建模資料: {len(age_counts)} 個整數歲 "
        f"(range {age_counts.index.min()}-{age_counts.index.max()})"
    )

    all_coefs = []
    nad_corrections: dict[str, list[float]] = defaultdict(list)
    acs_corrections: dict[str, list[float]] = defaultdict(list)
    p_corrections: dict[str, list[float]] = defaultdict(list)

    for i in range(n_iter):
        rng = np.random.default_rng(seed + i)
        df_sample = sample_one_per_age(df_nad_model, rng)
        a, b = fit_error_model(df_sample)
        all_coefs.append({"iter": i + 1, "a": a, "b": b})

        if (i + 1) % 100 == 0:
            logger.info(f"  Iter {i+1}/{n_iter}: e = {a:.4f}*y + {b:.4f}")

        trained_subjects = set(df_sample["subject"].values)

        for _, row in df_acs.iterrows():
            corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
            acs_corrections[row["ID"]].append(corrected_age)

        for _, row in df_p.iterrows():
            corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
            p_corrections[row["ID"]].append(corrected_age)

        for _, row in df_nad.iterrows():
            if row["subject"] not in trained_subjects:
                corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
                nad_corrections[row["ID"]].append(corrected_age)

    coefs_df = pd.DataFrame(all_coefs)
    a_mean = float(coefs_df["a"].mean())
    b_mean = float(coefs_df["b"].mean())
    logger.info(f"平均係數: a={a_mean:.4f}, b={b_mean:.4f}")

    return {
        "coefs": coefs_df,
        "acs_corrections": acs_corrections,
        "nad_corrections": nad_corrections,
        "p_corrections": p_corrections,
        "age_counts": age_counts,
    }


def build_results(
    df_acs: pd.DataFrame,
    df_nad: pd.DataFrame,
    df_p: pd.DataFrame,
    bootstrap_results: dict,
) -> pd.DataFrame:
    rows = []
    for df, group, corr_dict in [
        (df_acs, "ACS", bootstrap_results["acs_corrections"]),
        (df_nad, "NAD", bootstrap_results["nad_corrections"]),
        (df_p, "P", bootstrap_results["p_corrections"]),
    ]:
        for _, row in df.iterrows():
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
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap 專屬圖表
# ---------------------------------------------------------------------------


def plot_coefficients(coefs: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    a_vals = coefs["a"].values
    b_vals = coefs["b"].values

    axes[0].scatter(coefs["iter"], a_vals, color="#FF9800", s=60, zorder=5,
                    edgecolors="white", linewidth=0.5)
    axes[0].axhline(y=a_vals.mean(), color="red", linestyle="--",
                    label=f"Mean = {a_vals.mean():.4f}")
    axes[0].fill_between(
        [0.5, len(coefs) + 0.5],
        a_vals.mean() - a_vals.std(),
        a_vals.mean() + a_vals.std(),
        alpha=0.15, color="red", label=f"SD = {a_vals.std():.4f}",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Coefficient a")
    axes[0].set_title("Slope (a): error = a × real_age + b", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(coefs["iter"], b_vals, color="#9C27B0", s=60, zorder=5,
                    edgecolors="white", linewidth=0.5)
    axes[1].axhline(y=b_vals.mean(), color="red", linestyle="--",
                    label=f"Mean = {b_vals.mean():.4f}")
    axes[1].fill_between(
        [0.5, len(coefs) + 0.5],
        b_vals.mean() - b_vals.std(),
        b_vals.mean() + b_vals.std(),
        alpha=0.15, color="red", label=f"SD = {b_vals.std():.4f}",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Coefficient b")
    axes[1].set_title("Intercept (b): error = a × real_age + b", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Bootstrap Coefficient Stability ({len(coefs)} Iterations)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plots_dir / "bootstrap_coefficients.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("bootstrap_coefficients.png 已儲存")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = BOOTSTRAP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    df_acs, df_nad, df_p = load_demographics_for_calibration(
        DEMOGRAPHICS_DIR, PREDICTED_AGES_FILE,
    )

    # 篩選 NAD 60+
    df_nad_model = filter_nad_60plus(df_nad)

    age_counts = count_age_distribution(df_nad_model)
    logger.info(f"\nNAD (age>=60) 年齡分佈 (整數歲):")
    for age, cnt in age_counts.items():
        logger.info(f"  {age} 歲: {cnt} 筆")

    # Bootstrap 校正
    bootstrap_results = run_bootstrap(
        df_nad_model, df_acs, df_nad, df_p, n_iter=1000, seed=42,
    )

    # 平均校正結果
    df_result = build_results(df_acs, df_nad, df_p, bootstrap_results)

    # 統一輸出 (CSV + 統計表 + 3 張共用圖)
    save_and_plot_all(df_result, output_dir, method_name="Bootstrap Correction")

    # Bootstrap 專屬: 係數
    coefs = bootstrap_results["coefs"]
    coefs_with_summary = pd.concat([
        coefs,
        pd.DataFrame([{"iter": "mean", "a": coefs["a"].mean(), "b": coefs["b"].mean()}]),
        pd.DataFrame([{"iter": "std", "a": coefs["a"].std(), "b": coefs["b"].std()}]),
    ], ignore_index=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    coefs_with_summary.to_csv(
        data_dir / "bootstrap_coefficients.csv", index=False, encoding="utf-8-sig",
    )
    plot_coefficients(coefs, output_dir)

    logger.info(f"\n所有輸出已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
