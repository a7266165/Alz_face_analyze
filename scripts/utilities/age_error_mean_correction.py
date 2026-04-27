"""
scripts/utilities/age_error_mean_correction.py
Mean Correction：NAD (age>=60) 每個整數歲平均 error 後擬合線性迴歸，
再套用到全體 (ACS / NAD / P)。

步驟：
1. 篩選 NAD age>=60
2. 每個整數歲計算平均 error
3. 對 (age, mean_error) 擬合 error = a*age + b
4. 校正公式：corrected = predicted_age + (a * real_age + b)
5. 輸出 CSV + 繪圖

統一輸出至 mean_correction/ 子目錄：
  corrected_ages.csv, summary_stats.csv,
  scatter_before_after.png, error_distribution_before_after.png,
  residual_by_age_combined.png,
  + mean correction 專屬: regression_fit.png, per_age_mean_error.csv,
    mean_correction_coefficients.csv
"""

import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import (
    DEMOGRAPHICS_DIR,
    MEAN_CORRECTION_DIR,
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


def compute_per_age_mean_error(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("age_int")["error"].agg(["mean", "std", "count"])
    stats = stats.rename(columns={"mean": "mean_error", "std": "std_error", "count": "n"})
    return stats.sort_index().reset_index()


def fit_error_model(per_age: pd.DataFrame) -> tuple[float, float]:
    a, b = np.polyfit(per_age["age_int"].values, per_age["mean_error"].values, 1)
    return float(a), float(b)


def apply_correction(
    df_acs: pd.DataFrame, df_nad: pd.DataFrame, df_p: pd.DataFrame,
    a: float, b: float,
) -> pd.DataFrame:
    rows = []
    for df_grp in [df_acs, df_nad, df_p]:
        for _, row in df_grp.iterrows():
            fitted_error = a * row["real_age"] + b
            corrected = row["predicted_age"] + fitted_error
            rows.append({
                "ID": row["ID"],
                "subject": row["subject"],
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
# Mean Correction 專屬圖表
# ---------------------------------------------------------------------------


def plot_regression_fit(per_age: pd.DataFrame, a: float, b: float, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        per_age["age_int"], per_age["mean_error"],
        c=COLORS["NAD"], s=60, zorder=5, edgecolors="white", linewidth=0.5,
        label=f"Per-age mean error (n={len(per_age)} ages)",
    )
    ax.errorbar(
        per_age["age_int"], per_age["mean_error"],
        yerr=per_age["std_error"].fillna(0),
        fmt="none", ecolor=COLORS["NAD"], alpha=0.3, capsize=2,
    )

    x_line = np.array([per_age["age_int"].min() - 1, per_age["age_int"].max() + 1])
    ax.plot(
        x_line, a * x_line + b,
        color="#FF9800", linewidth=2.5,
        label=f"Fit: error = {a:.4f} × age + {b:.2f}",
    )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (integer)", fontsize=12)
    ax.set_ylabel("Mean Error (real − predicted)", fontsize=12)
    ax.set_title(
        f"NAD (age≥60) Per-Age Mean Error & Linear Fit\n"
        f"(a={a:.4f}, b={b:.2f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plots_dir / "regression_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("regression_fit.png 已儲存")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = MEAN_CORRECTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    df_acs, df_nad, df_p = load_demographics_for_calibration(
        DEMOGRAPHICS_DIR, PREDICTED_AGES_FILE,
    )

    # 篩選 NAD 60+
    df_nad_model = df_nad[df_nad["real_age"] >= 60].copy()
    logger.info(
        f"NAD age>=60: {len(df_nad_model)} visits "
        f"({df_nad_model['subject'].nunique()} subjects)"
    )

    # 每個整數歲平均 error
    per_age = compute_per_age_mean_error(df_nad_model)
    logger.info(f"\n每個整數歲平均 error ({len(per_age)} ages):")
    for _, row in per_age.iterrows():
        logger.info(
            f"  {int(row['age_int'])} 歲: n={int(row['n'])}, "
            f"mean_error={row['mean_error']:.2f}±{row['std_error']:.2f}"
        )

    # 擬合線性模型
    a, b = fit_error_model(per_age)
    logger.info(f"\n線性迴歸: error = {a:.4f} × age + {b:.2f}")

    # 套用校正到全體
    df_result = apply_correction(df_acs, df_nad, df_p, a, b)

    # 統一輸出 (CSV + 統計表 + 3 張共用圖)
    save_and_plot_all(df_result, output_dir, method_name="Mean Correction")

    # Mean Correction 專屬
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    per_age.to_csv(
        data_dir / "per_age_mean_error.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([{"a": a, "b": b}]).to_csv(
        data_dir / "mean_correction_coefficients.csv", index=False, encoding="utf-8-sig",
    )
    plot_regression_fit(per_age, a, b, output_dir)

    logger.info(f"\n所有輸出已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
