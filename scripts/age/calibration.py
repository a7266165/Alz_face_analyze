"""
執行三條年齡校正 pipeline：CalibrationModel / BootstrapCorrector / MeanCorrector

輸出目錄結構（在 AGE_CALIBRATION_DIR 下）：
  calibration/train90_val10/  — K-fold (Train 90% / Val 10%)
  calibration/train10_val90/  — K-fold (Train 10% / Val 90%)
  calibration/comparison.csv  — 兩種 calibration 的 MAE 比較
  bootstrap_correction/       — Bootstrap 1000 次
  mean_correction/            — 每整數歲平均 error 線性迴歸
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    AGE_CALIBRATION_DIR,
    BOOTSTRAP_DIR,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)
CALIBRATION_DIR = AGE_CALIBRATION_DIR / "calibration"
MEAN_CORRECTION_DIR = AGE_CALIBRATION_DIR / "mean_correction"
CORRECTION_DIR = AGE_CALIBRATION_DIR
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "calibration",
    str(Path(__file__).resolve().parent.parent.parent / "src" / "age" / "calibration.py"),
)
_cal = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cal)
BootstrapCorrector = _cal.BootstrapCorrector
CalibrationModel = _cal.CalibrationModel
MeanCorrector = _cal.MeanCorrector
load_demographics_for_calibration = _cal.load_demographics_for_calibration
run_multi_seed_calibration = _cal.run_multi_seed_calibration
save_and_plot_all = _cal.save_and_plot_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_calibration(df_matched: pd.DataFrame) -> None:
    """Pipeline 1: 10-Fold StratifiedKFold 校正（兩種模式 + 多種子）。"""
    for use_val_only, label in [(True, "train90_val10"), (False, "train10_val90")]:
        logger.info(f"=== Calibration: {label} ===")
        out_dir = CALIBRATION_DIR / label
        df_result = run_multi_seed_calibration(
            df_matched, n_splits=10, n_seeds=30, use_val_only=use_val_only,
        )
        save_and_plot_all(df_result, out_dir, method_name=f"Calibration ({label})")

    rows = []
    for label in ["train90_val10", "train10_val90"]:
        csv = CALIBRATION_DIR / label / "data" / "corrected_ages.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for grp in ["ACS", "NAD", "P", "All"]:
            sub = df if grp == "All" else df[df["group"] == grp]
            rows.append({
                "method": label,
                "group": grp,
                "n": len(sub),
                "MAE_before": round(sub["error_before"].abs().mean(), 2),
                "MAE_after": round(sub["error_after"].abs().mean(), 2),
                "mean_error_before": round(sub["error_before"].mean(), 2),
                "mean_error_after": round(sub["error_after"].mean(), 2),
            })
    pd.DataFrame(rows).to_csv(
        CALIBRATION_DIR / "comparison.csv", index=False, encoding="utf-8-sig",
    )
    logger.info("comparison.csv 已儲存")


def run_bootstrap(
    df_acs: pd.DataFrame, df_nad: pd.DataFrame, df_p: pd.DataFrame,
) -> None:
    """Pipeline 2: Bootstrap 隨機抽樣校正。"""
    logger.info("=== Bootstrap Correction ===")
    corrector = BootstrapCorrector(n_iter=1000, seed=42, min_age=60)
    results = corrector.run(df_acs, df_nad, df_p)
    df_result = BootstrapCorrector.build_results(df_acs, df_nad, df_p, results)
    save_and_plot_all(df_result, BOOTSTRAP_DIR, method_name="Bootstrap Correction")

    data_dir = BOOTSTRAP_DIR / "data"
    plots_dir = BOOTSTRAP_DIR / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results["coefs"].to_csv(
        data_dir / "bootstrap_coefficients.csv", index=False, encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in [
        (axes[0], "a", "Slope (a)"), (axes[1], "b", "Intercept (b)"),
    ]:
        ax.hist(results["coefs"][col], bins=40, color="#2196F3", alpha=0.7,
                edgecolor="white")
        mean_val = results["coefs"][col].mean()
        ax.axvline(mean_val, color="red", linestyle="--",
                   label=f"mean={mean_val:.4f}")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Bootstrap Coefficients Distribution (1000 iterations)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(plots_dir / "bootstrap_coefficients.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("bootstrap_coefficients.png 已儲存")


def run_mean_correction(
    df_acs: pd.DataFrame, df_nad: pd.DataFrame, df_p: pd.DataFrame,
) -> None:
    """Pipeline 3: 每整數歲平均 error 線性迴歸校正。"""
    logger.info("=== Mean Correction ===")
    corrector = MeanCorrector(min_age=60)
    a, b = corrector.fit(df_nad)
    df_result = corrector.transform(df_acs, df_nad, df_p)
    save_and_plot_all(df_result, MEAN_CORRECTION_DIR, method_name="Mean Correction")

    data_dir = MEAN_CORRECTION_DIR / "data"
    plots_dir = MEAN_CORRECTION_DIR / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"a": a, "b": b}]).to_csv(
        data_dir / "mean_correction_coefficients.csv", index=False,
    )
    corrector.per_age_stats.to_csv(
        data_dir / "per_age_mean_error.csv", index=False, encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    stats = corrector.per_age_stats
    ax.scatter(stats["age_int"], stats["mean_error"], s=stats["n"] * 3,
               c="#2196F3", alpha=0.6, edgecolors="white", linewidth=0.5)
    x_line = np.array([stats["age_int"].min(), stats["age_int"].max()])
    ax.plot(x_line, a * x_line + b, color="red", linewidth=2,
            label=f"y = {a:.4f}x + {b:.2f}")
    ax.set_xlabel("True Age (integer)")
    ax.set_ylabel("Mean Error (real − predicted)")
    ax.set_title("Mean Error per Integer Age — NAD (age≥60)\n"
                 "bubble size ∝ sample count")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(plots_dir / "regression_fit.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("regression_fit.png 已儲存")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Age correction pipelines")
    parser.add_argument(
        "--cohort-mode", default="all",
        help="Cohort mode: 'all' (default, backward-compat) or any of the "
             "16 canonical names (e.g. p_first_cdr05_hc_all_cdrall_or_mmseall)",
    )
    args = parser.parse_args()
    cohort_mode = args.cohort_mode

    logger.info(f"Demographics: {DEMOGRAPHICS_DIR}")
    logger.info(f"Predicted ages: {PREDICTED_AGES_FILE}")
    logger.info(f"Output root: {CORRECTION_DIR}")
    logger.info(f"Cohort mode: {cohort_mode}")

    df_acs, df_nad, df_p = load_demographics_for_calibration(
        DEMOGRAPHICS_DIR, PREDICTED_AGES_FILE, cohort_mode=cohort_mode,
    )
    df_matched = pd.concat([df_acs, df_nad, df_p], ignore_index=True)

    run_calibration(df_matched)
    run_bootstrap(df_acs, df_nad, df_p)
    run_mean_correction(df_acs, df_nad, df_p)

    logger.info("=== 三條 pipeline 全部完成 ===")


if __name__ == "__main__":
    main()
