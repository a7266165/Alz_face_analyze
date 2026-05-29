"""
BMI prediction error analysis by group (P / NAD / ACS).

Produces:
  1. bmi_error_stats.csv  — per-group + total error statistics
  2. bmi_error_by_age.png — prediction error vs true age (per-integer bin)
  3. bmi_error_by_bmi.png — prediction error vs true BMI (per-integer bin)

Usage:
    conda run -n Alz_face_main_analysis python scripts/bmi/plot_bmi_error_analysis.py
    conda run -n Alz_face_main_analysis python scripts/bmi/plot_bmi_error_analysis.py --model xgb
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BMI_ANALYSIS_DIR, DEMOGRAPHICS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORS = {"P": "#F44336", "NAD": "#2196F3", "ACS": "#4CAF50"}
GROUP_LABELS = {"P": "Dementia (P)", "NAD": "Non-Dementia (NAD)", "ACS": "ACS"}
MIN_BIN_COUNT = 3


def _parse_group(sid: str) -> str:
    if sid.startswith("P"):
        return "P"
    if sid.startswith("NAD"):
        return "NAD"
    if sid.startswith("ACS"):
        return "ACS"
    return "OTHER"


def load_oof_with_age(model_name: str) -> pd.DataFrame:
    oof_csv = BMI_ANALYSIS_DIR / f"oof_{model_name}.csv"
    if not oof_csv.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_csv}")
    df = pd.read_csv(oof_csv)
    df["group"] = df["ID"].apply(_parse_group)

    from src.common.cohort import load_demographics
    ages = load_demographics()[["ID", "Age"]]

    df = df.merge(ages, on="ID", how="left")
    df["error"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["error"].abs()
    df["age_int"] = df["Age"].round().astype("Int64")
    df["bmi_int"] = df["y_true"].round().astype("Int64")
    return df


# ---------------------------------------------------------------------------
# (1) Error statistics table
# ---------------------------------------------------------------------------

def build_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in ["P", "NAD", "ACS", "Total"]:
        sub = df if grp == "Total" else df[df["group"] == grp]
        n = len(sub)
        if n == 0:
            continue
        mae = sub["abs_error"].mean()
        rmse = np.sqrt((sub["error"] ** 2).mean())
        mean_err = sub["error"].mean()
        std_err = sub["error"].std()
        r, p = stats.pearsonr(sub["y_true"], sub["y_pred"])
        ss_res = (sub["error"] ** 2).sum()
        ss_tot = ((sub["y_true"] - sub["y_true"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({
            "Group": grp,
            "N": n,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "Error (Mean±Std)": f"{mean_err:+.3f}±{std_err:.3f}",
            "Pearson_r": round(r, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (2)(3) Error-by-X line plots
# ---------------------------------------------------------------------------

def plot_error_by_x(df, x_col, x_label, title, out_path, xlim=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    for grp in ["P", "NAD", "ACS"]:
        sub = df[df["group"] == grp].dropna(subset=[x_col])
        if len(sub) == 0:
            continue
        st = sub.groupby(x_col)["error"].agg(["mean", "std", "count"])
        st = st[st["count"] >= MIN_BIN_COUNT].sort_index()
        if len(st) == 0:
            continue

        xs = st.index.values.astype(float)
        means = st["mean"].values
        stds = st["std"].fillna(0).values

        color = COLORS[grp]
        label = f"{GROUP_LABELS[grp]} (n={len(sub)})"
        ax.plot(xs, means, color=color, linewidth=2,
                marker="o", markersize=4, label=label)
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Prediction Error (True − Predicted)", fontsize=12)
    ax.set_title(title, fontsize=13)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="svr",
                        choices=["ridge", "svr", "xgb"])
    args = parser.parse_args()

    logger.info(f"Loading OOF predictions for {args.model}...")
    df = load_oof_with_age(args.model)
    logger.info(f"Loaded {len(df)} predictions with age info")

    BMI_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # (1) Statistics table
    stats_df = build_error_stats(df)
    stats_csv = BMI_ANALYSIS_DIR / f"bmi_error_stats_{args.model}.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info(f"  Saved {stats_csv.name}")
    print(f"\n{stats_df.to_string(index=False)}\n")

    # (2) Error by age
    plot_error_by_x(
        df, x_col="age_int",
        x_label="True Age (years)",
        title=f"BMI Prediction Error by Age ({args.model.upper()})",
        out_path=BMI_ANALYSIS_DIR / f"bmi_error_by_age_{args.model}.png",
        xlim=(40, 100),
    )

    # (3) Error by BMI
    plot_error_by_x(
        df, x_col="bmi_int",
        x_label="True BMI",
        title=f"BMI Prediction Error by True BMI ({args.model.upper()})",
        out_path=BMI_ANALYSIS_DIR / f"bmi_error_by_bmi_{args.model}.png",
        xlim=(14, 40),
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
