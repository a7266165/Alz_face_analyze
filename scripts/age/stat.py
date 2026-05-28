"""
scripts/age/stat.py
Age prediction error statistics — stratified CSVs and correlation analysis.

Outputs (to stat/ directory):
  age_error_stat_2.csv          — age-stratified stats per group
  age_error_sliding_window.csv  — 10-year sliding window stats
  patient_cdr_age_error.csv     — Patient CDR-stratified stats
  patient_mmse_age_error.csv    — Patient MMSE-stratified stats
  patient_casi_age_error.csv    — Patient CASI-stratified stats
  patient_mmse_error_corr.csv   — MMSE-error correlation
  patient_casi_error_corr.csv   — CASI-error correlation
  patient_mmse_vs_error.png     — MMSE-error scatter
  patient_casi_vs_error.png     — CASI-error scatter

Usage:
  conda run -n Alz_face_age python scripts/age/stat.py
  conda run -n Alz_face_age python scripts/age/stat.py --cohort-mode p_first_cdrall_hc_all_cdrall_or_mmseall
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
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    DEMOGRAPHICS_DIR,
    AGE_STAT_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.calibrator import load_predicted_ages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── data loading ─────────────────────────────────────────────────────────────

def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep_cols = ["ID", "Age", "group", "MMSE", "CASI", "Global_CDR"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        df["group"] = csv_file.replace(".csv", "")
        for c in keep_cols:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df[keep_cols])
    return pd.concat(dfs, ignore_index=True)


def match_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    records = []
    for subject_id, pred_age in predicted_ages.items():
        row = demo[demo["ID"] == subject_id]
        if row.empty:
            continue
        real_age = row["Age"].values[0]
        if pd.isna(real_age):
            continue
        records.append({
            "ID": subject_id,
            "real_age": real_age,
            "predicted_age": pred_age,
            "group": row["group"].values[0],
            "error": real_age - pred_age,
            "MMSE": row["MMSE"].values[0],
            "CASI": row["CASI"].values[0],
            "Global_CDR": row["Global_CDR"].values[0],
        })
    return pd.DataFrame(records)


def filter_cohort(df: pd.DataFrame, cohort_mode: str) -> pd.DataFrame:
    if cohort_mode == "all":
        return df
    df_p = df[df["group"] == "P"].copy()
    df_hc = df[df["group"].isin(["NAD", "ACS"])].copy()
    df_p["subject"] = df_p["ID"].apply(lambda x: x.rsplit("-", 1)[0])
    if "cdr05" in cohort_mode:
        df_p = df_p[pd.to_numeric(df_p["Global_CDR"], errors="coerce") >= 0.5]
    pick = df_p.sort_values("ID").drop_duplicates("subject", keep="first")
    return pd.concat([pick.drop(columns=["subject"]), df_hc], ignore_index=True)

# ── constants ────────────────────────────────────────────────────────────────

AGE_BINS = [
    ("<65", lambda a: a < 65),
    ("65-75", lambda a: (a >= 65) & (a < 75)),
    ("75-85", lambda a: (a >= 75) & (a < 85)),
    (">=85", lambda a: a >= 85),
]
MMSE_BINS = [
    ("24-30", lambda s: (s >= 24) & (s <= 30)),
    ("18-23", lambda s: (s >= 18) & (s < 24)),
    ("10-17", lambda s: (s >= 10) & (s < 18)),
    ("0-9",   lambda s: (s >= 0) & (s < 10)),
]
CASI_BINS = [
    ("85-100", lambda s: (s >= 85) & (s <= 100)),
    ("70-84",  lambda s: (s >= 70) & (s < 85)),
    ("45-69",  lambda s: (s >= 45) & (s < 70)),
    ("0-44",   lambda s: (s >= 0) & (s < 45)),
]

# ── stat writers ─────────────────────────────────────────────────────────────

def _age_stratified_rows(df):
    rows = []
    for label, mask_fn in AGE_BINS:
        sub = df[mask_fn(df["real_age"])]
        if sub.empty:
            continue
        err = sub["error"]
        rows.append({
            "age_group": label, "n": len(sub),
            "real_age": f"{sub['real_age'].mean():.2f}±{sub['real_age'].std():.2f}",
            "pred_age": f"{sub['predicted_age'].mean():.2f}±{sub['predicted_age'].std():.2f}",
            "diff": f"{err.mean():.2f}±{err.std():.2f}",
            "MAE": f"{err.abs().mean():.2f}",
        })
    rows.append({
        "age_group": "Total", "n": len(df),
        "real_age": f"{df['real_age'].mean():.2f}±{df['real_age'].std():.2f}",
        "pred_age": f"{df['predicted_age'].mean():.2f}±{df['predicted_age'].std():.2f}",
        "diff": f"{df['error'].mean():.2f}±{df['error'].std():.2f}",
        "MAE": f"{df['error'].abs().mean():.2f}",
    })
    return rows


def write_age_error_stat(df_matched, stat_dir):
    all_rows = []
    for grp in ["ACS", "NAD", "P", "All"]:
        sub = df_matched if grp == "All" else df_matched[df_matched["group"] == grp]
        for r in _age_stratified_rows(sub):
            all_rows.append({"group": grp, **r})
    out = stat_dir / "age_error_stat_2.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"saved {out}")


def write_sliding_window(df_matched, stat_dir, window=10, step=1):
    start_min = int(np.floor(df_matched["real_age"].min()))
    start_max = int(np.floor(df_matched["real_age"].max())) - window + 1
    rows = []
    for label, df_grp in [(g, df_matched[df_matched["group"] == g]) for g in ["ACS", "NAD", "P"]] + [("All", df_matched)]:
        for s in range(start_min, start_max + 1, step):
            sub = df_grp[(df_grp["real_age"] >= s) & (df_grp["real_age"] < s + window)]
            if sub.empty:
                continue
            err = sub["error"]
            rows.append({
                "group": label, "age_range": f"{s}-{s+window-1}", "n": len(sub),
                "real_age": f"{sub['real_age'].mean():.2f}±{sub['real_age'].std():.2f}",
                "pred_age": f"{sub['predicted_age'].mean():.2f}±{sub['predicted_age'].std():.2f}",
                "diff": f"{err.mean():.2f}±{err.std():.2f}",
                "MAE": f"{err.abs().mean():.2f}",
            })
    out = stat_dir / "age_error_sliding_window.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"saved {out}")


def write_patient_cdr(df_matched, stat_dir):
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p["Global_CDR"] = pd.to_numeric(df_p["Global_CDR"], errors="coerce")
    df_p = df_p.dropna(subset=["Global_CDR"])
    rows = []
    for cdr in [0.0, 0.5, 1.0, 2.0, 3.0]:
        sub = df_p[df_p["Global_CDR"] == cdr]
        if sub.empty:
            continue
        err = sub["error"]
        rows.append({"CDR": cdr, "n": len(sub),
                     "real_age": f"{sub['real_age'].mean():.2f}±{sub['real_age'].std():.2f}",
                     "pred_age": f"{sub['predicted_age'].mean():.2f}±{sub['predicted_age'].std():.2f}",
                     "diff": f"{err.mean():.2f}±{err.std():.2f}",
                     "MAE": f"{err.abs().mean():.2f}"})
    out = stat_dir / "patient_cdr_age_error.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"saved {out}")


def write_patient_score(df_matched, score_col, bins, stat_dir):
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p[score_col] = pd.to_numeric(df_p[score_col], errors="coerce")
    df_v = df_p.dropna(subset=[score_col, "error"])
    rows = []
    for label, mask_fn in bins:
        sub = df_v[mask_fn(df_v[score_col])]
        if sub.empty:
            continue
        err = sub["error"]
        rows.append({score_col: label, "n": len(sub),
                     "real_age": f"{sub['real_age'].mean():.2f}±{sub['real_age'].std():.2f}",
                     "pred_age": f"{sub['predicted_age'].mean():.2f}±{sub['predicted_age'].std():.2f}",
                     "diff": f"{err.mean():.2f}±{err.std():.2f}",
                     "MAE": f"{err.abs().mean():.2f}"})
    out = stat_dir / f"patient_{score_col.lower()}_age_error.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"saved {out}")


def write_patient_corr(df_matched, score_col, stat_dir):
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p[score_col] = pd.to_numeric(df_p[score_col], errors="coerce")
    df_v = df_p.dropna(subset=[score_col, "error"])
    if len(df_v) < 3:
        return
    x, y = df_v[score_col].values, df_v["error"].values
    r, p = sp_stats.pearsonr(x, y)
    rho, p_rho = sp_stats.spearmanr(x, y)
    out = stat_dir / f"patient_{score_col.lower()}_error_corr.csv"
    pd.DataFrame([{"score": score_col, "n": len(df_v),
                    "pearson_r": f"{r:.4f}", "pearson_p": f"{p:.2e}",
                    "spearman_rho": f"{rho:.4f}", "spearman_p": f"{p_rho:.2e}"}
                  ]).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"saved {out}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c="#F44336", alpha=0.4, s=20, edgecolors="white", linewidth=0.3)
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + intercept, color="#FF9800", linewidth=2, alpha=0.8,
            label=f"y = {slope:.3f}x + {intercept:.2f}")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(score_col, fontsize=12)
    ax.set_ylabel("Age Prediction Error (pred - real)", fontsize=12)
    ax.set_title(f"Patient: {score_col} vs Age Prediction Error\n"
                 f"(n={len(df_v)}, Pearson r={r:.3f}, p={p:.2e})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = stat_dir / f"patient_{score_col.lower()}_vs_error.png"
    plt.savefig(str(png), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {png}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-mode", default="all",
                    choices=["all",
                             "p_first_cdr05_hc_all_cdrall_or_mmseall",
                             "p_first_cdrall_hc_all_cdrall_or_mmseall"])
    ap.add_argument("--stat-dir", type=Path, default=AGE_STAT_DIR)
    args = ap.parse_args()

    args.stat_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    df_matched = match_ages(preds, demo)
    df_matched = filter_cohort(df_matched, args.cohort_mode)
    logger.info(f"cohort={args.cohort_mode}, matched={len(df_matched)}")

    write_age_error_stat(df_matched, args.stat_dir)
    write_sliding_window(df_matched, args.stat_dir)
    write_patient_cdr(df_matched, args.stat_dir)
    write_patient_score(df_matched, "MMSE", MMSE_BINS, args.stat_dir)
    write_patient_score(df_matched, "CASI", CASI_BINS, args.stat_dir)
    write_patient_corr(df_matched, "MMSE", args.stat_dir)
    write_patient_corr(df_matched, "CASI", args.stat_dir)


if __name__ == "__main__":
    main()
