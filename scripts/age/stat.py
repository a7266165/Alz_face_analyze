"""
scripts/age/stat.py
Age prediction error statistics — stratified CSVs and correlation analysis.

Cohort is built with the canonical ``src.common.cohort.cohort_list`` — the same
gold-standard filtering ``histogram.py`` uses — so the CDR/MMSE/visit filters
actually applied match the output directory's cohort name. Errors are computed
directly from the raw MiVOLO predictions.

Outputs (to <AGE_ANALYSIS_DIR>/<visit_dir>/<cdr_mmse_dir>/stat/):
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
  conda run -n Alz_face_main_analysis python scripts/age/stat.py
  conda run -n Alz_face_main_analysis python scripts/age/stat.py \
      --cohort-mode p_all_cdrall_hc_all_cdrall_or_mmseall
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
    AGE_ANALYSIS_DIR,
    DEFAULT_COHORT_MODE,
    PREDICTED_AGES_FILE,
    VALID_COHORT_CHOICES,
    cohort_path,
    cohort_spec_from_name,
)
from src.age.utils import load_predicted_ages
from src.common.cohort import cohort_list

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── data loading ─────────────────────────────────────────────────────────────

def build_matched(preds: dict, cohort_mode: str) -> pd.DataFrame:
    """Canonical cohort × predicted ages → per-row error table.

    Cohort filtering (CDR / MMSE / visit selection) is delegated to
    ``cohort_list`` so it matches ``histogram.py`` exactly.
    """
    spec = cohort_spec_from_name(cohort_mode)
    cohort = cohort_list(
        f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
        "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    cohort["group"] = cohort["Group"]
    cohort["ID"] = (cohort["Group"] + cohort["ID"].astype(str)
                    + "-" + cohort["Photo_Session"].astype(str))
    df = cohort.copy()
    df["real_age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    df["error"] = df["real_age"] - df["predicted_age"]
    for c in ["MMSE", "CASI", "Global_CDR"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["ID", "real_age", "predicted_age", "group", "error",
               "MMSE", "CASI", "Global_CDR"]]

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
    ap.add_argument("--cohort-mode", default=DEFAULT_COHORT_MODE,
                    choices=VALID_COHORT_CHOICES,
                    help=f"canonical cohort name (預設: {DEFAULT_COHORT_MODE})")
    ap.add_argument("--stat-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort-mode 自動決定")
    args = ap.parse_args()

    stat_dir = args.stat_dir or (
        AGE_ANALYSIS_DIR / cohort_path(args.cohort_mode) / "stat")
    stat_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"cohort-mode = {args.cohort_mode}")
    logger.info(f"stat-dir    = {stat_dir}")

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    df_matched = build_matched(preds, args.cohort_mode)
    logger.info(f"matched={len(df_matched)} "
                f"({df_matched['group'].value_counts().to_dict()})")

    write_age_error_stat(df_matched, stat_dir)
    write_sliding_window(df_matched, stat_dir)
    write_patient_cdr(df_matched, stat_dir)
    write_patient_score(df_matched, "MMSE", MMSE_BINS, stat_dir)
    write_patient_score(df_matched, "CASI", CASI_BINS, stat_dir)
    write_patient_corr(df_matched, "MMSE", stat_dir)
    write_patient_corr(df_matched, "CASI", stat_dir)


if __name__ == "__main__":
    main()
