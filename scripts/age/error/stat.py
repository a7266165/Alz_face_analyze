"""Age-prediction error statistics for ACS / NAD / P (stratified CSVs + score correlation)
— full cohort + AD-vs-HC 1:1 age-matched subset.

Outputs under <AGE_ANALYSIS_DIR>/<cohort>/stat/{full,1by1matched}/:
  age_error_stat_2.csv / age_error_sliding_window.csv  — age strata / 10-y sliding window
  patient_{cdr,mmse,casi}_age_error.csv                — Patient CDR / MMSE / CASI strata
  patient_{mmse,casi}_error_corr.csv (+ _vs_error.png) — score–error correlation (+ scatter)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_ANALYSIS_DIR,
    cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.age.utils import build_cohort_with_age_error
from src.common.matching import match_by_age

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 資料載入 ─────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """error 表 → stat 用精簡表（age_error 改回 error 供既有 write_* 沿用）。

    吃已載入的 error 表（完整 cohort 或 1by1matched 子集皆可）。
    """
    return df.rename(columns={"age_error": "error"})[
        ["ID", "real_age", "predicted_age", "group", "error",
         "MMSE", "CASI", "Global_CDR"]]

# ── 常數 ────────────────────────────────────────────────────────────────

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

# ── 統計輸出 ─────────────────────────────────────────────────────────────

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
    ax.set_ylabel("Age Prediction Error (real - predicted)", fontsize=12)
    ax.set_title(f"Patient: {score_col} vs Age Prediction Error\n"
                 f"(n={len(df_v)}, Pearson r={r:.3f}, p={p:.2e})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = stat_dir / f"patient_{score_col.lower()}_vs_error.png"
    plt.savefig(str(png), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {png}")

# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    ap.add_argument("--stat-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort 自動決定")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    stat_dir = args.stat_dir or (
        AGE_ANALYSIS_DIR / cohort_path(*cohort) / "stat")

    logger.info(f"cohort = {cohort}")
    logger.info(f"stat-dir    = {stat_dir}")

    full = build_cohort_with_age_error(*cohort)
    p_ids, hc_ids = match_by_age(*cohort, priority=["ACS"])  # ACS 優先：稀少的 ACS 對照先配對
    matched = full[full["ID"].isin(set(p_ids) | set(hc_ids))].reset_index(drop=True)
    logger.info(f"full={len(full)} ({full['group'].value_counts().to_dict()}), "
                f"1by1matched={len(matched)} "
                f"({matched['group'].value_counts().to_dict()})")

    for sub_name, sub_df in [("full", full), ("1by1matched", matched)]:
        df = _prep(sub_df)
        sd = stat_dir / sub_name
        sd.mkdir(parents=True, exist_ok=True)
        write_age_error_stat(df, sd)
        write_sliding_window(df, sd)
        write_patient_cdr(df, sd)
        write_patient_score(df, "MMSE", MMSE_BINS, sd)
        write_patient_score(df, "CASI", CASI_BINS, sd)
        write_patient_corr(df, "MMSE", sd)
        write_patient_corr(df, "CASI", sd)


if __name__ == "__main__":
    main()
