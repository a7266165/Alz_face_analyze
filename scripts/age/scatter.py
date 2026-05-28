"""
scripts/age/scatter.py
Age prediction scatter plots and error statistics.

Subcommands:
  scatter  — Real vs Predicted age scatter plots (internal, with UTKFace, ACS vs EACS)
  stat     — Error statistics CSVs (age-stratified, CDR/MMSE/CASI stratified, correlations)
  all      — Both (default)

Usage:
  conda run -n Alz_face_age python scripts/age/scatter.py
  conda run -n Alz_face_age python scripts/age/scatter.py --only scatter
  conda run -n Alz_face_age python scripts/age/scatter.py --only stat
"""

import argparse
import json
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
    AGE_SCATTER_DIR,
    AGE_STAT_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.calibration import load_predicted_ages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── colours ──────────────────────────────────────────────────────────────────
EACS_SOURCE_COLORS = {
    "IMDB":        "#1f77b4",
    "MegaAge":     "#ff7f0e",
    "FairFace":    "#2ca02c",
    "UTKFace":     "#d62728",
    "SZU-EmoDage": "#9467bd",
    "AFAD":        "#8c564b",
    "DiverseAsian": "#e377c2",
    "AgeDB":       "#17becf",
    "APPA-REAL":   "#bcbd22",
}

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


def load_eacs(demo_dir: Path) -> pd.DataFrame:
    path = demo_dir / "EACS.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Source" not in df.columns:
        df["Source"] = "unknown"
    return df


def attach_pred(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    df = df.copy()
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)
    df = df.rename(columns={"Age": "real_age"})
    df["error"] = df["real_age"] - df["predicted_age"]
    return df

# ── scatter panel helper ─────────────────────────────────────────────────────

def _draw_panel(ax, df, title, colors, labels):
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["real_age"], sub["predicted_age"],
                   c=color, label=labels.get(grp, grp),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5
    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")

    x = df["real_age"].to_numpy(float)
    y = df["predicted_age"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    if ss_xx > 0:
        a = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_xx
        b = float(y.mean() - a * x.mean())
        xs = np.array([age_min, age_max])
        ax.plot(xs, a * xs + b, color="#FF9800", linewidth=2, alpha=0.8,
                label=f"y = {a:.2f}x + {b:.2f}")

    n = len(df)
    r = df["real_age"].corr(df["predicted_age"])
    mae = df["error"].abs().mean()
    ax.set_xlabel("Real Age", fontsize=12)
    ax.set_ylabel("Predicted Age (MiVOLO)", fontsize=12)
    ax.set_title(f"{title}\n(n={n}, r={r:.3f}, MAE={mae:.1f})", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def _draw_panel_single(ax, df, title, color_by="single", single_color="#4CAF50"):
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    if color_by == "source":
        for src, color in EACS_SOURCE_COLORS.items():
            sub = df[df["Source"] == src]
            if sub.empty:
                continue
            ax.scatter(sub["real_age"], sub["predicted_age"],
                       c=color, label=f"{src} (n={len(sub)})",
                       alpha=0.55, s=22, edgecolors="white", linewidth=0.25)
    else:
        ax.scatter(df["real_age"], df["predicted_age"],
                   c=single_color, alpha=0.6, s=30,
                   edgecolors="white", linewidth=0.3,
                   label=f"n={len(df)}")

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5
    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")
    x = df["real_age"].to_numpy(float)
    y = df["predicted_age"].to_numpy(float)
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    if ss_xx > 0:
        a = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_xx
        b = float(y.mean() - a * x.mean())
        xs = np.array([age_min, age_max])
        ax.plot(xs, a * xs + b, color="#FF9800", linewidth=2, alpha=0.85,
                label=f"y = {a:.2f}x + {b:.2f}")
    n = len(df)
    r = df["real_age"].corr(df["predicted_age"])
    mae = df["error"].abs().mean()
    ax.set_xlabel("Real Age", fontsize=12)
    ax.set_ylabel("Predicted Age (MiVOLO)", fontsize=12)
    ax.set_title(f"{title}\n(n={n}, r={r:.3f}, MAE={mae:.1f})", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

# ── scatter plots ────────────────────────────────────────────────────────────

def plot_main_scatter(df_matched, scatter_dir):
    df_hc = df_matched[df_matched["group"].isin(["ACS", "NAD"])]
    df_p = df_matched[df_matched["group"] == "P"]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_panel(ax_l, df_hc, "Healthy Controls (NAD + ACS)",
                {"NAD": "#2196F3", "ACS": "#4CAF50"},
                {"NAD": "NAD", "ACS": "ACS"})
    _draw_panel(ax_r, df_p, "Patients (P)",
                {"P": "#F44336"}, {"P": "Patient"})
    plt.tight_layout()
    out = scatter_dir / "predicted_ages_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")


def plot_scatter_with_utkface(preds, scatter_dir):
    internal = load_demographics(DEMOGRAPHICS_DIR)[["ID", "Age", "group"]]
    internal["Age"] = pd.to_numeric(internal["Age"], errors="coerce")
    internal = attach_pred(internal, preds)
    eacs = load_eacs(DEMOGRAPHICS_DIR)
    eacs = eacs[eacs["Source"] == "UTKFace"].copy()
    eacs["group"] = "UTKFace"
    eacs = attach_pred(eacs[["ID", "Age", "group"]], preds)

    hc = internal[internal["group"].isin(["NAD", "ACS"])]
    p = internal[internal["group"] == "P"]
    hc_plus = pd.concat([hc, eacs], ignore_index=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_panel(ax_l, hc_plus, "Healthy Controls (NAD + ACS + UTKFace)",
                {"NAD": "#2196F3", "ACS": "#4CAF50", "UTKFace": "#d62728"},
                {"NAD": "NAD", "ACS": "ACS", "UTKFace": "UTKFace (E-ACS)"})
    _draw_panel(ax_r, p, "Patients (P)",
                {"P": "#F44336"}, {"P": "Patient"})
    plt.tight_layout()
    out = scatter_dir / "predicted_ages_scatter_with_utkface.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")


def plot_acs_vs_eacs(preds, scatter_dir, mode="per_source",
                     color_by_source=False, min_pred=None):
    acs_df = pd.read_csv(DEMOGRAPHICS_DIR / "ACS.csv", encoding="utf-8-sig")
    acs_df["Age"] = pd.to_numeric(acs_df["Age"], errors="coerce")
    acs_df = acs_df[["ID", "Age"]].rename(columns={"Age": "real_age"})
    acs_df = acs_df.copy()
    acs_df["predicted_age"] = acs_df["ID"].map(preds)
    acs_df = acs_df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    acs_df["error"] = acs_df["real_age"] - acs_df["predicted_age"]

    eacs_df = load_eacs(DEMOGRAPHICS_DIR)
    eacs_df = eacs_df[["ID", "Age", "Source"]].rename(columns={"Age": "real_age"})
    eacs_df = eacs_df.copy()
    eacs_df["predicted_age"] = eacs_df["ID"].map(preds)
    eacs_df = eacs_df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    eacs_df["error"] = eacs_df["real_age"] - eacs_df["predicted_age"]

    if min_pred is not None:
        acs_df = acs_df[acs_df["predicted_age"] >= min_pred].reset_index(drop=True)
        eacs_df = eacs_df[eacs_df["predicted_age"] >= min_pred].reset_index(drop=True)

    if mode == "combined":
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
        _draw_panel_single(ax_l, acs_df, "ACS (internal)", single_color="#4CAF50")
        _draw_panel_single(ax_r, eacs_df, "E-ACS (external)",
                           color_by="source" if color_by_source else "single",
                           single_color="#2196F3")
        fname = "predicted_ages_scatter_acs_vs_eacs.png"
    else:
        sources = [
            ("ACS (internal)", acs_df, "#4CAF50"),
        ] + [
            (f"E-ACS | {src}", eacs_df[eacs_df["Source"] == src], EACS_SOURCE_COLORS[src])
            for src in EACS_SOURCE_COLORS
        ]
        fig, axes = plt.subplots(2, 5, figsize=(27, 11))
        for ax, (title, sub, color) in zip(axes.flat, sources):
            _draw_panel_single(ax, sub, title=title, single_color=color)
        fname = "predicted_ages_scatter_per_source.png"

    if min_pred is not None:
        fname = f"{Path(fname).stem}_minpred{int(min_pred)}.png"

    out = scatter_dir / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")

# ── stat CSVs ────────────────────────────────────────────────────────────────

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
    ap.add_argument("--only", choices=["scatter", "stat", "all"], default="all")
    ap.add_argument("--cohort-mode", default="all",
                    choices=["all",
                             "p_first_cdr05_hc_all_cdrall_or_mmseall",
                             "p_first_cdrall_hc_all_cdrall_or_mmseall"])
    ap.add_argument("--scatter-dir", type=Path, default=AGE_SCATTER_DIR)
    ap.add_argument("--stat-dir", type=Path, default=AGE_STAT_DIR)
    ap.add_argument("--eacs-mode", default="per_source",
                    choices=["combined", "per_source"])
    ap.add_argument("--min-pred", type=float, default=None)
    args = ap.parse_args()

    args.scatter_dir.mkdir(parents=True, exist_ok=True)
    args.stat_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    df_matched = match_ages(preds, demo)
    df_matched = filter_cohort(df_matched, args.cohort_mode)
    logger.info(f"cohort={args.cohort_mode}, matched={len(df_matched)}")

    if args.only in ("scatter", "all"):
        plot_main_scatter(df_matched, args.scatter_dir)
        plot_scatter_with_utkface(preds, args.scatter_dir)
        plot_acs_vs_eacs(preds, args.scatter_dir, mode=args.eacs_mode,
                         min_pred=args.min_pred)

    if args.only in ("stat", "all"):
        write_age_error_stat(df_matched, args.stat_dir)
        write_sliding_window(df_matched, args.stat_dir)
        write_patient_cdr(df_matched, args.stat_dir)
        write_patient_score(df_matched, "MMSE", MMSE_BINS, args.stat_dir)
        write_patient_score(df_matched, "CASI", CASI_BINS, args.stat_dir)
        write_patient_corr(df_matched, "MMSE", args.stat_dir)
        write_patient_corr(df_matched, "CASI", args.stat_dir)


if __name__ == "__main__":
    main()
