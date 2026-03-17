"""
scripts/run_m3m4_deep_analysis.py

M3 (Age Estimation) + M4 (Expression Analysis) deep analysis for
Alzheimer's disease detection paper. Generates publication-quality
figures and statistical test results.

Outputs:
  - paper/figures/fig_*.png          (8 figures)
  - workspace/statistics/m3m4_deep/  (6 CSVs)
"""

import sys
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
project_root = PROJECT_ROOT

from src.config import DEMOGRAPHICS_DIR, WORKSPACE_DIR, STATISTICS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CALIBRATED_AGES_FILE = WORKSPACE_DIR / "predicted_ages.json"
EMOTION_SCORES_FILE = WORKSPACE_DIR / "emotion_score_EmoNet.csv"
FIGURES_DIR = project_root / "paper" / "figures"
STATS_DIR = WORKSPACE_DIR / "statistics" / "m3m4_deep"

COLOR_CONTROL = "#2196F3"
COLOR_AD = "#F44336"
COLOR_CDR = {0.0: "#4CAF50", 0.5: "#FFC107", 1.0: "#FF9800", 2.0: "#F44336"}

EMOTION_COLS = [
    "Anger", "Contempt", "Disgust", "Fear", "Happiness",
    "Neutral", "Sadness", "Surprise", "Valence", "Arousal",
]

CDR_LEVELS = [0.0, 0.5, 1.0, 2.0]

# Publication figure settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Try Arial, fall back to sans-serif
try:
    import matplotlib.font_manager as fm
    if any("Arial" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"
    else:
        plt.rcParams["font.family"] = "sans-serif"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"


# ---------------------------------------------------------------------------
# FDR correction (with statsmodels fallback)
# ---------------------------------------------------------------------------
try:
    from statsmodels.stats.multitest import multipletests as _multipletests
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def fdr_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (p_corrected, reject)."""
    pvals = np.asarray(pvals, dtype=float)
    if _HAS_STATSMODELS:
        reject, p_corr, _, _ = _multipletests(pvals, alpha=alpha, method="fdr_bh")
        return p_corr, reject
    # Manual BH
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]
    p_corr = np.empty(n)
    p_corr[idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        p_corr[idx[i]] = min(sorted_p[i] * n / (i + 1), p_corr[idx[i + 1]])
    return p_corr, p_corr < alpha


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def cohens_d(x, y):
    """Cohen's d (positive = x > y)."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else 0.0


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_predicted_ages(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep = ["ID", "Age", "Sex", "group", "MMSE", "CASI", "Global_CDR"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        df["group"] = csv_file.replace(".csv", "")
        for c in keep:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df[keep])
    return pd.concat(dfs, ignore_index=True)


def load_emotion_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def build_merged_dataframe(predicted_ages, demo, emotions):
    """Merge predicted ages, demographics and emotion scores."""
    records = []
    emo_dict = emotions.set_index("subject_id").to_dict("index")
    demo_dict = demo.set_index("ID").to_dict("index")

    for sid, pred_age in predicted_ages.items():
        if sid not in demo_dict:
            continue
        d = demo_dict[sid]
        rec = {
            "ID": sid,
            "real_age": d["Age"],
            "predicted_age": pred_age,
            "age_error": d["Age"] - pred_age,
            "group": d["group"],
            "label": 1 if d["group"] == "P" else 0,
            "Sex": d["Sex"],
            "MMSE": d["MMSE"],
            "CASI": d["CASI"],
            "Global_CDR": d["Global_CDR"],
        }
        if sid in emo_dict:
            rec.update(emo_dict[sid])
        else:
            for c in EMOTION_COLS:
                rec[c] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    # Drop rows missing emotion scores
    df = df.dropna(subset=EMOTION_COLS).reset_index(drop=True)
    df["abs_age_error"] = df["age_error"].abs()
    logger.info(
        f"Merged: {len(df)} rows "
        f"(Control={int((df['label']==0).sum())}, AD={int(df['label'].sum())})"
    )
    return df


# ===================================================================
# Layer 1: Core analyses
# ===================================================================

def analyze_emotion_profiles(df, fig_dir, stat_dir):
    """Analysis 1: AD vs Control emotion profile comparison."""
    logger.info("[1] Emotion profiles: AD vs Control")

    ctrl = df[df["label"] == 0]
    ad = df[df["label"] == 1]
    rows = []
    pvals = []

    for col in EMOTION_COLS:
        x_ctrl = ctrl[col].dropna().values
        x_ad = ad[col].dropna().values
        u, p = stats.mannwhitneyu(x_ctrl, x_ad, alternative="two-sided")
        d = cohens_d(x_ad, x_ctrl)  # positive = AD higher
        rows.append({
            "feature": col,
            "n_control": len(x_ctrl),
            "mean_control": f"{x_ctrl.mean():.4f}",
            "std_control": f"{x_ctrl.std():.4f}",
            "n_ad": len(x_ad),
            "mean_ad": f"{x_ad.mean():.4f}",
            "std_ad": f"{x_ad.std():.4f}",
            "U_stat": u,
            "p_value": p,
            "cohens_d": round(d, 4),
        })
        pvals.append(p)

    p_corr, reject = fdr_correction(pvals)
    for i, r in enumerate(rows):
        r["p_fdr"] = round(p_corr[i], 6)
        r["significant"] = "yes" if reject[i] else "no"

    csv_path = stat_dir / "emotion_ad_vs_control.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    # --- Print summary ---
    for r in rows:
        s = sig_stars(r["p_fdr"])
        logger.info(
            f"  {r['feature']:>10s}: Control={r['mean_control']} AD={r['mean_ad']} "
            f"d={r['cohens_d']:+.3f} p_fdr={r['p_fdr']:.4f} {s}"
        )

    # --- Radar chart ---
    mean_ctrl = [ctrl[c].mean() for c in EMOTION_COLS]
    mean_ad = [ad[c].mean() for c in EMOTION_COLS]
    n_feat = len(EMOTION_COLS)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]
    mean_ctrl += mean_ctrl[:1]
    mean_ad += mean_ad[:1]

    labels = []
    for i, col in enumerate(EMOTION_COLS):
        s = sig_stars(p_corr[i])
        labels.append(f"{col} {s}" if s else col)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, mean_ctrl, "o-", color=COLOR_CONTROL, linewidth=2, label=f"Control (n={len(ctrl)})")
    ax.fill(angles, mean_ctrl, alpha=0.15, color=COLOR_CONTROL)
    ax.plot(angles, mean_ad, "o-", color=COLOR_AD, linewidth=2, label=f"AD (n={len(ad)})")
    ax.fill(angles, mean_ad, alpha=0.15, color=COLOR_AD)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title("Emotion Profile: AD vs Control", fontsize=14, pad=20)
    fig.savefig(str(fig_dir / "fig_emotion_radar.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_emotion_radar.png")

    # --- Effect size bar chart ---
    d_vals = [float(r["cohens_d"]) for r in rows]
    colors = [COLOR_AD if d > 0 else COLOR_CONTROL for d in d_vals]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(n_feat)
    bars = ax.barh(y_pos, d_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(EMOTION_COLS)
    ax.set_xlabel("Cohen's d (positive = higher in AD)")
    ax.set_title("Effect Size: Emotion Features (AD vs Control)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    for i, (d, r) in enumerate(zip(d_vals, rows)):
        s = sig_stars(r["p_fdr"])
        if s:
            ax.text(d + 0.01 * np.sign(d), i, s, va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_emotion_effect_sizes.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_emotion_effect_sizes.png")


def analyze_age_error(df, fig_dir, stat_dir):
    """Analysis 2: Age error group difference."""
    logger.info("[2] Age estimation error: AD vs Control")

    ctrl = df[df["label"] == 0]["age_error"].dropna()
    ad = df[df["label"] == 1]["age_error"].dropna()
    u, p = stats.mannwhitneyu(ctrl.values, ad.values, alternative="two-sided")
    d = cohens_d(ad, ctrl)

    row = {
        "group": ["Control", "AD"],
        "n": [len(ctrl), len(ad)],
        "mean": [round(ctrl.mean(), 3), round(ad.mean(), 3)],
        "std": [round(ctrl.std(), 3), round(ad.std(), 3)],
        "median": [round(ctrl.median(), 3), round(ad.median(), 3)],
        "q25": [round(ctrl.quantile(0.25), 3), round(ad.quantile(0.25), 3)],
        "q75": [round(ctrl.quantile(0.75), 3), round(ad.quantile(0.75), 3)],
    }
    csv_df = pd.DataFrame(row)
    csv_df["U_stat"] = u
    csv_df["p_value"] = p
    csv_df["cohens_d"] = round(d, 4)
    csv_path = stat_dir / "age_error_ad_vs_control.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    logger.info(
        f"  Control: {ctrl.mean():.2f}\u00b1{ctrl.std():.2f} (n={len(ctrl)})\n"
        f"  AD:      {ad.mean():.2f}\u00b1{ad.std():.2f} (n={len(ad)})\n"
        f"  U={u}, p={p:.2e}, Cohen's d={d:.3f}"
    )

    # --- Violin plot ---
    plot_df = df[["label", "age_error"]].dropna().copy()
    plot_df["Group"] = plot_df["label"].map({0: "Control", 1: "AD"})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=plot_df, x="Group", y="age_error", order=["Control", "AD"],
        palette={"Control": COLOR_CONTROL, "AD": COLOR_AD},
        inner="quartile", linewidth=1.2, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x="Group", y="age_error", order=["Control", "AD"],
        color="black", alpha=0.08, size=2, jitter=True, ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Age Prediction Error (years)")
    ax.set_xlabel("")
    ax.set_title("Age Estimation Error: AD vs Control")
    s = sig_stars(p)
    ax.text(
        0.95, 0.95,
        f"U={u:,.0f}\np={p:.2e}\nd={d:.3f} {s}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_age_error_violin.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_age_error_violin.png")


def plot_age_scatter_2group(df, fig_dir):
    """Analysis 3: Predicted vs Real age scatter (2-group)."""
    logger.info("[3] Predicted vs Real age scatter")

    fig, ax = plt.subplots(figsize=(8, 7))
    for label_val, name, color in [(0, "Control", COLOR_CONTROL), (1, "AD", COLOR_AD)]:
        sub = df[df["label"] == label_val]
        ax.scatter(
            sub["real_age"], sub["predicted_age"],
            c=color, label=name, alpha=0.4, s=20, edgecolors="white", linewidth=0.3,
        )
        # Regression line
        x = sub["real_age"].values.astype(float)
        y = sub["predicted_age"].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) > 2:
            slope, intercept = np.polyfit(x, y, 1)
            xl = np.array([x.min(), x.max()])
            r = np.corrcoef(x, y)[0, 1]
            mae = np.abs(x - y).mean()
            ax.plot(xl, slope * xl + intercept, color=color, linewidth=2, alpha=0.8,
                    label=f"{name}: y={slope:.2f}x+{intercept:.1f} (r={r:.3f}, MAE={mae:.1f})")

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5
    ax.plot([age_min, age_max], [age_min, age_max], "k--", alpha=0.4, linewidth=1, label="y = x")
    ax.set_xlabel("Chronological Age (years)")
    ax.set_ylabel("Predicted Age (years)")
    ax.set_title("Predicted vs Chronological Age")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_age_scatter.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_age_scatter.png")


# ===================================================================
# Layer 2: CDR stratification
# ===================================================================

def analyze_cdr_emotion_trends(df, fig_dir, stat_dir):
    """Analysis 4: Emotion trends across CDR levels."""
    logger.info("[4] Emotion trends by CDR level")

    df_cdr = df.copy()
    df_cdr["Global_CDR"] = pd.to_numeric(df_cdr["Global_CDR"], errors="coerce")
    df_cdr = df_cdr.dropna(subset=["Global_CDR"])
    # Keep only levels with sufficient samples
    valid_levels = [lv for lv in CDR_LEVELS if (df_cdr["Global_CDR"] == lv).sum() >= 5]

    rows = []
    pvals_kw = []
    pvals_sp = []

    for col in EMOTION_COLS:
        groups = [df_cdr[df_cdr["Global_CDR"] == lv][col].dropna().values for lv in valid_levels]
        # Kruskal-Wallis
        if all(len(g) >= 2 for g in groups):
            h, p_kw = stats.kruskal(*groups)
        else:
            h, p_kw = np.nan, np.nan
        # Spearman trend
        valid = df_cdr[["Global_CDR", col]].dropna()
        rho, p_sp = stats.spearmanr(valid["Global_CDR"], valid[col])

        rec = {"feature": col}
        for lv in valid_levels:
            sub = df_cdr[df_cdr["Global_CDR"] == lv][col].dropna()
            lv_str = str(lv).replace(".", "_")
            rec[f"cdr_{lv_str}_n"] = len(sub)
            rec[f"cdr_{lv_str}_mean"] = round(sub.mean(), 4)
            rec[f"cdr_{lv_str}_std"] = round(sub.std(), 4)
        rec["kruskal_H"] = round(h, 4) if not np.isnan(h) else np.nan
        rec["kruskal_p"] = p_kw
        rec["spearman_rho"] = round(rho, 4)
        rec["spearman_p"] = p_sp
        rows.append(rec)
        pvals_kw.append(p_kw if not np.isnan(p_kw) else 1.0)
        pvals_sp.append(p_sp)

    p_kw_corr, rej_kw = fdr_correction(pvals_kw)
    p_sp_corr, rej_sp = fdr_correction(pvals_sp)
    for i, r in enumerate(rows):
        r["kruskal_p_fdr"] = round(p_kw_corr[i], 6)
        r["spearman_p_fdr"] = round(p_sp_corr[i], 6)

    csv_path = stat_dir / "emotion_by_cdr.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    for r in rows:
        s = sig_stars(r["kruskal_p_fdr"])
        logger.info(
            f"  {r['feature']:>10s}: H={r['kruskal_H']:.2f} p_fdr={r['kruskal_p_fdr']:.4f} {s} "
            f"rho={r['spearman_rho']:+.3f}"
        )

    # --- 2x5 grid figure ---
    n_cols = 5
    n_rows_fig = 2
    fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(16, 7))
    axes = axes.flatten()

    for i, col in enumerate(EMOTION_COLS):
        ax = axes[i]
        means = []
        sems = []
        for lv in valid_levels:
            sub = df_cdr[df_cdr["Global_CDR"] == lv][col].dropna()
            means.append(sub.mean())
            sems.append(sub.sem())

        ax.errorbar(
            valid_levels, means, yerr=sems,
            fmt="o-", color="black", linewidth=1.5, markersize=5,
            capsize=3, capthick=1,
        )
        # Color points by CDR
        for j, lv in enumerate(valid_levels):
            ax.plot(lv, means[j], "o", color=COLOR_CDR.get(lv, "gray"), markersize=7, zorder=5)

        s = sig_stars(p_kw_corr[i])
        ax.set_title(f"{col} {s}", fontsize=11)
        ax.set_xlabel("CDR")
        if i % n_cols == 0:
            ax.set_ylabel("Score")
        ax.set_xticks(valid_levels)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Emotion Scores by CDR Level (mean \u00b1 SEM)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_emotion_cdr_trends.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_emotion_cdr_trends.png")


def analyze_cdr_age_error(df, fig_dir, stat_dir):
    """Analysis 5: Age error by CDR level."""
    logger.info("[5] Age error by CDR level")

    df_cdr = df.copy()
    df_cdr["Global_CDR"] = pd.to_numeric(df_cdr["Global_CDR"], errors="coerce")
    df_cdr = df_cdr.dropna(subset=["Global_CDR", "age_error"])
    valid_levels = [lv for lv in CDR_LEVELS if (df_cdr["Global_CDR"] == lv).sum() >= 5]

    groups = [df_cdr[df_cdr["Global_CDR"] == lv]["age_error"].values for lv in valid_levels]
    h, p_kw = stats.kruskal(*groups) if all(len(g) >= 2 for g in groups) else (np.nan, np.nan)
    rho, p_sp = stats.spearmanr(df_cdr["Global_CDR"], df_cdr["age_error"])

    rows = []
    for lv in valid_levels:
        sub = df_cdr[df_cdr["Global_CDR"] == lv]["age_error"]
        rows.append({
            "CDR": lv, "n": len(sub),
            "mean": round(sub.mean(), 3), "std": round(sub.std(), 3),
            "median": round(sub.median(), 3),
            "q25": round(sub.quantile(0.25), 3), "q75": round(sub.quantile(0.75), 3),
        })
    csv_df = pd.DataFrame(rows)
    csv_df["kruskal_H"] = round(h, 4) if not np.isnan(h) else np.nan
    csv_df["kruskal_p"] = p_kw
    csv_df["spearman_rho"] = round(rho, 4)
    csv_df["spearman_p"] = p_sp
    csv_path = stat_dir / "age_error_by_cdr.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    for r in rows:
        logger.info(f"  CDR={r['CDR']}: {r['mean']:.2f}\u00b1{r['std']:.2f} (n={r['n']})")
    logger.info(f"  Kruskal-Wallis H={h:.3f}, p={p_kw:.2e}")
    logger.info(f"  Spearman rho={rho:.3f}, p={p_sp:.2e}")

    # --- Box plot ---
    plot_df = df_cdr[df_cdr["Global_CDR"].isin(valid_levels)].copy()
    plot_df["CDR_str"] = plot_df["Global_CDR"].astype(str)
    cdr_order = [str(lv) for lv in valid_levels]
    palette = {str(lv): COLOR_CDR.get(lv, "gray") for lv in valid_levels}

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df, x="CDR_str", y="age_error", order=cdr_order,
        palette=palette, linewidth=1.2, fliersize=2, ax=ax,
    )
    # Trend line through means
    means = [df_cdr[df_cdr["Global_CDR"] == lv]["age_error"].mean() for lv in valid_levels]
    ax.plot(range(len(valid_levels)), means, "k--o", linewidth=1.5, markersize=6, zorder=5, label="Mean trend")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("CDR Level")
    ax.set_ylabel("Age Prediction Error (years)")
    ax.set_title("Age Estimation Error by CDR Level")
    s_kw = sig_stars(p_kw)
    ax.text(
        0.95, 0.95,
        f"Kruskal-Wallis H={h:.2f}\np={p_kw:.2e} {s_kw}\nSpearman \u03c1={rho:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_age_error_by_cdr.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_age_error_by_cdr.png")


# ===================================================================
# Layer 3: Clinical correlations
# ===================================================================

def analyze_cognitive_correlations(df, fig_dir, stat_dir):
    """Analysis 6: MMSE/CASI correlation with M3+M4 features."""
    logger.info("[6] Cognitive score correlations")

    feature_cols = ["predicted_age", "age_error"] + EMOTION_COLS
    score_cols = ["MMSE", "CASI"]
    rows = []
    pvals = []

    for score in score_cols:
        for feat in feature_cols:
            valid = df[[score, feat]].dropna()
            n = len(valid)
            if n < 10:
                rows.append({
                    "cognitive_score": score, "feature": feat, "n": n,
                    "spearman_rho": np.nan, "p_value": np.nan,
                })
                pvals.append(1.0)
                continue
            rho, p = stats.spearmanr(valid[score], valid[feat])
            rows.append({
                "cognitive_score": score, "feature": feat, "n": n,
                "spearman_rho": round(rho, 4), "p_value": p,
            })
            pvals.append(p)

    p_corr, reject = fdr_correction(pvals)
    for i, r in enumerate(rows):
        r["p_fdr"] = round(p_corr[i], 6)
        r["significant"] = "yes" if reject[i] else "no"

    csv_path = stat_dir / "cognitive_feature_correlations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    for r in rows:
        if not np.isnan(r["spearman_rho"]):
            s = sig_stars(r["p_fdr"])
            logger.info(
                f"  {r['cognitive_score']:>4s} x {r['feature']:>14s}: "
                f"rho={r['spearman_rho']:+.3f} p_fdr={r['p_fdr']:.4f} {s} (n={r['n']})"
            )

    # --- Heatmap ---
    rho_matrix = np.full((len(score_cols), len(feature_cols)), np.nan)
    sig_matrix = np.full((len(score_cols), len(feature_cols)), "", dtype=object)
    idx = 0
    for i, score in enumerate(score_cols):
        for j, feat in enumerate(feature_cols):
            r = rows[idx]
            rho_matrix[i, j] = r["spearman_rho"] if not np.isnan(r.get("spearman_rho", np.nan)) else 0
            sig_matrix[i, j] = sig_stars(r["p_fdr"])
            idx += 1

    # Build annotation labels
    annot = np.empty_like(rho_matrix, dtype=object)
    for i in range(rho_matrix.shape[0]):
        for j in range(rho_matrix.shape[1]):
            val = rho_matrix[i, j]
            s = sig_matrix[i, j]
            annot[i, j] = f"{val:.2f}{s}" if not np.isnan(val) else ""

    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.heatmap(
        rho_matrix, annot=annot, fmt="",
        xticklabels=[c.replace("predicted_age", "Pred. Age").replace("age_error", "Age Error")
                      for c in feature_cols],
        yticklabels=score_cols,
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Spearman \u03c1"},
        ax=ax,
    )
    ax.set_title("Correlation: Cognitive Scores vs M3+M4 Features")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_cognitive_correlations.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_cognitive_correlations.png")


def plot_feature_correlation_matrix(df, fig_dir, stat_dir):
    """Analysis 7: 12-feature correlation matrix."""
    logger.info("[7] Feature correlation matrix")

    feature_cols = ["predicted_age", "age_error"] + EMOTION_COLS
    clean_labels = {
        "predicted_age": "Pred. Age",
        "age_error": "Age Error",
    }
    labels = [clean_labels.get(c, c) for c in feature_cols]

    corr = df[feature_cols].corr(method="spearman")
    corr.columns = labels
    corr.index = labels

    # Save CSV
    csv_path = stat_dir / "feature_correlation_matrix.csv"
    corr.to_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"  CSV saved: {csv_path}")

    # Compute p-values
    n_feat = len(feature_cols)
    p_matrix = np.ones((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            valid = df[[feature_cols[i], feature_cols[j]]].dropna()
            if len(valid) > 2:
                _, p = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
                p_matrix[i, j] = p
                p_matrix[j, i] = p

    # FDR on upper triangle
    upper_idx = np.triu_indices(n_feat, k=1)
    upper_p = p_matrix[upper_idx]
    upper_p_corr, _ = fdr_correction(upper_p)
    p_corr_matrix = np.ones((n_feat, n_feat))
    for k, (i, j) in enumerate(zip(*upper_idx)):
        p_corr_matrix[i, j] = upper_p_corr[k]
        p_corr_matrix[j, i] = upper_p_corr[k]

    # Annotation with significance
    annot = np.empty((n_feat, n_feat), dtype=object)
    corr_vals = corr.values
    for i in range(n_feat):
        for j in range(n_feat):
            s = sig_stars(p_corr_matrix[i, j]) if i != j else ""
            annot[i, j] = f"{corr_vals[i, j]:.2f}{s}"

    # Lower triangular mask
    mask = np.triu(np.ones((n_feat, n_feat), dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr, mask=mask, annot=annot, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Spearman \u03c1"},
        square=True, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (M3 + M4)")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "fig_feature_correlation_matrix.png"))
    plt.close(fig)
    logger.info(f"  Figure saved: fig_feature_correlation_matrix.png")


# ===================================================================
# Main
# ===================================================================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data ...")
    predicted_ages = load_predicted_ages(CALIBRATED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    emotions = load_emotion_scores(EMOTION_SCORES_FILE)
    df = build_merged_dataframe(predicted_ages, demo, emotions)

    logger.info("=" * 60)
    logger.info("Layer 1: Core Analyses")
    logger.info("=" * 60)
    analyze_emotion_profiles(df, FIGURES_DIR, STATS_DIR)
    analyze_age_error(df, FIGURES_DIR, STATS_DIR)
    plot_age_scatter_2group(df, FIGURES_DIR)

    logger.info("=" * 60)
    logger.info("Layer 2: CDR Stratification")
    logger.info("=" * 60)
    analyze_cdr_emotion_trends(df, FIGURES_DIR, STATS_DIR)
    analyze_cdr_age_error(df, FIGURES_DIR, STATS_DIR)

    logger.info("=" * 60)
    logger.info("Layer 3: Clinical Correlations")
    logger.info("=" * 60)
    analyze_cognitive_correlations(df, FIGURES_DIR, STATS_DIR)
    plot_feature_correlation_matrix(df, FIGURES_DIR, STATS_DIR)

    logger.info("=" * 60)
    logger.info("All analyses complete.")
    logger.info(f"Figures: {FIGURES_DIR}")
    logger.info(f"Statistics: {STATS_DIR}")


if __name__ == "__main__":
    main()
