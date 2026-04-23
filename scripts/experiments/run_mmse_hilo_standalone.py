"""
Standalone MMSE hi-lo analysis: AD cohort × MMSE median split × 1:1 age-matched.

獨立於 4-arm deep-dive 的 within-AD 分析。比較 high-MMSE vs low-MMSE AD subjects 於：
  (1) raw age prediction error (real_age - predicted_age)
  (2) 8 emotion methods × 7 emotions × 4 stats (mean/std/range/entropy)
  (3) landmark asymmetry L2 + area_diff
  (4) 3-model embedding asymmetry L2

此腳本同時匯出 match_1to1 / bh_fdr / compare_groups 等 helper 給 4-arm 使用。

Usage:
    conda run -n Alz_face_test_2 python scripts/experiments/run_mmse_hilo_standalone.py
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
AGES_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "features" / "aggregated"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features.csv"
EMBEDDING_DIFF_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "age_ladder" / "mmse_hilo_standalone"

EMOTION_METHODS = [
    "openface", "libreface", "pyfeat", "dan",
    "hsemotion", "vit", "poster_pp", "fer",
]
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
STATS = ["mean", "std", "range", "entropy"]
LANDMARK_REGIONS = ["eye", "nose", "mouth", "face_oval"]
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Cohort
# ============================================================

def load_p_demographics():
    df = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
    df["MMSE"] = pd.to_numeric(df["MMSE"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Global_CDR"] = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    df["base_id"] = df["ID"].str.extract(r"^([A-Za-z]+\d+)")
    df["visit"] = df["ID"].str.extract(r"-(\d+)$").astype(float)
    return df


def select_visit(df, visit_selection="first"):
    """Pick one visit per base_id where MMSE and Age are present."""
    elig = df.dropna(subset=["MMSE", "Age"]).copy()
    elig = elig.sort_values(["base_id", "visit"])
    if visit_selection == "first":
        picked = elig.groupby("base_id", as_index=False).first()
    else:
        picked = elig.groupby("base_id", as_index=False).last()
    return picked


def split_by_mmse_median(cohort, tiebreak="high"):
    median = cohort["MMSE"].median()
    if tiebreak == "high":
        cohort["mmse_group"] = np.where(cohort["MMSE"] >= median, "high", "low")
    else:
        cohort["mmse_group"] = np.where(cohort["MMSE"] > median, "high", "low")
    return cohort, float(median)


# ============================================================
# 1:1 age nearest-neighbor matching
# ============================================================

def match_1to1(cohort, caliper=2.0, seed=42):
    rng = np.random.RandomState(seed)
    high = cohort[cohort["mmse_group"] == "high"].copy()
    low = cohort[cohort["mmse_group"] == "low"].copy()

    # Minor group drives the iteration
    if len(low) <= len(high):
        minor, major = low, high
        minor_label, major_label = "low", "high"
    else:
        minor, major = high, low
        minor_label, major_label = "high", "low"

    # Shuffle minor group order (small randomness so results don't depend on input order)
    minor_order = minor.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    available = major.copy().reset_index(drop=True)

    pairs = []
    for _, row in minor_order.iterrows():
        if len(available) == 0:
            break
        age = row["Age"]
        diffs = (available["Age"] - age).abs()
        min_diff = diffs.min()
        if min_diff > caliper:
            continue
        # Tie-break by subject_id lex order
        candidates = available[diffs == min_diff].sort_values("ID")
        picked = candidates.iloc[0]
        pairs.append({
            "pair_id": len(pairs),
            "minor_id": row["ID"],
            "minor_age": age,
            "minor_mmse": row["MMSE"],
            "major_id": picked["ID"],
            "major_age": picked["Age"],
            "major_mmse": picked["MMSE"],
            "age_diff": picked["Age"] - age,
        })
        available = available.drop(picked.name).reset_index(drop=True)

    pairs_df = pd.DataFrame(pairs)

    # Build long-form matched cohort
    records = []
    for _, p in pairs_df.iterrows():
        records.append({
            "pair_id": p["pair_id"], "ID": p["minor_id"],
            "Age": p["minor_age"], "MMSE": p["minor_mmse"],
            "mmse_group": minor_label,
        })
        records.append({
            "pair_id": p["pair_id"], "ID": p["major_id"],
            "Age": p["major_age"], "MMSE": p["major_mmse"],
            "mmse_group": major_label,
        })
    matched = pd.DataFrame(records)
    return matched, pairs_df, (minor_label, major_label)


# ============================================================
# Feature loading
# ============================================================

def load_age_error(subject_ids, demo):
    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    rows = []
    for sid in subject_ids:
        real_row = demo[demo["ID"] == sid]
        if len(real_row) == 0:
            continue
        real_age = real_row.iloc[0]["Age"]
        pred_age = pred_ages.get(sid)
        if pd.isna(real_age) or pred_age is None:
            continue
        rows.append({
            "subject_id": sid,
            "age_error": real_age - pred_age,
            "abs_age_error": abs(real_age - pred_age),
        })
    return pd.DataFrame(rows)


def load_landmark_asymmetry(subject_ids):
    """Per-subject landmark asymmetry summary scalars:
      - {region}_l2: L2 norm of all x/y pair diffs in that region
      - {region}_area_diff: signed area asymmetry
      - landmark_total_l2: L2 over all 4 regions combined
    """
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    df = df[df["subject_id"].isin(subject_ids)].copy()

    rows = []
    for _, r in df.iterrows():
        out = {"subject_id": r["subject_id"]}
        total_sq = 0.0
        for region in LANDMARK_REGIONS:
            xy_cols = [c for c in df.columns
                       if c.startswith(f"{region}_") and "area_diff" not in c]
            vec = r[xy_cols].to_numpy(dtype=float)
            l2 = float(np.sqrt(np.nansum(vec * vec)))
            out[f"landmark__{region}_l2"] = l2
            out[f"landmark__{region}_area_diff"] = float(r[f"{region}_area_diff"])
            total_sq += np.nansum(vec * vec)
        out["landmark__total_l2"] = float(np.sqrt(total_sq))
        rows.append(out)
    return pd.DataFrame(rows)


def load_embedding_asymmetry(subject_ids):
    """Per-subject embedding asymmetry scalar per model (L2 norm of mean-pooled diff)."""
    rows = []
    for sid in subject_ids:
        out = {"subject_id": sid}
        for model in EMBEDDING_MODELS:
            npy = EMBEDDING_DIFF_DIR / model / "difference" / f"{sid}.npy"
            if not npy.exists():
                out[f"embasym__{model}_l2"] = np.nan
                continue
            a = np.load(npy, allow_pickle=True)
            if a.dtype == object:
                a = list(a.item().values())[0]
            mean_diff = a.mean(axis=0) if a.ndim == 2 else a
            out[f"embasym__{model}_l2"] = float(np.linalg.norm(mean_diff))
        rows.append(out)
    return pd.DataFrame(rows)


def load_emotion_features():
    """Return dict method -> DataFrame with subject_id + 7 emotions × 4 stats (28 cols)."""
    emotion_cols = [f"{emo}_{stat}" for emo in EMOTIONS for stat in STATS]
    out = {}
    for method in EMOTION_METHODS:
        path = EMOTION_DIR / f"{method}_harmonized.csv"
        if not path.exists():
            logger.warning(f"Missing {path}; skipping")
            continue
        df = pd.read_csv(path)
        keep = ["subject_id"] + [c for c in emotion_cols if c in df.columns]
        missing = [c for c in emotion_cols if c not in df.columns]
        if missing:
            logger.warning(f"{method}: missing cols {missing[:3]}... ({len(missing)} total)")
        # Rename feature cols to be method-prefixed
        df = df[keep].copy()
        rename = {c: f"{method}__{c}" for c in keep if c != "subject_id"}
        df = df.rename(columns=rename)
        out[method] = df
    return out


# ============================================================
# Stats
# ============================================================

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the top
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def hedges_g(a, b):
    """Hedges' g = Cohen's d with small-sample bias correction (J factor)."""
    d = cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    na, nb = len(a), len(b)
    df = na + nb - 2
    if df < 1:
        return d
    J = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * J


def compare_groups(values_high, values_low, feature_name, stat_mode="auto",
                    paired_values=None):
    """Compare high vs low MMSE for one feature.

    paired_values : optional pairs_df for paired t-test. If provided, also
      computes paired-test stat using the pair_id alignment.
    """
    vh = np.asarray(values_high, dtype=float)
    vl = np.asarray(values_low, dtype=float)
    vh_valid = vh[~np.isnan(vh)]
    vl_valid = vl[~np.isnan(vl)]
    n_h, n_l = len(vh_valid), len(vl_valid)

    base = {
        "feature": feature_name,
        "n_high": n_h, "n_low": n_l,
        "mean_high": vh_valid.mean() if n_h else np.nan,
        "std_high": vh_valid.std(ddof=1) if n_h > 1 else np.nan,
        "mean_low": vl_valid.mean() if n_l else np.nan,
        "std_low": vl_valid.std(ddof=1) if n_l > 1 else np.nan,
    }

    if n_h < 2 or n_l < 2:
        return {**base, "test": "skip", "stat": np.nan, "pvalue": np.nan,
                "cohen_d": np.nan, "hedges_g": np.nan,
                "paired_t_stat": np.nan, "paired_t_p": np.nan, "n_paired": 0}

    use_mwu = stat_mode == "mannwhitney" or (stat_mode == "auto" and min(n_h, n_l) < 20)
    if use_mwu:
        stat_val, pval = stats.mannwhitneyu(vh_valid, vl_valid,
                                             alternative="two-sided")
        test_name = "mannwhitney_u"
    else:
        stat_val, pval = stats.ttest_ind(vh_valid, vl_valid, equal_var=False)
        test_name = "welch_t"

    # Paired t-test on the subset where both members of each pair have data
    paired_stat, paired_p, n_paired = np.nan, np.nan, 0
    if paired_values is not None and len(vh) == len(vl):
        valid_mask = ~np.isnan(vh) & ~np.isnan(vl)
        if valid_mask.sum() >= 2:
            paired_stat, paired_p = stats.ttest_rel(vh[valid_mask], vl[valid_mask])
            n_paired = int(valid_mask.sum())

    return {
        **base,
        "test": test_name, "stat": float(stat_val), "pvalue": float(pval),
        "cohen_d": cohens_d(vh_valid, vl_valid),
        "hedges_g": hedges_g(vh_valid, vl_valid),
        "paired_t_stat": float(paired_stat) if not np.isnan(paired_stat) else np.nan,
        "paired_t_p": float(paired_p) if not np.isnan(paired_p) else np.nan,
        "n_paired": n_paired,
    }


# ============================================================
# Plots
# ============================================================

def plot_age_error_violin(matched_df, out_path):
    high = matched_df[matched_df["mmse_group"] == "high"]["age_error"].dropna()
    low = matched_df[matched_df["mmse_group"] == "low"]["age_error"].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([high, low], showmedians=True)
    for pc, color in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"High MMSE\n(n={len(high)})", f"Low MMSE\n(n={len(low)})"])
    ax.set_ylabel("Age error (real − predicted)")
    ax.set_title("Age prediction error by MMSE group\n(age-matched 1:1 within AD)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mmse_figure(cohort, matched_df, median_val, out_path):
    """MMSE-only figure: pre-match distribution + post-match boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) MMSE histogram (pre-matching), median split annotated
    ax = axes[0]
    hi = cohort[cohort["mmse_group"] == "high"]["MMSE"]
    lo = cohort[cohort["mmse_group"] == "low"]["MMSE"]
    bins = np.arange(0, 31, 1)
    ax.hist([lo, hi], bins=bins, stacked=True,
            color=["#C44E52", "#4C72B0"],
            label=[f"Low MMSE (n={len(lo)})", f"High MMSE (n={len(hi)})"])
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.2,
               label=f"median={median_val:.1f}")
    ax.set_xlabel("MMSE")
    ax.set_ylabel("# subjects")
    ax.set_title("(a) MMSE distribution (pre-matching AD cohort)")
    ax.legend()

    # (b) MMSE boxplot post-match
    ax = axes[1]
    hi_m = matched_df[matched_df["mmse_group"] == "high"]["MMSE"]
    lo_m = matched_df[matched_df["mmse_group"] == "low"]["MMSE"]
    bp = ax.boxplot([hi_m, lo_m], patch_artist=True,
                    tick_labels=[f"High\n(n={len(hi_m)})", f"Low\n(n={len(lo_m)})"])
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("MMSE")
    ax.set_title(f"(b) MMSE by group (post-matching)\n"
                 f"High: {hi_m.mean():.1f}±{hi_m.std():.1f}  |  "
                 f"Low: {lo_m.mean():.1f}±{lo_m.std():.1f}")
    ax.axhline(median_val, color="black", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_age_figure(cohort, matched_df, median_val, out_path):
    """Age-only figure: pre/post distribution + Age×MMSE scatter showing matching losses."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Age distribution: pre vs post matching, per group
    ax = axes[0]
    bins_age = np.arange(50, 100, 2)
    hi_pre = cohort[cohort["mmse_group"] == "high"]["Age"]
    lo_pre = cohort[cohort["mmse_group"] == "low"]["Age"]
    hi_post = matched_df[matched_df["mmse_group"] == "high"]["Age"]
    lo_post = matched_df[matched_df["mmse_group"] == "low"]["Age"]
    ax.hist(hi_pre, bins=bins_age, alpha=0.3, color="#4C72B0",
            label=f"High pre (n={len(hi_pre)})", histtype="stepfilled")
    ax.hist(lo_pre, bins=bins_age, alpha=0.3, color="#C44E52",
            label=f"Low pre (n={len(lo_pre)})", histtype="stepfilled")
    ax.hist(hi_post, bins=bins_age, histtype="step", color="#4C72B0",
            linewidth=2, label=f"High post (n={len(hi_post)})")
    ax.hist(lo_post, bins=bins_age, histtype="step", color="#C44E52",
            linewidth=2, label=f"Low post (n={len(lo_post)})")
    ax.set_xlabel("Age")
    ax.set_ylabel("# subjects")
    ax.set_title("(a) Age distribution: pre (fill) vs post (line) matching")
    ax.legend(fontsize=8)

    # (b) Age vs MMSE scatter — pre/post coloring
    ax = axes[1]
    matched_ids = set(matched_df["ID"])
    unmatched = cohort[~cohort["ID"].isin(matched_ids)]
    ax.scatter(unmatched["Age"], unmatched["MMSE"], color="lightgray",
               s=18, alpha=0.5, label=f"unmatched (n={len(unmatched)})")
    hi_mm = matched_df[matched_df["mmse_group"] == "high"]
    lo_mm = matched_df[matched_df["mmse_group"] == "low"]
    ax.scatter(hi_mm["Age"], hi_mm["MMSE"], color="#4C72B0", s=22,
               alpha=0.7, label=f"High matched (n={len(hi_mm)})")
    ax.scatter(lo_mm["Age"], lo_mm["MMSE"], color="#C44E52", s=22,
               alpha=0.7, label=f"Low matched (n={len(lo_mm)})")
    ax.axhline(median_val, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Age")
    ax.set_ylabel("MMSE")
    ax.set_title("(b) Age × MMSE (matched highlighted)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_emotion_heatmap(summary_df, out_path):
    """8 methods × 7 emotions heatmap of Δmean (high − low) on the _mean stat only."""
    diff = np.full((len(EMOTION_METHODS), len(EMOTIONS)), np.nan)
    sig = np.full_like(diff, "", dtype=object)
    for i, method in enumerate(EMOTION_METHODS):
        for j, emo in enumerate(EMOTIONS):
            feat = f"{method}__{emo}_mean"
            row = summary_df[summary_df["feature"] == feat]
            if len(row) == 0:
                continue
            diff[i, j] = row["mean_high"].iloc[0] - row["mean_low"].iloc[0]
            q = row["qvalue"].iloc[0]
            if not pd.isna(q):
                if q < 0.001: sig[i, j] = "***"
                elif q < 0.01: sig[i, j] = "**"
                elif q < 0.05: sig[i, j] = "*"

    vmax = np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(EMOTION_METHODS)))
    ax.set_yticklabels(EMOTION_METHODS)
    for i in range(len(EMOTION_METHODS)):
        for j in range(len(EMOTIONS)):
            if sig[i, j]:
                ax.text(j, i, sig[i, j], ha="center", va="center", fontsize=10)
    ax.set_title("Δ mean (High MMSE − Low MMSE) of emotion probabilities\n* q<0.05, ** q<0.01, *** q<0.001")
    fig.colorbar(im, ax=ax, label="Δ mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visit-selection", choices=["first", "latest"], default="first")
    parser.add_argument("--median-tiebreak", choices=["high", "low"], default="high",
                        help="Where to place MMSE == median subjects")
    parser.add_argument("--caliper", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stat", choices=["ttest", "mannwhitney", "auto"], default="auto")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cohort
    demo = load_p_demographics()
    cohort = select_visit(demo, args.visit_selection)
    logger.info(f"P patients with MMSE+Age ({args.visit_selection} visit): n={len(cohort)}")

    # 2. Filter to those with predicted age
    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    cohort = cohort[cohort["ID"].isin(pred_ages)].copy()
    logger.info(f"After requiring predicted_age: n={len(cohort)}")

    # 3. Split by MMSE median
    cohort, median_val = split_by_mmse_median(cohort, args.median_tiebreak)
    n_high = (cohort["mmse_group"] == "high").sum()
    n_low = (cohort["mmse_group"] == "low").sum()
    logger.info(f"MMSE median={median_val:.1f}; high n={n_high}, low n={n_low}")

    cohort.to_csv(OUTPUT_DIR / "cohort.csv", index=False)

    # 4. 1:1 age matching
    matched, pairs_df, (minor_label, major_label) = match_1to1(
        cohort, caliper=args.caliper, seed=args.seed
    )
    pairs_df.to_csv(OUTPUT_DIR / "matched_pairs.csv", index=False)

    n_pairs = len(pairs_df)
    logger.info(f"Matched pairs: {n_pairs} (caliper={args.caliper}, minor={minor_label})")

    # 5. Load features and merge
    age_err = load_age_error(matched["ID"].tolist(), demo)
    emo_dfs = load_emotion_features()
    lmk = load_landmark_asymmetry(matched["ID"].tolist())
    emb_asym = load_embedding_asymmetry(matched["ID"].tolist())

    merged = matched.merge(age_err, left_on="ID", right_on="subject_id", how="left")
    for method, edf in emo_dfs.items():
        merged = merged.merge(edf, left_on="ID", right_on="subject_id",
                              how="left", suffixes=("", f"_{method}_sid"))
        if f"subject_id_{method}_sid" in merged.columns:
            merged = merged.drop(columns=[f"subject_id_{method}_sid"])
    merged = merged.merge(lmk, left_on="ID", right_on="subject_id",
                          how="left", suffixes=("", "_lmk_sid"))
    if "subject_id_lmk_sid" in merged.columns:
        merged = merged.drop(columns=["subject_id_lmk_sid"])
    merged = merged.merge(emb_asym, left_on="ID", right_on="subject_id",
                          how="left", suffixes=("", "_emb_sid"))
    if "subject_id_emb_sid" in merged.columns:
        merged = merged.drop(columns=["subject_id_emb_sid"])
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("subject_id")])

    merged.to_csv(OUTPUT_DIR / "matched_features.csv", index=False)

    # 6. Matching report
    with open(OUTPUT_DIR / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write("=== AD MMSE age-balanced cohort ===\n")
        f.write(f"Visit selection: {args.visit_selection}\n")
        f.write(f"MMSE median: {median_val:.2f} (tiebreak={args.median_tiebreak})\n")
        f.write(f"Pre-matching: high n={n_high}, low n={n_low}\n")
        f.write(f"Post-matching: {n_pairs} pairs ({2*n_pairs} subjects)\n")
        f.write(f"Caliper (years): {args.caliper}\n\n")
        high_m = matched[matched["mmse_group"] == "high"]
        low_m = matched[matched["mmse_group"] == "low"]
        f.write(f"Post-match age: high {high_m['Age'].mean():.2f}±{high_m['Age'].std():.2f}, "
                f"low {low_m['Age'].mean():.2f}±{low_m['Age'].std():.2f}\n")
        if len(high_m) > 1 and len(low_m) > 1:
            t, p = stats.ttest_ind(high_m["Age"], low_m["Age"], equal_var=False)
            f.write(f"Age t-test p={p:.4f} (should be >>0.05 if matching worked)\n")
        f.write(f"Post-match MMSE: high {high_m['MMSE'].mean():.2f}±{high_m['MMSE'].std():.2f}, "
                f"low {low_m['MMSE'].mean():.2f}±{low_m['MMSE'].std():.2f}\n")

    # 7. Per-feature stats
    feature_cols = ["age_error", "abs_age_error"]
    for method in EMOTION_METHODS:
        for emo in EMOTIONS:
            for stat in STATS:
                col = f"{method}__{emo}_{stat}"
                if col in merged.columns:
                    feature_cols.append(col)
    for region in LANDMARK_REGIONS:
        for suffix in ("l2", "area_diff"):
            col = f"landmark__{region}_{suffix}"
            if col in merged.columns:
                feature_cols.append(col)
    if "landmark__total_l2" in merged.columns:
        feature_cols.append("landmark__total_l2")
    for model in EMBEDDING_MODELS:
        col = f"embasym__{model}_l2"
        if col in merged.columns:
            feature_cols.append(col)

    # Build paired-aligned arrays by pair_id (for paired t-test)
    high_by_pair = merged[merged["mmse_group"] == "high"].set_index("pair_id")
    low_by_pair = merged[merged["mmse_group"] == "low"].set_index("pair_id")
    common_pairs = high_by_pair.index.intersection(low_by_pair.index)
    high_aligned = high_by_pair.loc[common_pairs].sort_index()
    low_aligned = low_by_pair.loc[common_pairs].sort_index()

    rows = []
    for col in feature_cols:
        if col not in merged.columns:
            continue
        rows.append(compare_groups(
            high_aligned[col].to_numpy(dtype=float),
            low_aligned[col].to_numpy(dtype=float),
            col, stat_mode=args.stat,
            paired_values=True,
        ))
    summary = pd.DataFrame(rows)

    # 8. FDR correction across all features (BH-FDR applied independently
    # to the Welch p-values and the paired t-test p-values)
    for pcol, qcol in [("pvalue", "qvalue"), ("paired_t_p", "paired_t_q")]:
        valid = summary[pcol].notna()
        summary[qcol] = np.nan
        if valid.sum() > 0:
            summary.loc[valid, qcol] = bh_fdr(summary.loc[valid, pcol].values)

    summary.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)

    # 9. Continuous MMSE correlations (robustness)
    cont_rows = []
    for col in feature_cols:
        if col not in merged.columns:
            continue
        sub = merged[[col, "MMSE"]].dropna()
        if len(sub) < 5:
            continue
        r_p, p_p = stats.pearsonr(sub["MMSE"], sub[col])
        r_s, p_s = stats.spearmanr(sub["MMSE"], sub[col])
        cont_rows.append({
            "feature": col, "n": len(sub),
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
        })
    pd.DataFrame(cont_rows).to_csv(OUTPUT_DIR / "mmse_continuous_corr.csv", index=False)

    # 10. Plots
    plot_mmse_figure(cohort, matched, median_val, OUTPUT_DIR / "fig_mmse_overview.png")
    plot_age_figure(cohort, matched, median_val, OUTPUT_DIR / "fig_age_overview.png")
    plot_age_error_violin(merged, OUTPUT_DIR / "fig_age_error_violin.png")
    plot_emotion_heatmap(summary, OUTPUT_DIR / "fig_emotion_grid_heatmap.png")

    logger.info(f"Done. Outputs at {OUTPUT_DIR}")
    logger.info(f"  Pairs: {n_pairs}, features tested: {len(summary)}")
    sig = summary[summary["qvalue"] < 0.05]
    logger.info(f"  Significant (q<0.05): {len(sig)} features")
    if len(sig) > 0:
        logger.info(f"  Top 5: {sig.nsmallest(5, 'qvalue')['feature'].tolist()}")


if __name__ == "__main__":
    main()
