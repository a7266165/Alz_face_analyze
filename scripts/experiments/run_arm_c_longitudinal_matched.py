"""
Arm C: MMSE baseline-matched longitudinal comparison.

Parallels Arm B's strict 1:1 age-matching design but on the longitudinal
axis. Splits multi-visit AD patients by baseline MMSE median (High vs Low),
matches 1:1 on baseline_age with caliper = 2y, then compares annualized
delta features (drift-rate-style) between the two matched groups.

Primary question: Does baseline severity predict the rate of face drift
over 2 years, above and beyond age?

Usage:
    conda run -n Alz_face_test_2 python scripts/experiments/run_arm_c_longitudinal_matched.py
"""

import argparse
import importlib.util
import logging
import os
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

# Reuse utilities from Arm A (cv_eval, bootstrap_auc_ci)
_spec = importlib.util.spec_from_file_location(
    "arm_a_ad_vs_hc", Path(__file__).parent / "run_arm_a_ad_vs_hc.py"
)
_arm_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arm_a)
cv_eval = _arm_a.cv_eval
bootstrap_auc_ci = _arm_a.bootstrap_auc_ci
cohens_d = _arm_a.cohens_d
hedges_g = _arm_a.hedges_g

DELTAS_CSV = PROJECT_ROOT / "workspace" / "longitudinal" / "patient_deltas.csv"
LANDMARK_LONG_CSV = (PROJECT_ROOT / "workspace" / "asymmetry" / "analysis" /
                     "longitudinal_landmark_deltas.csv")
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "arms_analysis" / "per_arm" / "arm_c"

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness",
            "surprise", "neutral"]
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]
LANDMARK_DELTA_COLS = ["eye_l2", "nose_l2", "mouth_l2", "face_oval_l2",
                       "landmark_l2_norm", "landmark_mean_abs"]

# Hi-lo split metric: MMSE (default) or CASI. Configurable via env var.
METRIC = os.environ.get("HILO_METRIC", "MMSE")
METRIC_LOW = METRIC.lower()
GROUP_COL = f"{METRIC_LOW}_group"
COMPARISON_NAME = f"{METRIC_LOW}_high_vs_low"
FIRST_METRIC = f"first_{METRIC}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


# ============================================================
# Cohort build
# ============================================================

def load_arm_c_cohort(min_follow_up_days=180):
    df = pd.read_csv(DELTAS_CSV)
    need = [FIRST_METRIC, "first_age", "baseline_predicted_age",
            "emb_cosine_dist", "follow_up_days", "follow_up_years"]
    for col in need:
        if col not in df.columns:
            raise RuntimeError(f"patient_deltas missing required col: {col} "
                                f"(re-run build_longitudinal_dataset.py)")
    mask = (df[FIRST_METRIC].notna() &
            df["first_age"].notna() &
            df["baseline_predicted_age"].notna() &
            df["emb_cosine_dist"].notna() &
            (df["follow_up_days"] >= min_follow_up_days))
    cohort = df[mask].copy().reset_index(drop=True)

    # Merge landmark first-to-last deltas (by base_id); columns come through
    # un-prefixed so we rename them to lmk_delta_* to avoid ambiguity.
    if LANDMARK_LONG_CSV.exists():
        lmk = pd.read_csv(LANDMARK_LONG_CSV)
        keep = ["base_id"] + [c for c in LANDMARK_DELTA_COLS if c in lmk.columns]
        lmk = lmk[keep].copy()
        rename = {c: f"lmk_delta_{c}" for c in LANDMARK_DELTA_COLS if c in lmk.columns}
        lmk = lmk.rename(columns=rename)
        cohort = cohort.merge(lmk, on="base_id", how="left")
    else:
        logger.warning(f"Landmark longitudinal deltas not found: {LANDMARK_LONG_CSV}")

    return cohort


def split_by_first_metric(cohort, tiebreak="high"):
    median = cohort[FIRST_METRIC].median()
    if tiebreak == "high":
        cohort[GROUP_COL] = np.where(
            cohort[FIRST_METRIC] >= median, "high", "low")
    else:
        cohort[GROUP_COL] = np.where(
            cohort[FIRST_METRIC] > median, "high", "low")
    return cohort, float(median)


# Backward-compat alias
split_by_first_mmse = split_by_first_metric


# ============================================================
# 1:1 age matching (same logic as Arm B, on first_age)
# ============================================================

def match_1to1(cohort, caliper=2.0, seed=42):
    rng = np.random.RandomState(seed)
    high = cohort[cohort[GROUP_COL] == "high"].copy()
    low = cohort[cohort[GROUP_COL] == "low"].copy()

    if len(low) <= len(high):
        minor, major = low, high
        minor_label, major_label = "low", "high"
    else:
        minor, major = high, low
        minor_label, major_label = "high", "low"

    minor_order = minor.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    available = major.copy().reset_index(drop=True)

    pairs = []
    for _, row in minor_order.iterrows():
        if len(available) == 0:
            break
        age = row["first_age"]
        diffs = (available["first_age"] - age).abs()
        min_diff = diffs.min()
        if min_diff > caliper:
            continue
        candidates = available[diffs == min_diff].sort_values("base_id")
        picked = candidates.iloc[0]
        pairs.append({
            "pair_id": len(pairs),
            "minor_id": row["base_id"], "minor_age": age,
            f"minor_{METRIC_LOW}": row[FIRST_METRIC],
            "minor_follow_up_years": row["follow_up_years"],
            "major_id": picked["base_id"],
            "major_age": picked["first_age"],
            f"major_{METRIC_LOW}": picked[FIRST_METRIC],
            "major_follow_up_years": picked["follow_up_years"],
            "age_diff": picked["first_age"] - age,
        })
        available = available.drop(picked.name).reset_index(drop=True)

    pairs_df = pd.DataFrame(pairs)

    records = []
    for _, p in pairs_df.iterrows():
        records.append({
            "pair_id": p["pair_id"], "base_id": p["minor_id"],
            GROUP_COL: minor_label,
        })
        records.append({
            "pair_id": p["pair_id"], "base_id": p["major_id"],
            GROUP_COL: major_label,
        })
    matched = pd.DataFrame(records)
    return matched, pairs_df, (minor_label, major_label)


# ============================================================
# Annualized feature construction
# ============================================================

def build_annualized_features(cohort):
    """Add annualized_* columns = (delta col) / follow_up_years."""
    df = cohort.copy()
    y = df["follow_up_years"].replace(0, np.nan)
    candidate_delta = (["delta_age_error", "delta_predicted_age"] +
                       [f"delta_{e}" for e in EMOTIONS] +
                       ["delta_MMSE", "delta_CASI", "delta_CDR_SB"])
    for col in candidate_delta:
        if col in df.columns:
            df[f"ann_{col}"] = df[col] / y

    # Embedding cosine drift per model (back-compat alias + 3 new columns)
    df["ann_emb_cosine_dist"] = df["emb_cosine_dist"] / y
    for model in EMBEDDING_MODELS:
        col = f"emb_cosine_dist_{model}"
        if col in df.columns:
            df[f"ann_{col}"] = df[col] / y

    # Embedding asymmetry delta per model (Δ L2 / year)
    for model in EMBEDDING_MODELS:
        col = f"delta_embasym_{model}"
        if col in df.columns:
            df[f"ann_{col}"] = df[col] / y

    # Landmark first-to-last deltas → annualized
    for c in LANDMARK_DELTA_COLS:
        src = f"lmk_delta_{c}"
        if src in df.columns:
            df[f"ann_{src}"] = df[src] / y
    return df


# ============================================================
# Per-feature comparison (high vs low matched groups)
# ============================================================

def compare_groups_paired(values_high, values_low, feature_name):
    """Paired comparison; inputs already aligned by pair_id."""
    vh = np.asarray(values_high, dtype=float)
    vl = np.asarray(values_low, dtype=float)
    valid_mask = ~np.isnan(vh) & ~np.isnan(vl)
    vh_v, vl_v = vh[valid_mask], vl[valid_mask]
    n = len(vh_v)

    base = {
        "feature": feature_name, "n_pairs": n,
        "mean_high": float(vh_v.mean()) if n > 0 else np.nan,
        "std_high": float(vh_v.std(ddof=1)) if n > 1 else np.nan,
        "mean_low": float(vl_v.mean()) if n > 0 else np.nan,
        "std_low": float(vl_v.std(ddof=1)) if n > 1 else np.nan,
    }
    if n < 5:
        return {**base, "welch_t": np.nan, "welch_p": np.nan,
                "paired_t": np.nan, "paired_p": np.nan,
                "cohen_d": np.nan, "hedges_g": np.nan}

    wt, wp = stats.ttest_ind(vh_v, vl_v, equal_var=False)
    pt, pp = stats.ttest_rel(vh_v, vl_v)
    return {
        **base,
        "welch_t": float(wt), "welch_p": float(wp),
        "paired_t": float(pt), "paired_p": float(pp),
        "cohen_d": cohens_d(vh_v, vl_v),
        "hedges_g": hedges_g(vh_v, vl_v),
    }


# ============================================================
# Classification AUC on matched cohort
# ============================================================

def classification_auc(feat_df, label, groups, model_cls="xgb",
                        n_folds=5, seed=42):
    """Return AUC + CI; feat_df is a DataFrame aligned with label."""
    X = feat_df.to_numpy(dtype=float)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(label)
    X, y, g = X[mask], label[mask].astype(int), groups[mask]
    if len(X) < 50 or len(np.unique(y)) < 2:
        return {"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "balacc": np.nan, "mcc": np.nan, "n": len(X)}
    m = cv_eval(X, y, g, model_cls=model_cls, n_folds=n_folds, seed=seed,
                return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=seed)
    return {"n": int(mask.sum()),
            "auc_ci_low": ci_low, "auc_ci_high": ci_high,
            **m}


# ============================================================
# Plots
# ============================================================

def plot_drift_by_group(merged, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # ann emb_cosine_dist
    ax = axes[0]
    parts = ax.violinplot(
        [merged[merged[GROUP_COL] == "high"]["ann_emb_cosine_dist"].dropna(),
         merged[merged[GROUP_COL] == "low"]["ann_emb_cosine_dist"].dropna()],
        showmedians=True,
    )
    for pc, c in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"High {METRIC}", f"Low {METRIC}"])
    ax.set_ylabel("Annualized embedding cosine drift (per year)")
    ax.set_title(f"Embedding drift rate by baseline {METRIC} group")

    # ann delta_age_error
    ax = axes[1]
    parts = ax.violinplot(
        [merged[merged[GROUP_COL] == "high"]["ann_delta_age_error"].dropna(),
         merged[merged[GROUP_COL] == "low"]["ann_delta_age_error"].dropna()],
        showmedians=True,
    )
    for pc, c in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"High {METRIC}", f"Low {METRIC}"])
    ax.set_ylabel("Annualized Δ age_error (y/y)")
    ax.set_title(f"Age-error drift rate by baseline {METRIC} group")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caliper", type=float, default=2.0)
    parser.add_argument("--min-follow-up-days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiebreak", choices=["high", "low"], default="high")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_dir = OUTPUT_DIR / COMPARISON_NAME
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for d in ("age", "embedding_mean", "embedding_asymmetry",
              "landmark_asymmetry", "emotion"):
        (comparison_dir / d).mkdir(parents=True, exist_ok=True)

    # 1. Cohort
    cohort = load_arm_c_cohort(min_follow_up_days=args.min_follow_up_days)
    logger.info(f"Eligible longitudinal cohort: {len(cohort)}")

    # 2. Annualized features
    cohort = build_annualized_features(cohort)

    # 3. Metric median split
    cohort, median_val = split_by_first_metric(cohort, args.tiebreak)
    n_high = (cohort[GROUP_COL] == "high").sum()
    n_low = (cohort[GROUP_COL] == "low").sum()
    logger.info(f"Baseline {METRIC} median={median_val:.1f}; high={n_high}, low={n_low}")
    cohort.to_csv(comparison_dir / "cohort_longitudinal.csv", index=False)

    # 4. 1:1 age match on first_age
    matched, pairs_df, (minor, major) = match_1to1(
        cohort, caliper=args.caliper, seed=args.seed
    )
    pairs_df.to_csv(comparison_dir / "matched_pairs_longitudinal.csv", index=False)
    logger.info(f"Matched pairs: {len(pairs_df)} "
                f"(caliper={args.caliper}, minor={minor})")

    # Merge full cohort info into matched (drop cohort's group col to avoid
    # _x/_y collision; matched already has the authoritative group label).
    merged = matched.merge(
        cohort.drop(columns=[GROUP_COL]), on="base_id", how="left"
    )
    merged.to_csv(comparison_dir / "matched_features_longitudinal.csv", index=False)

    # 5. Matching report
    high_m = merged[merged[GROUP_COL] == "high"]
    low_m = merged[merged[GROUP_COL] == "low"]
    with open(comparison_dir / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Arm C longitudinal matched cohort ===\n")
        f.write(f"Eligible n: {len(cohort)}\n")
        f.write(f"{METRIC} median={median_val:.2f}; pre-match high={n_high}, low={n_low}\n")
        f.write(f"Post-match: {len(pairs_df)} pairs ({2*len(pairs_df)} subjects)\n\n")

        f.write(f"Baseline age: high {high_m['first_age'].mean():.2f}±"
                f"{high_m['first_age'].std():.2f}, "
                f"low {low_m['first_age'].mean():.2f}±"
                f"{low_m['first_age'].std():.2f}\n")
        if len(high_m) > 1 and len(low_m) > 1:
            t, p = stats.ttest_ind(high_m["first_age"], low_m["first_age"],
                                    equal_var=False)
            f.write(f"  Age Welch t-test p={p:.4f} (>>0.05 = matching OK)\n")

        f.write(f"\nFollow-up (years): high {high_m['follow_up_years'].mean():.2f}±"
                f"{high_m['follow_up_years'].std():.2f}, "
                f"low {low_m['follow_up_years'].mean():.2f}±"
                f"{low_m['follow_up_years'].std():.2f}\n")
        tt, tp = stats.ttest_ind(high_m["follow_up_years"],
                                  low_m["follow_up_years"], equal_var=False)
        f.write(f"  Follow-up Welch t-test p={tp:.4f}\n")

        f.write(f"\nBaseline {METRIC}: high {high_m[FIRST_METRIC].mean():.2f}±"
                f"{high_m[FIRST_METRIC].std():.2f}, "
                f"low {low_m[FIRST_METRIC].mean():.2f}±"
                f"{low_m[FIRST_METRIC].std():.2f}\n")

    # 6. Per-feature annualized comparisons (paired, aligned by pair_id)
    high_align = (merged[merged[GROUP_COL] == "high"]
                  .set_index("pair_id").sort_index())
    low_align = (merged[merged[GROUP_COL] == "low"]
                 .set_index("pair_id").sort_index())
    common = high_align.index.intersection(low_align.index)
    high_align = high_align.loc[common]
    low_align = low_align.loc[common]

    # Features to compare: annualized drift features
    compare_features = ["ann_emb_cosine_dist", "ann_delta_age_error"]
    for e in EMOTIONS:
        if f"ann_delta_{e}" in merged.columns:
            compare_features.append(f"ann_delta_{e}")
    for cog in ["ann_delta_MMSE", "ann_delta_CASI", "ann_delta_CDR_SB"]:
        if cog in merged.columns:
            compare_features.append(cog)
    # Per-model embedding cosine drift rates
    for model in EMBEDDING_MODELS:
        col = f"ann_emb_cosine_dist_{model}"
        if col in merged.columns:
            compare_features.append(col)
    # Per-model embedding asymmetry delta rates
    for model in EMBEDDING_MODELS:
        col = f"ann_delta_embasym_{model}"
        if col in merged.columns:
            compare_features.append(col)
    # Landmark annualized deltas
    for c in LANDMARK_DELTA_COLS:
        col = f"ann_lmk_delta_{c}"
        if col in merged.columns:
            compare_features.append(col)

    per_feat_rows = []
    for col in compare_features:
        if col not in high_align.columns:
            continue
        per_feat_rows.append(compare_groups_paired(
            high_align[col].values, low_align[col].values, col
        ))
    pf = pd.DataFrame(per_feat_rows)
    for pcol, qcol in [("welch_p", "welch_q"), ("paired_p", "paired_q")]:
        valid = pf[pcol].notna()
        pf[qcol] = np.nan
        if valid.sum() > 0:
            pf.loc[valid, qcol] = bh_fdr(pf.loc[valid, pcol].values)
    pf.to_csv(comparison_dir / "per_feature_stats.csv", index=False)

    # 7. Modality-level classification AUC (High vs Low MMSE predicted from
    # each annualized feature set)
    merged["label"] = (merged[GROUP_COL] == "high").astype(int)
    base_ids = merged["base_id"].to_numpy()
    labels = merged["label"].to_numpy()

    mod_rows = []

    def eval_mod(name, feat_cols, model_cls="xgb"):
        feats = merged[feat_cols].copy()
        r = classification_auc(feats, labels, base_ids, model_cls=model_cls,
                                n_folds=5, seed=args.seed)
        return {"modality": name, "n_features": len(feat_cols), **r}

    if "first_age" in merged.columns:
        mod_rows.append(eval_mod("age_only", ["first_age"], "logistic"))
    mod_rows.append(eval_mod("age_error_annualized",
                              ["ann_delta_age_error"], "logistic"))
    emo_ann_cols = [f"ann_delta_{e}" for e in EMOTIONS
                    if f"ann_delta_{e}" in merged.columns]
    if emo_ann_cols:
        mod_rows.append(eval_mod("emotion_annualized", emo_ann_cols, "xgb"))

    # Embedding cosine drift rate, per model
    # (keep back-compat `embedding_drift_rate` = arcface)
    mod_rows.append(eval_mod("embedding_drift_rate",
                              ["ann_emb_cosine_dist"], "logistic"))
    for model in EMBEDDING_MODELS:
        col = f"ann_emb_cosine_dist_{model}"
        if col in merged.columns:
            mod_rows.append(eval_mod(f"embedding_{model}_drift_rate",
                                      [col], "logistic"))

    # Embedding asymmetry annualized delta (3 scalars → XGBoost)
    emb_asym_cols = [f"ann_delta_embasym_{m}" for m in EMBEDDING_MODELS
                     if f"ann_delta_embasym_{m}" in merged.columns]
    if emb_asym_cols:
        mod_rows.append(eval_mod("embedding_asymmetry_annualized",
                                  emb_asym_cols, "xgb"))

    # Landmark asymmetry annualized delta (multi-feature → XGBoost)
    lmk_cols = [f"ann_lmk_delta_{c}" for c in LANDMARK_DELTA_COLS
                if f"ann_lmk_delta_{c}" in merged.columns]
    if lmk_cols:
        mod_rows.append(eval_mod("landmark_asymmetry_annualized",
                                  lmk_cols, "xgb"))

    mod_df = pd.DataFrame(mod_rows)
    mod_df.to_csv(comparison_dir / "summary_per_modality.csv", index=False)

    # 8. Plots
    plot_drift_by_group(merged, comparison_dir / "embedding_mean" / "fig_drift_rate_by_group.png")

    logger.info(f"Done. Outputs at {OUTPUT_DIR}")
    sig_feat = pf[pf["paired_q"] < 0.05]
    logger.info(f"Features with paired_q<0.05: {len(sig_feat)}")
    if len(sig_feat) > 0:
        logger.info("Top-5 by paired_q:")
        for _, r in sig_feat.nsmallest(5, "paired_q").iterrows():
            logger.info(f"  {r['feature']:<30} d={r['cohen_d']:+.3f}  "
                        f"paired_p={r['paired_p']:.4f}  q={r['paired_q']:.4f}")
    for _, r in mod_df.iterrows():
        auc = r.get("auc", np.nan)
        ci_l = r.get("auc_ci_low", np.nan); ci_h = r.get("auc_ci_high", np.nan)
        logger.info(f"  {r['modality']:<25} AUC={auc:.3f} [{ci_l:.3f}, {ci_h:.3f}]  "
                    f"n={r['n']}  #feats={r['n_features']}")


if __name__ == "__main__":
    main()
