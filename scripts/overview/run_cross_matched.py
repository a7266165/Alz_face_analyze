"""
Cross-sectional age-matched + AD hi-lo split analyses.

Comparisons (--comparison):
  ad_vs_{hc, nad, acs}  AD vs HC-group, 1:1 age-matched on first visit.
                        Only matching artifacts + age-error violin (薄包裝；cohort
                        comes from scripts.utilities.cohort.build_cohort_ad_vs_HCgroup
                        with design='cross_matched').
  {mmse, casi}_hilo     AD within-cohort hi-lo split (median on metric), 1:1
                        age-matched. Full multi-modality analysis:
                          (1) raw age-prediction error
                          (2) 8 emotion methods × 7 emotions × 4 stats
                          (3) landmark asymmetry L2 + area_diff
                          (4) 3-model embedding asymmetry L2
                        Plus per-modality AUC supplement: age_only, age_error,
                        emotion, landmark_asymmetry, embedding_asymmetry,
                        embedding_{arcface,topofr,dlib}_mean.

All cohort builders / matching / feature loaders / stat helpers live in
scripts/utilities/{cohort, feature_loaders, stats_helpers}.py. This script
only contains plot helpers + the comparison-level orchestration.

Usage:
    conda run -n Alz_face_main_analysis python scripts/overview/run_cross_matched.py --comparison mmse_hilo
    conda run -n Alz_face_main_analysis python scripts/overview/run_cross_matched.py --comparison ad_vs_hc
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    AGE_PRED_ERROR_STAT_DIR, EMO_AU_FEATURE_STAT_DIR,
    OVERVIEW_DIR, VALID_COHORT_CHOICES, cohort_path, cohort_spec_from_name,
)
from scripts.utilities.cohort import (
    _keep_visits_with_features, _pick_first_visit_with_features,
    apply_p_cdr_filter,
    build_cohort_ad_vs_HCgroup, load_p_demographics, match_1to1,
    select_visit, split_by_metric_median,
)
from scripts.utilities.feature_loaders import (
    EMBEDDING_MODELS, EMOTION_METHODS, EMOTION_STATS, EMOTIONS, LANDMARK_REGIONS,
    load_age_error, load_embedding_asymmetry, load_embedding_mean,
    load_emotion_features, load_landmark_matrix,
)
from scripts.utilities.stats_helpers import (
    bh_fdr, bootstrap_auc_ci, cohens_d, compare_groups, cv_eval,
)

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
from src.config import PREDICTED_AGES_FILE as AGES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Plots
# ============================================================

def plot_age_error_violin(matched_df, out_path, metric, group_col):
    high = matched_df[matched_df[group_col] == "high"]["age_error"].dropna()
    low = matched_df[matched_df[group_col] == "low"]["age_error"].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([high, low], showmedians=True)
    for pc, color in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"High {metric}\n(n={len(high)})",
                        f"Low {metric}\n(n={len(low)})"])
    ax.set_ylabel("Age error (real − predicted)")
    ax.set_title(f"Age prediction error by {metric} group\n"
                 f"(age-matched 1:1 within AD)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_figure(cohort, matched_df, median_val, out_path,
                       metric, group_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    hi = cohort[cohort[group_col] == "high"][metric]
    lo = cohort[cohort[group_col] == "low"][metric]
    metric_max = max(30, int(np.ceil(cohort[metric].max() / 5.0)) * 5)
    bins = np.arange(0, metric_max + 1, max(1, metric_max // 30))
    ax.hist([lo, hi], bins=bins, stacked=True,
            color=["#C44E52", "#4C72B0"],
            label=[f"Low {metric} (n={len(lo)})",
                   f"High {metric} (n={len(hi)})"])
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.2,
               label=f"median={median_val:.1f}")
    ax.set_xlabel(metric); ax.set_ylabel("# subjects")
    ax.set_title(f"(a) {metric} distribution (pre-matching AD cohort)")
    ax.legend()

    ax = axes[1]
    hi_m = matched_df[matched_df[group_col] == "high"][metric]
    lo_m = matched_df[matched_df[group_col] == "low"][metric]
    bp = ax.boxplot([hi_m, lo_m], patch_artist=True,
                    tick_labels=[f"High\n(n={len(hi_m)})",
                                 f"Low\n(n={len(lo_m)})"])
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel(metric)
    ax.set_title(f"(b) {metric} by group (post-matching)\n"
                 f"High: {hi_m.mean():.1f}±{hi_m.std():.1f}  |  "
                 f"Low: {lo_m.mean():.1f}±{lo_m.std():.1f}")
    ax.axhline(median_val, color="black", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_age_figure(cohort, matched_df, median_val, out_path, metric, group_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bins_age = np.arange(50, 100, 2)
    hi_pre = cohort[cohort[group_col] == "high"]["Age"]
    lo_pre = cohort[cohort[group_col] == "low"]["Age"]
    hi_post = matched_df[matched_df[group_col] == "high"]["Age"]
    lo_post = matched_df[matched_df[group_col] == "low"]["Age"]
    ax.hist(hi_pre, bins=bins_age, alpha=0.3, color="#4C72B0",
            label=f"High pre (n={len(hi_pre)})", histtype="stepfilled")
    ax.hist(lo_pre, bins=bins_age, alpha=0.3, color="#C44E52",
            label=f"Low pre (n={len(lo_pre)})", histtype="stepfilled")
    ax.hist(hi_post, bins=bins_age, histtype="step", color="#4C72B0",
            linewidth=2, label=f"High post (n={len(hi_post)})")
    ax.hist(lo_post, bins=bins_age, histtype="step", color="#C44E52",
            linewidth=2, label=f"Low post (n={len(lo_post)})")
    ax.set_xlabel("Age"); ax.set_ylabel("# subjects")
    ax.set_title("(a) Age distribution: pre (fill) vs post (line) matching")
    ax.legend(fontsize=8)

    ax = axes[1]
    matched_ids = set(matched_df["ID"])
    unmatched = cohort[~cohort["ID"].isin(matched_ids)]
    ax.scatter(unmatched["Age"], unmatched[metric], color="lightgray",
               s=18, alpha=0.5, label=f"unmatched (n={len(unmatched)})")
    hi_mm = matched_df[matched_df[group_col] == "high"]
    lo_mm = matched_df[matched_df[group_col] == "low"]
    ax.scatter(hi_mm["Age"], hi_mm[metric], color="#4C72B0", s=22,
               alpha=0.7, label=f"High matched (n={len(hi_mm)})")
    ax.scatter(lo_mm["Age"], lo_mm[metric], color="#C44E52", s=22,
               alpha=0.7, label=f"Low matched (n={len(lo_mm)})")
    ax.axhline(median_val, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Age"); ax.set_ylabel(metric)
    ax.set_title(f"(b) Age × {metric} (matched highlighted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_emotion_heatmap(summary_df, out_path, metric):
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
    vmax = (np.nanmax(np.abs(diff))
            if np.isfinite(np.nanmax(np.abs(diff))) else 1.0)
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
    ax.set_title(f"Δ mean (High {metric} − Low {metric}) of emotion probabilities\n"
                 "* q<0.05, ** q<0.01, *** q<0.001")
    fig.colorbar(im, ax=ax, label="Δ mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# AD vs {HC, NAD, ACS} comparison
# ============================================================

def _hc_groups_per_group_stats(cohort, col):
    a = cohort[cohort["label"] == 1][col].dropna().astype(float)
    b = cohort[cohort["label"] == 0][col].dropna().astype(float)
    out = {
        "ad_n": len(a), "ad_mean": float(a.mean()) if len(a) else np.nan,
        "ad_std": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
        "hc_n": len(b), "hc_mean": float(b.mean()) if len(b) else np.nan,
        "hc_std": float(b.std(ddof=1)) if len(b) > 1 else np.nan,
    }
    if len(a) > 1 and len(b) > 1:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        out["welch_t"] = float(t); out["welch_p"] = float(p)
    else:
        out["welch_t"] = np.nan; out["welch_p"] = np.nan
    out["cohen_d"] = cohens_d(a, b)
    return out


def _hc_groups_fmt_stats(label, s, fmt=".2f"):
    return (f"{label}: AD {s['ad_mean']:{fmt}} ± {s['ad_std']:{fmt}} vs "
            f"{s['hc_mean']:{fmt}} ± {s['hc_std']:{fmt}}  "
            f"(Welch t={s['welch_t']:.2f}, p={s['welch_p']:.3g}, "
            f"d={s['cohen_d']:.3f})")


def _hc_groups_plot_violin(cohort, comparison_name, out_path, caliper):
    hc = cohort[cohort["label"] == 0]["age_error"].dropna()
    ad = cohort[cohort["label"] == 1]["age_error"].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([hc, ad], showmedians=True)
    for pc, color in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{comparison_name}\n(n={len(hc)})",
                        f"AD\n(n={len(ad)})"])
    ax.set_ylabel("Age error (real − predicted)")
    ax.set_title(f"Age prediction error: {comparison_name} vs AD\n"
                 f"(age-matched 1:1, caliper={caliper}y)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_hc_groups_comparison(comparison, cohort_dir, cohort_mode,
                             hc_source_mode="ACS", caliper=2.0):
    """ad_vs_{hc, nad, acs} 1:1 age-matched analysis.

    comparison ∈ {'HC', 'NAD', 'ACS'} (uppercase). Produces:
      overview/<cohort>/cross_matched/ad_vs_<comparison>/
        {matched_features.csv, matched_pairs.csv, matching_report.txt,
         summary_stats.csv}
      age/analysis/pred_error_stat/<cohort>/ad_vs_<comparison>/matched_violin.png
    """
    logger.info(f"=== AD vs {comparison} ===")
    cohort, pairs = build_cohort_ad_vs_HCgroup(
        comparison, design="cross_matched",
        cohort_mode=cohort_mode, hc_source_mode=hc_source_mode,
        caliper=caliper,
    )
    n_pairs_raw = len(pairs) if pairs is not None else len(cohort) // 2

    from src.cohort import filter_pairs_by_predicted_age
    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    cohort, pairs = filter_pairs_by_predicted_age(cohort, pairs, pred_ages)
    n_pairs = len(pairs) if pairs is not None else len(cohort) // 2
    logger.info(f"Matched pairs: {n_pairs}/{n_pairs_raw} after predicted_age filter "
                f"(caliper={caliper}y)")

    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["ID"].map(pred_ages)
    cohort["age_error"] = cohort["Age"] - cohort["predicted_age"]

    partition = f"ad_vs_{comparison.lower()}"
    artifacts_dir = OVERVIEW_DIR / cohort_dir / "cross_matched" / partition
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    age_stat_dir = AGE_PRED_ERROR_STAT_DIR / cohort_dir / "1by1match" / partition
    age_stat_dir.mkdir(parents=True, exist_ok=True)

    cohort.to_csv(artifacts_dir / "matched_features.csv", index=False)
    if pairs is not None:
        pairs.to_csv(artifacts_dir / "matched_pairs.csv", index=False)

    age_s = _hc_groups_per_group_stats(cohort, "Age")
    mmse_s = _hc_groups_per_group_stats(cohort, "MMSE")
    err_s = _hc_groups_per_group_stats(cohort, "age_error")

    summary = pd.DataFrame([
        {"variable": "Age", **age_s},
        {"variable": "MMSE", **mmse_s},
        {"variable": "age_error", **err_s},
    ])
    summary.to_csv(artifacts_dir / "summary_stats.csv", index=False)

    with open(artifacts_dir / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write(f"AD vs {comparison} 1:1 age-matched (caliper={caliper}y)\n")
        f.write(f"Matched pairs: {n_pairs}\n\n")
        f.write(_hc_groups_fmt_stats("Age   ", age_s) + "\n")
        f.write(_hc_groups_fmt_stats("MMSE  ", mmse_s) + "\n")
        f.write(_hc_groups_fmt_stats("AgeErr", err_s) + "\n")

    _hc_groups_plot_violin(cohort, comparison,
                           age_stat_dir / "matched_violin.png", caliper)
    logger.info(f"Saved {artifacts_dir}/ + {age_stat_dir}/matched_violin.png")
    logger.info("  " + _hc_groups_fmt_stats("Age   ", age_s))
    logger.info("  " + _hc_groups_fmt_stats("MMSE  ", mmse_s))
    logger.info("  " + _hc_groups_fmt_stats("AgeErr", err_s))
    return summary


# ============================================================
# AUC supplement
# ============================================================

def _auc_for_modality(name, X_df, matched, model_cls="xgb"):
    """Eval one modality DataFrame against the matched cohort label."""
    merged = matched[["ID", "base_id", "label"]].merge(
        X_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns
                 if c not in ("ID", "base_id", "label", "subject_id")]
    merged = merged.dropna(subset=feat_cols)
    if len(merged) < 50:
        return {"modality": name, "n": len(merged), "auc": np.nan,
                "balacc": np.nan, "mcc": np.nan, "n_features": len(feat_cols),
                "auc_ci_low": np.nan, "auc_ci_high": np.nan}
    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    g = merged["base_id"].to_numpy()
    m = cv_eval(X, y, g, model_cls=model_cls, n_folds=5, seed=42, return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=42)
    return {"modality": name, "n": len(merged),
            "n_high": int((y == 1).sum()), "n_low": int((y == 0).sum()),
            "n_features": len(feat_cols), **m,
            "auc_ci_low": ci_low, "auc_ci_high": ci_high}


def _run_auc_supplement(matched, comparison_dir, group_col):
    """Per-modality AUC on the matched hi-lo cohort."""
    matched = matched.copy()
    matched["label"] = (matched[group_col] == "high").astype(int)
    if "base_id" not in matched.columns:
        matched["base_id"] = matched["ID"].str.extract(r"^([A-Za-z]+\d+)")
    ids = matched["ID"].tolist()
    logger.info(f"AUC supplement on matched cohort: {len(matched)} "
                f"(high={int(matched['label'].sum())}, "
                f"low={int((1 - matched['label']).sum())})")

    results = []

    # Age-only baseline
    X_age = matched[["Age"]].to_numpy(dtype=float)
    y = matched["label"].to_numpy(dtype=int)
    g = matched["base_id"].to_numpy()
    m = cv_eval(X_age, y, g, model_cls="logistic", n_folds=5, seed=42,
                return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=42)
    results.append({"modality": "age_only", "n": len(matched),
                    "n_high": int(y.sum()), "n_low": int((1 - y).sum()),
                    "n_features": 1, **m,
                    "auc_ci_low": ci_low, "auc_ci_high": ci_high})

    # age_error
    if "age_error" in matched.columns:
        sub = matched.dropna(subset=["age_error"])
        if len(sub) >= 50:
            Xe = sub[["age_error"]].to_numpy(dtype=float)
            ye = sub["label"].to_numpy(dtype=int)
            ge = sub["base_id"].to_numpy()
            m = cv_eval(Xe, ye, ge, model_cls="logistic", n_folds=5, seed=42,
                        return_preds=True)
            y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
            ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=42)
            results.append({"modality": "age_error", "n": len(sub),
                            "n_high": int(ye.sum()), "n_low": int((1 - ye).sum()),
                            "n_features": 1, **m,
                            "auc_ci_low": ci_low, "auc_ci_high": ci_high})

    emo_cols = [c for c in matched.columns if "__" in c and
                not c.startswith(("landmark__", "embasym__"))]
    emo_cols = [c for c in emo_cols if c != "mmse_group"]
    if emo_cols:
        emo_df = matched[["ID"] + emo_cols].rename(columns={"ID": "subject_id"})
        results.append(_auc_for_modality("emotion_8methods", emo_df, matched))

    lmk_cols = [c for c in matched.columns if c.startswith("landmark__")]
    if lmk_cols:
        lmk_df = matched[["ID"] + lmk_cols].rename(columns={"ID": "subject_id"})
        results.append(_auc_for_modality("landmark_asymmetry", lmk_df, matched))

    emb_asym_cols = [c for c in matched.columns if c.startswith("embasym__")]
    if emb_asym_cols:
        emb_asym_df = matched[["ID"] + emb_asym_cols].rename(columns={"ID": "subject_id"})
        results.append(_auc_for_modality("embedding_asymmetry", emb_asym_df, matched))

    for model in EMBEDDING_MODELS:
        emb = load_embedding_mean(ids, model)
        if emb is not None:
            results.append(_auc_for_modality(f"embedding_{model}_mean", emb, matched))

    df = pd.DataFrame(results)
    out_csv = comparison_dir / "summary_per_modality_auc.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")
    for r in results:
        logger.info(f"  {r['modality']:<25} AUC={r.get('auc', float('nan')):.3f}  "
                    f"BalAcc={r.get('balacc', float('nan')):.3f}  "
                    f"n={r['n']}  #feats={r['n_features']}")
    return df


# ============================================================
# Hi-lo split full multi-modality analysis
# ============================================================

def _select_ad_for_hilo(demo, metric, cohort_mode):
    """Pick AD subjects for hi-lo splits per cohort_mode (V2.2 spec-aware).

    P-side CDR filter is read from ``CohortSpec.p_cdr`` (cdr05 default;
    cdrall opt-out for sensitivity). P-visit selection from ``CohortSpec.p_visit``.
    """
    spec = cohort_spec_from_name(cohort_mode)
    demo = demo.copy()
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    if spec.p_visit == "all":
        elig = demo[demo[metric].notna() & demo["Age"].notna()].copy()
        elig = apply_p_cdr_filter(elig, spec)
        elig = elig.sort_values(["base_id", "visit"])
        return _keep_visits_with_features(elig)
    if spec.hc_visit == "all":
        elig = demo[demo[metric].notna() & demo["Age"].notna()].copy()
        elig = apply_p_cdr_filter(elig, spec)
        elig = elig.sort_values(["base_id", "visit"])
        return _pick_first_visit_with_features(elig)
    # p_visit=first + hc_visit=first: first visit per base_id + CDR filter
    cohort = demo.dropna(subset=[metric, "Age"]).copy()
    cohort = cohort.sort_values(["base_id", "visit"]).groupby(
        "base_id", as_index=False).first()
    return apply_p_cdr_filter(cohort, spec).copy()


def _run_hilo(args, metric):
    """Hi-lo (MMSE / CASI median split) full multi-modality analysis."""
    metric_low = metric.lower()
    group_col = f"{metric_low}_group"
    comparison_name = f"{metric_low}_high_vs_low"

    cohort_dir = cohort_path(args.cohort_mode)
    comparison_dir = OVERVIEW_DIR / cohort_dir / "cross_matched" / comparison_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    age_stat_dir = AGE_PRED_ERROR_STAT_DIR / cohort_dir / "1by1match" / comparison_name
    age_stat_dir.mkdir(parents=True, exist_ok=True)
    emo_stat_dir = EMO_AU_FEATURE_STAT_DIR / cohort_dir / comparison_name
    emo_stat_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cohort
    demo = load_p_demographics()
    spec = cohort_spec_from_name(args.cohort_mode)
    if spec.hc_visit == "all" or spec.p_visit == "all":
        cohort = _select_ad_for_hilo(demo, metric, args.cohort_mode)
        logger.info(f"P patients ({args.cohort_mode}, CDR≥0.5 + features): "
                    f"n={len(cohort)}")
    else:
        cohort = select_visit(demo, args.visit_selection, metric=metric)
        logger.info(f"P patients with {metric}+Age "
                    f"({args.visit_selection} visit): n={len(cohort)}")

    # 2. Filter to those with predicted age
    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    cohort = cohort[cohort["ID"].isin(pred_ages)].copy()
    logger.info(f"After requiring predicted_age: n={len(cohort)}")

    # 3. Median split
    cohort, median_val = split_by_metric_median(
        cohort, args.median_tiebreak, metric=metric, group_col=group_col)
    n_high = (cohort[group_col] == "high").sum()
    n_low = (cohort[group_col] == "low").sum()
    logger.info(f"{metric} median={median_val:.1f}; high n={n_high}, low n={n_low}")

    cohort.to_csv(comparison_dir / "cohort.csv", index=False)

    # 4. 1:1 age matching
    matched, pairs_df, (minor_label, major_label) = match_1to1(
        cohort, caliper=args.caliper, seed=args.seed,
        metric=metric, group_col=group_col,
    )
    pairs_df.to_csv(comparison_dir / "matched_pairs.csv", index=False)
    n_pairs = len(pairs_df)
    logger.info(f"Matched pairs: {n_pairs} (caliper={args.caliper}, minor={minor_label})")

    # 5. Load + merge per-modality features
    age_err = load_age_error(matched["ID"].tolist(), demo)
    emo_dfs = load_emotion_features()
    lmk = load_landmark_matrix(matched["ID"].tolist())
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
    merged = merged.drop(columns=[c for c in merged.columns
                                  if c.startswith("subject_id")])
    merged.to_csv(comparison_dir / "matched_features.csv", index=False)

    # 6. Matching report
    with open(comparison_dir / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== AD {metric} age-balanced cohort ===\n")
        f.write(f"Visit selection: {args.visit_selection}\n")
        f.write(f"{metric} median: {median_val:.2f} (tiebreak={args.median_tiebreak})\n")
        f.write(f"Pre-matching: high n={n_high}, low n={n_low}\n")
        f.write(f"Post-matching: {n_pairs} pairs ({2*n_pairs} subjects)\n")
        f.write(f"Caliper (years): {args.caliper}\n\n")
        high_m = matched[matched[group_col] == "high"]
        low_m = matched[matched[group_col] == "low"]
        f.write(f"Post-match age: high {high_m['Age'].mean():.2f}±{high_m['Age'].std():.2f}, "
                f"low {low_m['Age'].mean():.2f}±{low_m['Age'].std():.2f}\n")
        if len(high_m) > 1 and len(low_m) > 1:
            t, p = stats.ttest_ind(high_m["Age"], low_m["Age"], equal_var=False)
            f.write(f"Age t-test p={p:.4f} (should be >>0.05 if matching worked)\n")
        f.write(f"Post-match {metric}: high {high_m[metric].mean():.2f}±{high_m[metric].std():.2f}, "
                f"low {low_m[metric].mean():.2f}±{low_m[metric].std():.2f}\n")

    # 7. Per-feature stats
    feature_cols = ["age_error", "abs_age_error"]
    for method in EMOTION_METHODS:
        for emo in EMOTIONS:
            for s in EMOTION_STATS:
                col = f"{method}__{emo}_{s}"
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

    high_by_pair = merged[merged[group_col] == "high"].set_index("pair_id")
    low_by_pair = merged[merged[group_col] == "low"].set_index("pair_id")
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

    # 8. FDR correction
    for pcol, qcol in [("pvalue", "qvalue"), ("paired_t_p", "paired_t_q")]:
        valid = summary[pcol].notna()
        summary[qcol] = np.nan
        if valid.sum() > 0:
            summary.loc[valid, qcol] = bh_fdr(summary.loc[valid, pcol].values)
    summary.to_csv(comparison_dir / "summary_stats.csv", index=False)

    # 9. Continuous-metric correlations (robustness)
    cont_rows = []
    for col in feature_cols:
        if col not in merged.columns:
            continue
        sub = merged[[col, metric]].dropna()
        if len(sub) < 5:
            continue
        r_p, p_p = stats.pearsonr(sub[metric], sub[col])
        r_s, p_s = stats.spearmanr(sub[metric], sub[col])
        cont_rows.append({
            "feature": col, "n": len(sub),
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
        })
    pd.DataFrame(cont_rows).to_csv(
        comparison_dir / f"{metric_low}_continuous_corr.csv", index=False)

    # 10. Plots
    plot_metric_figure(cohort, matched, median_val,
                       comparison_dir / f"fig_{metric_low}_overview.png",
                       metric=metric, group_col=group_col)
    plot_age_figure(cohort, matched, median_val,
                    comparison_dir / "fig_age_overview.png",
                    metric=metric, group_col=group_col)
    plot_age_error_violin(merged, age_stat_dir / "matched_violin.png",
                          metric=metric, group_col=group_col)
    plot_emotion_heatmap(summary, emo_stat_dir / "fig_emotion_grid_heatmap.png",
                         metric=metric)

    # 11. AUC supplement
    _run_auc_supplement(merged, comparison_dir, group_col)

    logger.info(f"Done. Artifacts at {comparison_dir}; "
                f"per-modality fanout to age/emo_au feature_stat trees.")
    logger.info(f"  Pairs: {n_pairs}, features tested: {len(summary)}")
    sig = summary[summary["qvalue"] < 0.05]
    logger.info(f"  Significant (q<0.05): {len(sig)} features")
    if len(sig) > 0:
        logger.info(f"  Top 5: {sig.nsmallest(5, 'qvalue')['feature'].tolist()}")


# ============================================================
# Main
# ============================================================

COMPARISON_CHOICES = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", required=True, choices=COMPARISON_CHOICES,
                        help="ad_vs_{hc,nad,acs}: AD vs HC-group matching artifacts; "
                             "{mmse,casi}_hilo: AD within-cohort hi-lo full analysis "
                             "+ AUC supplement.")
    parser.add_argument("--cohort-mode",
                        choices=VALID_COHORT_CHOICES,
                        default="p_first_cdr05_hc_first_cdrall_or_mmseall")
    parser.add_argument("--hc-source-mode", choices=["ACS", "ACS_ext", "EACS"],
                        default="ACS",
                        help="Only affects ad_vs_{hc,nad,acs} comparisons.")
    parser.add_argument("--caliper", type=float, default=2.0)
    parser.add_argument("--visit-selection", choices=["first", "latest"], default="first",
                        help="hi-lo only.")
    parser.add_argument("--median-tiebreak", choices=["high", "low"], default="high",
                        help="hi-lo only: where metric==median subjects go.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stat", choices=["ttest", "mannwhitney", "auto"], default="auto")
    args = parser.parse_args()

    cohort_dir = cohort_path(args.cohort_mode)
    if args.comparison.startswith("ad_vs_"):
        comparison_label = args.comparison.replace("ad_vs_", "").upper()
        run_hc_groups_comparison(comparison_label, cohort_dir,
                                 args.cohort_mode,
                                 hc_source_mode=args.hc_source_mode,
                                 caliper=args.caliper)
    else:
        metric = "MMSE" if args.comparison == "mmse_hilo" else "CASI"
        _run_hilo(args, metric)


if __name__ == "__main__":
    main()
