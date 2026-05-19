"""
Cross-sectional naive AD vs HC classification (was run_arm_a_ad_vs_hc.py).

No age balance — intentionally so, to demonstrate the age-confound baseline.
Each feature modality gets 5-fold GroupKFold XGBoost (AUC/BalAcc/MCC/F1/Sens/Spec
+ bootstrap AUC 95% CI); age-only logistic baseline is also reported so that
per-modality contribution can be measured as (modality AUC − age-only AUC).

Also reports per-feature Cohen's d / Hedges' g (AD mean − HC mean), and
per-modality ROC + PR curves (PNG).

Cohort comes from scripts.utilities.cohort.build_cohort_ad_vs_HCgroup with
design='cross_naive'. cohort_mode controls AD/HC visit-selection behavior;
hc_source_mode controls ACS-group composition (internal vs external).

Outputs (per cohort_mode + ad_vs_hc partition):
  overview/<cohort>/cross_naive/cohort.csv
  overview/<cohort>/cross_naive/ad_vs_hc/summary_per_modality.csv
  age/analysis/pred_error_stat/<cohort>/ad_vs_hc/<modality>.png
  emo_au/analysis/feature_stat/<cohort>/ad_vs_hc/{<modality>.png, per_feature_*.csv}
  asymmetry/analysis/feature_stat/<cohort>/ad_vs_hc/...
  embedding/analysis/feature_stat/{original,difference}/<cohort>/ad_vs_hc/...

Usage:
    conda run -n Alz_face_main_analysis python scripts/overview/run_cross_naive.py \\
        --cohort-mode default
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
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    AGE_PRED_ERROR_STAT_DIR, ASYMMETRY_FEATURE_STAT_DIR,
    EMBEDDING_FEATURE_STAT_DIR, EMO_AU_FEATURE_STAT_DIR,
    OVERVIEW_DIR, VALID_COHORT_CHOICES, cohort_name,
)
from scripts.utilities.cohort import build_cohort_ad_vs_HCgroup
from scripts.utilities.feature_loaders import (
    EMBEDDING_MODELS,
    load_age_error_pairs,
    load_embedding_asymmetry,
    load_embedding_mean,
    load_emotion_matrix,
    load_landmark_matrix,
)
from scripts.utilities.stats_helpers import (
    bootstrap_auc_ci, cohens_d, cv_eval, hedges_g,
)

# 5 modality directions for per-direction subfolder layout
MODALITY_DIRECTION = {
    "age_only": "age",
    "age_error": "age",
    "embedding_arcface_mean": "embedding_mean",
    "embedding_dlib_mean": "embedding_mean",
    "embedding_topofr_mean": "embedding_mean",
    "embedding_asymmetry": "embedding_asymmetry",
    "landmark_asymmetry": "landmark_asymmetry",
    "emotion_8methods": "emotion",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Per-modality CV + per-feature effect sizes
# ============================================================

def _plot_roc_pr(y_true, y_prob, modality, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#4C72B0", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"{modality}\nROC AUC = {auc_val:.3f}")
    axes[1].plot(rec, prec, color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{modality}\nPR curve")
    fig.tight_layout()
    fig.savefig(out_dir / f"{modality}.png", dpi=120)
    plt.close(fig)


def run_modality(name, X_df, cohort, model_cls="xgb", n_folds=5, seed=42,
                 compute_ci=True, save_curve_dir=None):
    merged = cohort[["ID", "base_id", "label"]].merge(
        X_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns if c not in
                 ["ID", "base_id", "label", "subject_id"]]
    merged = merged.dropna(subset=feat_cols)
    if len(merged) < 50:
        return {"modality": name, "n": len(merged), "auc": np.nan,
                "n_features": len(feat_cols)}
    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    g = merged["base_id"].to_numpy()
    m = cv_eval(X, y, g, model_cls=model_cls, n_folds=n_folds, seed=seed,
                return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci_low, ci_high = (bootstrap_auc_ci(y_true, y_prob, seed=seed)
                       if compute_ci else (np.nan, np.nan))
    if save_curve_dir is not None:
        save_curve_dir.mkdir(parents=True, exist_ok=True)
        _plot_roc_pr(y_true, y_prob, name, save_curve_dir)
    return {
        "modality": name, "n": len(merged),
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "n_features": len(feat_cols),
        **m,
        "auc_ci_low": ci_low, "auc_ci_high": ci_high,
    }


def per_feature_effect_sizes(modality, feat_df, cohort, out_dir, top_n=10):
    """Per-feature AD vs HC Cohen's d & Welch p; save CSV."""
    merged = cohort[["ID", "label"]].merge(
        feat_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns if c not in
                 ["ID", "label", "subject_id"]]
    rows = []
    for col in feat_cols:
        ad = merged.loc[merged["label"] == 1, col].dropna().values
        hc = merged.loc[merged["label"] == 0, col].dropna().values
        if len(ad) < 5 or len(hc) < 5:
            continue
        d = cohens_d(ad, hc); g = hedges_g(ad, hc)
        t, p = stats.ttest_ind(ad, hc, equal_var=False)
        rows.append({
            "modality": modality, "feature": col,
            "n_ad": len(ad), "n_hc": len(hc),
            "mean_ad": float(ad.mean()), "mean_hc": float(hc.mean()),
            "cohen_d": d, "hedges_g": g,
            "welch_t": float(t), "welch_p": float(p),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"per_feature_{modality}.csv", index=False)
    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ci", action="store_true",
                        help="skip bootstrap AUC CI (for quick smoke test)")
    parser.add_argument("--cohort-mode",
                        choices=VALID_COHORT_CHOICES,
                        default="default")
    parser.add_argument("--hc-source-mode",
                        choices=["ACS", "ACS_ext", "EACS"], default="ACS")
    args = parser.parse_args()

    cohort_dir = cohort_name(args.cohort_mode)
    partition = "ad_vs_hc"
    design_dir = OVERVIEW_DIR / cohort_dir / "cross_naive"
    partition_dir = design_dir / partition
    partition_dir.mkdir(parents=True, exist_ok=True)

    def dir_for(modality):
        direction = MODALITY_DIRECTION[modality]
        if direction == "age":
            d = AGE_PRED_ERROR_STAT_DIR / cohort_dir / partition
        elif direction == "emotion":
            d = EMO_AU_FEATURE_STAT_DIR / cohort_dir / partition
        elif direction == "landmark_asymmetry":
            d = ASYMMETRY_FEATURE_STAT_DIR / cohort_dir / partition
        elif direction == "embedding_mean":
            d = EMBEDDING_FEATURE_STAT_DIR / "original" / cohort_dir / partition
        elif direction == "embedding_asymmetry":
            d = EMBEDDING_FEATURE_STAT_DIR / "difference" / cohort_dir / partition
        else:
            raise ValueError(f"unknown modality direction {direction!r}")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Cohort: cross_naive design (no matching), HC group = NAD ∪ ACS
    cohort, _ = build_cohort_ad_vs_HCgroup(
        "HC", design="cross_naive",
        cohort_mode=args.cohort_mode,
        hc_source_mode=args.hc_source_mode,
    )
    cohort["base_id"] = cohort["ID"].astype(str).str.extract(r"^(.+)-\d+$")
    n_ad = int((cohort["label"] == 1).sum())
    n_hc = int((cohort["label"] == 0).sum())
    logger.info(f"Cohort ({args.cohort_mode}): AD={n_ad}, HC={n_hc}")
    cohort.to_csv(design_dir / "cohort.csv", index=False)

    age_mean_diff = (cohort[cohort["label"] == 1]["Age"].mean() -
                     cohort[cohort["label"] == 0]["Age"].mean())
    logger.info(f"Age gap AD − HC: {age_mean_diff:+.2f} years")

    ids = cohort["ID"].tolist()
    results = []
    compute_ci = not args.skip_ci

    # Age-only baseline
    X_age = cohort[["Age"]].to_numpy(dtype=float)
    y_all = cohort["label"].to_numpy(dtype=int)
    g_all = cohort["base_id"].to_numpy()
    m = cv_eval(X_age, y_all, g_all, model_cls="logistic",
                n_folds=args.n_folds, seed=args.seed, return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci = (bootstrap_auc_ci(y_true, y_prob, seed=args.seed)
          if compute_ci else (np.nan, np.nan))
    results.append({"modality": "age_only", "n": len(cohort),
                    "n_pos": n_ad, "n_neg": n_hc, "n_features": 1,
                    **m, "auc_ci_low": ci[0], "auc_ci_high": ci[1]})
    logger.info(f"age_only AUC={m['auc']:.3f}  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    # age_error
    age_pairs = load_age_error_pairs(ids)
    age_err_df = cohort[["ID"]].merge(age_pairs, on="ID", how="left")
    age_err_df["age_error"] = cohort["Age"].to_numpy() - age_err_df["predicted_age"]
    age_err_df = age_err_df.dropna(subset=["age_error"]).rename(columns={"ID": "subject_id"})
    results.append(run_modality(
        "age_error", age_err_df[["subject_id", "age_error"]], cohort,
        model_cls="logistic", n_folds=args.n_folds, seed=args.seed,
        compute_ci=compute_ci, save_curve_dir=dir_for("age_error")))

    # Emotion
    emo = load_emotion_matrix(ids)
    if emo is not None:
        results.append(run_modality(
            "emotion_8methods", emo, cohort,
            n_folds=args.n_folds, seed=args.seed,
            compute_ci=compute_ci,
            save_curve_dir=dir_for("emotion_8methods")))
        per_feature_effect_sizes("emotion_8methods", emo, cohort,
                                 dir_for("emotion_8methods"))

    # Landmark
    lmk = load_landmark_matrix(ids)
    results.append(run_modality(
        "landmark_asymmetry", lmk, cohort,
        n_folds=args.n_folds, seed=args.seed, compute_ci=compute_ci,
        save_curve_dir=dir_for("landmark_asymmetry")))
    per_feature_effect_sizes("landmark_asymmetry", lmk, cohort,
                             dir_for("landmark_asymmetry"))

    # Embedding asymmetry
    emb_asym = load_embedding_asymmetry(ids)
    results.append(run_modality(
        "embedding_asymmetry", emb_asym, cohort,
        n_folds=args.n_folds, seed=args.seed, compute_ci=compute_ci,
        save_curve_dir=dir_for("embedding_asymmetry")))
    per_feature_effect_sizes("embedding_asymmetry", emb_asym, cohort,
                             dir_for("embedding_asymmetry"))

    # Embedding mean per model
    for model in EMBEDDING_MODELS:
        emb = load_embedding_mean(ids, model)
        if emb is not None:
            modality = f"embedding_{model}_mean"
            results.append(run_modality(
                modality, emb, cohort,
                n_folds=args.n_folds, seed=args.seed, compute_ci=compute_ci,
                save_curve_dir=dir_for(modality)))

    df = pd.DataFrame(results)
    df["age_gap_years"] = age_mean_diff
    age_auc = df.loc[df["modality"] == "age_only", "auc"].iloc[0]
    df["delta_vs_age_only"] = df["auc"] - age_auc
    df.to_csv(partition_dir / "summary_per_modality.csv", index=False)

    logger.info(f"Done. overview design dir={design_dir}; per-modality fanout under "
                f"age/emo_au/asymmetry/embedding feature_stat trees.")
    for r in results:
        logger.info(
            f"  {r['modality']:<25} AUC={r.get('auc', np.nan):.3f} "
            f"[{r.get('auc_ci_low', np.nan):.3f}, {r.get('auc_ci_high', np.nan):.3f}]  "
            f"BalAcc={r.get('balacc', np.nan):.3f}  MCC={r.get('mcc', np.nan):.3f}  "
            f"F1={r.get('f1', np.nan):.3f}  n={r['n']}"
        )


if __name__ == "__main__":
    main()
