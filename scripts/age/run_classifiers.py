"""
Per-design × per-comparison binary classifier on 2-3 simple age/MMSE features.

For each design ∈ {cross_naive, cross_matched, longitudinal_naive,
longitudinal_matched} × comparison ∈ {HC, NAD, ACS}:
  Feature sets:
    set1 = [Age, MMSE]                   (2 features)
    set2 = [Age, MMSE, age_error]        (3 features)
  Models (run order: XGBoost first, then TabPFN):
    xgb     XGBClassifier(n_est=200, depth=4, lr=0.1)
    tabpfn  TabPFNClassifier(ignore_pretraining_limits=True)

Cohorts come from scripts.utilities.cohort.build_cohort_ad_vs_HCgroup. For
longitudinal designs the "Age"/"MMSE" features map to first_age/first_MMSE,
and age_error = first_age − baseline_predicted_age.

Outputs:
  cross_naive / cross_matched     → age/analysis/classification/...
  longitudinal_*                  → longitudinal/analysis/age/classification/...

  <root>/<cohort>/ad_vs_<cmp>/<bucket>/<set>/<model>_oof.csv  per-subject OOF
  <root>/<cohort>/ad_vs_<cmp>/summary_<bucket>.csv            AUC/CI/BalAcc/MCC
  overview/<cohort>/classifier_summary_all.csv                grand summary

Usage:
    conda run -n Alz_face_main_analysis python scripts/age/run_classifiers.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    AGE_CLASSIFICATION_DIR, LONGI_AGE_CLASSIFICATION_DIR,
    OVERVIEW_DIR, cohort_name,
)
from scripts.utilities.cohort import build_cohort_ad_vs_HCgroup, VALID_DESIGNS
from scripts.utilities.stats_helpers import bootstrap_auc_ci

AGES_FILE = (PROJECT_ROOT / "workspace" / "age" / "predictions" /
             "p_first_hc_strict" / "predicted_ages.json")
COMPARISONS = ["HC", "NAD", "ACS"]
ALL_DESIGNS = list(VALID_DESIGNS)
N_FOLDS = 10
SEED = 42

# design → (classification root, bucket subdir name)
DESIGN_TO_ROOT_BUCKET = {
    "cross_naive":          (AGE_CLASSIFICATION_DIR, "full"),
    "cross_matched":        (AGE_CLASSIFICATION_DIR, "matched"),
    "longitudinal_naive":   (LONGI_AGE_CLASSIFICATION_DIR, "full"),
    "longitudinal_matched": (LONGI_AGE_CLASSIFICATION_DIR, "matched"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Cohort → feature DataFrame
# ============================================================

def _attach_age_error_xsec(cohort, pred_ages):
    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["ID"].map(pred_ages)
    cohort["age_error"] = cohort["Age"] - cohort["predicted_age"]
    return cohort


def _attach_age_error_long(cohort, pred_ages):
    """Longitudinal: baseline age_error = first_age − pred_age(first_visit_id)."""
    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["first_visit_id"].map(pred_ages)
    cohort["age_error"] = cohort["first_age"] - cohort["predicted_age"]
    cohort["Age"] = cohort["first_age"]
    cohort["MMSE"] = cohort["first_MMSE"]
    if "ID" not in cohort.columns:
        cohort["ID"] = cohort["base_id"]
    return cohort


def build_feature_df(design, hc_source, pred_ages, cohort_mode, hc_source_mode):
    """Return DataFrame with [ID, base_id, Age, MMSE, age_error, label].

    Pre-filter: drop subjects missing ANY of {Age, MMSE, age_error} so that
    2-feat and 3-feat see the same N. For matched designs, drop the entire
    pair if either side fails the filter (preserves pairing).
    """
    cohort, pairs = build_cohort_ad_vs_HCgroup(
        hc_source, design=design,
        cohort_mode=cohort_mode, hc_source_mode=hc_source_mode,
    )
    if cohort is None or len(cohort) == 0:
        return None
    if design in ("cross_naive", "cross_matched"):
        cohort = _attach_age_error_xsec(cohort, pred_ages)
    else:
        cohort = _attach_age_error_long(cohort, pred_ages)
    if "base_id" not in cohort.columns:
        cohort["base_id"] = cohort["ID"].astype(str).str.extract(
            r"^([A-Za-z_]+\d+)")

    has_all = (cohort["Age"].notna() &
               cohort["MMSE"].notna() &
               cohort["age_error"].notna())
    n_raw = len(cohort)

    if design in ("cross_matched", "longitudinal_matched") and pairs is not None \
            and "pair_id" in cohort.columns:
        passing_ids = set(cohort.loc[has_all, "ID"])
        keep = (pairs["minor_id"].isin(passing_ids) &
                pairs["major_id"].isin(passing_ids))
        kept_ids = set(pairs.loc[keep, "minor_id"]).union(
            pairs.loc[keep, "major_id"])
        cohort = cohort[cohort["ID"].isin(kept_ids)].copy()
        n_pairs_kept = int(keep.sum()); n_pairs_raw = len(pairs)
        logger.info(f"  pair filter: kept {n_pairs_kept}/{n_pairs_raw} pairs "
                    f"(={2 * n_pairs_kept} subjects from raw {n_raw})")
    else:
        cohort = cohort[has_all].copy()
        logger.info(f"  subject filter: kept {len(cohort)}/{n_raw} subjects "
                    f"(needed all of Age/MMSE/age_error)")

    keep_cols = ["ID", "base_id", "Age", "MMSE", "age_error", "label"]
    return cohort[[c for c in keep_cols if c in cohort.columns]].copy()


# ============================================================
# CV evaluation (custom — different from stats_helpers.cv_eval which only
# does 5-fold pooled metrics; this returns per-subject OOF probs)
# ============================================================

def get_classifier(name, seed):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=seed, n_jobs=2, eval_metric="logloss",
            use_label_encoder=False,
        )
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=seed,
                                ignore_pretraining_limits=True)
    raise ValueError(f"unknown classifier: {name}")


def _cv_eval(feat_df, feat_cols, model_name, n_folds=N_FOLDS, seed=SEED):
    """GroupKFold CV; returns (per-subject OOF DataFrame, summary dict)."""
    df = feat_df.dropna(subset=feat_cols + ["label"]).copy()
    if df.empty:
        return None, {"model": model_name, "n_features": len(feat_cols),
                      "n": 0, "auc": float("nan"), "auc_ci_low": float("nan"),
                      "auc_ci_high": float("nan"), "balacc": float("nan"),
                      "mcc": float("nan"),
                      "skip_reason": "all-NaN features"}
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["base_id"].to_numpy()

    n_groups = pd.Series(groups).nunique()
    n_folds_eff = min(n_folds, n_groups)
    if n_folds_eff < 2 or len(np.unique(y)) < 2:
        return None, {"model": model_name, "n_features": len(feat_cols),
                      "n": len(df), "auc": float("nan"),
                      "auc_ci_low": float("nan"), "auc_ci_high": float("nan"),
                      "balacc": float("nan"), "mcc": float("nan"),
                      "skip_reason": f"n_groups={n_groups} insufficient"}

    gkf = GroupKFold(n_splits=n_folds_eff)
    y_prob = np.full(len(df), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = get_classifier(model_name, seed=seed + fold)
        clf.fit(X[tr], y[tr])
        y_prob[te] = clf.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(y_prob)
    if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
        return None, {"model": model_name, "n_features": len(feat_cols),
                      "n": int(valid.sum()), "auc": float("nan"),
                      "auc_ci_low": float("nan"), "auc_ci_high": float("nan"),
                      "balacc": float("nan"), "mcc": float("nan"),
                      "skip_reason": "insufficient OOF coverage"}
    auc = roc_auc_score(y[valid], y_prob[valid])
    ci_lo, ci_hi = bootstrap_auc_ci(y[valid], y_prob[valid], seed=seed)
    y_pred = (y_prob[valid] >= 0.5).astype(int)
    balacc = balanced_accuracy_score(y[valid], y_pred)
    mcc = matthews_corrcoef(y[valid], y_pred)

    oof = pd.DataFrame({
        "ID": df["ID"].values, "base_id": df["base_id"].values,
        "label": y, "prob": y_prob,
    })
    summary = {
        "model": model_name, "n_features": len(feat_cols),
        "feat_cols": "+".join(feat_cols), "n": int(valid.sum()),
        "n_pos": int((y[valid] == 1).sum()),
        "n_neg": int((y[valid] == 0).sum()),
        "auc": float(auc), "auc_ci_low": ci_lo, "auc_ci_high": ci_hi,
        "balacc": float(balacc), "mcc": float(mcc),
        "n_folds": n_folds_eff,
    }
    return oof, summary


# ============================================================
# Per-cell driver
# ============================================================

FEATURE_SETS = [
    ("2feat", ["Age", "MMSE"]),
    ("3feat", ["Age", "MMSE", "age_error"]),
]
MODEL_ORDER = ["xgb", "tabpfn"]


def run_cell(design, hc_source, pred_ages, cohort_dir, cohort_mode, hc_source_mode):
    """Layout: <root>/<cohort>/ad_vs_<cmp>/<bucket>/<set>/<model>_oof.csv
    + <root>/<cohort>/ad_vs_<cmp>/summary_<bucket>.csv"""
    root, bucket = DESIGN_TO_ROOT_BUCKET[design]
    partition = f"ad_vs_{hc_source.lower()}"
    partition_dir = root / cohort_dir / partition
    bucket_dir = partition_dir / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== {design} × {hc_source} → "
                f"{bucket_dir.relative_to(PROJECT_ROOT)} ===")
    try:
        feat_df = build_feature_df(design, hc_source, pred_ages,
                                   cohort_mode, hc_source_mode)
    except Exception as e:
        logger.error(f"  cohort build failed: {e}")
        return [{"design": design, "comparison": partition,
                 "model": "—", "n_features": 0, "n": 0,
                 "auc": float("nan"), "skip_reason": f"cohort error: {e}"}]
    if feat_df is None or feat_df.empty:
        logger.warning(f"  empty cohort, skipping")
        return [{"design": design, "comparison": partition,
                 "model": "—", "n_features": 0, "n": 0,
                 "auc": float("nan"), "skip_reason": "empty cohort"}]

    rows = []
    for set_name, feat_cols in FEATURE_SETS:
        feat_dir = bucket_dir / set_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        for model in MODEL_ORDER:
            oof, s = _cv_eval(feat_df, feat_cols, model)
            s.update({"design": design, "comparison": partition,
                      "feature_set": set_name})
            if oof is not None:
                oof.to_csv(feat_dir / f"{model}_oof.csv", index=False)
            logger.info(
                f"  {set_name:6s} {model:6s}: n={s['n']:4d}  "
                f"AUC={s.get('auc', float('nan')):.3f} "
                f"[{s.get('auc_ci_low', float('nan')):.3f}, "
                f"{s.get('auc_ci_high', float('nan')):.3f}]  "
                f"BalAcc={s.get('balacc', float('nan')):.3f}  "
                f"MCC={s.get('mcc', float('nan')):.3f}"
                f"{'  SKIP=' + s['skip_reason'] if s.get('skip_reason') else ''}")
            rows.append(s)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(partition_dir / f"summary_{bucket}.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-mode",
                        choices=["default", "p_first_hc_all", "p_all_hc_all"],
                        default="default")
    parser.add_argument("--hc-source-mode",
                        choices=["ACS", "ACS_ext", "EACS"], default="ACS")
    parser.add_argument("--designs", nargs="+", choices=ALL_DESIGNS,
                        default=ALL_DESIGNS,
                        help="只跑指定 designs (default: all 4)")
    args = parser.parse_args()

    cohort_dir = cohort_name(args.cohort_mode)
    logger.info(f"cohort_mode={args.cohort_mode}  designs={args.designs}  "
                f"cohort_dir={cohort_dir}")

    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    logger.info(f"Loaded {len(pred_ages)} predicted ages")

    all_rows = []
    for design in args.designs:
        for hc_source in COMPARISONS:
            rows = run_cell(design, hc_source, pred_ages, cohort_dir,
                            args.cohort_mode, args.hc_source_mode)
            all_rows.extend(rows)

    grand = pd.DataFrame(all_rows)
    out_csv = OVERVIEW_DIR / cohort_dir / "classifier_summary_all.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grand.to_csv(out_csv, index=False)
    logger.info(f"\nDone. Grand summary: {out_csv}")


if __name__ == "__main__":
    main()
