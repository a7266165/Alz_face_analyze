"""
Per-arm × per-comparison binary classifier on 2-3 simple features.

For each arm ∈ {A, B, C, D} × comparison ∈ {HC, NAD, ACS}:
  Feature sets:
    set1 = [Age, MMSE]                  (2 features)
    set2 = [Age, MMSE, age_error]       (3 features)
  Models (run order: XGBoost first, then TabPFN):
    xgb     : XGBClassifier(n_est=200, depth=4, lr=0.1)
    tabpfn  : TabPFNClassifier(ignore_pretraining_limits=True)

Cohorts are built via the grid script's `build_cohort_ad_vs_HCgroup` helper
so naive (A) / cross-sec matched (B) / longitudinal naive (C) / longitudinal
matched (D) are all handled consistently. For longitudinal arms (C/D) the
"Age"/"MMSE" features map to first_age/first_MMSE, and age_error =
first_age − baseline_predicted_age.

Outputs to workspace/arms_analysis/per_arm/<arm>/ad_vs_<cmp_lower>/age/:
  classifier_<set>_<model>_oof.csv  — per-subject out-of-fold predictions
  classifier_summary.csv            — AUC + 95% CI + BalAcc + MCC for 4 combos

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/experiments/run_arm_age_classifiers.py
"""
import argparse
import importlib.util
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_grid = _load_module("run_4arm_deep_dive",
                      PROJECT_ROOT / "scripts" / "experiments" /
                      "run_4arm_deep_dive.py")
build_cohort_ad_vs_HCgroup = _grid.build_cohort_ad_vs_HCgroup

DEFAULT_ARMS_ROOT = (PROJECT_ROOT / "workspace" / "arms_analysis" /
                     "p_first_hc_strict" / "per_arm")
AGES_FILE = (PROJECT_ROOT / "workspace" / "age" / "age_prediction" /
             "predicted_ages.json")
COMPARISONS = ["HC", "NAD", "ACS"]
ALL_ARMS = ["A", "B", "C", "D"]
N_FOLDS = 10
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Cohort → feature DataFrame (per arm)
# ============================================================

def _attach_age_error_xsec(cohort, pred_ages):
    """Cross-sectional age_error = Age − predicted_age."""
    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["ID"].map(pred_ages)
    cohort["age_error"] = cohort["Age"] - cohort["predicted_age"]
    return cohort


def _attach_age_error_long(cohort, pred_ages):
    """Longitudinal: baseline age_error = first_age − predicted_age(first_visit_id).

    Renames first_age → Age, first_MMSE → MMSE so feature names are uniform.
    """
    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["first_visit_id"].map(pred_ages)
    cohort["age_error"] = cohort["first_age"] - cohort["predicted_age"]
    cohort["Age"] = cohort["first_age"]
    cohort["MMSE"] = cohort["first_MMSE"]
    if "ID" not in cohort.columns:
        cohort["ID"] = cohort["base_id"]
    return cohort


def build_feature_df(arm, hc_source, pred_ages):
    """Return DataFrame with [ID, base_id, Age, MMSE, age_error, label].

    Pre-filter pattern (consistent with run_arm_b_ad_vs_hcgroups.py):
      - Drop subjects missing ANY of {Age, MMSE, age_error}, so 2-feat and
        3-feat see the same N subjects.
      - For matched arms (B, D), drop entire pair (both sides) if either side
        fails the filter, preserving pairing.
    """
    cohort, pairs = build_cohort_ad_vs_HCgroup(hc_source, arm=arm)
    if cohort is None or len(cohort) == 0:
        return None
    if arm in ("A", "B"):
        cohort = _attach_age_error_xsec(cohort, pred_ages)
    else:
        cohort = _attach_age_error_long(cohort, pred_ages)
    if "base_id" not in cohort.columns:
        cohort["base_id"] = cohort["ID"].astype(str).str.extract(
            r"^([A-Za-z_]+\d+)")

    # Subject-level pass filter: all 3 features present
    has_all = (cohort["Age"].notna() &
               cohort["MMSE"].notna() &
               cohort["age_error"].notna())
    n_raw = len(cohort)

    # If matched, drop entire pair when either side fails
    if arm in ("B", "D") and pairs is not None and "pair_id" in cohort.columns:
        passing_ids = set(cohort.loc[has_all, "ID"])
        keep = (pairs["minor_id"].isin(passing_ids) &
                pairs["major_id"].isin(passing_ids))
        kept_ids = set(pairs.loc[keep, "minor_id"]).union(
            pairs.loc[keep, "major_id"])
        cohort = cohort[cohort["ID"].isin(kept_ids)].copy()
        n_pairs_kept = int(keep.sum())
        n_pairs_raw = len(pairs)
        logger.info(f"  pair filter: kept {n_pairs_kept}/{n_pairs_raw} pairs "
                    f"(={2 * n_pairs_kept} subjects from raw {n_raw})")
    else:
        cohort = cohort[has_all].copy()
        logger.info(f"  subject filter: kept {len(cohort)}/{n_raw} subjects "
                    f"(needed all of Age/MMSE/age_error)")

    keep_cols = ["ID", "base_id", "Age", "MMSE", "age_error", "label"]
    feat_df = cohort[[c for c in keep_cols if c in cohort.columns]].copy()
    return feat_df


# ============================================================
# CV evaluation
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


def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cv_eval(feat_df, feat_cols, model_name, n_folds=N_FOLDS, seed=SEED):
    """GroupKFold CV; returns OOF preds DataFrame + summary dict."""
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
                       "auc_ci_low": float("nan"),
                       "auc_ci_high": float("nan"),
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
                       "auc_ci_low": float("nan"),
                       "auc_ci_high": float("nan"),
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
        "n_pos": int((y[valid] == 1).sum()), "n_neg": int((y[valid] == 0).sum()),
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
MODEL_ORDER = ["xgb", "tabpfn"]  # XGB first as requested


def run_cell(arm, hc_source, pred_ages, arms_root):
    cmp_dir = (arms_root / f"arm_{arm.lower()}" /
               f"ad_vs_{hc_source.lower()}" / "age")
    cmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Arm {arm} × {hc_source} → {cmp_dir.relative_to(PROJECT_ROOT)} ===")
    try:
        feat_df = build_feature_df(arm, hc_source, pred_ages)
    except Exception as e:
        logger.error(f"  cohort build failed: {e}")
        return [{"arm": arm, "comparison": f"ad_vs_{hc_source.lower()}",
                  "model": "—", "n_features": 0, "n": 0,
                  "auc": float("nan"), "skip_reason": f"cohort error: {e}"}]
    if feat_df is None or feat_df.empty:
        logger.warning(f"  empty cohort, skipping")
        return [{"arm": arm, "comparison": f"ad_vs_{hc_source.lower()}",
                  "model": "—", "n_features": 0, "n": 0,
                  "auc": float("nan"), "skip_reason": "empty cohort"}]

    rows = []
    for set_name, feat_cols in FEATURE_SETS:
        for model in MODEL_ORDER:
            oof, s = cv_eval(feat_df, feat_cols, model)
            s.update({"arm": arm,
                      "comparison": f"ad_vs_{hc_source.lower()}",
                      "feature_set": set_name})
            if oof is not None:
                oof_path = cmp_dir / f"classifier_{set_name}_{model}_oof.csv"
                oof.to_csv(oof_path, index=False)
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
    summary_df.to_csv(cmp_dir / "classifier_summary.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-mode", choices=["default", "p_first_hc_all"],
                         default="default",
                         help="default=原 strict HC + first-visit；"
                              "p_first_hc_all=first-visit P + ALL NAD/ACS")
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=ALL_ARMS,
                         help="只跑指定 arms (default: A B C D)")
    parser.add_argument("--arms-root", type=Path, default=DEFAULT_ARMS_ROOT,
                         help="per_arm 根目錄 (default: workspace/arms_analysis/"
                              "per_arm)；grand summary 寫到 <arms_root>/.."
                              "/classifier_summary_all.csv")
    args = parser.parse_args()

    _grid.COHORT_MODE = args.cohort_mode
    args.arms_root = args.arms_root.resolve()
    logger.info(f"cohort_mode={args.cohort_mode}  arms={args.arms}  "
                 f"arms_root={args.arms_root}")

    with open(AGES_FILE) as f:
        pred_ages = json.load(f)
    logger.info(f"Loaded {len(pred_ages)} predicted ages")

    all_rows = []
    for arm in args.arms:
        for hc_source in COMPARISONS:
            rows = run_cell(arm, hc_source, pred_ages, args.arms_root)
            all_rows.extend(rows)

    grand = pd.DataFrame(all_rows)
    out_csv = args.arms_root.parent / "classifier_summary_all.csv"
    grand.to_csv(out_csv, index=False)
    logger.info(f"\nDone. Grand summary: {out_csv}")


if __name__ == "__main__":
    main()
