"""
Arm B supplementary: Per-modality AUC on the MMSE high/low matched cohort.

Arm B's primary output is per-feature 2-group statistics (Cohen's d, q-value).
This script augments with ML-style AUC per feature modality on the same
matched_features.csv, so metrics are directly comparable with Arm A.

Uses matched_features.csv (1:1 age-matched pairs) labelled by mmse_group:
  high=1, low=0.

Usage:
    conda run -n Alz_face_test_2 python scripts/experiments/run_arm_b_auc_supplement.py
"""

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse loaders + cv_eval from Arm A
_spec = importlib.util.spec_from_file_location(
    "arm_a_ad_vs_hc", Path(__file__).parent / "run_arm_a_ad_vs_hc.py"
)
_l1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l1)
cv_eval = _l1.cv_eval
bootstrap_auc_ci = _l1.bootstrap_auc_ci
load_emotion_matrix = _l1.load_emotion_matrix
load_landmark_matrix = _l1.load_landmark_matrix
load_embedding_mean = _l1.load_embedding_mean
load_embedding_asymmetry = _l1.load_embedding_asymmetry
EMBEDDING_MODELS = _l1.EMBEDDING_MODELS

L3_DIR = PROJECT_ROOT / "workspace" / "age_ladder" / "mmse_hilo_standalone"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def eval_modality(name, X_df, matched, model_cls="xgb"):
    merged = matched[["ID", "base_id", "label"]].merge(
        X_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns if c not in
                 ["ID", "base_id", "label", "subject_id"]]
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


def main():
    matched = pd.read_csv(L3_DIR / "matched_features.csv")
    matched["label"] = (matched["mmse_group"] == "high").astype(int)
    matched["base_id"] = matched["ID"].str.extract(r"^([A-Za-z]+\d+)")

    ids = matched["ID"].tolist()
    logger.info(f"Matched cohort: {len(matched)} (high={matched['label'].sum()}, "
                f"low={(1-matched['label']).sum()})")

    results = []

    # Age-only baseline (should be ~0.5 if age-balance is perfect)
    X_age = matched[["Age"]].to_numpy(dtype=float)
    y = matched["label"].to_numpy(dtype=int)
    g = matched["base_id"].to_numpy()
    m = cv_eval(X_age, y, g, model_cls="logistic", n_folds=5, seed=42,
                return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=42)
    results.append({"modality": "age_only", "n": len(matched),
                    "n_high": int(y.sum()), "n_low": int((1-y).sum()),
                    "n_features": 1, **m,
                    "auc_ci_low": ci_low, "auc_ci_high": ci_high})

    # age_error (already in matched_features)
    if "age_error" in matched.columns:
        sub = matched.dropna(subset=["age_error"])
        Xe = sub[["age_error"]].to_numpy(dtype=float)
        ye = sub["label"].to_numpy(dtype=int)
        ge = sub["base_id"].to_numpy()
        m = cv_eval(Xe, ye, ge, model_cls="logistic", n_folds=5, seed=42,
                    return_preds=True)
        y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
        ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, seed=42)
        results.append({"modality": "age_error", "n": len(sub),
                        "n_high": int(ye.sum()), "n_low": int((1-ye).sum()),
                        "n_features": 1, **m,
                        "auc_ci_low": ci_low, "auc_ci_high": ci_high})

    # Emotion (columns already in matched_features from method__emotion_stat)
    emo_cols = [c for c in matched.columns if "__" in c and
                not c.startswith(("landmark__", "embasym__"))]
    emo_cols = [c for c in emo_cols if c != "mmse_group"]
    emo_df = matched[["ID"] + emo_cols].rename(columns={"ID": "subject_id"})
    results.append(eval_modality("emotion_8methods", emo_df, matched))

    # Landmark asymmetry (landmark__* cols from matched_features)
    lmk_cols = [c for c in matched.columns if c.startswith("landmark__")]
    lmk_df = matched[["ID"] + lmk_cols].rename(columns={"ID": "subject_id"})
    results.append(eval_modality("landmark_asymmetry", lmk_df, matched))

    # Embedding asymmetry (embasym__* cols)
    emb_asym_cols = [c for c in matched.columns if c.startswith("embasym__")]
    emb_asym_df = matched[["ID"] + emb_asym_cols].rename(columns={"ID": "subject_id"})
    results.append(eval_modality("embedding_asymmetry", emb_asym_df, matched))

    # Full embedding mean per model (not in matched_features; load from scratch)
    for model in EMBEDDING_MODELS:
        emb = load_embedding_mean(ids, model)
        if emb is not None:
            results.append(eval_modality(f"embedding_{model}_mean", emb, matched))

    df = pd.DataFrame(results)
    out_csv = L3_DIR / "summary_per_modality_auc.csv"
    df.to_csv(out_csv, index=False)

    logger.info(f"Saved {out_csv}")
    for r in results:
        logger.info(f"  {r['modality']:<25} AUC={r.get('auc', float('nan')):.3f}  "
                    f"BalAcc={r.get('balacc', float('nan')):.3f}  "
                    f"n={r['n']}  #feats={r['n_features']}")


if __name__ == "__main__":
    main()
