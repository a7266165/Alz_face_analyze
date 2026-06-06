"""
Train ArcFace → BMI regression models (Ridge + SVR + XGBoost).

10-fold GroupKFold cross-validation (grouped by base_id to prevent leakage).
Saves trained models, OOF predictions, and per-fold / aggregate metrics.

Usage:
    conda run -n Alz_face_main_analysis python scripts/bmi/train_bmi.py
    conda run -n Alz_face_main_analysis python scripts/bmi/train_bmi.py --models ridge
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BMI_ANALYSIS_DIR,
    BMI_MODELS_DIR,
    EMBEDDING_FEATURES_DIR,
)
from src.bmi import (
    build_embedding_dataset,
    cross_validate,
    load_arcface_features,
    load_bmi_subjects,
    train_final,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", default=["ridge", "svr", "xgb"],
        choices=["ridge", "svr", "xgb"],
        help="Models to train (default: all three)")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load dataset ────────────────────────────────────
    logger.info("Loading ArcFace embeddings + BMI demographics...")
    ids, bmi, base_ids = load_bmi_subjects()
    feats = load_arcface_features(ids, EMBEDDING_FEATURES_DIR)
    X, y, groups, ids = build_embedding_dataset(ids, bmi, base_ids, feats)
    logger.info(f"Dataset ready: {X.shape[0]} visits, {len(set(groups))} subjects")

    # ── Ensure output dirs ──────────────────────────────
    BMI_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    BMI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        # ── Cross-validation ────────────────────────────
        cv = cross_validate(X, y, groups, model_name,
                            n_splits=args.n_folds, seed=args.seed)

        # ── Save per-fold metrics ───────────────────────
        fold_df = pd.DataFrame(cv["fold_metrics"])
        fold_csv = BMI_ANALYSIS_DIR / f"cv_folds_{model_name}.csv"
        fold_df.to_csv(fold_csv, index=False)
        logger.info(f"Fold metrics → {fold_csv}")

        # ── Save OOF predictions ────────────────────────
        oof_df = pd.DataFrame({
            "ID": ids,
            "y_true": cv["oof_true"],
            "y_pred": cv["oof_pred"],
            "fold": cv["oof_fold"].astype(int),
        })
        oof_csv = BMI_ANALYSIS_DIR / f"oof_{model_name}.csv"
        oof_df.to_csv(oof_csv, index=False)
        logger.info(f"OOF predictions → {oof_csv}")

        # ── Aggregate summary ───────────────────────────
        agg = cv["aggregate"]
        agg["model"] = model_name
        all_summaries.append(agg)
        logger.info(
            f"\n  {model_name} aggregate:  "
            f"MAE={agg['mae']:.2f}  RMSE={agg['rmse']:.2f}  "
            f"R²={agg['r2']:.3f}  r={agg['pearson_r']:.3f}  "
            f"(p={agg['pearson_p']:.2e})  N={agg['n']}")

        # ── Train final model on all data ───────────────
        model, scaler = train_final(X, y, model_name)

        import joblib
        model_path = BMI_MODELS_DIR / f"{model_name}_model.joblib"
        scaler_path = BMI_MODELS_DIR / f"{model_name}_scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Model → {model_path}")

        # ── Save resubstitution predictions (for scatter) ─
        y_resub = model.predict(scaler.transform(X))
        resub_df = pd.DataFrame({
            "ID": ids,
            "y_true": y,
            "y_pred": y_resub,
        })
        resub_csv = BMI_ANALYSIS_DIR / f"resub_{model_name}.csv"
        resub_df.to_csv(resub_csv, index=False)

    # ── Combined summary CSV ────────────────────────────
    summary_df = pd.DataFrame(all_summaries)
    summary_csv = BMI_ANALYSIS_DIR / "cv_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nCombined summary → {summary_csv}")

    # ── Print final comparison table ────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<8} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'r':>7} {'N':>6}")
    print(f"{'-'*60}")
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<8} {row['mae']:>6.2f} {row['rmse']:>6.2f} "
              f"{row['r2']:>7.3f} {row['pearson_r']:>7.3f} {int(row['n']):>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
