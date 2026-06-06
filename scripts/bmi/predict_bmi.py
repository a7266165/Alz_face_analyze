"""
Batch BMI inference using a trained ArcFace→BMI model.

Loads the trained model + scaler, predicts BMI for every subject with
an ArcFace embedding, and saves the result as JSON.

Usage:
    conda run -n Alz_face_main_analysis python scripts/bmi/predict_bmi.py
    conda run -n Alz_face_main_analysis python scripts/bmi/predict_bmi.py --model ridge
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BMI_MODELS_DIR,
    BMI_PREDICTIONS_DIR,
    EMBEDDING_FEATURES_DIR,
)
from src.bmi import load_arcface_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="ridge", choices=["ridge", "svr", "xgb"],
        help="Which trained model to use (default: ridge)")
    args = parser.parse_args()

    model_path = BMI_MODELS_DIR / f"{args.model}_model.joblib"
    scaler_path = BMI_MODELS_DIR / f"{args.model}_scaler.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run scripts/bmi/train_bmi.py first.")

    logger.info(f"Loading {args.model} model...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    emb_dir = EMBEDDING_FEATURES_DIR / "arcface" / "original"
    all_ids = sorted(p.stem for p in emb_dir.glob("*.npy"))
    logger.info(f"Found {len(all_ids)} subjects with ArcFace embeddings")

    feats = load_arcface_features(all_ids, EMBEDDING_FEATURES_DIR)
    logger.info(f"Loaded embeddings for {len(feats)} subjects")

    ordered_ids = sorted(feats.keys())
    X = np.stack([feats[sid] for sid in ordered_ids], axis=0)
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)

    predictions = {sid: round(float(y_pred[i]), 2) for i, sid in enumerate(ordered_ids)}

    BMI_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BMI_PREDICTIONS_DIR / f"predicted_bmi_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    pred_arr = np.array(list(predictions.values()))
    logger.info(
        f"Saved {len(predictions)} predictions → {out_path}\n"
        f"  Range: [{pred_arr.min():.1f}, {pred_arr.max():.1f}]  "
        f"Mean: {pred_arr.mean():.1f}  Std: {pred_arr.std():.1f}")


if __name__ == "__main__":
    main()
