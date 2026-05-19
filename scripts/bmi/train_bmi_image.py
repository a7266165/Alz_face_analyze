"""
Train end-to-end face image → BMI regression (ResNet-50 / EfficientNet-B0).

10-fold GroupKFold cross-validation (grouped by base_id).
Saves OOF predictions in the same CSV format as the embedding pipeline,
so plot_bmi_results.py works directly.

Usage:
    conda run -n Alz_face_bmi python scripts/bmi/train_bmi_image.py
    conda run -n Alz_face_bmi python scripts/bmi/train_bmi_image.py --arch efficientnet_b0
    conda run -n Alz_face_bmi python scripts/bmi/train_bmi_image.py --arch resnet50 --finetune-epochs 20
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ALIGNED_DIR,
    BMI_ANALYSIS_DIR,
    BMI_MODELS_DIR,
    DEMOGRAPHICS_DIR,
)
from src.bmi.image_trainer import (
    build_image_dataset_info,
    cross_validate_image,
    train_final_image,
)
from src.bmi.trainer import regression_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--skip-final-train", action="store_true",
                        help="Skip training the final model on all data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load dataset info ───────────────────────────────
    logger.info("Loading demographics + BMI...")
    ids, bmi, groups = build_image_dataset_info(DEMOGRAPHICS_DIR)
    logger.info(f"Dataset: {len(ids)} visits, {len(set(groups))} subjects, "
                f"BMI [{bmi.min():.1f}, {bmi.max():.1f}]")

    BMI_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    BMI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_tag = f"image_{args.arch}"

    # ── Cross-validation ────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Warmup: {args.warmup_epochs} ep, Finetune: {args.finetune_epochs} ep")
    logger.info(f"{'='*60}")

    cv = cross_validate_image(
        ids, bmi, groups, ALIGNED_DIR,
        arch=args.arch,
        n_splits=args.n_folds,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        device=device,
        input_size=args.input_size,
    )

    # ── Save per-fold metrics ───────────────────────────
    fold_df = pd.DataFrame(cv["fold_metrics"])
    fold_csv = BMI_ANALYSIS_DIR / f"cv_folds_{model_tag}.csv"
    fold_df.to_csv(fold_csv, index=False)

    # ── Save OOF predictions ────────────────────────────
    oof_df = pd.DataFrame({
        "ID": cv["oof_ids"],
        "y_true": cv["oof_true"],
        "y_pred": cv["oof_pred"],
        "fold": cv["oof_fold"].astype(int),
    })
    oof_csv = BMI_ANALYSIS_DIR / f"oof_{model_tag}.csv"
    oof_df.to_csv(oof_csv, index=False)

    agg = cv["aggregate"]
    logger.info(
        f"\n  {model_tag} aggregate:  "
        f"MAE={agg['mae']:.2f}  RMSE={agg['rmse']:.2f}  "
        f"R²={agg['r2']:.3f}  r={agg['pearson_r']:.3f}  "
        f"(p={agg['pearson_p']:.2e})  N={agg['n']}")

    # ── Train final model ───────────────────────────────
    if not args.skip_final_train:
        logger.info("Training final model on all data...")
        model = train_final_image(
            ids, bmi, ALIGNED_DIR,
            arch=args.arch,
            warmup_epochs=args.warmup_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_finetune=args.lr_finetune,
            device=device,
            input_size=args.input_size,
        )
        model_path = BMI_MODELS_DIR / f"{model_tag}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model → {model_path}")

    # ── Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'r':>7} {'N':>6}")
    print(f"{'-'*60}")
    print(f"{model_tag:<25} {agg['mae']:>6.2f} {agg['rmse']:>6.2f} "
          f"{agg['r2']:>7.3f} {agg['pearson_r']:>7.3f} {int(agg['n']):>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
