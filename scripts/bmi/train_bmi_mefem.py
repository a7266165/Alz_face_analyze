"""
MeFEm (ViT-Base) embedding → BMI regression with SVR.

Extracts 768-dim face embeddings using the MeFEm-B model (HuggingFace:
boretsyury/MeFEm), then runs the same SVR + GroupKFold protocol as the
ArcFace baseline for an apple-to-apple comparison.

Usage:
    conda run -n Alz_face_bmi python scripts/bmi/train_bmi_mefem.py
    conda run -n Alz_face_bmi python scripts/bmi/train_bmi_mefem.py --variant small
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    preprocess_dir,
    BMI_ANALYSIS_DIR,
    BMI_MODELS_DIR,
    DEMOGRAPHICS_DIR,
)

ALIGNED_DIR = preprocess_dir("aligned")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VARIANTS = {
    "small": ("vit_small_patch16_224", "MeFEm-S.pth.tar", 384),
    "base":  ("vit_base_patch16_224",  "MeFEm-B.pth.tar", 768),
}


# ---------------------------------------------------------------------------
# MeFEm model loading
# ---------------------------------------------------------------------------

def load_mefem(variant: str = "base", device: str = "cuda"):
    import timm
    from huggingface_hub import hf_hub_download

    model_name, filename, emb_dim = VARIANTS[variant]
    logger.info(f"Downloading MeFEm-{variant} weights...")
    ckpt_path = hf_hub_download(repo_id="boretsyury/MeFEm", filename=filename)

    model = timm.create_model(model_name, pretrained=False,
                              num_classes=0, global_pool="token")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    logger.info(f"MeFEm-{variant} loaded: {model_name}, emb_dim={emb_dim}")
    return model, emb_dim


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_mefem_embeddings(
    model, ids, aligned_dir, device="cuda", batch_size=64,
):
    """Extract MeFEm embeddings for each ID, mean-pooling over 10 photos."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    embeddings = {}
    total = len(ids)
    for idx, sid in enumerate(ids):
        subdir = aligned_dir / sid
        if not subdir.is_dir():
            continue
        pngs = sorted(subdir.glob("*.png"))
        if not pngs:
            continue

        tensors = []
        for p in pngs:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))

        batch = torch.stack(tensors).to(device)
        emb = model(batch)
        embeddings[sid] = emb.mean(dim=0).cpu().numpy()

        if (idx + 1) % 200 == 0:
            logger.info(f"  Extracted {idx+1}/{total}")

    logger.info(f"  Extracted {len(embeddings)}/{total} subjects")
    return embeddings


# ---------------------------------------------------------------------------
# Dataset building + CV (reuse from trainer.py)
# ---------------------------------------------------------------------------

def build_dataset_from_embeddings(embeddings, demographics_dir):
    """Pair MeFEm embeddings with BMI labels."""
    from src.common.cohort import load_demographics
    demo = load_demographics()
    demo = demo[["ID", "base_id", "BMI"]].dropna(subset=["BMI"])

    rows_x, rows_y, rows_g, rows_id = [], [], [], []
    for _, row in demo.iterrows():
        sid = row["ID"]
        if sid not in embeddings:
            continue
        rows_x.append(embeddings[sid])
        rows_y.append(float(row["BMI"]))
        rows_g.append(row["base_id"])
        rows_id.append(sid)

    X = np.stack(rows_x, axis=0)
    y = np.array(rows_y, dtype=np.float64)
    base_id_to_int = {b: i for i, b in enumerate(sorted(set(rows_g)))}
    groups = np.array([base_id_to_int[g] for g in rows_g])

    logger.info(f"Dataset: {X.shape[0]} visits, {len(base_id_to_int)} subjects, "
                f"emb_dim={X.shape[1]}, BMI [{y.min():.1f}, {y.max():.1f}]")
    return X, y, groups, rows_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="base", choices=["small", "base"])
    parser.add_argument("--models", nargs="*", default=["svr", "ridge"],
                        choices=["ridge", "svr", "xgb"])
    parser.add_argument("--n-folds", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load MeFEm ──────────────────────────────────────
    model, emb_dim = load_mefem(args.variant, device)

    # ── Load all IDs with BMI ───────────────────────────
    from src.common.cohort import load_demographics
    all_ids = load_demographics().dropna(subset=["BMI"])["ID"].tolist()
    logger.info(f"IDs with BMI: {len(all_ids)}")

    # ── Extract embeddings ──────────────────────────────
    logger.info("Extracting MeFEm embeddings...")
    embeddings = extract_mefem_embeddings(model, all_ids, ALIGNED_DIR, device)
    del model
    torch.cuda.empty_cache()

    # ── Build dataset ───────────────────────────────────
    X, y, groups, ids = build_dataset_from_embeddings(embeddings, DEMOGRAPHICS_DIR)

    # ── Run CV for each model ───────────────────────────
    from src.bmi.trainer import cross_validate, regression_metrics, train_final

    BMI_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    BMI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for model_name in args.models:
        tag = f"mefem_{args.variant}_{model_name}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {tag}")
        logger.info(f"{'='*60}")

        cv = cross_validate(X, y, groups, model_name, n_splits=args.n_folds)

        fold_df = pd.DataFrame(cv["fold_metrics"])
        fold_df.to_csv(BMI_ANALYSIS_DIR / f"cv_folds_{tag}.csv", index=False)

        oof_df = pd.DataFrame({
            "ID": ids,
            "y_true": cv["oof_true"],
            "y_pred": cv["oof_pred"],
            "fold": cv["oof_fold"].astype(int),
        })
        oof_df.to_csv(BMI_ANALYSIS_DIR / f"oof_{tag}.csv", index=False)

        agg = cv["aggregate"]
        agg["model"] = tag
        all_summaries.append(agg)
        logger.info(f"\n  {tag}:  MAE={agg['mae']:.2f}  R²={agg['r2']:.3f}  "
                    f"r={agg['pearson_r']:.3f}  N={agg['n']}")

        import joblib
        trained_model, scaler = train_final(X, y, model_name)
        joblib.dump(trained_model, BMI_MODELS_DIR / f"{tag}_model.joblib")
        joblib.dump(scaler, BMI_MODELS_DIR / f"{tag}_scaler.joblib")

    # ── Summary table ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'r':>7} {'N':>6}")
    print(f"{'-'*60}")
    for s in all_summaries:
        print(f"{s['model']:<25} {s['mae']:>6.2f} {s['rmse']:>6.2f} "
              f"{s['r2']:>7.3f} {s['pearson_r']:>7.3f} {int(s['n']):>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
