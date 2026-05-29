"""
Batch BMI inference using a trained end-to-end face image model.

Loads the trained model, predicts BMI for every subject with aligned
face images (averaging over 10 photos), saves as JSON.

Usage:
    conda run -n Alz_face_bmi python scripts/bmi/predict_bmi_image.py
    conda run -n Alz_face_bmi python scripts/bmi/predict_bmi_image.py --arch efficientnet_b0
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    preprocess_dir,
    BMI_MODELS_DIR,
    BMI_PREDICTIONS_DIR,
)

ALIGNED_DIR = preprocess_dir("aligned")
from src.bmi.image_dataset import BMIFaceEvalDataset, eval_transforms
from src.bmi.image_trainer import make_model

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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    model_tag = f"image_{args.arch}"
    model_path = BMI_MODELS_DIR / f"{model_tag}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run scripts/bmi/train_bmi_image.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {args.arch} model...")
    model = make_model(args.arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_ids = sorted(
        p.name for p in ALIGNED_DIR.iterdir()
        if p.is_dir() and any(p.glob("*.png"))
    )
    logger.info(f"Found {len(all_ids)} subjects with aligned images")

    dummy_bmi = np.zeros(len(all_ids), dtype=np.float64)
    ds = BMIFaceEvalDataset(all_ids, dummy_bmi, ALIGNED_DIR,
                            transform=eval_transforms(args.input_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_preds, all_sids = [], []
    with torch.no_grad():
        for imgs, _, sids in loader:
            imgs = imgs.to(device)
            out = model(imgs).squeeze(1).cpu().numpy()
            all_preds.extend(out.tolist())
            all_sids.extend(sids)

    import pandas as pd
    df = pd.DataFrame({"sid": all_sids, "pred": all_preds})
    subj = df.groupby("sid")["pred"].mean()

    predictions = {sid: round(float(v), 2) for sid, v in subj.items()}

    BMI_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BMI_PREDICTIONS_DIR / f"predicted_bmi_{model_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    pred_arr = np.array(list(predictions.values()))
    logger.info(
        f"Saved {len(predictions)} predictions → {out_path}\n"
        f"  Range: [{pred_arr.min():.1f}, {pred_arr.max():.1f}]  "
        f"Mean: {pred_arr.mean():.1f}  Std: {pred_arr.std():.1f}")


if __name__ == "__main__":
    main()
