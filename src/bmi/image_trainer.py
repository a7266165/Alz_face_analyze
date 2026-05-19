"""
End-to-end face image → BMI regression trainer.

Fine-tunes a pretrained CNN (ResNet-50 / EfficientNet-B0) with a
single-output regression head. Uses GroupKFold by base_id to prevent
leakage — identical protocol to the embedding-based pipeline.

Training strategy:
    Phase 1 (warmup):  freeze backbone, train head only (higher LR)
    Phase 2 (finetune): unfreeze all, train with lower LR + cosine decay
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .image_dataset import (
    BMIFaceDataset,
    BMIFaceEvalDataset,
    eval_transforms,
    train_transforms,
)

logger = logging.getLogger(__name__)

ModelArch = Literal["resnet50", "efficientnet_b0"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(arch: ModelArch, pretrained: bool = True) -> nn.Module:
    if arch == "resnet50":
        import torchvision.models as models
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if arch == "efficientnet_b0":
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
        return model

    raise ValueError(f"Unknown architecture: {arch}")


def _freeze_backbone(model: nn.Module, arch: ModelArch):
    if arch == "resnet50":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    elif arch == "efficientnet_b0":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False


def _unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Data building (reuse demographics loading from trainer.py)
# ---------------------------------------------------------------------------

def build_image_dataset_info(
    demographics_dir: Path,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load IDs + BMI + group indices from demographics CSVs.

    Returns (ids, bmi_values, groups) — same as trainer.build_dataset but
    without loading embeddings.
    """
    dfs = []
    for csv_name, id_prefix_re in [
        ("P.csv", r"^(P\d+)"),
        ("NAD.csv", r"^(NAD\d+)"),
        ("ACS.csv", r"^(ACS\d+)"),
    ]:
        path = demographics_dir / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "BMI" not in df.columns:
            continue
        df = df[["ID", "BMI"]].dropna(subset=["BMI"])
        df["base_id"] = df["ID"].str.extract(id_prefix_re)
        dfs.append(df)

    demo = pd.concat(dfs, ignore_index=True)
    ids = demo["ID"].tolist()
    bmi = demo["BMI"].to_numpy(dtype=np.float64)
    base_ids = demo["base_id"].tolist()
    base_id_to_int = {b: i for i, b in enumerate(sorted(set(base_ids)))}
    groups = np.array([base_id_to_int[b] for b in base_ids])

    return ids, bmi, groups


# ---------------------------------------------------------------------------
# Metrics (reuse from trainer.py)
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from .trainer import regression_metrics as _metrics
    return _metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def _eval_subject_level(model, eval_dataset, device, batch_size=64):
    """Predict per-image, then average to per-subject predictions."""
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    all_preds = []
    all_sids = []
    all_bmi = []
    for imgs, bmi, sids in loader:
        imgs = imgs.to(device)
        out = model(imgs).squeeze(1).cpu().numpy()
        all_preds.extend(out.tolist())
        all_sids.extend(sids)
        all_bmi.extend(bmi.numpy().tolist())

    df = pd.DataFrame({"sid": all_sids, "pred": all_preds, "true": all_bmi})
    subj = df.groupby("sid").agg({"pred": "mean", "true": "first"}).reset_index()
    return subj["true"].to_numpy(), subj["pred"].to_numpy(), subj["sid"].tolist()


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_image(
    ids: List[str],
    bmi: np.ndarray,
    groups: np.ndarray,
    aligned_dir: Path,
    arch: ModelArch = "resnet50",
    n_splits: int = 10,
    warmup_epochs: int = 3,
    finetune_epochs: int = 12,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = "cuda",
    input_size: int = 224,
) -> Dict:
    """GroupKFold CV for end-to-end image-based BMI regression."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    oof_true_all, oof_pred_all, oof_ids_all, oof_fold_all = [], [], [], []

    ids_arr = np.array(ids)
    bmi_arr = np.array(bmi, dtype=np.float64)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(ids_arr, bmi_arr, groups)):
        logger.info(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

        train_ids = ids_arr[train_idx].tolist()
        train_bmi = bmi_arr[train_idx]
        test_ids = ids_arr[test_idx].tolist()
        test_bmi = bmi_arr[test_idx]

        train_ds = BMIFaceDataset(train_ids, train_bmi, aligned_dir,
                                  transform=train_transforms(input_size))
        test_ds = BMIFaceEvalDataset(test_ids, test_bmi, aligned_dir,
                                     transform=eval_transforms(input_size))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)

        model = make_model(arch, pretrained=True).to(device)
        criterion = nn.SmoothL1Loss()

        # Phase 1: head only
        _freeze_backbone(model, arch)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)
        for ep in range(warmup_epochs):
            loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Phase 2: full finetune
        _unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=1e-6)
        best_loss = float("inf")
        best_state = None
        for ep in range(finetune_epochs):
            loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # Evaluate
        y_true, y_pred, sids = _eval_subject_level(model, test_ds, device, batch_size)
        metrics = regression_metrics(y_true, y_pred)
        metrics["fold"] = fold_i
        fold_metrics.append(metrics)
        logger.info(f"    MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}  "
                    f"r={metrics['pearson_r']:.3f}")

        oof_true_all.extend(y_true.tolist())
        oof_pred_all.extend(y_pred.tolist())
        oof_ids_all.extend(sids)
        oof_fold_all.extend([fold_i] * len(sids))

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    oof_true = np.array(oof_true_all)
    oof_pred = np.array(oof_pred_all)
    aggregate = regression_metrics(oof_true, oof_pred)
    logger.info(f"  Aggregate: MAE={aggregate['mae']:.2f}  R2={aggregate['r2']:.3f}  "
                f"r={aggregate['pearson_r']:.3f}")

    return {
        "fold_metrics": fold_metrics,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_ids": oof_ids_all,
        "oof_fold": np.array(oof_fold_all),
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Full training (for inference)
# ---------------------------------------------------------------------------

def train_final_image(
    ids: List[str],
    bmi: np.ndarray,
    aligned_dir: Path,
    arch: ModelArch = "resnet50",
    warmup_epochs: int = 3,
    finetune_epochs: int = 12,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = "cuda",
    input_size: int = 224,
) -> nn.Module:
    """Train on entire dataset, return the model."""
    ds = BMIFaceDataset(ids, np.array(bmi, dtype=np.float64), aligned_dir,
                        transform=train_transforms(input_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    model = make_model(arch, pretrained=True).to(device)
    criterion = nn.SmoothL1Loss()

    _freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)
    for ep in range(warmup_epochs):
        _train_one_epoch(model, loader, criterion, optimizer, device)

    _unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=1e-6)
    for ep in range(finetune_epochs):
        loss = _train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        logger.info(f"  Final train epoch {ep}: loss={loss:.4f}")

    return model
