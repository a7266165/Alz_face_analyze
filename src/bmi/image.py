"""端到端 人臉影像 → BMI 迴歸。

對 pretrained CNN（ResNet-50 / EfficientNet-B0）接單輸出迴歸頭做 fine-tune，協定與
embedding 路線一致（10-fold GroupKFold by base_id）。兩階段訓練：先凍 backbone 只訓 head
（較大 LR），再全解凍以較小 LR + cosine decay。
"""

import logging
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .core import regression_metrics
from .image_data import BMIFaceDataset, BMIFaceEvalDataset, eval_transforms, train_transforms

logger = logging.getLogger(__name__)

ModelArch = Literal["resnet50", "efficientnet_b0"]


def make_image_model(arch: ModelArch, pretrained: bool = True) -> nn.Module:
    """resnet50 | efficientnet_b0 → 單輸出迴歸 CNN。"""
    if arch == "resnet50":
        import torchvision.models as models
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "efficientnet_b0":
        import timm
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
    raise ValueError(f"Unknown architecture: {arch}")


def _freeze_backbone(model: nn.Module, arch: ModelArch):
    if arch == "resnet50":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")
    elif arch == "efficientnet_b0":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name


def _unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def _eval_subject_level(model, eval_dataset, device, batch_size=64):
    """逐影像預測後對 subject 平均。"""
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    preds, sids, bmis = [], [], []
    for imgs, bmi, sid in loader:
        out = model(imgs.to(device)).squeeze(1).cpu().numpy()
        preds.extend(out.tolist())
        sids.extend(sid)
        bmis.extend(bmi.numpy().tolist())

    df = pd.DataFrame({"sid": sids, "pred": preds, "true": bmis})
    subj = df.groupby("sid").agg({"pred": "mean", "true": "first"}).reset_index()
    return subj["true"].to_numpy(), subj["pred"].to_numpy(), subj["sid"].tolist()


def _finetune(model, train_loader, arch, device, *, warmup_epochs, finetune_epochs,
              lr_head, lr_finetune, track_best):
    """兩階段訓練：warmup（只訓 head）→ 全解凍 + cosine decay。track_best 時保留最佳 train-loss 權重。"""
    criterion = nn.SmoothL1Loss()

    _freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)
    for _ in range(warmup_epochs):
        _train_one_epoch(model, train_loader, criterion, optimizer, device)

    _unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=1e-6)
    best_loss, best_state = float("inf"), None
    for ep in range(finetune_epochs):
        loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if track_best and loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif not track_best:
            logger.info(f"  Final train epoch {ep}: loss={loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


def cross_validate_image(
    ids: List[str],
    bmi: np.ndarray,
    groups: np.ndarray,
    aligned_dir,
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
    """端到端影像迴歸的 GroupKFold CV。"""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics, oof_true, oof_pred, oof_ids, oof_fold = [], [], [], [], []

    ids_arr = np.array(ids)
    bmi_arr = np.array(bmi, dtype=np.float64)

    for fold_i, (tr, te) in enumerate(gkf.split(ids_arr, bmi_arr, groups)):
        logger.info(f"  Fold {fold_i}: train={len(tr)}, test={len(te)}")

        train_ds = BMIFaceDataset(ids_arr[tr].tolist(), bmi_arr[tr], aligned_dir,
                                  transform=train_transforms(input_size))
        test_ds = BMIFaceEvalDataset(ids_arr[te].tolist(), bmi_arr[te], aligned_dir,
                                     transform=eval_transforms(input_size))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)

        model = make_image_model(arch, pretrained=True).to(device)
        _finetune(model, train_loader, arch, device,
                  warmup_epochs=warmup_epochs, finetune_epochs=finetune_epochs,
                  lr_head=lr_head, lr_finetune=lr_finetune, track_best=True)

        y_true, y_pred, sids = _eval_subject_level(model, test_ds, device, batch_size)
        m = regression_metrics(y_true, y_pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        logger.info(f"    MAE={m['mae']:.2f}  R2={m['r2']:.3f}  r={m['pearson_r']:.3f}")

        oof_true.extend(y_true.tolist())
        oof_pred.extend(y_pred.tolist())
        oof_ids.extend(sids)
        oof_fold.extend([fold_i] * len(sids))

        del model
        torch.cuda.empty_cache()

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)
    aggregate = regression_metrics(oof_true, oof_pred)
    logger.info(f"  Aggregate: MAE={aggregate['mae']:.2f}  R2={aggregate['r2']:.3f}  "
                f"r={aggregate['pearson_r']:.3f}")

    return {
        "fold_metrics": fold_metrics,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_ids": oof_ids,
        "oof_fold": np.array(oof_fold),
        "aggregate": aggregate,
    }


def train_final_image(
    ids: List[str],
    bmi: np.ndarray,
    aligned_dir,
    arch: ModelArch = "resnet50",
    warmup_epochs: int = 3,
    finetune_epochs: int = 12,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = "cuda",
    input_size: int = 224,
) -> nn.Module:
    """全資料訓練，回 model。"""
    ds = BMIFaceDataset(ids, np.array(bmi, dtype=np.float64), aligned_dir,
                        transform=train_transforms(input_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    model = make_image_model(arch, pretrained=True).to(device)
    _finetune(model, loader, arch, device,
              warmup_epochs=warmup_epochs, finetune_epochs=finetune_epochs,
              lr_head=lr_head, lr_finetune=lr_finetune, track_best=False)
    return model
