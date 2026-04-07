"""
TabPFN 512-d direct classification for M1 (original) and M2 (abs_rel_diff).

No RFE feature selection — feeds all 512 dimensions directly into TabPFN.
Uses fold assignments from the LR prediction CSVs for fold-aligned 5-fold CV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
from _utils import find_latest_dir

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
FEATURES_DIR = WORKSPACE_DIR / "embedding" / "features"

# 自動掃描最新 analysis 目錄，手動指定時填入 Path
ANALYSIS_DIR = None
if ANALYSIS_DIR is None:
    ANALYSIS_DIR = find_latest_dir(WORKSPACE_DIR, "analysis_")
PRED_DIR = ANALYSIS_DIR / "pred_probability" / "n_features_132"

EMBEDDING_MODELS = ["arcface", "topofr"]
FEATURE_TYPES = {
    "original": "original",                              # M1
    "absolute_relative_differences": "absolute_relative_differences",  # M2
}
RANDOM_SEED = 42


def load_fold_assignments(model: str, feature_type: str) -> dict[str, int]:
    """
    Load fold assignments from the LR test CSV.
    Each subject appears exactly once in the test set (for its assigned fold).
    Returns {subject_id: fold_idx}.
    """
    csv_name = f"{model}_{feature_type}_cdr0_test.csv"
    csv_path = PRED_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={"個案編號": "subject_id"})
    return dict(zip(df["subject_id"], df["fold"]))


def load_npy(path: Path) -> np.ndarray:
    """Load .npy — handles both plain arrays and dict-wrapped object arrays."""
    try:
        arr = np.load(path)
    except ValueError:
        arr = np.load(path, allow_pickle=True)

    if arr.ndim == 0:
        d = arr.item()
        arr = list(d.values())[0]
    return arr


def load_features(model: str, feature_type: str, subject_ids: list[str]) -> np.ndarray:
    """
    Load 512-d features for given subjects.
    Each .npy has shape (n_images, 512) — average across images.
    Returns (n_subjects, 512) array.
    """
    feat_dir = FEATURES_DIR / model / feature_type
    features = []
    for sid in subject_ids:
        npy_path = feat_dir / f"{sid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing: {npy_path}")
        arr = load_npy(npy_path)  # (n_images, 512)
        features.append(arr.mean(axis=0))  # → (512,)
    return np.array(features, dtype=np.float32)


def infer_label(subject_id: str) -> int:
    if subject_id.startswith("P"):
        return 1  # AD
    elif subject_id.startswith("ACS") or subject_id.startswith("NAD"):
        return 0  # Control
    else:
        return -1


def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    # Sensitivity / Specificity
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "mcc": mcc,
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def run_experiment(model: str, feature_type: str):
    print(f"\n{'='*70}")
    print(f"  Model: {model} | Feature: {feature_type}")
    print(f"{'='*70}")

    # 1. Get fold assignments
    fold_map = load_fold_assignments(model, feature_type)
    all_sids = list(fold_map.keys())
    all_labels = np.array([infer_label(s) for s in all_sids])

    # Filter out invalid labels
    valid_mask = all_labels != -1
    all_sids = [s for s, v in zip(all_sids, valid_mask) if v]
    all_labels = all_labels[valid_mask]
    all_folds = np.array([fold_map[s] for s in all_sids])

    print(f"  Total subjects: {len(all_sids)} (AD: {(all_labels==1).sum()}, "
          f"Control: {(all_labels==0).sum()})")

    # 2. Load ALL features at once
    print("  Loading 512-d features...")
    all_features = load_features(model, feature_type, all_sids)
    print(f"  Feature matrix: {all_features.shape}")

    # 3. 5-fold CV
    unique_folds = sorted(set(all_folds))
    fold_results = []

    for fold_idx in unique_folds:
        test_mask = all_folds == fold_idx
        train_mask = ~test_mask

        X_train = all_features[train_mask]
        y_train = all_labels[train_mask]
        X_test = all_features[test_mask]
        y_test = all_labels[test_mask]

        print(f"\n  Fold {fold_idx}: train={X_train.shape[0]}, test={X_test.shape[0]}")

        clf = TabPFNClassifier(random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate(y_test, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"    Acc={metrics['accuracy']:.4f}  MCC={metrics['mcc']:.4f}  "
              f"AUC={metrics['auc']:.4f}  Sens={metrics['sensitivity']:.4f}  "
              f"Spec={metrics['specificity']:.4f}  F1={metrics['f1']:.4f}")

    # 4. Summary
    print(f"\n  {'─'*60}")
    print(f"  MEAN ± STD across {len(unique_folds)} folds:")
    for key in ["accuracy", "mcc", "auc", "sensitivity", "specificity", "f1"]:
        vals = [r[key] for r in fold_results]
        print(f"    {key:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_results


def main():
    all_results = {}

    for model in EMBEDDING_MODELS:
        for feat_type in FEATURE_TYPES:
            key = f"{model}_{feat_type}"
            try:
                all_results[key] = run_experiment(model, feat_type)
            except Exception as e:
                print(f"\n  ERROR for {key}: {e}")
                all_results[key] = None

    # Final summary table
    print(f"\n\n{'='*80}")
    print("  FINAL SUMMARY — TabPFN 512-d Direct Classification")
    print(f"{'='*80}")
    print(f"  {'Config':<45s} {'Acc':>8s} {'MCC':>8s} {'AUC':>8s} "
          f"{'Sens':>8s} {'Spec':>8s} {'F1':>8s}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for key, results in all_results.items():
        if results is None:
            print(f"  {key:<45s}  FAILED")
            continue
        means = {k: np.mean([r[k] for r in results]) for k in results[0]}
        stds = {k: np.std([r[k] for r in results]) for k in results[0]}
        print(
            f"  {key:<45s} "
            f"{means['accuracy']*100:>5.2f}%  "
            f"{means['mcc']:>6.3f}  "
            f"{means['auc']:>6.3f}  "
            f"{means['sensitivity']*100:>5.2f}%  "
            f"{means['specificity']*100:>5.2f}%  "
            f"{means['f1']:>6.3f}"
        )


if __name__ == "__main__":
    main()
