"""
Full-features direct classification: 512 + 512 + 2 + 10 = 1036-d.

Concatenates all raw features without stacking:
  - M1: 512-d original face embeddings
  - M2: 512-d abs_rel_diff face embeddings
  - M3: 2 features (real_age, age_error)
  - M4: 10 features (emotion scores)

Classifiers: TabPFN and XGBoost.
Uses fold assignments from LR prediction CSVs for fold-aligned 5-fold CV.
"""

import json
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
from _utils import find_latest_dir

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
FEATURES_DIR = WORKSPACE_DIR / "features"

# 自動掃描最新 analysis 目錄，手動指定時填入 Path
ANALYSIS_DIR = None
if ANALYSIS_DIR is None:
    ANALYSIS_DIR = find_latest_dir(WORKSPACE_DIR, "analysis_")
PRED_DIR = ANALYSIS_DIR / "pred_probability" / "n_features_132"

EMOTION_FILE = WORKSPACE_DIR / "emotion_score_EmoNet.csv"
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
PREDICTED_AGES_FILE = WORKSPACE_DIR / "predicted_ages.json"

EMBEDDING_MODELS = ["arcface", "topofr"]
RANDOM_SEED = 42


def load_fold_assignments(model: str) -> dict[str, int]:
    """Load fold assignments from the original LR test CSV."""
    csv_path = PRED_DIR / f"{model}_original_cdr0_test.csv"
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


def load_512d_features(model: str, feature_type: str, subject_ids: list[str]) -> np.ndarray:
    """Load and mean-average (10, 512) → (512,) per subject."""
    feat_dir = FEATURES_DIR / model / feature_type
    features = []
    for sid in subject_ids:
        npy_path = feat_dir / f"{sid}.npy"
        if not npy_path.exists():
            return None  # subject missing
        arr = load_npy(npy_path)
        features.append(arr.mean(axis=0))
    return np.array(features, dtype=np.float32)


def load_emotion_scores() -> pd.DataFrame:
    """Load emotion scores CSV."""
    df = pd.read_csv(EMOTION_FILE, encoding="utf-8-sig")
    emotion_cols = [
        "Anger", "Contempt", "Disgust", "Fear", "Happiness",
        "Neutral", "Sadness", "Surprise", "Valence", "Arousal",
    ]
    return df[["subject_id"] + emotion_cols]


def load_age_data() -> pd.DataFrame:
    """Load demographics + predicted ages → real_age, age_error."""
    dfs = []
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        csv_path = DEMOGRAPHICS_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df[["ID", "Age"]])
    demo_df = pd.concat(dfs, ignore_index=True)
    demo_df = demo_df.rename(columns={"ID": "subject_id", "Age": "real_age"})
    demo_df["real_age"] = pd.to_numeric(demo_df["real_age"], errors="coerce")

    with open(PREDICTED_AGES_FILE, "r") as f:
        predicted_ages = json.load(f)

    demo_df["predicted_age"] = demo_df["subject_id"].map(predicted_ages)
    demo_df = demo_df.dropna(subset=["real_age", "predicted_age"])
    demo_df["age_error"] = demo_df["real_age"] - demo_df["predicted_age"]
    return demo_df[["subject_id", "real_age", "age_error"]]


def infer_label(subject_id: str) -> int:
    if subject_id.startswith("P"):
        return 1
    elif subject_id.startswith("ACS") or subject_id.startswith("NAD"):
        return 0
    return -1


def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc, "mcc": mcc, "auc": auc, "f1": f1,
        "sensitivity": sensitivity, "specificity": specificity,
    }


def run_experiment(model: str):
    print(f"\n{'='*70}")
    print(f"  Model: {model} | Full Features (1036-d)")
    print(f"{'='*70}")

    # 1. Fold assignments
    fold_map = load_fold_assignments(model)
    all_sids = sorted(fold_map.keys())

    # 2. Filter valid labels
    labels = {s: infer_label(s) for s in all_sids}
    all_sids = [s for s in all_sids if labels[s] != -1]

    # 3. Load emotion + age data
    emotion_df = load_emotion_scores()
    age_df = load_age_data()
    emotion_map = emotion_df.set_index("subject_id")
    age_map = age_df.set_index("subject_id")

    # 4. Filter subjects that have ALL features
    valid_sids = []
    for sid in all_sids:
        if sid not in emotion_map.index or sid not in age_map.index:
            continue
        # Check .npy files exist
        orig_path = FEATURES_DIR / model / "original" / f"{sid}.npy"
        asym_path = FEATURES_DIR / model / "absolute_relative_differences" / f"{sid}.npy"
        if orig_path.exists() and asym_path.exists():
            valid_sids.append(sid)

    print(f"  Valid subjects with all features: {len(valid_sids)} "
          f"(from {len(all_sids)} total)")

    # 5. Build feature matrix
    print("  Loading features...")
    feat_original = load_512d_features(model, "original", valid_sids)
    feat_asymmetry = load_512d_features(model, "absolute_relative_differences", valid_sids)

    emotion_vals = emotion_map.loc[valid_sids].values.astype(np.float32)  # (n, 10)
    age_vals = age_map.loc[valid_sids][["real_age", "age_error"]].values.astype(np.float32)  # (n, 2)

    X_all = np.hstack([feat_original, feat_asymmetry, age_vals, emotion_vals])
    y_all = np.array([labels[s] for s in valid_sids], dtype=np.int32)
    folds_all = np.array([fold_map[s] for s in valid_sids])

    print(f"  Feature matrix: {X_all.shape} "
          f"(M1:512 + M2:512 + M3:2 + M4:10 = {X_all.shape[1]})")
    print(f"  AD: {(y_all==1).sum()}, Control: {(y_all==0).sum()}")

    # 6. Run 5-fold CV with both classifiers
    unique_folds = sorted(set(folds_all))
    results = {"TabPFN": [], "XGBoost": []}

    for fold_idx in unique_folds:
        test_mask = folds_all == fold_idx
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        print(f"\n  Fold {fold_idx}: train={X_train.shape[0]}, test={X_test.shape[0]}")

        # TabPFN
        clf_tab = TabPFNClassifier(random_state=RANDOM_SEED)
        clf_tab.fit(X_train, y_train)
        y_prob_tab = clf_tab.predict_proba(X_test)[:, 1]
        y_pred_tab = (y_prob_tab >= 0.5).astype(int)
        m_tab = evaluate(y_test, y_pred_tab, y_prob_tab)
        results["TabPFN"].append(m_tab)
        print(f"    TabPFN:  Acc={m_tab['accuracy']:.4f}  MCC={m_tab['mcc']:.4f}  "
              f"AUC={m_tab['auc']:.4f}  Sens={m_tab['sensitivity']:.4f}  "
              f"Spec={m_tab['specificity']:.4f}")

        # XGBoost
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        clf_xgb = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            n_jobs=2, eval_metric="logloss",
        )
        clf_xgb.fit(X_train, y_train)
        y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
        m_xgb = evaluate(y_test, y_pred_xgb, y_prob_xgb)
        results["XGBoost"].append(m_xgb)
        print(f"    XGBoost: Acc={m_xgb['accuracy']:.4f}  MCC={m_xgb['mcc']:.4f}  "
              f"AUC={m_xgb['auc']:.4f}  Sens={m_xgb['sensitivity']:.4f}  "
              f"Spec={m_xgb['specificity']:.4f}")

    # 7. Summary
    print(f"\n  {'─'*60}")
    for clf_name, fold_results in results.items():
        print(f"\n  {clf_name} — MEAN ± STD across {len(unique_folds)} folds:")
        for key in ["accuracy", "mcc", "auc", "sensitivity", "specificity", "f1"]:
            vals = [r[key] for r in fold_results]
            print(f"    {key:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return results


def main():
    all_results = {}
    for model in EMBEDDING_MODELS:
        try:
            all_results[model] = run_experiment(model)
        except Exception as e:
            print(f"\n  ERROR for {model}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*80}")
    print("  FINAL SUMMARY — Full Features 1036-d Direct Classification")
    print(f"{'='*80}")
    print(f"  {'Config':<30s} {'Classifier':<10s} {'Acc':>8s} {'MCC':>8s} "
          f"{'AUC':>8s} {'Sens':>8s} {'Spec':>8s} {'F1':>8s}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for model, clf_results in all_results.items():
        for clf_name, fold_res in clf_results.items():
            means = {k: np.mean([r[k] for r in fold_res]) for k in fold_res[0]}
            stds = {k: np.std([r[k] for r in fold_res]) for k in fold_res[0]}
            print(
                f"  {model:<30s} {clf_name:<10s} "
                f"{means['accuracy']*100:>5.2f}%  "
                f"{means['mcc']:>6.3f}  "
                f"{means['auc']:>6.3f}  "
                f"{means['sensitivity']*100:>5.2f}%  "
                f"{means['specificity']*100:>5.2f}%  "
                f"{means['f1']:>6.3f}"
            )


if __name__ == "__main__":
    main()
