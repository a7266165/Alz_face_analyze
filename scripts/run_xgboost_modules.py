"""
XGBoost 獨立模組分類器

對 Module 3（年齡特徵）和 Module 4（表情特徵）各自訓練 XGBoost，
使用與 TabPFN Meta-Analysis 相同的 fold-aligned 5-fold CV。

使用方式:
    poetry run python scripts/run_xgboost_modules.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

# 加入專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ========== 路徑設定 ==========
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
PREDICTED_AGES_FILE = WORKSPACE_DIR / "predicted_ages.json"
EMOTION_SCORES_FILE = WORKSPACE_DIR / "emotion_score_EmoNet.csv"

# LR 預測分數目錄（取 fold 分配）
PREDICTIONS_DIR = (
    WORKSPACE_DIR
    / "analysis_20260228_154726_logistic_balancing_False_allvisits_True"
    / "pred_probability"
)
N_FEATURES = 132
MODEL = "topofr"
CDR = "cdr0"

# ========== 模組定義 ==========
MODULE_DEFS = {
    "M3_Age": ["real_age", "age_error"],
    "M4_Expression": [
        "Anger", "Contempt", "Disgust", "Fear", "Happiness",
        "Neutral", "Sadness", "Surprise", "Valence", "Arousal",
    ],
}


def extract_base_id(subject_id: str) -> str:
    match = re.match(r"^([A-Za-z]+\d+)", subject_id)
    return match.group(1) if match else subject_id


def infer_label(subject_id: str) -> int:
    if subject_id.startswith("P"):
        return 1
    elif subject_id.startswith("ACS") or subject_id.startswith("NAD"):
        return 0
    return -1


def load_fold_assignments() -> pd.DataFrame:
    """從 LR 預測檔案取得 fold 分配（test split）"""
    nf_dir = PREDICTIONS_DIR / f"n_features_{N_FEATURES}"
    test_file = nf_dir / f"{MODEL}_original_{CDR}_test.csv"
    train_file = nf_dir / f"{MODEL}_original_{CDR}_train.csv"

    test_df = pd.read_csv(test_file, encoding="utf-8-sig")
    test_df = test_df.rename(columns={"個案編號": "subject_id"})
    test_df["split"] = "test"

    train_df = pd.read_csv(train_file, encoding="utf-8-sig")
    train_df = train_df.rename(columns={"個案編號": "subject_id"})
    train_df["split"] = "train"

    combined = pd.concat([test_df, train_df], ignore_index=True)
    return combined[["subject_id", "fold", "split"]]


def load_age_data() -> pd.DataFrame:
    """載入年齡資料並計算 age_error"""
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


def load_emotion_data() -> pd.DataFrame:
    """載入表情分數"""
    emotion_cols = MODULE_DEFS["M4_Expression"]
    df = pd.read_csv(EMOTION_SCORES_FILE, encoding="utf-8-sig")
    return df[["subject_id"] + emotion_cols]


def compute_metrics(y_true, y_pred, y_prob):
    """計算所有評估指標"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_xgboost_module(module_name, feature_cols, fold_df, feature_df):
    """對指定模組訓練 fold-aligned XGBoost"""
    print(f"\n{'='*60}")
    print(f"  {module_name}: features = {feature_cols}")
    print(f"{'='*60}")

    # 合併 fold 分配與特徵
    merged = pd.merge(fold_df, feature_df, on="subject_id", how="inner")
    merged["label"] = merged["subject_id"].apply(infer_label)
    merged = merged[merged["label"] >= 0]

    folds = sorted(merged["fold"].unique())
    print(f"  Folds: {folds}, Total samples: {len(merged)}")

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_metrics = []

    for fold_idx in folds:
        fold_data = merged[merged["fold"] == fold_idx]
        train_data = fold_data[fold_data["split"] == "train"]
        test_data = fold_data[fold_data["split"] == "test"]

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data["label"].values
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data["label"].values

        # XGBoost with scale_pos_weight for class imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold_idx}: acc={metrics['accuracy']:.4f}, "
              f"MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}, "
              f"sens={metrics['sensitivity']:.4f}, spec={metrics['specificity']:.4f} "
              f"(train={len(train_data)}, test={len(test_data)})")

    # 整體指標
    overall = compute_metrics(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)
    )

    # 各 fold 平均
    avg_metrics = {}
    for key in ["accuracy", "mcc", "auc", "sensitivity", "specificity", "precision", "f1"]:
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    print(f"\n  --- {module_name} Overall (pooled) ---")
    print(f"  Accuracy:    {overall['accuracy']:.4f}")
    print(f"  MCC:         {overall['mcc']:.4f}")
    print(f"  AUC:         {overall['auc']:.4f}")
    print(f"  Sensitivity: {overall['sensitivity']:.4f}")
    print(f"  Specificity: {overall['specificity']:.4f}")
    print(f"  Precision:   {overall['precision']:.4f}")
    print(f"  F1:          {overall['f1']:.4f}")
    print(f"  TP={overall['tp']}, TN={overall['tn']}, FP={overall['fp']}, FN={overall['fn']}")

    print(f"\n  --- {module_name} Fold Average ± Std ---")
    for key in ["accuracy", "mcc", "auc", "sensitivity", "specificity"]:
        print(f"  {key:12s}: {avg_metrics[key]:.4f} ± {avg_metrics[f'{key}_std']:.4f}")

    return overall, avg_metrics


def main():
    print("Loading fold assignments...")
    fold_df = load_fold_assignments()
    print(f"  Fold assignments: {len(fold_df)} records")

    print("Loading age data...")
    age_df = load_age_data()
    print(f"  Age data: {len(age_df)} subjects")

    print("Loading emotion data...")
    emotion_df = load_emotion_data()
    print(f"  Emotion data: {len(emotion_df)} subjects")

    # Module 3: Age
    age_feature_df = age_df.copy()
    m3_overall, m3_avg = train_xgboost_module(
        "M3_Age", MODULE_DEFS["M3_Age"], fold_df, age_feature_df
    )

    # Module 4: Expression
    m4_overall, m4_avg = train_xgboost_module(
        "M4_Expression", MODULE_DEFS["M4_Expression"], fold_df, emotion_df
    )

    # 比較表
    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':>14s} | {'M3_Age (pooled)':>16s} | {'M4_Expression (pooled)':>22s}")
    print("-" * 60)
    for key in ["accuracy", "mcc", "auc", "sensitivity", "specificity", "precision", "f1"]:
        print(f"{key:>14s} | {m3_overall[key]:>16.4f} | {m4_overall[key]:>22.4f}")


if __name__ == "__main__":
    main()
