"""
從 xgboost meta-analysis summary.csv 產生 by_n_features 圖表，
風格與 logistic analysis 的 by_n_features 圖一致。
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
from _utils import find_latest_dir

# ── 設定 ──────────────────────────────────────────────
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# 自動掃描最新 xgboost_meta_analysis 目錄，手動指定時填入 Path
RESULT_DIR = None
if RESULT_DIR is None:
    RESULT_DIR = find_latest_dir(WORKSPACE_DIR, "xgboost_meta_analysis_")
SUMMARY_CSV = RESULT_DIR / "summary.csv"
OUTPUT_DIR = RESULT_DIR / "plots" / "by_n_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["accuracy", "mcc", "sensitivity", "specificity"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}

# ── 讀取資料 ──────────────────────────────────────────
df = pd.read_csv(SUMMARY_CSV)

# ── 按 model / method / cdr_threshold 分組畫圖 ────────
for (model, method, cdr), grp in df.groupby(["model", "method", "cdr_threshold"]):
    grp_sorted = grp.sort_values("n_features", ascending=False)
    n_features = grp_sorted["n_features"].values

    dataset_key = f"{model}_{method}_cdr{cdr}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{dataset_key} (XGBoost Meta-Analysis)", fontsize=14)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        values = grp_sorted[metric].values

        ax.plot(
            n_features, values,
            "o-", linewidth=2, markersize=4,
            label="Test", color="blue",
        )

        ax.set_xlabel("Number of Features")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.invert_xaxis()
        ax.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{dataset_key}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

print(f"\nAll plots saved to: {OUTPUT_DIR}")
