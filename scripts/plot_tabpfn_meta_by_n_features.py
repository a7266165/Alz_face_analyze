"""
從 TabPFN meta-analysis summary.csv + report 產生 by_n_features 圖表，
風格與 logistic analysis 的 by_n_features 圖一致。
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── 中文字體 ──────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────
DEFAULT_RESULT_DIR = Path(
    r"C:\Users\4080\Desktop\Alz_face_analyze\workspace"
    r"\tabpfn_meta_analysis_20260226_151915"
)

METRICS = ["accuracy", "mcc", "sensitivity", "specificity"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def parse_train_metrics_from_report(report_path: Path) -> dict:
    """從 report.txt 解析訓練集效能指標。"""
    text = report_path.read_text(encoding="utf-8")

    # 擷取「訓練集效能」區塊
    match = re.search(r"訓練集效能.*?\n-+\n(.*?)(?:\n\n|\Z)", text, re.DOTALL)
    if not match:
        return {}

    block = match.group(1)
    metrics = {}
    for metric in METRICS:
        m = re.search(rf"{metric}:\s*([\d.]+)", block)
        if m:
            metrics[f"train_{metric}"] = float(m.group(1))
    return metrics


def load_train_metrics(result_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """遍歷 report 檔，將 train metrics merge 回 DataFrame。"""
    rows = []
    for _, row in df.iterrows():
        n_features = int(row["n_features"])
        model = row["model"]
        cdr = int(row["cdr_threshold"])
        dataset_key = f"{model}_cdr{cdr}"

        report_path = (
            result_dir / "reports" / f"n_features_{n_features}" / f"{dataset_key}_report.txt"
        )
        train = parse_train_metrics_from_report(report_path) if report_path.exists() else {}
        rows.append(train)

    train_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), train_df], axis=1)


def plot_by_n_features(result_dir: Path):
    """讀取資料並產生 by_n_features 圖表。"""
    summary_csv = result_dir / "summary.csv"
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")

    # 合併 train metrics
    df = load_train_metrics(result_dir, df)
    has_train = "train_accuracy" in df.columns

    output_dir = result_dir / "plots" / "by_n_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    for (model, cdr), grp in df.groupby(["model", "cdr_threshold"]):
        grp_sorted = grp.sort_values("n_features", ascending=False)
        n_feats = grp_sorted["n_features"].values
        dataset_key = f"{model}_cdr{int(cdr)}"

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{dataset_key} (TabPFN Meta-Analysis)", fontsize=14)

        for idx, metric in enumerate(METRICS):
            ax = axes[idx // 2, idx % 2]

            # Test
            test_vals = grp_sorted[metric].values
            ax.plot(
                n_feats, test_vals, "o-",
                linewidth=2, markersize=4, label="Test", color="blue",
            )

            # Train
            train_col = f"train_{metric}"
            if has_train and train_col in grp_sorted.columns:
                train_vals = grp_sorted[train_col].values
                ax.plot(
                    n_feats, train_vals, "s--",
                    linewidth=2, markersize=4, label="Train",
                    color="green", alpha=0.7,
                )

            ax.set_xlabel("Number of Features")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(METRIC_LABELS[metric])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.invert_xaxis()
            ax.legend()

        plt.tight_layout()
        out_path = output_dir / f"{dataset_key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    if not (result_dir / "summary.csv").exists():
        print(f"Error: summary.csv not found in {result_dir}")
        sys.exit(1)
    plot_by_n_features(result_dir)
