"""
scripts/utilities/calibrate_age_prediction.py
使用健康族群 (ACS + NAD) 建立 K-fold 誤差校正模型。

兩種方法：
  1. Train 90% / Val 10% — 健康組用 1 val fold 校正
  2. Train 10% / Val 90% — 健康組用 9 val folds 平均

兩者各跑 30 個隨機種子取平均。

統一輸出至 calibration/ 子目錄：
  calibration/
  ├── train90_val10/   (corrected_ages.csv, summary_stats.csv, 3 plots)
  ├── train10_val90/   (同上)
  └── comparison.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# 專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import (
    DEMOGRAPHICS_DIR,
    CALIBRATION_DIR,
    PREDICTED_AGES_FILE,
)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "calibration",
    PROJECT_ROOT / "src" / "extractor" / "features" / "age" / "calibration.py",
)
_cal = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cal)

load_predicted_ages = _cal.load_predicted_ages
run_multi_seed_calibration = _cal.run_multi_seed_calibration
save_and_plot_all = _cal.save_and_plot_all
export_summary_stats = _cal.export_summary_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N_SEEDS = 30
N_SPLITS = 10


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------


def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep_cols = ["ID", "Age", "group"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        group = csv_file.replace(".csv", "")
        df["group"] = group
        dfs.append(df[keep_cols])
    return pd.concat(dfs, ignore_index=True)


def match_predicted_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    records = []
    for subject_id, pred_age in predicted_ages.items():
        row = demo[demo["ID"] == subject_id]
        if not row.empty:
            real_age = row["Age"].values[0]
            if pd.isna(real_age) or pd.isna(pred_age):
                continue
            group = row["group"].values[0]
            subject = subject_id.rsplit("-", 1)[0]
            records.append({
                "ID": subject_id,
                "subject": subject,
                "real_age": real_age,
                "predicted_age": pred_age,
                "group": group,
                "error": real_age - pred_age,
                "age_int": int(np.floor(real_age)),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = CALIBRATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    predicted_ages = load_predicted_ages(PREDICTED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    df_matched = match_predicted_ages(predicted_ages, demo)

    logger.info(f"總配對筆數: {len(df_matched)}")

    methods = [
        ("train90_val10", True, "10-Fold (Train 90% / Val 10%)"),
        ("train10_val90", False, "10-Fold (Train 10% / Val 90%)"),
    ]

    comparison_rows = []

    for dir_name, use_val_only, method_name in methods:
        sub_dir = output_dir / dir_name
        logger.info(f"\n{'='*60}")
        logger.info(f"方法: {method_name} × {N_SEEDS} seeds")
        logger.info(f"{'='*60}")

        df_result = run_multi_seed_calibration(
            df_matched,
            n_splits=N_SPLITS,
            n_seeds=N_SEEDS,
            use_val_only=use_val_only,
            base_seed=42,
        )

        save_and_plot_all(df_result, sub_dir, method_name=method_name)

        # 收集 comparison 資料
        for grp in ["ACS", "NAD", "P", "All"]:
            sub = df_result if grp == "All" else df_result[df_result["group"] == grp]
            if len(sub) == 0:
                continue
            comparison_rows.append({
                "method": dir_name,
                "group": grp,
                "n": len(sub),
                "MAE_before": f"{sub['error_before'].abs().mean():.2f}",
                "MAE_after": f"{sub['error_after'].abs().mean():.2f}",
                "mean_error_before": f"{sub['error_before'].mean():.2f}",
                "mean_error_after": f"{sub['error_after'].mean():.2f}",
            })

    # 對比表
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = output_dir / "comparison.csv"
    df_comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n方法對比已儲存: {comp_path}")

    # 印出對比
    logger.info("\n" + "=" * 60)
    logger.info("兩種方法對比")
    logger.info("=" * 60)
    for _, row in df_comp.iterrows():
        logger.info(
            f"  {row['method']:>15s} | {row['group']:>5s} (n={row['n']}): "
            f"MAE {row['MAE_before']} → {row['MAE_after']}, "
            f"Mean {row['mean_error_before']} → {row['mean_error_after']}"
        )

    logger.info(f"\n所有輸出已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
