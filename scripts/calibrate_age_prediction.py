"""
scripts/calibrate_age_prediction.py
使用健康族群 (ACS + NAD) 建立 10-fold 誤差校正模型，
並將校正應用到 Patient 族群。
"""

import sys
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    DEMOGRAPHICS_DIR,
    WORKSPACE_DIR,
    STATISTICS_DIR,
    PREDICTED_AGES_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------


def load_predicted_ages(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep_cols = ["ID", "Age", "group"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        group = csv_file.replace(".csv", "")
        df["group"] = group
        dfs.append(df[keep_cols])
    return pd.concat(dfs, ignore_index=True)


def match_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    records = []
    for subject_id, pred_age in predicted_ages.items():
        row = demo[demo["ID"] == subject_id]
        if not row.empty:
            real_age = row["Age"].values[0]
            group = row["group"].values[0]
            # 從 ID 提取 subject number (例如 ACS1-2 -> ACS1, P123-4 -> P123)
            subject = subject_id.rsplit("-", 1)[0]
            rec = {
                "ID": subject_id,
                "subject": subject,
                "real_age": real_age,
                "predicted_age": pred_age,
                "group": group,
                "error": real_age - pred_age,  # error = real - predicted
            }
            records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 10-Fold 校正
# ---------------------------------------------------------------------------


def _get_age_stratum(age: float) -> str:
    """將年齡分類到年齡層。"""
    if age < 65:
        return "<65"
    elif age < 75:
        return "65-75"
    elif age < 85:
        return "75-85"
    else:
        return ">=85"


def calibrate_with_10fold(df_matched: pd.DataFrame) -> pd.DataFrame:
    """
    使用健康族群 (ACS + NAD) 進行 10-fold 校正（按年齡分層抽樣）。
    Patient 族群會被每個 fold 的模型校正，最後取平均。
    """
    # 分離健康組和病人組
    df_healthy = df_matched[df_matched["group"].isin(["ACS", "NAD"])].copy()
    df_patient = df_matched[df_matched["group"] == "P"].copy()

    logger.info(f"健康族群: {len(df_healthy)} 人次")
    logger.info(f"Patient 族群: {len(df_patient)} 人次")

    # 取得健康族群的唯一 subject，並計算每個人的代表年齡（用於分層）
    # 使用該 subject 所有 visits 的平均年齡
    subject_ages = df_healthy.groupby("subject")["real_age"].mean()
    healthy_subjects = subject_ages.index.values
    subject_strata = np.array([_get_age_stratum(subject_ages[s]) for s in healthy_subjects])

    logger.info(f"健康族群唯一受試者數: {len(healthy_subjects)}")

    # 統計各年齡層人數
    unique, counts = np.unique(subject_strata, return_counts=True)
    for stratum, cnt in zip(unique, counts):
        logger.info(f"  年齡層 {stratum}: {cnt} 人")

    # 10-fold 按人切分（依年齡分層）
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 儲存健康組 validation 的校正結果
    healthy_calibrated = []

    # 儲存 Patient 組每個 fold 的校正結果
    patient_fold_calibrated = {pid: [] for pid in df_patient["ID"]}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(healthy_subjects, subject_strata)):
        train_subjects = healthy_subjects[train_idx]
        val_subjects = healthy_subjects[val_idx]

        # 取得 train 和 validation 的所有 visits
        df_train = df_healthy[df_healthy["subject"].isin(train_subjects)]
        df_val = df_healthy[df_healthy["subject"].isin(val_subjects)]

        logger.info(
            f"Fold {fold_idx + 1}: train={len(df_train)} visits, "
            f"val={len(df_val)} visits"
        )

        # 擬合 error ~ predicted (線性迴歸)
        x_train = df_train["predicted_age"].values
        y_train = df_train["error"].values  # error = real - predicted

        # y = a*x + b
        a, b = np.polyfit(x_train, y_train, 1)
        logger.info(f"  迴歸係數: error = {a:.4f} * predicted + {b:.4f}")

        # 校正 validation 組
        for _, row in df_val.iterrows():
            pred = row["predicted_age"]
            fitted_error = a * pred + b
            calibrated = pred + fitted_error
            healthy_calibrated.append({
                "ID": row["ID"],
                "subject": row["subject"],
                "group": row["group"],
                "real_age": row["real_age"],
                "predicted_age": pred,
                "calibrated_age": calibrated,
                "error_before": row["error"],
                "error_after": row["real_age"] - calibrated,
                "fold": fold_idx + 1,
            })

        # 校正 Patient 組
        for _, row in df_patient.iterrows():
            pred = row["predicted_age"]
            fitted_error = a * pred + b
            calibrated = pred + fitted_error
            patient_fold_calibrated[row["ID"]].append(calibrated)

    # 健康組 validation 結果
    df_healthy_result = pd.DataFrame(healthy_calibrated)

    # Patient 組：取 10 個 fold 的平均
    patient_results = []
    for _, row in df_patient.iterrows():
        calibrated_values = patient_fold_calibrated[row["ID"]]
        calibrated_mean = np.mean(calibrated_values)
        patient_results.append({
            "ID": row["ID"],
            "subject": row["subject"],
            "group": row["group"],
            "real_age": row["real_age"],
            "predicted_age": row["predicted_age"],
            "calibrated_age": calibrated_mean,
            "error_before": row["error"],
            "error_after": row["real_age"] - calibrated_mean,
            "fold": "avg",
        })

    df_patient_result = pd.DataFrame(patient_results)

    # 合併結果
    df_result = pd.concat([df_healthy_result, df_patient_result], ignore_index=True)

    return df_result


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------


def plot_calibrated_scatter(df_result: pd.DataFrame, output_path: Path):
    """繪製校正後的 real vs calibrated 散佈圖。"""
    df_healthy = df_result[df_result["group"].isin(["ACS", "NAD"])]
    df_patient = df_result[df_result["group"] == "P"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左圖: 健康族群 (validation 校正結果)
    _draw_panel(
        axes[0],
        df_healthy,
        title="Healthy Controls (ACS + NAD) - Calibrated",
        colors={"ACS": "#4CAF50", "NAD": "#2196F3"},
    )

    # 右圖: Patient (平均校正結果)
    _draw_panel(
        axes[1],
        df_patient,
        title="Patients (P) - Calibrated",
        colors={"P": "#F44336"},
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"校正後散佈圖已儲存: {output_path}")


def _draw_panel(ax, df: pd.DataFrame, title: str, colors: dict):
    """繪製單一面板的 scatter plot。"""
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if len(sub) > 0:
            ax.scatter(
                sub["real_age"],
                sub["calibrated_age"],
                c=color,
                label=grp,
                alpha=0.6,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )

    # 計算範圍
    all_ages = pd.concat([df["real_age"], df["calibrated_age"]])
    age_min = all_ages.min() - 5
    age_max = all_ages.max() + 5

    # y = x 對角線
    ax.plot(
        [age_min, age_max], [age_min, age_max],
        "k--", alpha=0.5, linewidth=1, label="y = x",
    )

    # Regression line
    x = df["real_age"].values.astype(np.float64)
    y = df["calibrated_age"].values.astype(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) > 2:
        a, b = np.polyfit(x, y, 1)
        x_line = np.array([age_min, age_max])
        ax.plot(
            x_line, a * x_line + b,
            color="#FF9800", linewidth=2, alpha=0.8,
            label=f"y = {a:.2f}x + {b:.2f}",
        )

    # 統計
    n = len(df)
    corr = df["real_age"].corr(df["calibrated_age"])
    mae_before = df["error_before"].abs().mean()
    mae_after = df["error_after"].abs().mean()

    ax.set_xlabel("Real Age", fontsize=12)
    ax.set_ylabel("Calibrated Age", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"(n={n}, r={corr:.3f}, MAE: {mae_before:.2f} → {mae_after:.2f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 統計摘要
# ---------------------------------------------------------------------------


AGE_BINS = [
    ("<65", lambda a: a < 65),
    ("65-75", lambda a: (a >= 65) & (a < 75)),
    ("75-85", lambda a: (a >= 75) & (a < 85)),
    (">=85", lambda a: a >= 85),
]


def export_calibrated_stats(df_result: pd.DataFrame, output_dir: Path):
    """輸出校正後的統計表格 CSV（族群 + 年齡分層，按 age_group 排序）。"""
    rows = []

    # 定義 age_group 排序順序
    age_group_order = ["Total", "<65", "65-75", "75-85", ">=85"]

    # 按族群 + 年齡分層統計（包含 Total）
    for grp in ["ACS", "NAD", "ACS+NAD", "P", "All"]:
        if grp == "All":
            df_grp = df_result
        elif grp == "ACS+NAD":
            df_grp = df_result[df_result["group"].isin(["ACS", "NAD"])]
        else:
            df_grp = df_result[df_result["group"] == grp]

        # Total
        n = len(df_grp)
        if n > 0:
            rows.append(_compute_stats_row(df_grp, grp, "Total"))

        # 各年齡層
        for age_label, mask_fn in AGE_BINS:
            sub = df_grp[mask_fn(df_grp["real_age"])]
            n = len(sub)
            if n == 0:
                continue
            rows.append(_compute_stats_row(sub, grp, age_label))

    df_stats = pd.DataFrame(rows)

    # 按 group 和 age_group 排序
    group_order = {"ACS": 0, "NAD": 1, "ACS+NAD": 2, "P": 3, "All": 4}
    age_order = {ag: i for i, ag in enumerate(age_group_order)}
    df_stats["_group_order"] = df_stats["group"].map(group_order)
    df_stats["_age_order"] = df_stats["age_group"].map(age_order)
    df_stats = df_stats.sort_values(["_group_order", "_age_order"])
    df_stats = df_stats.drop(columns=["_group_order", "_age_order"])

    csv_path = output_dir / "calibrated_age_error_stat.csv"
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"校正後統計表 CSV 已儲存: {csv_path}")

    return df_stats


def _compute_stats_row(sub: pd.DataFrame, group: str, age_group: str) -> dict:
    """計算單一子集的統計數據。"""
    n = len(sub)
    real = sub["real_age"]
    pred = sub["predicted_age"]
    calib = sub["calibrated_age"]
    err_before = sub["error_before"]  # real - predicted
    err_after = sub["error_after"]    # real - calibrated

    return {
        "group": group,
        "age_group": age_group,
        "n": n,
        "real_age": f"{real.mean():.2f}±{real.std():.2f}",
        "pred_age": f"{pred.mean():.2f}±{pred.std():.2f}",
        "pred_age_calibrated": f"{calib.mean():.2f}±{calib.std():.2f}",
        "diff": f"{err_before.mean():.2f}±{err_before.std():.2f}",
        "diff_calibrated": f"{err_after.mean():.2f}±{err_after.std():.2f}",
        "MAE": f"{err_before.abs().mean():.2f}",
        "MAE_calibrated": f"{err_after.abs().mean():.2f}",
    }


def print_summary(df_result: pd.DataFrame):
    """印出校正前後的統計摘要。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("校正前後統計比較")
    logger.info("=" * 60)

    for grp in ["ACS", "NAD", "P", "All"]:
        if grp == "All":
            sub = df_result
        else:
            sub = df_result[df_result["group"] == grp]

        n = len(sub)
        if n == 0:
            continue

        mae_before = sub["error_before"].abs().mean()
        mae_after = sub["error_after"].abs().mean()
        diff_before = sub["error_before"].mean()
        diff_after = sub["error_after"].mean()
        std_before = sub["error_before"].std()
        std_after = sub["error_after"].std()

        logger.info(
            f"{grp:>5s} (n={n}): "
            f"MAE {mae_before:.2f} → {mae_after:.2f}, "
            f"diff {diff_before:.2f}±{std_before:.2f} → {diff_after:.2f}±{std_after:.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 使用 config 常數
    predicted_ages_file = PREDICTED_AGES_FILE
    demo_dir = DEMOGRAPHICS_DIR
    output_dir = STATISTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    predicted_ages = load_predicted_ages(predicted_ages_file)
    demo = load_demographics(demo_dir)
    df_matched = match_ages(predicted_ages, demo)

    logger.info(f"總配對筆數: {len(df_matched)}")

    # 10-fold 校正
    df_result = calibrate_with_10fold(df_matched)

    # 輸出 CSV
    csv_path = output_dir / "calibrated_ages.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"校正結果 CSV 已儲存: {csv_path}")

    # 統計摘要
    print_summary(df_result)

    # 輸出校正後統計表 CSV
    export_calibrated_stats(df_result, output_dir)

    # 繪製散佈圖
    plot_path = WORKSPACE_DIR / "predicted_ages_scatter_calibrated.png"
    plot_calibrated_scatter(df_result, plot_path)


if __name__ == "__main__":
    main()
