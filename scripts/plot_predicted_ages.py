"""
scripts/plot_predicted_ages.py
統計 predicted_ages.json 的年齡分佈，
並繪製 真實年齡 vs 預測年齡 scatter plot
"""

import sys
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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


def load_predicted_ages(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep_cols = ["ID", "Age", "group", "MMSE", "CASI", "Global_CDR"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        group = csv_file.replace(".csv", "")
        df["group"] = group
        for c in keep_cols:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df[keep_cols])
    return pd.concat(dfs, ignore_index=True)


def match_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    records = []
    for subject_id, pred_age in predicted_ages.items():
        row = demo[demo["ID"] == subject_id]
        if not row.empty:
            real_age = row["Age"].values[0]
            group = row["group"].values[0]
            rec = {
                "ID": subject_id,
                "real_age": real_age,
                "predicted_age": pred_age,
                "group": group,
                "error": real_age - pred_age,
                "MMSE": row["MMSE"].values[0],
                "CASI": row["CASI"].values[0],
                "Global_CDR": row["Global_CDR"].values[0],
            }
            records.append(rec)
    return pd.DataFrame(records)


AGE_BINS = [
    ("<65", lambda a: a < 65),
    ("65-75", lambda a: (a >= 65) & (a < 75)),
    ("75-85", lambda a: (a >= 75) & (a < 85)),
    (">=85", lambda a: a >= 85),
]


def _group_stats(df: pd.DataFrame) -> str:
    """回傳單一子集的 n, real, pred, diff, MAE, r 摘要字串"""
    n = len(df)
    if n == 0:
        return "n=0"
    real = df["real_age"]
    pred = df["predicted_age"]
    err = df["error"]
    r = real.corr(pred) if n > 2 else float("nan")
    return (
        f"n={n}, "
        f"真實={real.mean():.2f}\u00b1{real.std():.2f}, "
        f"預測={pred.mean():.2f}\u00b1{pred.std():.2f}, "
        f"diff={err.mean():.2f}\u00b1{err.std():.2f}, "
        f"MAE={err.abs().mean():.2f}, r={r:.4f}"
    )


def _age_stratified_rows(df: pd.DataFrame) -> list:
    """產生年齡分層的統計列（供 CSV 輸出）"""
    rows = []
    for label, mask_fn in AGE_BINS:
        sub = df[mask_fn(df["real_age"])]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "age_group": label,
                "n": len(sub),
                "real_age": f"{sub['real_age'].mean():.2f}\u00b1{sub['real_age'].std():.2f}",
                "pred_age": f"{sub['predicted_age'].mean():.2f}\u00b1{sub['predicted_age'].std():.2f}",
                "diff": f"{sub['error'].mean():.2f}\u00b1{sub['error'].std():.2f}",
                "MAE": f"{sub['error'].abs().mean():.2f}",
            }
        )
    # Total
    rows.append(
        {
            "age_group": "Total",
            "n": len(df),
            "real_age": f"{df['real_age'].mean():.2f}\u00b1{df['real_age'].std():.2f}",
            "pred_age": f"{df['predicted_age'].mean():.2f}\u00b1{df['predicted_age'].std():.2f}",
            "diff": f"{df['error'].mean():.2f}\u00b1{df['error'].std():.2f}",
            "MAE": f"{df['error'].abs().mean():.2f}",
        }
    )
    return rows


def print_statistics(
    predicted_ages: dict, df_matched: pd.DataFrame, output_dir: Path
):
    logger.info("=" * 60)
    logger.info("predicted_ages.json 統計")
    logger.info("=" * 60)
    logger.info(f"預測年齡筆數: {len(predicted_ages)}")
    logger.info(f"成功配對筆數: {len(df_matched)}")
    logger.info(f"未配對筆數:   {len(predicted_ages) - len(df_matched)}")
    logger.info("")

    # --- 整體統計 ---
    logger.info("[整體] " + _group_stats(df_matched))
    logger.info("")

    # --- 各族群整體統計 ---
    logger.info("各族群整體統計:")
    for grp in ["ACS", "NAD", "P"]:
        sub = df_matched[df_matched["group"] == grp]
        logger.info(f"  {grp}: " + _group_stats(sub))
    logger.info("")

    # --- 各族群年齡分層統計 ---
    all_csv_rows = []

    for grp in ["ACS", "NAD", "P"]:
        sub = df_matched[df_matched["group"] == grp]
        logger.info(f"{grp} 年齡分層:")
        rows = _age_stratified_rows(sub)
        for r in rows:
            logger.info(
                f"  {r['age_group']:>5s}: n={r['n']}, "
                f"真實={r['real_age']}, 預測={r['pred_age']}, "
                f"diff={r['diff']}, MAE={r['MAE']}"
            )
            all_csv_rows.append({"group": grp, **r})
        logger.info("")

    # 整體年齡分層
    logger.info("全部 年齡分層:")
    rows = _age_stratified_rows(df_matched)
    for r in rows:
        logger.info(
            f"  {r['age_group']:>5s}: n={r['n']}, "
            f"真實={r['real_age']}, 預測={r['pred_age']}, "
            f"diff={r['diff']}, MAE={r['MAE']}"
        )
        all_csv_rows.append({"group": "All", **r})

    # 儲存 CSV
    csv_path = output_dir / "age_error_stat_2.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n統計 CSV 已儲存: {csv_path}")


def _draw_one_panel(ax, df: pd.DataFrame, title: str, colors: dict, labels: dict):
    """在單一 ax 上畫 scatter + y=x + regression line + 統計"""
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if len(sub) > 0:
            ax.scatter(
                sub["real_age"],
                sub["predicted_age"],
                c=color,
                label=labels[grp],
                alpha=0.6,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5

    # y = x 對角線
    ax.plot(
        [age_min, age_max], [age_min, age_max],
        "k--", alpha=0.5, linewidth=1, label="y = x",
    )

    # Regression line (手動計算，避開 LAPACK 問題)
    try:
        x = df["real_age"].values.astype(np.float64)
        y = df["predicted_age"].values.astype(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = float(np.sum((x - x_mean) ** 2))
        if ss_xx > 0:
            a = float(np.sum((x - x_mean) * (y - y_mean))) / ss_xx
            b = y_mean - a * x_mean
            x_line = np.array([age_min, age_max])
            ax.plot(
                x_line, a * x_line + b,
                color="#FF9800", linewidth=2, alpha=0.8,
                label=f"y = {a:.2f}x + {b:.2f}",
            )
        else:
            logger.warning(f"無法計算迴歸線: x 方差為零")
    except Exception as e:
        logger.warning(f"迴歸線計算失敗: {e}")

    n = len(df)
    corr = df["real_age"].corr(df["predicted_age"])
    mae = df["error"].abs().mean()

    ax.set_xlabel("Real Age", fontsize=12)
    ax.set_ylabel("Predicted Age (MiVOLO)", fontsize=12)
    ax.set_title(f"{title}\n(n={n}, r={corr:.3f}, MAE={mae:.1f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_scatter(df_matched: pd.DataFrame, output_path: Path):
    df_healthy = df_matched[df_matched["group"].isin(["ACS", "NAD"])]
    df_patient = df_matched[df_matched["group"] == "P"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

    _draw_one_panel(
        ax_left,
        df_healthy,
        title="Healthy Controls (NAD + ACS)",
        colors={"NAD": "#2196F3", "ACS": "#4CAF50"},
        labels={"NAD": "NAD", "ACS": "ACS"},
    )

    _draw_one_panel(
        ax_right,
        df_patient,
        title="Patients (P)",
        colors={"P": "#F44336"},
        labels={"P": "Patient"},
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"圖表已儲存: {output_path}")


# ---------------------------------------------------------------------------
# Patient 專屬分析: CDR 分層誤差、MMSE/CASI 相關
# ---------------------------------------------------------------------------

CDR_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]


def patient_cdr_error_stats(df_matched: pd.DataFrame, output_dir: Path):
    """Patient 族群依 CDR 分層的年齡預測誤差統計，輸出 CSV。"""
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p["Global_CDR"] = pd.to_numeric(df_p["Global_CDR"], errors="coerce")
    df_p = df_p.dropna(subset=["Global_CDR"])

    rows = []
    for cdr in CDR_LEVELS:
        sub = df_p[df_p["Global_CDR"] == cdr]
        n = len(sub)
        if n == 0:
            continue
        err = sub["error"]
        rows.append({
            "CDR": cdr,
            "n": n,
            "real_age": f"{sub['real_age'].mean():.2f}\u00b1{sub['real_age'].std():.2f}",
            "pred_age": f"{sub['predicted_age'].mean():.2f}\u00b1{sub['predicted_age'].std():.2f}",
            "diff": f"{err.mean():.2f}\u00b1{err.std():.2f}",
            "MAE": f"{err.abs().mean():.2f}",
        })
    # Total
    err_all = df_p["error"]
    rows.append({
        "CDR": "Total",
        "n": len(df_p),
        "real_age": f"{df_p['real_age'].mean():.2f}\u00b1{df_p['real_age'].std():.2f}",
        "pred_age": f"{df_p['predicted_age'].mean():.2f}\u00b1{df_p['predicted_age'].std():.2f}",
        "diff": f"{err_all.mean():.2f}\u00b1{err_all.std():.2f}",
        "MAE": f"{err_all.abs().mean():.2f}",
    })

    csv_path = output_dir / "patient_cdr_age_error.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Patient CDR 分層誤差 CSV 已儲存: {csv_path}")

    # 印出摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("Patient CDR 分層年齡預測誤差")
    logger.info("=" * 60)
    for r in rows:
        logger.info(
            f"  CDR={str(r['CDR']):>5s}: n={r['n']}, "
            f"real={r['real_age']}, pred={r['pred_age']}, "
            f"diff={r['diff']}, MAE={r['MAE']}"
        )


MMSE_BINS = [
    ("24-30", lambda s: (s >= 24) & (s <= 30)),
    ("18-23", lambda s: (s >= 18) & (s < 24)),
    ("10-17", lambda s: (s >= 10) & (s < 18)),
    ("0-9",   lambda s: (s >= 0) & (s < 10)),
]

CASI_BINS = [
    ("85-100", lambda s: (s >= 85) & (s <= 100)),
    ("70-84",  lambda s: (s >= 70) & (s < 85)),
    ("45-69",  lambda s: (s >= 45) & (s < 70)),
    ("0-44",   lambda s: (s >= 0) & (s < 45)),
]


def patient_score_stratified_error(
    df_matched: pd.DataFrame, score_col: str, bins: list,
    output_dir: Path,
):
    """Patient 族群依 score_col 分層的年齡預測誤差統計，輸出 CSV。"""
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p[score_col] = pd.to_numeric(df_p[score_col], errors="coerce")
    df_valid = df_p.dropna(subset=[score_col, "error"])

    rows = []
    for label, mask_fn in bins:
        sub = df_valid[mask_fn(df_valid[score_col])]
        n = len(sub)
        if n == 0:
            continue
        err = sub["error"]
        rows.append({
            score_col: label,
            "n": n,
            "real_age": f"{sub['real_age'].mean():.2f}\u00b1{sub['real_age'].std():.2f}",
            "pred_age": f"{sub['predicted_age'].mean():.2f}\u00b1{sub['predicted_age'].std():.2f}",
            "diff": f"{err.mean():.2f}\u00b1{err.std():.2f}",
            "MAE": f"{err.abs().mean():.2f}",
        })
    # Total
    err_all = df_valid["error"]
    rows.append({
        score_col: "Total",
        "n": len(df_valid),
        "real_age": f"{df_valid['real_age'].mean():.2f}\u00b1{df_valid['real_age'].std():.2f}",
        "pred_age": f"{df_valid['predicted_age'].mean():.2f}\u00b1{df_valid['predicted_age'].std():.2f}",
        "diff": f"{err_all.mean():.2f}\u00b1{err_all.std():.2f}",
        "MAE": f"{err_all.abs().mean():.2f}",
    })

    csv_path = output_dir / f"patient_{score_col.lower()}_age_error.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Patient {score_col} 分層誤差 CSV 已儲存: {csv_path}")

    # 印出摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Patient {score_col} 分層年齡預測誤差")
    logger.info("=" * 60)
    for r in rows:
        logger.info(
            f"  {score_col}={str(r[score_col]):>6s}: n={r['n']}, "
            f"real={r['real_age']}, pred={r['pred_age']}, "
            f"diff={r['diff']}, MAE={r['MAE']}"
        )


def patient_score_error_correlation(
    df_matched: pd.DataFrame, score_col: str, output_dir: Path,
):
    """計算 Patient 族群中 score_col 與 age prediction error 的相關係數，
    輸出 CSV 並繪製 scatter plot。"""
    df_p = df_matched[df_matched["group"] == "P"].copy()
    df_p[score_col] = pd.to_numeric(df_p[score_col], errors="coerce")
    df_valid = df_p.dropna(subset=[score_col, "error"])

    n = len(df_valid)
    if n < 3:
        logger.warning(f"{score_col}: 有效樣本不足 (n={n})，跳過。")
        return

    x = df_valid[score_col].values
    y = df_valid["error"].values
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)

    # CSV
    csv_path = output_dir / f"patient_{score_col.lower()}_error_corr.csv"
    pd.DataFrame([{
        "score": score_col,
        "n": n,
        "pearson_r": f"{r:.4f}",
        "pearson_p": f"{p:.2e}",
        "spearman_rho": f"{rho:.4f}",
        "spearman_p": f"{p_rho:.2e}",
    }]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Patient {score_col}-error 相關 CSV 已儲存: {csv_path}")

    # 印出
    logger.info("")
    logger.info(f"Patient {score_col} vs Age Prediction Error (n={n}):")
    logger.info(f"  Pearson  r = {r:.4f}  (p = {p:.2e})")
    logger.info(f"  Spearman ρ = {rho:.4f}  (p = {p_rho:.2e})")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c="#F44336", alpha=0.4, s=20, edgecolors="white", linewidth=0.3)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept,
            color="#FF9800", linewidth=2, alpha=0.8,
            label=f"y = {slope:.3f}x + {intercept:.2f}")

    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(score_col, fontsize=12)
    ax.set_ylabel("Age Prediction Error (pred − real)", fontsize=12)
    ax.set_title(
        f"Patient: {score_col} vs Age Prediction Error\n"
        f"(n={n}, Pearson r={r:.3f}, p={p:.2e})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / f"patient_{score_col.lower()}_vs_error.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"散佈圖已儲存: {png_path}")


def main():
    # 使用 config 常數
    predicted_ages_file = PREDICTED_AGES_FILE
    demo_dir = DEMOGRAPHICS_DIR
    output_dir = WORKSPACE_DIR
    output_path = output_dir / "predicted_ages_scatter.png"

    predicted_ages = load_predicted_ages(predicted_ages_file)
    demo = load_demographics(demo_dir)
    df_matched = match_ages(predicted_ages, demo)

    print_statistics(predicted_ages, df_matched, output_dir)
    plot_scatter(df_matched, output_path)

    # Patient 專屬分析
    stat_dir = STATISTICS_DIR
    stat_dir.mkdir(parents=True, exist_ok=True)
    patient_score_stratified_error(df_matched, "MMSE", MMSE_BINS, stat_dir)
    patient_score_stratified_error(df_matched, "CASI", CASI_BINS, stat_dir)
    patient_score_error_correlation(df_matched, "MMSE", stat_dir)
    patient_score_error_correlation(df_matched, "CASI", stat_dir)
    patient_cdr_error_stats(df_matched, stat_dir)


if __name__ == "__main__":
    main()
