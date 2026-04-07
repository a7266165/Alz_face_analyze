"""
scripts/utilities/age_error_mean_correction.py
Mean Correction：NAD (age>=60) 每個整數歲平均 error 後擬合線性迴歸，
再套用到全體 (ACS / NAD / P)。

步驟：
1. 篩選 NAD age>=60
2. 每個整數歲計算平均 error
3. 對 (age, mean_error) 擬合 error = a*age + b
4. 校正公式：corrected = predicted_age + (a * real_age + b)
5. 輸出 CSV + 繪圖
"""

import sys
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import (
    DEMOGRAPHICS_DIR,
    MEAN_CORRECTION_DIR,
    PREDICTED_AGES_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 配色
COLORS = {"ACS": "#4CAF50", "NAD": "#2196F3", "P": "#F44336"}


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------


def load_data(
    demo_dir: Path, predicted_ages_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """載入人口學資料和預測年齡，回傳 (df_acs, df_nad, df_p)。"""
    predicted_ages = json.loads(predicted_ages_file.read_text(encoding="utf-8"))

    dfs = {}
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        group = csv_name.replace(".csv", "")
        df = pd.read_csv(demo_dir / csv_name, encoding="utf-8-sig")
        df["group"] = group
        df["subject"] = df["ID"].apply(lambda x: x.rsplit("-", 1)[0])
        df["predicted_age"] = df["ID"].map(predicted_ages)
        df = df.dropna(subset=["predicted_age", "Age"])
        df["real_age"] = df["Age"]
        df["error"] = df["real_age"] - df["predicted_age"]
        df["age_int"] = df["real_age"].apply(lambda x: int(np.floor(x)))
        dfs[group] = df[
            ["ID", "subject", "group", "real_age", "predicted_age", "error", "age_int"]
        ].copy()

    logger.info(
        f"載入完成: ACS={len(dfs['ACS'])}, NAD={len(dfs['NAD'])}, P={len(dfs['P'])}"
    )
    return dfs["ACS"], dfs["NAD"], dfs["P"]


# ---------------------------------------------------------------------------
# 核心演算法
# ---------------------------------------------------------------------------


def compute_per_age_mean_error(df: pd.DataFrame) -> pd.DataFrame:
    """計算每個整數歲的平均 error。"""
    stats = df.groupby("age_int")["error"].agg(["mean", "std", "count"])
    stats = stats.rename(columns={"mean": "mean_error", "std": "std_error", "count": "n"})
    return stats.sort_index().reset_index()


def fit_error_model(per_age: pd.DataFrame) -> tuple[float, float]:
    """對每個整數歲的平均 error 擬合 error = a * age + b。"""
    a, b = np.polyfit(per_age["age_int"].values, per_age["mean_error"].values, 1)
    return float(a), float(b)


def apply_correction(
    df_acs: pd.DataFrame, df_nad: pd.DataFrame, df_p: pd.DataFrame,
    a: float, b: float,
) -> pd.DataFrame:
    """套用校正到全體。"""
    rows = []
    for df_grp in [df_acs, df_nad, df_p]:
        for _, row in df_grp.iterrows():
            fitted_error = a * row["real_age"] + b
            corrected = row["predicted_age"] + fitted_error
            rows.append({
                "ID": row["ID"],
                "group": row["group"],
                "real_age": row["real_age"],
                "predicted_age": row["predicted_age"],
                "corrected_predicted_age": corrected,
                "error_before": row["error"],
                "error_after": row["real_age"] - corrected,
                "age_int": row["age_int"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 統計摘要
# ---------------------------------------------------------------------------


def print_summary(df_result: pd.DataFrame):
    """印出校正前後統計摘要。"""
    logger.info("\n" + "=" * 60)
    logger.info("Mean Correction 校正結果摘要")
    logger.info("=" * 60)

    for grp in ["ACS", "NAD", "P", "All"]:
        sub = df_result if grp == "All" else df_result[df_result["group"] == grp]
        n = len(sub)
        if n == 0:
            continue

        mae_before = sub["error_before"].abs().mean()
        mae_after = sub["error_after"].abs().mean()
        mean_before = sub["error_before"].mean()
        mean_after = sub["error_after"].mean()
        std_before = sub["error_before"].std()
        std_after = sub["error_after"].std()

        logger.info(
            f"{grp:>5s} (n={n}): "
            f"MAE {mae_before:.2f} -> {mae_after:.2f}, "
            f"Mean {mean_before:.2f}±{std_before:.2f} -> "
            f"{mean_after:.2f}±{std_after:.2f}"
        )


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------


def plot_regression_fit(per_age: pd.DataFrame, a: float, b: float, output_dir: Path):
    """每個整數歲平均 error 散點 + 擬合直線。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        per_age["age_int"], per_age["mean_error"],
        c=COLORS["NAD"], s=60, zorder=5, edgecolors="white", linewidth=0.5,
        label=f"Per-age mean error (n={len(per_age)} ages)",
    )
    ax.errorbar(
        per_age["age_int"], per_age["mean_error"],
        yerr=per_age["std_error"].fillna(0),
        fmt="none", ecolor=COLORS["NAD"], alpha=0.3, capsize=2,
    )

    x_line = np.array([per_age["age_int"].min() - 1, per_age["age_int"].max() + 1])
    ax.plot(
        x_line, a * x_line + b,
        color="#FF9800", linewidth=2.5,
        label=f"Fit: error = {a:.4f} × age + {b:.2f}",
    )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (integer)", fontsize=12)
    ax.set_ylabel("Mean Error (real − predicted)", fontsize=12)
    ax.set_title(
        f"NAD (age≥60) Per-Age Mean Error & Linear Fit\n"
        f"(a={a:.4f}, b={b:.2f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(str(output_dir / "regression_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("迴歸擬合圖已儲存")


def plot_before_after(df_result: pd.DataFrame, output_dir: Path):
    """校正前後誤差分佈直方圖（ACS / NAD / P 三面板）。"""
    groups = ["ACS", "NAD", "P"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col_idx, grp in enumerate(groups):
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue

        color = COLORS[grp]
        bins = 30

        # 上排: 校正前
        ax_before = axes[0, col_idx]
        ax_before.hist(sub["error_before"], bins=bins, color=color, alpha=0.7,
                       edgecolor="white", linewidth=0.5)
        mae_b = sub["error_before"].abs().mean()
        mean_b = sub["error_before"].mean()
        ax_before.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax_before.set_title(
            f"{grp} Before (n={len(sub)})\n"
            f"Mean={mean_b:.2f}, MAE={mae_b:.2f}",
            fontsize=11,
        )
        ax_before.set_xlabel("Error (real − predicted)")
        ax_before.grid(True, alpha=0.3)

        # 下排: 校正後
        ax_after = axes[1, col_idx]
        ax_after.hist(sub["error_after"], bins=bins, color=color, alpha=0.7,
                      edgecolor="white", linewidth=0.5)
        mae_a = sub["error_after"].abs().mean()
        mean_a = sub["error_after"].mean()
        ax_after.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax_after.set_title(
            f"{grp} After\n"
            f"Mean={mean_a:.2f}, MAE={mae_a:.2f}",
            fontsize=11,
        )
        ax_after.set_xlabel("Corrected Error")
        ax_after.grid(True, alpha=0.3)

    fig.suptitle("Age Error Distribution: Before vs After Mean Correction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "error_distribution_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("校正前後誤差分佈圖已儲存")


def plot_scatter(df_result: pd.DataFrame, output_dir: Path):
    """校正前後散佈圖：real_age vs predicted/corrected_predicted_age。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    groups = ["ACS", "NAD", "P"]
    for col_idx, grp in enumerate(groups):
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        color = COLORS[grp]

        for row_idx, (y_col, label, mae_col) in enumerate([
            ("predicted_age", "Before Correction", "error_before"),
            ("corrected_predicted_age", "After Correction", "error_after"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                sub["real_age"], sub[y_col],
                c=color, alpha=0.4, s=20, edgecolors="white", linewidth=0.3,
            )

            vmin, vmax = 20, 100
            ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1,
                    label="y = x")

            x = sub["real_age"].values.astype(np.float64)
            y = sub[y_col].values.astype(np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.array([vmin, vmax])
                ax.plot(x_line, slope * x_line + intercept, color="#FF9800",
                        linewidth=2, alpha=0.8,
                        label=f"y = {slope:.2f}x + {intercept:.2f}")

            corr = sub["real_age"].corr(sub[y_col])
            mae = sub[mae_col].abs().mean()
            mean_err = sub[mae_col].mean()

            ax.set_xlabel("Real Age", fontsize=11)
            ax.set_ylabel("Predicted Age" if row_idx == 0 else "Corrected Age",
                          fontsize=11)
            ax.set_title(
                f"{grp} - {label}\n"
                f"(n={len(sub)}, r={corr:.3f}, MAE={mae:.2f}, "
                f"Mean Err={mean_err:.2f})",
                fontsize=11,
            )
            ax.legend(fontsize=9)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Real Age vs Predicted Age: Before & After Mean Correction",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "scatter_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("校正前後散佈圖已儲存")


def plot_residual_by_age_combined(df_result: pd.DataFrame, output_dir: Path):
    """三組 (ACS/NAD/P) 疊合的校正後殘差折線圖。"""
    fig, ax = plt.subplots(figsize=(14, 5))

    for grp, color in COLORS.items():
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        st = sub.groupby("age_int")["error_after"].agg(["mean", "std", "count"])
        st = st[st["count"] >= 3].sort_index()
        if len(st) == 0:
            continue

        ages = st.index.values
        means = st["mean"].values
        stds = st["std"].fillna(0).values

        ax.plot(ages, means, color=color, linewidth=2,
                marker="o", markersize=4, label=f"{grp} (n={len(sub)})")
        ax.fill_between(ages, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (y)", fontsize=12)
    ax.set_ylabel("Residual ε = y − corrected", fontsize=12)
    ax.set_title(
        "Correction Residual by True Age — ACS / NAD / P (mean ± std)",
        fontsize=13,
    )
    ax.set_xlim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(str(output_dir / "residual_by_age_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("殘差折線圖已儲存")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = MEAN_CORRECTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    df_acs, df_nad, df_p = load_data(DEMOGRAPHICS_DIR, PREDICTED_AGES_FILE)

    # 篩選 NAD 60+
    df_nad_model = df_nad[df_nad["real_age"] >= 60].copy()
    logger.info(
        f"NAD age>=60: {len(df_nad_model)} visits "
        f"({df_nad_model['subject'].nunique()} subjects)"
    )

    # 每個整數歲平均 error
    per_age = compute_per_age_mean_error(df_nad_model)
    logger.info(f"\n每個整數歲平均 error ({len(per_age)} ages):")
    for _, row in per_age.iterrows():
        logger.info(
            f"  {int(row['age_int'])} 歲: n={int(row['n'])}, "
            f"mean_error={row['mean_error']:.2f}±{row['std_error']:.2f}"
        )

    # 擬合線性模型
    a, b = fit_error_model(per_age)
    logger.info(f"\n線性迴歸: error = {a:.4f} × age + {b:.2f}")

    # 套用校正到全體
    df_result = apply_correction(df_acs, df_nad, df_p, a, b)

    # 統計摘要
    print_summary(df_result)

    # 輸出 CSV
    per_age.to_csv(
        output_dir / "per_age_mean_error.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"每個整數歲平均 error CSV 已儲存")

    pd.DataFrame([{"a": a, "b": b}]).to_csv(
        output_dir / "mean_correction_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"迴歸係數 CSV 已儲存")

    df_result.to_csv(
        output_dir / "corrected_ages_mean.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"校正結果 CSV 已儲存")

    # 繪圖
    plot_regression_fit(per_age, a, b, output_dir)
    plot_before_after(df_result, output_dir)
    plot_scatter(df_result, output_dir)
    plot_residual_by_age_combined(df_result, output_dir)

    logger.info(f"\n所有輸出已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
