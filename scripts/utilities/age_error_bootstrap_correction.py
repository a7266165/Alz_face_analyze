"""
scripts/utilities/age_error_bootstrap_correction.py
使用 NAD (age>=60) 進行 bootstrap 年齡誤差校正：
1. 篩選 NAD 年齡 >= 60 的受試者
2. 每個整數歲隨機抽 1 人，擬合 e = a*y + b（y = 真實年齡）
3. 重複 10 次取 (a, b) 平均值
4. 校正公式：corrected = yp - (a*y + b)
5. 繪製校正前後結果圖 + 殘差折線圖
"""

import sys
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import (
    DEMOGRAPHICS_DIR,
    BOOTSTRAP_DIR,
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


def count_age_distribution(df: pd.DataFrame) -> pd.Series:
    """統計每個整數歲的樣本數。"""
    return df.groupby("age_int").size().sort_index()


def filter_nad_60plus(df_nad: pd.DataFrame) -> pd.DataFrame:
    """篩選 NAD 中 real_age >= 60 的子集作為建模資料。"""
    df = df_nad[df_nad["real_age"] >= 60].copy()
    logger.info(f"NAD age>=60: {len(df)} visits ({df['subject'].nunique()} subjects)")
    return df


def sample_one_per_age(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """每個整數歲隨機抽 1 位 subject 的 1 筆紀錄。"""
    rows = []
    for age_int, group_df in df.groupby("age_int"):
        subjects = group_df["subject"].unique()
        chosen_subject = rng.choice(subjects)
        sub_rows = group_df[group_df["subject"] == chosen_subject]
        rows.append(sub_rows.iloc[[rng.integers(len(sub_rows))]])
    return pd.concat(rows, ignore_index=True)


def fit_error_model(df_train: pd.DataFrame) -> tuple[float, float]:
    """擬合 error = a * real_age + b。"""
    a, b = np.polyfit(df_train["real_age"].values, df_train["error"].values, 1)
    return float(a), float(b)


def apply_correction(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """校正公式：corrected = yp - (a*y + b)。"""
    df = df.copy()
    fitted_error = a * df["real_age"] + b
    df["corrected_predicted_age"] = df["predicted_age"] + fitted_error
    df["corrected_error"] = df["real_age"] - df["corrected_predicted_age"]
    return df


def run_bootstrap(
    df_nad_model: pd.DataFrame,
    df_acs: pd.DataFrame,
    df_nad: pd.DataFrame,
    df_p: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 校正：每次抽 1 人/歲建模，NAD 剔除訓練樣本後校正。"""
    age_counts = count_age_distribution(df_nad_model)
    logger.info(
        f"建模資料: {len(age_counts)} 個整數歲 "
        f"(range {age_counts.index.min()}-{age_counts.index.max()})"
    )

    all_coefs = []
    # 記錄每個 ID 的校正結果（NAD 只收未被訓練的迭代）
    nad_corrections: dict[str, list[float]] = defaultdict(list)
    acs_corrections: dict[str, list[float]] = defaultdict(list)
    p_corrections: dict[str, list[float]] = defaultdict(list)

    for i in range(n_iter):
        rng = np.random.default_rng(seed + i)
        df_sample = sample_one_per_age(df_nad_model, rng)
        a, b = fit_error_model(df_sample)
        all_coefs.append({"iter": i + 1, "a": a, "b": b})

        if (i + 1) % 100 == 0:
            logger.info(f"  Iter {i+1}/{n_iter}: e = {a:.4f}*y + {b:.4f}")

        # 本次被抽到的 subject
        trained_subjects = set(df_sample["subject"].values)

        # ACS / P: 全部校正
        for _, row in df_acs.iterrows():
            corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
            acs_corrections[row["ID"]].append(corrected_age)

        for _, row in df_p.iterrows():
            corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
            p_corrections[row["ID"]].append(corrected_age)

        # NAD: 剔除被訓練的 subject
        for _, row in df_nad.iterrows():
            if row["subject"] not in trained_subjects:
                corrected_age = row["predicted_age"] + (a * row["real_age"] + b)
                nad_corrections[row["ID"]].append(corrected_age)

    coefs_df = pd.DataFrame(all_coefs)
    a_mean = float(coefs_df["a"].mean())
    b_mean = float(coefs_df["b"].mean())
    logger.info(f"平均係數: a={a_mean:.4f}, b={b_mean:.4f}")

    return {
        "coefs": coefs_df,
        "acs_corrections": acs_corrections,
        "nad_corrections": nad_corrections,
        "p_corrections": p_corrections,
        "age_counts": age_counts,
    }


def build_results(
    df_acs: pd.DataFrame,
    df_nad: pd.DataFrame,
    df_p: pd.DataFrame,
    bootstrap_results: dict,
) -> pd.DataFrame:
    """平均各 ID 的 bootstrap 校正結果。"""
    rows = []
    for df, group, corr_dict in [
        (df_acs, "ACS", bootstrap_results["acs_corrections"]),
        (df_nad, "NAD", bootstrap_results["nad_corrections"]),
        (df_p, "P", bootstrap_results["p_corrections"]),
    ]:
        for _, row in df.iterrows():
            corrections = corr_dict.get(row["ID"], [])
            if not corrections:
                continue
            corrected = np.mean(corrections)
            rows.append({
                "ID": row["ID"],
                "subject": row["subject"],
                "group": group,
                "real_age": row["real_age"],
                "predicted_age": row["predicted_age"],
                "corrected_predicted_age": corrected,
                "error_before": row["error"],
                "error_after": row["real_age"] - corrected,
                "age_int": row["age_int"],
                "n_iters": len(corrections),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------


def plot_age_distribution(
    age_counts: pd.Series,
    target_count: int | None,
    output_dir: Path,
    group_name: str = "ACS",
    color: str = "#4CAF50",
):
    """各整數歲樣本數長條圖。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(age_counts.index, age_counts.values, color=color, alpha=0.8)
    if target_count is not None:
        ax.axhline(y=target_count, color="red", linestyle="--", linewidth=1.5,
                   label=f"Target = {target_count}")
    ax.set_xlabel("Integer Age (real_age)", fontsize=12)
    ax.set_ylabel("Number of Visits", fontsize=12)
    title = (
        f"{group_name} Age Distribution (n={age_counts.sum()}, "
        f"{len(age_counts)} unique ages"
    )
    if target_count is not None:
        title += f", target={target_count}"
    title += ")"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for age, count in age_counts.items():
        ax.text(age, count + 0.3, str(count), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{group_name.lower()}_age_distribution.png"
    fig.savefig(str(output_dir / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{group_name} 年齡分佈圖已儲存")


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
        ax_before.set_xlabel("Error (real - predicted)")
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

    fig.suptitle("Age Error Distribution: Before vs After Bootstrap Correction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "error_distribution_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("校正前後誤差分佈圖已儲存")


AGE_BINS_LABELS = [
    ("<65", lambda a: a < 65),
    ("65-75", lambda a: (a >= 65) & (a < 75)),
    ("75-85", lambda a: (a >= 75) & (a < 85)),
    (">=85", lambda a: a >= 85),
]


def plot_error_by_age(df_result: pd.DataFrame, output_dir: Path):
    """分年齡層箱型圖：校正前後 x ACS/NAD/P。"""
    groups = ["ACS", "NAD", "P"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for col_idx, grp in enumerate(groups):
        ax = axes[col_idx]
        sub = df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue

        data_before, data_after, labels = [], [], []
        for label, mask_fn in AGE_BINS_LABELS:
            bin_data = sub[mask_fn(sub["real_age"])]
            if len(bin_data) == 0:
                continue
            data_before.append(bin_data["error_before"].values)
            data_after.append(bin_data["error_after"].values)
            labels.append(f"{label}\n(n={len(bin_data)})")

        if not labels:
            continue

        n_bins = len(labels)
        positions_before = np.arange(n_bins) * 3
        positions_after = positions_before + 1

        bp1 = ax.boxplot(data_before, positions=positions_before, widths=0.8,
                         patch_artist=True,
                         boxprops=dict(facecolor=COLORS[grp], alpha=0.4),
                         medianprops=dict(color="black"))
        bp2 = ax.boxplot(data_after, positions=positions_after, widths=0.8,
                         patch_artist=True,
                         boxprops=dict(facecolor=COLORS[grp], alpha=0.8),
                         medianprops=dict(color="black"))

        ax.set_xticks(positions_before + 0.5)
        ax.set_xticklabels(labels)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(f"{grp}", fontsize=13)
        ax.set_ylabel("Error (real - predicted)")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Before", "After"],
                  fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Error by Age Group: Before vs After Correction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "error_by_age_group.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("分年齡層誤差箱型圖已儲存")


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

            # y = x 對角線
            vmin, vmax = 20, 100
            ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1,
                    label="y = x")

            # Regression line
            x = sub["real_age"].values.astype(np.float64)
            y = sub[y_col].values.astype(np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) > 2:
                a, b = np.polyfit(x, y, 1)
                x_line = np.array([vmin, vmax])
                ax.plot(x_line, a * x_line + b, color="#FF9800", linewidth=2,
                        alpha=0.8, label=f"y = {a:.2f}x + {b:.2f}")

            # 統計
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

    fig.suptitle("Real Age vs Predicted Age: Before & After Bootstrap Correction",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "scatter_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("校正前後散佈圖已儲存")


def plot_residual_by_age(df_result: pd.DataFrame, output_dir: Path):
    """校正後殘差 ε = y - corrected 按真實年齡的折線圖，四張獨立圖。"""
    groups = [
        ("ACS", COLORS["ACS"]),
        ("NAD", COLORS["NAD"]),
        ("P", COLORS["P"]),
        ("All", "#333333"),
    ]

    for grp, color in groups:
        sub = df_result if grp == "All" else df_result[df_result["group"] == grp]
        if len(sub) == 0:
            continue
        stats = sub.groupby("age_int")["error_after"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= 3].sort_index()
        if len(stats) == 0:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        ages = stats.index.values
        means = stats["mean"].values
        stds = stats["std"].fillna(0).values

        ax.plot(ages, means, color=color, linewidth=2,
                marker="o", markersize=4, label=f"mean (n={len(sub)})")
        ax.fill_between(ages, means - stds, means + stds,
                        color=color, alpha=0.2, label="± 1 std")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
        ax.set_xlabel("True Age (y)", fontsize=12)
        ax.set_ylabel("Residual ε = y − corrected", fontsize=12)
        ax.set_title(
            f"{grp} — Correction Residual by True Age (mean ± std)",
            fontsize=13,
        )
        ax.set_xlim(50, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"residual_by_age_{grp.lower()}.png"
        fig.savefig(str(output_dir / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("殘差折線圖 (4 張) 已儲存")


def plot_coefficients(coefs: pd.DataFrame, output_dir: Path):
    """Bootstrap 係數穩定性圖。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    a_vals = coefs["a"].values
    b_vals = coefs["b"].values

    # 左圖: a 的分佈
    axes[0].scatter(coefs["iter"], a_vals, color="#FF9800", s=60, zorder=5,
                    edgecolors="white", linewidth=0.5)
    axes[0].axhline(y=a_vals.mean(), color="red", linestyle="--",
                    label=f"Mean = {a_vals.mean():.4f}")
    axes[0].fill_between(
        [0.5, len(coefs) + 0.5],
        a_vals.mean() - a_vals.std(),
        a_vals.mean() + a_vals.std(),
        alpha=0.15, color="red", label=f"SD = {a_vals.std():.4f}",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Coefficient a")
    axes[0].set_title("Slope (a): error = a * predicted_age + b", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右圖: b 的分佈
    axes[1].scatter(coefs["iter"], b_vals, color="#9C27B0", s=60, zorder=5,
                    edgecolors="white", linewidth=0.5)
    axes[1].axhline(y=b_vals.mean(), color="red", linestyle="--",
                    label=f"Mean = {b_vals.mean():.4f}")
    axes[1].fill_between(
        [0.5, len(coefs) + 0.5],
        b_vals.mean() - b_vals.std(),
        b_vals.mean() + b_vals.std(),
        alpha=0.15, color="red", label=f"SD = {b_vals.std():.4f}",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Coefficient b")
    axes[1].set_title("Intercept (b): error = a * predicted_age + b", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Bootstrap Coefficient Stability (10 Iterations)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "bootstrap_coefficients.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Bootstrap 係數穩定性圖已儲存")


# ---------------------------------------------------------------------------
# 統計摘要
# ---------------------------------------------------------------------------


def print_summary(df_result: pd.DataFrame):
    """印出校正前後的統計摘要。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Bootstrap 校正前後統計比較")
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
            f"Mean {mean_before:.2f}+-{std_before:.2f} -> "
            f"{mean_after:.2f}+-{std_after:.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    demo_dir = DEMOGRAPHICS_DIR
    predicted_ages_file = PREDICTED_AGES_FILE
    output_dir = BOOTSTRAP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    df_acs, df_nad, df_p = load_data(demo_dir, predicted_ages_file)

    # 篩選 NAD 60+ 作為建模資料
    df_nad_model = filter_nad_60plus(df_nad)

    # 統計建模資料年齡分佈
    age_counts = count_age_distribution(df_nad_model)
    logger.info(f"\nNAD (age>=60) 年齡分佈 (整數歲):")
    for age, cnt in age_counts.items():
        logger.info(f"  {age} 歲: {cnt} 筆")

    sparse_ages = age_counts[age_counts < 2]
    if len(sparse_ages) > 0:
        logger.warning(
            f"以下年齡只有 1 位受試者，bootstrap 將無隨機性: "
            f"{dict(sparse_ages)}"
        )

    # Bootstrap 校正 (1000 次, NAD 剔除訓練樣本)
    bootstrap_results = run_bootstrap(
        df_nad_model, df_acs, df_nad, df_p, n_iter=1000, seed=42
    )

    # 平均校正結果
    df_result = build_results(df_acs, df_nad, df_p, bootstrap_results)

    # 統計摘要
    print_summary(df_result)

    # 輸出 CSV
    csv_path = output_dir / "corrected_ages_bootstrap.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"校正結果已儲存: {csv_path}")

    coefs = bootstrap_results["coefs"]
    coefs_path = output_dir / "bootstrap_coefficients.csv"
    coefs_with_summary = pd.concat([
        coefs,
        pd.DataFrame([{
            "iter": "mean",
            "a": coefs["a"].mean(),
            "b": coefs["b"].mean(),
        }]),
        pd.DataFrame([{
            "iter": "std",
            "a": coefs["a"].std(),
            "b": coefs["b"].std(),
        }]),
    ], ignore_index=True)
    coefs_with_summary.to_csv(coefs_path, index=False, encoding="utf-8-sig")
    logger.info(f"Bootstrap 係數已儲存: {coefs_path}")

    # 繪圖
    plot_age_distribution(
        bootstrap_results["age_counts"],
        None,
        output_dir,
        group_name="NAD_60plus",
        color=COLORS["NAD"],
    )
    plot_before_after(df_result, output_dir)
    plot_error_by_age(df_result, output_dir)
    plot_scatter(df_result, output_dir)
    plot_coefficients(coefs, output_dir)
    plot_residual_by_age(df_result, output_dir)

    logger.info(f"\n所有輸出已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
