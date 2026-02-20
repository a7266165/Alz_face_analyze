"""
scripts/demographics_statistics.py
針對 aligned 資料夾中有預處理影像的個案，
統計年齡、性別、CDR、MMSE、CASI 的分佈情形。
分別以「人次」與「人」兩種層級呈現。
"""

import sys
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ALIGNED_DIR, DEMOGRAPHICS_DIR, STATISTICS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 使用 config 常數
DEMO_DIR = DEMOGRAPHICS_DIR
OUTPUT_DIR = STATISTICS_DIR


# ---------------------------------------------------------------------------
# 1. 載入資料
# ---------------------------------------------------------------------------

def load_aligned_ids(aligned_dir: Path) -> set[str]:
    """掃描 aligned 資料夾，回傳所有個案 ID (資料夾名稱)。"""
    ids = set()
    for p in aligned_dir.iterdir():
        if p.is_dir():
            ids.add(p.name)
    return ids


def load_demographics(demo_dir: Path) -> pd.DataFrame:
    """讀取三份 demographics CSV，合併為一個 DataFrame。"""
    dfs = []

    # ACS — 沒有 CDR 欄位
    df_acs = pd.read_csv(demo_dir / "ACS.csv", encoding="utf-8-sig")
    df_acs["group"] = "ACS"
    df_acs.rename(columns={"ACS": "Subject_No"}, inplace=True)
    dfs.append(df_acs)

    # NAD
    df_nad = pd.read_csv(demo_dir / "NAD.csv", encoding="utf-8-sig")
    df_nad["group"] = "NAD"
    df_nad.rename(columns={"NAD": "Subject_No"}, inplace=True)
    dfs.append(df_nad)

    # P
    df_p = pd.read_csv(demo_dir / "P.csv", encoding="utf-8-sig")
    df_p["group"] = "P"
    df_p.rename(columns={"Patient": "Subject_No"}, inplace=True)
    dfs.append(df_p)

    df = pd.concat(dfs, ignore_index=True)
    return df


def parse_subject_id(visit_id: str) -> str:
    """從人次 ID (如 ACS1-2, NAD10-1, P5-3) 萃取個人 ID (ACS1, NAD10, P5)。"""
    m = re.match(r"^([A-Za-z]+\d+)-\d+$", visit_id)
    if m:
        return m.group(1)
    return visit_id


# ---------------------------------------------------------------------------
# 2. 匹配與篩選
# ---------------------------------------------------------------------------

def match_and_filter(
    aligned_ids: set[str], demo: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """以 aligned 資料夾為準，匹配 demographics。回傳匹配後的 DataFrame 及報告。"""
    demo_ids = set(demo["ID"].values)
    matched_ids = aligned_ids & demo_ids
    aligned_only = aligned_ids - demo_ids  # aligned 有但 demographics 沒有
    demo_only = demo_ids - aligned_ids     # demographics 有但 aligned 沒有

    df_matched = demo[demo["ID"].isin(matched_ids)].copy()
    df_matched["subject_id"] = df_matched["ID"].apply(parse_subject_id)

    report = {
        "aligned_total": len(aligned_ids),
        "demo_total": len(demo_ids),
        "matched": len(matched_ids),
        "aligned_only": len(aligned_only),
        "demo_only": len(demo_only),
        "aligned_only_ids": sorted(aligned_only),
        "demo_only_ids_count": len(demo_only),
    }
    return df_matched, report


# ---------------------------------------------------------------------------
# 3. 統計函式
# ---------------------------------------------------------------------------

def _continuous_stats(series: pd.Series) -> dict:
    """計算連續變項的統計量，排除 NaN。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    n_valid = len(s)
    n_missing = len(series) - n_valid
    if n_valid == 0:
        return {
            "n_valid": 0, "n_missing": n_missing,
            "mean": None, "std": None, "median": None,
            "min": None, "max": None,
        }
    return {
        "n_valid": n_valid,
        "n_missing": n_missing,
        "mean": round(s.mean(), 2),
        "std": round(s.std(), 2),
        "median": round(s.median(), 2),
        "min": round(s.min(), 2),
        "max": round(s.max(), 2),
    }


def _categorical_counts(series: pd.Series) -> dict:
    """計算類別變項的次數分佈。"""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    counts = s.value_counts().sort_index()
    n_missing = len(series) - len(s)
    return {"counts": counts.to_dict(), "n_missing": n_missing, "n_valid": len(s)}


def compute_stats_for_subset(df: pd.DataFrame, group_label: str) -> dict:
    """針對一個子集，計算所有指標。"""
    has_cdr = "Global_CDR" in df.columns
    result = {
        "group": group_label,
        "n": len(df),
        "Age": _continuous_stats(df["Age"]),
        "Sex": _categorical_counts(df["Sex"]),
        "MMSE": _continuous_stats(df["MMSE"]),
        "CASI": _continuous_stats(df["CASI"]),
    }
    if has_cdr:
        result["Global_CDR"] = _categorical_counts(df["Global_CDR"])
    else:
        result["Global_CDR"] = {"counts": {}, "n_missing": len(df), "n_valid": 0}
    return result


def get_baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    """取每位個案的最新一次拍照紀錄 (Photo_Session 最大者) 作為人層級資料。"""
    df_sorted = df.sort_values(["subject_id", "Photo_Session"])
    return df_sorted.drop_duplicates(subset="subject_id", keep="last")


# ---------------------------------------------------------------------------
# 4. 格式化輸出
# ---------------------------------------------------------------------------

def _fmt_continuous(stats: dict) -> str:
    if stats["n_valid"] == 0:
        return f"N/A (missing={stats['n_missing']})"
    return (
        f"{stats['mean']} ± {stats['std']}  "
        f"(median={stats['median']}, range={stats['min']}–{stats['max']}, "
        f"n={stats['n_valid']}, missing={stats['n_missing']})"
    )


def _fmt_categorical(stats: dict) -> str:
    parts = [f"{k}: {v}" for k, v in stats["counts"].items()]
    return (
        ", ".join(parts)
        + f"  (n={stats['n_valid']}, missing={stats['n_missing']})"
    )


def _fmt_cdr(stats: dict) -> str:
    if stats["n_valid"] == 0:
        return f"N/A (missing={stats['n_missing']})"
    return _fmt_categorical(stats)


def print_stats_block(title: str, stats_list: list[dict]):
    """印出一個統計區塊 (人次 or 人)。"""
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)

    for s in stats_list:
        logger.info(f"\n--- {s['group']} (n={s['n']}) ---")
        logger.info(f"  Age:        {_fmt_continuous(s['Age'])}")
        logger.info(f"  Sex:        {_fmt_categorical(s['Sex'])}")
        logger.info(f"  MMSE:       {_fmt_continuous(s['MMSE'])}")
        logger.info(f"  CASI:       {_fmt_continuous(s['CASI'])}")
        logger.info(f"  Global_CDR: {_fmt_cdr(s['Global_CDR'])}")

    logger.info("")


# ---------------------------------------------------------------------------
# 5. 匯出 CSV
# ---------------------------------------------------------------------------

def _stats_to_rows(stats_list: list[dict], level: str) -> list[dict]:
    """將統計結果轉成扁平 dict list，方便匯出 CSV。"""
    rows = []
    for s in stats_list:
        row = {"level": level, "group": s["group"], "n": s["n"]}
        for var in ["Age", "MMSE", "CASI"]:
            st = s[var]
            row[f"{var}_n"] = st["n_valid"]
            row[f"{var}_missing"] = st["n_missing"]
            row[f"{var}_mean"] = st["mean"]
            row[f"{var}_std"] = st["std"]
            row[f"{var}_median"] = st["median"]
            row[f"{var}_min"] = st["min"]
            row[f"{var}_max"] = st["max"]
        # Sex
        sex = s["Sex"]
        row["Sex_F"] = sex["counts"].get("F", 0)
        row["Sex_M"] = sex["counts"].get("M", 0)
        row["Sex_missing"] = sex["n_missing"]
        # CDR
        cdr = s["Global_CDR"]
        row["CDR_n"] = cdr["n_valid"]
        row["CDR_missing"] = cdr["n_missing"]
        for k, v in cdr["counts"].items():
            row[f"CDR_{k}"] = v
        rows.append(row)
    return rows


def export_csv(
    visit_stats: list[dict],
    person_stats: list[dict],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _stats_to_rows(visit_stats, "visit") + _stats_to_rows(person_stats, "person")
    df = pd.DataFrame(rows)
    path = output_dir / "demographics_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"統計摘要 CSV 已儲存: {path}")


# ---------------------------------------------------------------------------
# 6. 繪圖
# ---------------------------------------------------------------------------

GROUP_COLORS = {"ACS": "#4CAF50", "NAD": "#2196F3", "P": "#F44336"}
GROUP_LABELS = {"ACS": "ACS (Healthy)", "NAD": "NAD (Non-AD Dementia)", "P": "P (AD Patient)"}


def _plot_continuous_hist_row(
    axes, df: pd.DataFrame, col: str, bins, bin_min, bin_max,
    xlabel: str, ylabel: str,
):
    """在一列 3 個 axes 上畫三組的連續變項直方圖。"""
    groups = ["ACS", "NAD", "P"]
    for ax, grp in zip(axes, groups):
        sub = df[df["group"] == grp]
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()

        ax.hist(
            vals, bins=bins, color=GROUP_COLORS[grp],
            edgecolor="white", linewidth=0.8, alpha=0.85,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        n_missing = len(sub) - len(vals)
        title_stats = f"n={len(vals)} (missing={n_missing})"
        if len(vals) > 0:
            title_stats += f"\nmean={vals.mean():.1f} ± {vals.std():.1f}, median={vals.median():.1f}"
        ax.set_title(f"{GROUP_LABELS[grp]}\n{title_stats}", fontsize=11)
        ax.set_xlim(bin_min, bin_max)
        ax.grid(axis="y", alpha=0.3)


def _plot_cdr_bar_row(axes, df: pd.DataFrame, ylabel: str):
    """在一列 3 個 axes 上畫三組的 CDR 長條圖。"""
    groups = ["ACS", "NAD", "P"]
    cdr_levels = ["0", "0.5", "1", "2", "3"]
    for ax, grp in zip(axes, groups):
        sub = df[df["group"] == grp]
        cdr_vals = pd.to_numeric(sub.get("Global_CDR"), errors="coerce").dropna()
        n_valid = len(cdr_vals)

        counts = [int((cdr_vals == float(lv)).sum()) for lv in cdr_levels]

        bars = ax.bar(cdr_levels, counts, color=GROUP_COLORS[grp],
                      edgecolor="white", linewidth=0.8, alpha=0.85)
        ax.set_xlabel("Global CDR", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        n_missing = len(sub) - n_valid
        ax.set_title(
            f"{GROUP_LABELS[grp]}\nn={n_valid} (missing={n_missing})",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.3)

        # 在每個 bar 上標數字
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(c), ha="center", va="bottom", fontsize=9)


def _plot_2x3(df_visit, df_person, col, xlabel, bins, bin_min, bin_max,
              output_dir, filename):
    """通用的 2x3 連續變項直方圖 (上=visit, 下=person)。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.text(0.02, 0.75, "Visit-level", fontsize=14, fontweight="bold",
             rotation=90, va="center")
    fig.text(0.02, 0.28, "Person-level", fontsize=14, fontweight="bold",
             rotation=90, va="center")

    _plot_continuous_hist_row(axes[0], df_visit, col, bins, bin_min, bin_max,
                             xlabel, "Count (visits)")
    _plot_continuous_hist_row(axes[1], df_person, col, bins, bin_min, bin_max,
                             xlabel, "Count (persons)")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    path = output_dir / filename
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"直方圖已儲存: {path}")


def plot_all_histograms(df_matched: pd.DataFrame, df_baseline: pd.DataFrame, output_dir: Path):
    """繪製 Age, MMSE, CASI, CDR 的分佈直方圖。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Age ---
    all_ages = pd.to_numeric(df_matched["Age"], errors="coerce").dropna()
    age_min = int(np.floor(all_ages.min() / 5) * 5)
    age_max = int(np.ceil(all_ages.max() / 5) * 5)
    _plot_2x3(df_matched, df_baseline, "Age", "Age (years)",
              np.arange(age_min, age_max + 5, 5), age_min, age_max,
              output_dir, "age_distribution_histograms.png")

    # --- MMSE (0-30) ---
    _plot_2x3(df_matched, df_baseline, "MMSE", "MMSE score",
              np.arange(-0.5, 32, 3), -0.5, 31,
              output_dir, "mmse_distribution_histograms.png")

    # --- CASI (0-100) ---
    _plot_2x3(df_matched, df_baseline, "CASI", "CASI score",
              np.arange(-2.5, 105, 10), -2.5, 102.5,
              output_dir, "casi_distribution_histograms.png")

    # --- CDR (categorical bar chart) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.text(0.02, 0.75, "Visit-level", fontsize=14, fontweight="bold",
             rotation=90, va="center")
    fig.text(0.02, 0.28, "Person-level", fontsize=14, fontweight="bold",
             rotation=90, va="center")
    _plot_cdr_bar_row(axes[0], df_matched, "Count (visits)")
    _plot_cdr_bar_row(axes[1], df_baseline, "Count (persons)")
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    path = output_dir / "cdr_distribution_histograms.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"直方圖已儲存: {path}")


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main():
    logger.info("掃描 aligned 資料夾 ...")
    aligned_ids = load_aligned_ids(ALIGNED_DIR)
    logger.info(f"aligned 資料夾個案數 (人次): {len(aligned_ids)}")

    logger.info("讀取 demographics CSV ...")
    demo = load_demographics(DEMO_DIR)
    logger.info(f"demographics 總紀錄數 (人次): {len(demo)}")

    logger.info("匹配中 ...")
    df_matched, report = match_and_filter(aligned_ids, demo)

    # --- 匹配報告 ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("匹配報告")
    logger.info("=" * 70)
    logger.info(f"  aligned 人次數:     {report['aligned_total']}")
    logger.info(f"  demographics 人次數: {report['demo_total']}")
    logger.info(f"  成功匹配人次數:     {report['matched']}")
    logger.info(f"  aligned 有但 demographics 無: {report['aligned_only']}")
    logger.info(f"  demographics 有但 aligned 無: {report['demo_only']}")
    if report["aligned_only"] > 0:
        logger.info(f"  未匹配 aligned IDs (前20): {report['aligned_only_ids'][:20]}")
    logger.info("")

    # --- 人次層級統計 ---
    groups = ["ACS", "NAD", "P"]
    visit_stats = []
    for grp in groups:
        sub = df_matched[df_matched["group"] == grp]
        visit_stats.append(compute_stats_for_subset(sub, grp))
    visit_stats.append(compute_stats_for_subset(df_matched, "Total"))

    print_stats_block("人次層級統計 (每筆 visit 為一單位)", visit_stats)

    # --- 人層級統計 (baseline) ---
    df_baseline = get_baseline_df(df_matched)
    person_stats = []
    for grp in groups:
        sub = df_baseline[df_baseline["group"] == grp]
        person_stats.append(compute_stats_for_subset(sub, grp))
    person_stats.append(compute_stats_for_subset(df_baseline, "Total"))

    print_stats_block("人層級統計 (每人取最新 visit)", person_stats)

    # --- 各組人次數統計 ---
    logger.info("=" * 70)
    logger.info("各組人次分佈")
    logger.info("=" * 70)
    for grp in groups:
        sub = df_matched[df_matched["group"] == grp]
        n_subjects = sub["subject_id"].nunique()
        n_visits = len(sub)
        visits_per_subject = sub.groupby("subject_id").size()
        logger.info(
            f"  {grp}: {n_subjects} 人, {n_visits} 人次, "
            f"每人平均 {visits_per_subject.mean():.2f} 次 "
            f"(range={visits_per_subject.min()}–{visits_per_subject.max()})"
        )
    total_subjects = df_matched["subject_id"].nunique()
    total_visits = len(df_matched)
    vps = df_matched.groupby("subject_id").size()
    logger.info(
        f"  Total: {total_subjects} 人, {total_visits} 人次, "
        f"每人平均 {vps.mean():.2f} 次 "
        f"(range={vps.min()}–{vps.max()})"
    )
    logger.info("")

    # --- 匯出 CSV ---
    export_csv(visit_stats, person_stats, OUTPUT_DIR)

    # --- 繪圖 ---
    plot_all_histograms(df_matched, df_baseline, OUTPUT_DIR)

    logger.info("統計完成。")


if __name__ == "__main__":
    main()
