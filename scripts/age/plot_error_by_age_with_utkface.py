"""
在原 error_by_age_combined.png 基礎上，把 E-ACS 的 UTKFace 子集
加成第四條線（與 ACS/NAD/P 並列，使用相同的 bootstrap 平均係數校正）。

校正公式（與 plot_predicted_ages.plot_error_by_age_combined 一致）：
    corrected = predicted + (a_mean * real_age + b_mean)
    error_after = real - corrected
a_mean, b_mean 取自 bootstrap_coefficients.csv 的 mean 列。

輸出：<output-dir>/error_by_age/{lines,merged}_<suffix>.png
       <output-dir>/error_by_age/sw10/...
       <output-dir>/residual_by_age/{lines,merged}_<suffix>.png
       <output-dir>/residual_by_age/sw10/...

預設讀 BOOTSTRAP_DIR / data / {bootstrap_coefficients,corrected_ages}.csv，
寫到 AGE_PREDICTION_DIR/{error_by_age,residual_by_age}/。可用
--bootstrap-dir / --output-dir 切換到 V2 路徑（age_prediction_p_first_hc_all）。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_PREDICTION_DIR,
    BOOTSTRAP_DIR,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLORS = {
    "ACS":       "#4CAF50",  # green
    "NAD":       "#2196F3",  # blue
    "P":         "#F44336",  # red
    "UTKFace":   "#9C27B0",  # purple
    "AgeDB":     "#17becf",  # cyan
    "APPA-REAL": "#bcbd22",  # olive
}


def load_bootstrap_mean_coefs(coefs_csv: Path) -> tuple[float, float]:
    df = pd.read_csv(coefs_csv, encoding="utf-8-sig")
    mean_row = df[df["iter"].astype(str) == "mean"].iloc[0]
    a, b = float(mean_row["a"]), float(mean_row["b"])
    logger.info(f"bootstrap 平均係數: a={a:.4f}, b={b:.4f}")
    return a, b


def load_corrected_internal(csv_path: Path) -> pd.DataFrame:
    """讀取 ACS/NAD/P 的 bootstrap 校正結果。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"internal corrected: {len(df)} rows "
                f"(ACS={int((df['group']=='ACS').sum())}, "
                f"NAD={int((df['group']=='NAD').sum())}, "
                f"P={int((df['group']=='P').sum())})")
    return df


def build_external_corrected(
    source: str, a: float, b: float, preds: dict | None = None
) -> pd.DataFrame:
    """載入指定 EACS Source 的 subjects，套用 bootstrap 平均係數校正。"""
    demo = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv", encoding="utf-8-sig")
    demo = demo[demo["Source"] == source].copy()
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")

    if preds is None:
        with open(PREDICTED_AGES_FILE, "r", encoding="utf-8") as f:
            preds = json.load(f)

    demo["predicted_age"] = demo["ID"].map(preds)
    demo = demo.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)

    real = demo["Age"].astype(float)
    pred = demo["predicted_age"].astype(float)
    corrected = pred + (a * real + b)

    out = pd.DataFrame({
        "ID": demo["ID"],
        "group": source,
        "real_age": real,
        "predicted_age": pred,
        "corrected_age": corrected,
        "error_before": real - pred,
        "error_after": real - corrected,
        "age_int": real.astype(int),
    })
    logger.info(f"{source} corrected: {len(out)} rows "
                f"(age {out['real_age'].min():.0f}-{out['real_age'].max():.0f})")
    return out


# 向下相容
def build_utkface_corrected(a: float, b: float) -> pd.DataFrame:
    return build_external_corrected("UTKFace", a, b)


def plot_sliding_window(
    df_all: pd.DataFrame,
    output_path: Path,
    groups: list[str],
    title: str,
    ylabel: str,
    y_col: str,
    group_labels: dict[str, str] | None = None,
    window: int = 10,
    step: int = 1,
    min_count: int = 5,
    xlim: tuple[int, int] = (50, 100),
):
    """10 年滑動視窗版本：每個 x 點代表 [x-w/2, x+w/2) 範圍內 subjects 的 mean±std。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = group_labels or {}

    lo_x, hi_x = xlim
    starts = np.arange(lo_x, hi_x - window + 1 + step, step)

    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if len(sub) == 0:
            continue

        xs, ms, ss = [], [], []
        real = sub["real_age"].values
        vals = sub[y_col].values
        for s in starts:
            mask = (real >= s) & (real < s + window)
            if mask.sum() < min_count:
                continue
            xs.append(s + window / 2.0)
            ms.append(vals[mask].mean())
            ss.append(vals[mask].std(ddof=0))

        if not xs:
            continue
        xs = np.array(xs)
        ms = np.array(ms)
        ss = np.array(ss)

        display = labels.get(grp, grp)
        ax.plot(xs, ms, color=color, linewidth=2,
                marker="o", markersize=4, label=f"{display} (n={len(sub)})")
        ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"True Age (y) — window center ({window}-y sliding)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(lo_x, hi_x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")


def plot_combined(
    df_all: pd.DataFrame,
    output_path: Path,
    groups: list[str],
    title: str,
    ylabel: str = "Error (after bootstrap correction)",
    y_col: str = "error_after",
    group_labels: dict[str, str] | None = None,
):
    """以指定 groups 畫多條 error-by-age 折線。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = group_labels or {}

    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if len(sub) == 0:
            continue
        st = sub.groupby("age_int")[y_col].agg(["mean", "std", "count"])
        st = st[st["count"] >= 3].sort_index()
        if len(st) == 0:
            continue

        ages = st.index.values
        means = st["mean"].values
        stds = st["std"].fillna(0).values

        display = labels.get(grp, grp)
        label_name = f"{display} (n={len(sub)})"
        ax.plot(ages, means, color=color, linewidth=2,
                marker="o", markersize=4, label=label_name)
        ax.fill_between(ages, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (y)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")


EXTERNAL_SOURCES = ["AgeDB", "APPA-REAL", "UTKFace"]


def combo_suffix(sources: tuple) -> str:
    """('AgeDB','APPA-REAL') → 'agedb_appareal'。空 tuple → 'internal'。"""
    if not sources:
        return "internal"
    return "_".join(
        s.lower().replace("-", "") for s in sorted(sources, key=str.lower)
    )


def combo_label(sources: tuple) -> str:
    """('AgeDB','APPA-REAL') → 'AgeDB + APPA-REAL'。空 tuple → ''。"""
    return " + ".join(sorted(sources, key=str.lower))


def all_combos() -> list:
    """返回 8 個 source combinations（含空集）。"""
    from itertools import combinations
    return [()] + [tuple(c) for k in (1, 2, 3)
                   for c in combinations(EXTERNAL_SOURCES, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dir", type=Path, default=BOOTSTRAP_DIR,
                        help="Directory holding data/bootstrap_coefficients.csv "
                             "+ data/corrected_ages.csv (output of "
                             "age_error_bootstrap_correction.py).")
    parser.add_argument("--output-dir", type=Path, default=AGE_PREDICTION_DIR,
                        help="Root dir under which error_by_age/ and "
                             "residual_by_age/ are written.")
    args = parser.parse_args()

    DIR_ERR = args.output_dir / "error_by_age"
    DIR_ERR_SW = DIR_ERR / "sw10"
    DIR_RES = args.output_dir / "residual_by_age"
    DIR_RES_SW = DIR_RES / "sw10"

    logger.info(f"bootstrap-dir = {args.bootstrap_dir}")
    logger.info(f"output-dir    = {args.output_dir}")

    a, b = load_bootstrap_mean_coefs(args.bootstrap_dir / "data" / "bootstrap_coefficients.csv")
    df_internal = load_corrected_internal(args.bootstrap_dir / "data" / "corrected_ages.csv")

    with open(PREDICTED_AGES_FILE, "r", encoding="utf-8") as f:
        preds = json.load(f)

    keep_cols = ["group", "real_age", "predicted_age", "corrected_age",
                 "error_before", "error_after", "age_int"]

    df_internal_only = df_internal[keep_cols]
    df_externals = {
        src: build_external_corrected(src, a, b, preds)[keep_cols]
        for src in EXTERNAL_SOURCES
    }

    # 兩種輸出設定：(corrected, uncorrected)
    settings = [
        # (dir_per_year, dir_sw,     ylabel,                                   y_col,         title_state)
        (DIR_ERR,        DIR_ERR_SW,
         "Error (after bootstrap correction)",     "error_after",
         "Corrected Error"),
        (DIR_RES,        DIR_RES_SW,
         "Residual ε = y − predicted",             "error_before",
         "Prediction Residual (Before Correction)"),
    ]

    for combo in all_combos():
        # 構造 df：internal + 該組合內的 external
        parts = [df_internal_only]
        for src in combo:
            parts.append(df_externals[src])
        df_combo = pd.concat(parts, ignore_index=True)

        groups_lines = ["ACS", "NAD", "P"] + sorted(combo, key=str.lower)
        suffix = combo_suffix(combo)
        label = combo_label(combo)

        # Merged 版：把外部 groups 改寫成 "ACS"
        df_merged = df_combo.copy()
        if combo:
            df_merged.loc[df_merged["group"].isin(combo), "group"] = "ACS"
        merged_label = f"ACS + {label}" if combo else "ACS"

        for dir_py, dir_sw, ylabel, y_col, title_state in settings:
            # ---- lines view ----
            title_l = (f"{title_state} by True Age — "
                       f"{' / '.join(groups_lines)} (mean ± std)")
            plot_combined(
                df_combo, dir_py / f"lines_{suffix}.png",
                groups=groups_lines, title=title_l,
                ylabel=ylabel, y_col=y_col,
            )
            plot_sliding_window(
                df_combo, dir_sw / f"lines_{suffix}_sw10.png",
                groups=groups_lines,
                title=(f"{title_state} by True Age — "
                       f"{' / '.join(groups_lines)} "
                       f"(10-y sliding window, mean ± std)"),
                ylabel=ylabel, y_col=y_col,
            )

            # ---- merged view（僅當 combo 非空時有意義） ----
            if combo:
                title_m = (f"{title_state} by True Age — "
                           f"{merged_label} / NAD / P (mean ± std)")
                plot_combined(
                    df_merged, dir_py / f"merged_{suffix}.png",
                    groups=["ACS", "NAD", "P"], title=title_m,
                    ylabel=ylabel, y_col=y_col,
                    group_labels={"ACS": merged_label},
                )
                plot_sliding_window(
                    df_merged, dir_sw / f"merged_{suffix}_sw10.png",
                    groups=["ACS", "NAD", "P"],
                    title=(f"{title_state} by True Age — "
                           f"{merged_label} / NAD / P "
                           f"(10-y sliding window, mean ± std)"),
                    ylabel=ylabel, y_col=y_col,
                    group_labels={"ACS": merged_label},
                )


if __name__ == "__main__":
    main()
