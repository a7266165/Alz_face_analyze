"""
Plot meta-learner asymmetry variant comparison.

Layout:
    2 columns (full union eval, matched eval) × 3 rows (AUC, BalAcc, MCC)
    3 classifiers as different line styles
    For matched eval: 3 partition lines (HC, NAD, ACS)

Output per (normalize, match_level, eval_unit):
    _summary/{normalize}/{match_level}/{eval_unit}/
        variant_plot_hc_family_{emb}_{match_strategy}.png

Usage:
    conda run -n Alz_face_main_analysis python scripts/meta/plot_meta_variant_comparison.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

from src.config import cohort_dirs

COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
VISIT_DIR, CDR_MMSE_DIR = cohort_dirs(*COHORT)
BG_MODE = "background"
PHOTO_MODE = "mean"
REDUCER = "no_drop"

META_ROOT = PROJECT_ROOT / "workspace" / "meta" / "analysis"

VARIANT_ORDER = [
    "difference",
    "absolute_difference",
    "relative_differences",
    "absolute_relative_differences",
]
VARIANT_SHORT = {
    "difference": "diff",
    "absolute_difference": "|diff|",
    "relative_differences": "rel_diff",
    "absolute_relative_differences": "|rel_diff|",
}

PARTITION_STYLE = {
    "ad_vs_hc": {"label": "HC (NAD+ACS)", "color": "#1f77b4", "marker": "o"},
    "ad_vs_nad": {"label": "NAD only", "color": "#ff7f0e", "marker": "s"},
    "ad_vs_acs": {"label": "ACS only", "color": "#2ca02c", "marker": "^"},
}

CLF_LINESTYLE = {
    "tabpfn": {"ls": "-", "label": "TabPFN"},
    "logistic": {"ls": "--", "label": "LR"},
    "xgboost": {"ls": ":", "label": "XGB"},
}

METRICS = ["auc", "balacc", "mcc"]
METRIC_LABELS = {"auc": "AUC", "balacc": "BalAcc", "mcc": "MCC"}
METRIC_YLIM = {"auc": (0.3, 0.85), "balacc": (0.5, 0.8), "mcc": (0.1, 0.6)}


def load_data():
    cohort_dir = META_ROOT / VISIT_DIR / CDR_MMSE_DIR

    matched_path = cohort_dir / "summary_all_metrics.csv"
    matched_df = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()

    full_rows = []
    for summary in (cohort_dir / BG_MODE).rglob("summary.csv"):
        fwd_dir = summary.parent
        if fwd_dir.name == "_summary":
            continue
        parts = fwd_dir.relative_to(cohort_dir / BG_MODE).parts
        # emb / asym / photo / reducer / base_clf / base_param / fwd / norm / clf
        if len(parts) < 9:
            continue
        emb, asym, photo, reducer, base_clf, base_param, _, norm, clf = parts[:9]
        df = pd.read_csv(summary)
        df["asymmetry_variant"] = asym
        df["normalize"] = norm
        df["balacc"] = (df["sensitivity"] + df["specificity"]) / 2
        full_rows.append(df)

    full_df = pd.concat(full_rows, ignore_index=True) if full_rows else pd.DataFrame()
    return matched_df, full_df


def plot_single_clf(
    matched_sub, full_sub, clf_name,
    emb_model, norm_tag, match_strategy, out_dir,
):
    clf_full = full_sub[full_sub["meta_classifier"] == clf_name]
    clf_matched = matched_sub[matched_sub["meta_classifier"] == clf_name]

    if clf_full.empty and clf_matched.empty:
        return

    clf_label = CLF_LINESTYLE.get(clf_name, {"label": clf_name})["label"]

    fig, axes = plt.subplots(
        len(METRICS), 2,
        figsize=(9, 9.5), sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.28},
    )

    x_ticks = np.arange(len(VARIANT_ORDER))
    x_labels = [VARIANT_SHORT[v] for v in VARIANT_ORDER]

    # ---- Left column: full union eval ----
    for row_idx, metric in enumerate(METRICS):
        ax = axes[row_idx, 0]
        vals = []
        for variant in VARIANT_ORDER:
            row = clf_full[clf_full["asymmetry_variant"] == variant]
            vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)

        ax.plot(
            x_ticks, vals,
            color="#555555", marker="*", markersize=7,
            linewidth=1.5, label="full union",
        )
        for xi, v in zip(x_ticks, vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.3f}", (xi, v),
                    textcoords="offset points", xytext=(-5, 6),
                    fontsize=7, color="#555555", fontweight="bold",
                )

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.set_title("full union eval", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
        if row_idx == len(METRICS) - 1:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=8)

    # ---- Right column: matched eval ----
    for row_idx, metric in enumerate(METRICS):
        ax = axes[row_idx, 1]

        for partition, pstyle in PARTITION_STYLE.items():
            part_df = clf_matched[clf_matched["partition"] == partition]
            vals = []
            ci_low_vals = []
            ci_high_vals = []

            for variant in VARIANT_ORDER:
                row = part_df[part_df["asymmetry_variant"] == variant]
                if row.empty:
                    vals.append(np.nan)
                    ci_low_vals.append(np.nan)
                    ci_high_vals.append(np.nan)
                else:
                    vals.append(float(row[metric].iloc[0]))
                    if metric == "auc":
                        cl = row["auc_ci_low"].iloc[0]
                        ch = row["auc_ci_high"].iloc[0]
                        ci_low_vals.append(float(cl) if pd.notna(cl) else np.nan)
                        ci_high_vals.append(float(ch) if pd.notna(ch) else np.nan)

            ax.plot(
                x_ticks, vals,
                color=pstyle["color"], marker=pstyle["marker"],
                markersize=5, linewidth=1.5, label=pstyle["label"],
            )

            if metric == "auc" and any(not np.isnan(v) for v in ci_low_vals):
                ax.fill_between(
                    x_ticks, ci_low_vals, ci_high_vals,
                    color=pstyle["color"], alpha=0.12,
                )

            for xi, v in zip(x_ticks, vals):
                if not np.isnan(v):
                    ax.annotate(
                        f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(-5, 6),
                        fontsize=7, color=pstyle["color"],
                    )

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.set_title("matched eval", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
        if row_idx == len(METRICS) - 1:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=8)

    for row_idx, metric in enumerate(METRICS):
        ylim = METRIC_YLIM[metric]
        axes[row_idx, 0].set_ylim(ylim)
        axes[row_idx, 1].set_ylim(ylim)

    match_short = match_strategy.replace("match_", "")
    fig.suptitle(
        f"HC family — {emb_model} / {clf_label} / {norm_tag} ({match_short})",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"variant_plot_hc_family_{emb_model}_{clf_name}_{match_short}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / fname}")


def plot_variant_comparison(
    matched_df, full_df, emb_model, norm_tag,
    match_level, eval_unit, match_strategy, out_dir,
):
    matched_sub = matched_df[
        (matched_df["emb_model"] == emb_model)
        & (matched_df["normalize"] == norm_tag)
        & (matched_df["match_level"] == match_level)
        & (matched_df["eval_unit"] == eval_unit)
        & (matched_df["match_strategy"] == match_strategy)
        & (matched_df["eval_strategy"] == "1by1matched")
    ].copy()

    full_sub = full_df[
        (full_df["emb_model"] == emb_model)
        & (full_df["normalize"] == norm_tag)
    ].copy()

    if matched_sub.empty and full_sub.empty:
        return

    classifiers = sorted(full_sub["meta_classifier"].unique()) if not full_sub.empty else []
    for clf_name in classifiers:
        plot_single_clf(
            matched_sub, full_sub, clf_name,
            emb_model, norm_tag, match_strategy, out_dir,
        )


def main():
    matched_df, full_df = load_data()

    summary_root = META_ROOT / VISIT_DIR / CDR_MMSE_DIR / "_summary"

    emb_models = full_df["emb_model"].unique() if not full_df.empty else []
    norm_tags = sorted(full_df["normalize"].unique()) if not full_df.empty else []
    match_levels = matched_df["match_level"].unique() if not matched_df.empty else []
    eval_units = matched_df["eval_unit"].unique() if not matched_df.empty else []

    match_strategy = "priority_acs"

    for emb in emb_models:
        for norm in norm_tags:
            for ml in match_levels:
                for eu in eval_units:
                    out_dir = summary_root / norm / ml / eu
                    plot_variant_comparison(
                        matched_df, full_df, emb, norm, ml, eu,
                        match_strategy, out_dir,
                    )


if __name__ == "__main__":
    main()
