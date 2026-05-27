"""
Build hierarchical _summary.csv for meta analysis eval chain.

Walks all metrics.json under the eval chain tree and:
  1. Builds a flat all_metrics.csv at fwd/_summary/
  2. Saves _summary.csv at each intermediate directory level
     (match_strategy, eval_unit, match_level, eval_strategy)

Directory structure:
  .../tabpfn/fwd/
    {eval_strategy}/{match_level}/{eval_unit}/{match_strategy}/{partition}/
      metrics.json

Usage:
    conda run -n Alz_face_main_analysis python scripts/meta/build_meta_summary.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

from src.config import cohort_name, cohort_spec_from_name

# ========== 設定 ==========

COHORT_MODE = "p_first_cdrall_hc_all_cdrall_or_mmseall"
BG_MODE = "background"
PHOTO_MODE = "mean"
REDUCER = "no_drop"

EMB_MODELS = ["arcface"]
META_CLASSIFIERS = ["tabpfn", "logistic", "xgboost"]
NORMALIZE_OPTIONS = ["raw", "minmax"]
BASE_CLASSIFIER = "logistic"
BASE_CLASSIFIER_PARAM = "C_1"
ASYMMETRY_VARIANTS = [
    "difference",
    "absolute_difference",
    "relative_differences",
    "absolute_relative_differences",
]

META_ROOT = PROJECT_ROOT / "workspace" / "meta" / "analysis"

EVAL_CHAIN_LEVELS = [
    "eval_strategy",
    "match_level",
    "eval_unit",
    "match_strategy",
    "partition",
]

METRIC_FIELDS = [
    "n", "n_pos", "n_neg",
    "auc", "auc_ci_low", "auc_ci_high",
    "balacc", "mcc", "f1", "sens", "spec",
]

CM_FIELDS = ["TN", "FP", "FN", "TP"]

WILCOXON_FIELDS = ["wilcoxon_W", "wilcoxon_p", "n_pairs", "mean_diff"]

OUTPUT_COLUMNS = (
    ["partition", "emb_model", "meta_classifier",
     "eval_strategy", "match_level", "eval_unit", "match_strategy"]
    + METRIC_FIELDS + CM_FIELDS + WILCOXON_FIELDS
)


def parse_metrics_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    row = {}
    for key in ["partition", "eval_strategy", "match_level",
                "eval_unit", "match_strategy", "emb_model", "meta_classifier"]:
        row[key] = data.get(key)

    block = data.get("metrics_matched") or {}
    for field in METRIC_FIELDS:
        row[field] = block.get(field)

    cm = block.get("confusion_matrix")
    if cm and len(cm) == 2 and all(len(r) == 2 for r in cm):
        row["TN"], row["FP"] = cm[0]
        row["FN"], row["TP"] = cm[1]
    else:
        for f in CM_FIELDS:
            row[f] = None

    paired = data.get("paired_wilcoxon") or {}
    row["wilcoxon_W"] = paired.get("W")
    row["wilcoxon_p"] = paired.get("p")
    row["n_pairs"] = paired.get("n_pairs")
    row["mean_diff"] = paired.get("mean_diff")

    return row


def collect_all_metrics(fwd_dir: Path) -> pd.DataFrame:
    rows = []
    for mj in fwd_dir.rglob("metrics.json"):
        rel = mj.relative_to(fwd_dir)
        parts = rel.parts
        if len(parts) != 6:
            continue
        try:
            row = parse_metrics_json(mj)
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].sort_values(
        ["partition", "eval_strategy", "match_level",
         "eval_unit", "match_strategy"]
    ).reset_index(drop=True)


def save_hierarchical_summaries(fwd_dir: Path, full_df: pd.DataFrame):
    summary_dir = fwd_dir / "_summary"
    summary_dir.mkdir(exist_ok=True)
    full_df.to_csv(
        summary_dir / "all_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"  [all] {summary_dir / 'all_metrics.csv'} ({len(full_df)} rows)")

    level_cols = list(EVAL_CHAIN_LEVELS)
    for depth in range(len(level_cols)):
        group_keys = level_cols[: depth + 1]
        remaining = level_cols[depth + 1:]

        for vals, sub_df in full_df.groupby(group_keys, dropna=False):
            if not isinstance(vals, tuple):
                vals = (vals,)

            rel_path = Path(*vals)
            target_dir = fwd_dir / rel_path
            if not target_dir.exists():
                continue

            out_path = target_dir / "_summary.csv"
            sub_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  [{'/'.join(group_keys[-1:])}] {out_path} ({len(sub_df)} rows)")


def main():
    spec = cohort_spec_from_name(cohort_name(COHORT_MODE))

    for norm_tag in NORMALIZE_OPTIONS:
        for asym_variant in ASYMMETRY_VARIANTS:
            for emb_model in EMB_MODELS:
                for meta_clf in META_CLASSIFIERS:
                    fwd_dir = (
                        META_ROOT / spec.visit_dir / spec.cdr_mmse_dir
                        / BG_MODE / emb_model / asym_variant
                        / PHOTO_MODE / REDUCER
                        / BASE_CLASSIFIER / BASE_CLASSIFIER_PARAM / "fwd"
                        / norm_tag / meta_clf
                    )
                    if not fwd_dir.exists():
                        print(f"SKIP {fwd_dir}")
                        continue

                    print(f"\n=== {norm_tag} / {emb_model} / {asym_variant} / {meta_clf} ===")
                    full_df = collect_all_metrics(fwd_dir)
                    if full_df.empty:
                        print("  (no metrics found)")
                        continue

                    save_hierarchical_summaries(fwd_dir, full_df)

    combined_root = META_ROOT / spec.visit_dir / spec.cdr_mmse_dir
    all_rows = []
    for summary_file in combined_root.rglob("_summary/all_metrics.csv"):
        fwd_dir = summary_file.parent.parent
        rel = fwd_dir.relative_to(combined_root)
        parts = rel.parts
        # bg / emb / asym / photo / reducer / base_clf / base_param / fwd / norm / clf
        if len(parts) < 10:
            continue
        bg, emb, asym, photo, reducer, base_clf, base_param, _, norm, clf = parts[:10]
        df = pd.read_csv(summary_file)
        df["asymmetry_variant"] = asym
        df["normalize"] = norm
        all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        cols_front = ["normalize", "asymmetry_variant"] + OUTPUT_COLUMNS
        extra = [c for c in combined.columns if c not in cols_front]
        combined = combined[cols_front + extra]
        combined = combined.sort_values(["normalize", "asymmetry_variant", "partition", "eval_strategy"])
        out = combined_root / "summary_all_metrics.csv"
        combined.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n=== Combined: {out} ({len(combined)} rows) ===")


if __name__ == "__main__":
    main()
