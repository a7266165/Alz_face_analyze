"""
Build _summary for meta analysis (v3).

Walks all metrics.json under the eval chain tree and produces:
  1. _summary/all_metrics.csv  — one big flat table (all combos × partitions)
  2. Hierarchical _summary.csv at each intermediate level

Path structure (v3):
  {visit}/{cdr}/{bg_mode}/{extra_feat}/{emb_model}/{asym_variant}/{scoring_method}/
    {photo}/{reducer}/fwd/{normalize}/{meta_clf}/
      1by1matched/subject_match/eval_by_subject/{match_strategy}/{partition}/
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

from src.config import cohort_dirs

COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
META_ROOT = PROJECT_ROOT / "workspace" / "meta" / "analysis"

METRIC_FIELDS = [
    "n", "n_pos", "n_neg",
    "auc", "auc_ci_low", "auc_ci_high",
    "balacc", "mcc", "f1", "sens", "spec",
]
CM_FIELDS = ["TN", "FP", "FN", "TP"]
WILCOXON_FIELDS = ["wilcoxon_W", "wilcoxon_p", "n_pairs", "mean_diff"]

SWEEP_COLS = [
    "bg_mode", "extra_features", "emb_model",
    "asymmetry_variant", "scoring_method",
    "normalize", "meta_classifier",
]
EVAL_COLS = [
    "eval_strategy", "match_level", "eval_unit",
    "match_strategy", "partition",
]
OUTPUT_COLUMNS = SWEEP_COLS + EVAL_COLS + METRIC_FIELDS + CM_FIELDS + WILCOXON_FIELDS


def parse_metrics_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    row = {}
    for key in EVAL_COLS:
        row[key] = data.get(key)
    row["emb_model"] = data.get("emb_model")
    row["meta_classifier"] = data.get("meta_classifier")

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


def extract_sweep_dims(metrics_path: Path, cohort_root: Path) -> dict:
    """Extract sweep dimensions from the file path."""
    rel = metrics_path.relative_to(cohort_root)
    parts = rel.parts
    # bg_mode/extra_feat/emb_model/asym_variant/scoring_method/
    #   photo/reducer/fwd/normalize/meta_clf/
    #     eval_strategy/match_level/eval_unit/match_strategy/partition/metrics.json
    # indices: 0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15
    if len(parts) < 16:
        return None
    return {
        "bg_mode": parts[0],
        "extra_features": parts[1],
        "emb_model": parts[2],
        "asymmetry_variant": parts[3],
        "scoring_method": parts[4],
        "normalize": parts[8],
        "meta_classifier": parts[9],
    }


def collect_all_metrics(cohort_root: Path) -> pd.DataFrame:
    rows = []
    for mj in cohort_root.rglob("metrics.json"):
        sweep = extract_sweep_dims(mj, cohort_root)
        if sweep is None:
            continue
        try:
            row = parse_metrics_json(mj)
            row.update(sweep)
            rows.append(row)
        except Exception as e:
            print(f"  WARN: {mj}: {e}")
            continue

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].sort_values(
        SWEEP_COLS + ["partition"]
    ).reset_index(drop=True)


def save_summaries(cohort_root: Path, full_df: pd.DataFrame):
    summary_dir = cohort_root / "_summary"
    summary_dir.mkdir(exist_ok=True)

    out_path = summary_dir / "all_metrics.csv"
    full_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [all] {out_path} ({len(full_df)} rows)")

    for col in SWEEP_COLS:
        by_dir = summary_dir / f"by_{col}"
        by_dir.mkdir(parents=True, exist_ok=True)
        for val, sub_df in full_df.groupby(col, dropna=False):
            sub_df.to_csv(by_dir / f"{val}.csv", index=False, encoding="utf-8-sig")
        print(f"  [by_{col}] {len(full_df[col].unique())} files")


def main():
    visit_dir, cdr_mmse_dir = cohort_dirs(*COHORT)
    cohort_root = META_ROOT / visit_dir / cdr_mmse_dir

    print(f"Scanning: {cohort_root}")
    full_df = collect_all_metrics(cohort_root)

    if full_df.empty:
        print("No metrics found!")
        return

    print(f"Found {len(full_df)} metric rows")
    save_summaries(cohort_root, full_df)

    print(f"\nTop 10 by MCC (ad_vs_hc / priority_acs):")
    mask = (full_df["partition"] == "ad_vs_hc") & (full_df["match_strategy"] == "priority_acs")
    top = full_df[mask].nlargest(10, "mcc")
    for _, r in top.iterrows():
        print(f"  {r['bg_mode']}/{r['extra_features']}/{r['emb_model']}/"
              f"{r['asymmetry_variant']}/{r['scoring_method']}/"
              f"{r['normalize']}/{r['meta_classifier']}: "
              f"MCC={r['mcc']:.4f}, AUC={r['auc']:.4f}")


if __name__ == "__main__":
    main()
