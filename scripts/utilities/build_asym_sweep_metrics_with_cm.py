"""
Build per-cell `all_metrics_with_cm.csv` for an embedding sweep cohort.

Walks the per-cell metrics.json tree and aggregates into the verbose-scope
schema used by plot_pca_components_sweep.py. Does NOT rely on
combined_summary.csv (which gets overwritten by partial re-runs).

For each (reducer, variant) cell tree:
    fwd/<part>/<emb>/<clf>/forward_matched_metrics.json
        -> rows for scope ∈ {forward_matched, forward_full}
    rev/<part>/<emb>/<clf>/ensemble/metrics.json
        -> rows for scope ∈ {reverse_ensemble_matched_oof,
                             reverse_ensemble_full,
                             reverse_ensemble_unmatched}
    rev/<part>/<emb>/<clf>/single/metrics.json
        -> rows for scope ∈ {reverse_single_matched_train,
                             reverse_single_unmatched}

Output:
    embedding_classification/<reducer>/_summary/all_metrics_with_cm.csv
    embedding_asymmetry_classification/<reducer>/<variant>/_summary/all_metrics_with_cm.csv

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/utilities/build_asym_sweep_metrics_with_cm.py \
        --cohort-mode p_first_hc_all
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common metric fields to copy from each block (when present).
METRIC_FIELDS = [
    "n", "n_pos", "n_neg",
    "auc", "auc_ci_low", "auc_ci_high",
    "balacc", "mcc", "f1", "sens", "spec",
]

# Block extractors per scope: (file rel path, json key, strategy, method)
EXTRACTORS = [
    ("fwd",            "forward_matched_metrics.json", "metrics_matched_subset",
     "forward_matched", "forward", ""),
    ("fwd",            "forward_matched_metrics.json", "metrics_full_cohort",
     "forward_full", "forward", ""),
    ("rev/ensemble",   "metrics.json", "metrics_matched_oof",
     "reverse_ensemble_matched_oof", "reverse", "ensemble"),
    ("rev/ensemble",   "metrics.json", "metrics_full",
     "reverse_ensemble_full", "reverse", "ensemble"),
    ("rev/ensemble",   "metrics.json", "metrics_unmatched",
     "reverse_ensemble_unmatched", "reverse", "ensemble"),
    ("rev/single",     "metrics.json", "metrics_matched_train",
     "reverse_single_matched_train", "reverse", "single"),
    ("rev/single",     "metrics.json", "metrics_unmatched",
     "reverse_single_unmatched", "reverse", "single"),
]


def _row_from_block(block, partition, emb, clf, strategy, method, scope):
    if not block:
        return None
    row = {
        "partition": partition,
        "embedding": emb,
        "classifier": clf,
        "strategy": strategy,
        "scope": scope,
    }
    for f in METRIC_FIELDS:
        row[f] = block.get(f)
    cm = block.get("confusion_matrix")
    if cm is not None and len(cm) == 2 and len(cm[0]) == 2:
        row["TN"] = int(cm[0][0])
        row["FP"] = int(cm[0][1])
        row["FN"] = int(cm[1][0])
        row["TP"] = int(cm[1][1])
    else:
        row["TN"] = row["FP"] = row["FN"] = row["TP"] = pd.NA
    row["method"] = method
    return row


def _scan_cells(variant_dir):
    """Walk the cell tree under variant_dir and return rows per scope."""
    rows = []
    for fwd_or_rev in ("fwd", "rev"):
        top = variant_dir / fwd_or_rev
        if not top.is_dir():
            continue
        for partition_d in sorted(top.iterdir()):
            if not partition_d.is_dir():
                continue
            for emb_d in sorted(partition_d.iterdir()):
                if not emb_d.is_dir():
                    continue
                for clf_d in sorted(emb_d.iterdir()):
                    if not clf_d.is_dir():
                        continue
                    partition = partition_d.name
                    emb = emb_d.name
                    clf = clf_d.name
                    for (rel, fname, key, scope, strategy, method) in EXTRACTORS:
                        if not rel.startswith(fwd_or_rev):
                            continue
                        rest = rel[len(fwd_or_rev) + 1:] if "/" in rel else ""
                        json_path = clf_d / rest / fname if rest else clf_d / fname
                        if not json_path.exists():
                            continue
                        try:
                            d = json.loads(json_path.read_text())
                        except (ValueError, json.JSONDecodeError):
                            continue
                        block = d.get(key)
                        # Wilcoxon stats (forward only)
                        wilc = d.get("paired_wilcoxon", {}) if scope.startswith("forward") else {}
                        row = _row_from_block(block, partition, emb, clf,
                                                 strategy, method, scope)
                        if row is None:
                            continue
                        if wilc:
                            row["wilcoxon_W"] = wilc.get("W")
                            row["wilcoxon_p"] = wilc.get("p")
                            row["n_pairs"] = wilc.get("n_pairs")
                            row["mean_diff"] = wilc.get("mean_diff")
                        rows.append(row)
    return rows


def build_one(variant_dir):
    rows = _scan_cells(variant_dir)
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # Order columns sensibly
    col_order = (["partition", "embedding", "classifier", "strategy", "scope"]
                 + METRIC_FIELDS
                 + ["TN", "FP", "FN", "TP",
                    "wilcoxon_W", "wilcoxon_p", "n_pairs", "mean_diff",
                    "method"])
    cols = [c for c in col_order if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    df = df[cols + extras]
    out = variant_dir / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_metrics_with_cm.csv", index=False)
    return len(df)


def walk_root(root, label):
    if not root.exists():
        return 0, 0
    n_files, n_rows = 0, 0
    for reducer in sorted(root.iterdir()):
        if not reducer.is_dir() or reducer.name.startswith("_"):
            continue
        # Determine if this is the "original" layout (cells direct under reducer)
        # or the asymmetry layout (cells under variant subdir).
        if (reducer / "fwd").is_dir() or (reducer / "rev").is_dir():
            candidates = [reducer]
        else:
            candidates = [v for v in sorted(reducer.iterdir())
                           if v.is_dir() and ((v / "fwd").is_dir() or (v / "rev").is_dir())]
        for vdir in candidates:
            n = build_one(vdir)
            if n > 0:
                n_files += 1
                n_rows += n
                rel = vdir.relative_to(root)
                print(f"  {label}/{rel}: wrote {n} rows")
    return n_files, n_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort-mode", default="default",
                    choices=["default", "p_first_hc_all"])
    args = p.parse_args()
    cohort_dir = ("p_first_hc_all" if args.cohort_mode == "p_first_hc_all"
                  else "p_first_hc_strict")
    base = PROJECT_ROOT / "workspace" / "arms_analysis" / cohort_dir

    total_files, total_rows = 0, 0
    for sub in ("embedding_classification", "embedding_asymmetry_classification"):
        f, r = walk_root(base / sub, sub)
        total_files += f
        total_rows += r
    print(f"\nTOTAL: {total_files} files, {total_rows} rows")


if __name__ == "__main__":
    main()
