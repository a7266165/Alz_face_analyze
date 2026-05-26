"""
Build per-cell `all_metrics_with_cm.csv` for an embedding sweep cohort.

Walks the per-cell metrics.json tree and aggregates into the long-form schema
consumed by plot_pca_*.py / plot_dropcorr_*.py. Does NOT rely on
combined_summary.csv (which gets overwritten by partial re-runs).

Layout:
    embedding/analysis/classification/<variant>/<cohort>/<reducer_path>/<part>/{fwd,rev}/<emb>/<clf>[/C_X]/

`<reducer_path>` is one of:
    no_drop
    pca/n_components_<int>
    pca/var_ratio_<float>
    drop_feats/pearson_r_<float>

For each cell:
    fwd/<part>/<emb>/<clf>[/C_X]/forward_matched_metrics.json
        -> rows for scope ∈ {forward_matched, forward_full}
    rev/<part>/<emb>/<clf>[/C_X]/metrics.json
        -> rows for scope ∈ {reverse_matched_oof, reverse_unmatched}

Output:
    <reducer_dir>/_summary/all_metrics_with_cm.csv

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/utilities/build_asym_sweep_metrics_with_cm.py \
        --cohort-mode p_all_cdr05_hc_all_cdrall_or_mmseall
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

METRIC_FIELDS = [
    "n", "n_pos", "n_neg",
    "auc", "auc_ci_low", "auc_ci_high",
    "balacc", "mcc", "f1", "sens", "spec",
]

# (bucket, fname, json_key, scope, strategy)
EXTRACTORS = [
    ("fwd", "forward_matched_metrics.json", "metrics_matched_subset",
     "forward_matched", "forward"),
    ("fwd", "forward_matched_metrics.json", "metrics_full_cohort",
     "forward_full", "forward"),
    ("rev", "metrics.json", "metrics_matched_oof",
     "reverse_matched_oof", "reverse"),
    ("rev", "metrics.json", "metrics_unmatched",
     "reverse_unmatched", "reverse"),
]


def _row_from_block(block, partition, emb, clf, lr_C, xgb_params,
                    strategy, scope):
    if not block:
        return None
    xp = xgb_params or {}
    row = {
        "partition": partition,
        "embedding": emb,
        "classifier": clf,
        "lr_C": lr_C,
        "xgb_n_estimators": xp.get("n_estimators"),
        "xgb_max_depth": xp.get("max_depth"),
        "xgb_learning_rate": xp.get("learning_rate"),
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
    return row


def _has_metrics_json(d):
    return ((d / "forward_matched_metrics.json").exists()
            or (d / "metrics.json").exists())


def _iter_cell_dirs(parent_d):
    """Yield (cell_dir, classifier) for each cell under parent_d.

    Generic walk: a "cell" is any directory that directly contains
    forward_matched_metrics.json or metrics.json. We accept two depths
    under parent_d:
      1. parent_d/<classifier>/                → flat (legacy or hyperparam-less)
      2. parent_d/<classifier>/<param_tag>/    → hyperparam-tagged leaf
                                              (C_<lr_C>, ne_X_md_Y_lr_Z, ...)

    Hyperparam values are read from the JSON itself (lr_C / xgb_params
    fields), not parsed from the path — keeps this walker classifier-agnostic
    and robust to format changes.
    """
    for clf_d in sorted(parent_d.iterdir()):
        if not clf_d.is_dir():
            continue
        clf = clf_d.name
        if _has_metrics_json(clf_d):
            yield clf_d, clf
        for sub_d in sorted(clf_d.iterdir()):
            if sub_d.is_dir() and _has_metrics_json(sub_d):
                yield sub_d, clf


def _scan_cells(reducer_dir):
    """Walk the cell tree under a reducer-leaf dir and return rows per scope.

    Layout: reducer_dir/<clf>/<param>/<fwd|rev>/<eval_method>/<match_level>/<eval_unit>/<match_strategy>/<partition>/<files>
    Embedding info is read from the JSON payload.
    """
    rows = []
    for json_file in reducer_dir.rglob("*.json"):
        fname = json_file.name
        matching_extractors = [e for e in EXTRACTORS if e[1] == fname]
        if not matching_extractors:
            continue
        try:
            d = json.loads(json_file.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
        rel = json_file.parent.relative_to(reducer_dir)
        parts = rel.parts
        bucket = None
        bucket_idx = None
        partition_name = parts[-1] if len(parts) > 0 else "unknown"
        for i, p in enumerate(parts):
            if p in ("fwd", "rev"):
                bucket = p
                bucket_idx = i
                break
        if bucket is None:
            continue
        # Extract eval_method / match_level / eval_unit / match_strategy
        # from parts after fwd/rev: [eval_method, match_level, eval_unit,
        #                             match_strategy, partition]
        after_bucket = parts[bucket_idx + 1:]
        eval_method = after_bucket[0] if len(after_bucket) > 0 else "unknown"
        match_level = after_bucket[1] if len(after_bucket) > 1 else "unknown"
        eval_unit = after_bucket[2] if len(after_bucket) > 2 else "unknown"
        match_strategy = after_bucket[3] if len(after_bucket) > 3 else "unknown"

        for (b, fn, key, scope, strategy) in matching_extractors:
            if b != bucket:
                continue
            block = d.get(key)
            lr_C = d.get("lr_C")
            xgb_params = d.get("xgb_params")
            emb = d.get("embedding", "unknown")
            clf = d.get("classifier", "unknown")
            row = _row_from_block(block, partition_name,
                                   emb, clf, lr_C,
                                   xgb_params, strategy, scope)
            if row is None:
                continue
            row["eval_method"] = eval_method
            row["match_level"] = match_level
            row["eval_unit"] = eval_unit
            row["match_strategy"] = match_strategy
            if scope.startswith("forward"):
                wilc = d.get("paired_wilcoxon", {}) or {}
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
    col_order = (["partition", "embedding", "classifier",
                  "lr_C",
                  "xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate",
                  "strategy", "scope",
                  "eval_method", "match_level", "eval_unit", "match_strategy"]
                 + METRIC_FIELDS
                 + ["TN", "FP", "FN", "TP",
                    "wilcoxon_W", "wilcoxon_p", "n_pairs", "mean_diff"])
    cols = [c for c in col_order if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    df = df[cols + extras]
    out = variant_dir / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_metrics_with_cm.csv", index=False)
    return len(df)


_REDUCER_PREFIXES = {"pca", "drop_feats"}


def _iter_reducer_dirs(class_root):
    """Yield reducer-leaf dirs under a classification root.

    Layout: class_root/<reducer>/<clf>/<param>/fwd|rev/...
    Reducer is depth-1 (no_drop) or depth-2 (pca/*, drop_feats/*).

    Skips paths containing any underscore-prefixed segment (_summary, etc.).
    """
    if not class_root.is_dir():
        return
    seen = set()
    for marker_name in ("fwd", "rev"):
        for marker in class_root.rglob(marker_name):
            if not marker.is_dir():
                continue
            parts = marker.relative_to(class_root).parts
            if not parts:
                continue
            if parts[0] in _REDUCER_PREFIXES:
                if len(parts) < 2:
                    continue
                reducer = class_root / parts[0] / parts[1]
            else:
                reducer = class_root / parts[0]
            seen.add(reducer)
    for reducer in sorted(seen):
        rel_parts = reducer.relative_to(class_root).parts
        if any(p.startswith("_") for p in rel_parts):
            continue
        yield reducer


def walk_root(root, label):
    """Walk a per-cohort root (root = <tree>/<cohort>/ or <tree>/<variant>/<cohort>/);
    children directly under root are reducer top-levels (no_drop, pca, drop_feats)."""
    if not root.exists():
        return 0, 0
    n_files, n_rows = 0, 0
    candidates = list(_iter_reducer_dirs(root))
    for vdir in candidates:
        n = build_one(vdir)
        if n > 0:
            n_files += 1
            n_rows += n
            rel = vdir.relative_to(root)
            print(f"  {label}/{rel}: wrote {n} rows")
    return n_files, n_rows


def main():
    from src.config import (
        EMBEDDING_CLASSIFICATION_DIR,
        VALID_COHORT_CHOICES,
        cohort_name,
    )
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort-mode", default="p_first_cdr05_hc_first_cdrall_or_mmseall",
                    choices=VALID_COHORT_CHOICES)
    args = p.parse_args()
    cohort_dir = cohort_name(args.cohort_mode)
    classification_root = EMBEDDING_CLASSIFICATION_DIR

    total_files, total_rows = 0, 0
    # New layout: <visit>/<cdr_mmse>/<bg_mode>/<embedding>/<variant>/<photo>/<reducer>/...
    # Cohort is split into visit + cdr_mmse directories.
    from src.config import cohort_spec_from_name
    spec = cohort_spec_from_name(cohort_dir)
    visit_dir = classification_root / spec.visit_dir / spec.cdr_mmse_dir
    if visit_dir.is_dir():
        for bg_dir in sorted(visit_dir.iterdir()):
            if not bg_dir.is_dir() or bg_dir.name.startswith("_"):
                continue
            for emb_dir in sorted(bg_dir.iterdir()):
                if not emb_dir.is_dir() or emb_dir.name.startswith("_"):
                    continue
                for variant_dir in sorted(emb_dir.iterdir()):
                    if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
                        continue
                    for photo_dir in sorted(variant_dir.iterdir()):
                        if not photo_dir.is_dir() or photo_dir.name.startswith("_"):
                            continue
                        label = (f"{bg_dir.name}/{emb_dir.name}/"
                                 f"{variant_dir.name}/{photo_dir.name}")
                        f, r = walk_root(photo_dir, label)
                        total_files += f
                        total_rows += r
    print(f"\nTOTAL: {total_files} files, {total_rows} rows")


if __name__ == "__main__":
    main()
