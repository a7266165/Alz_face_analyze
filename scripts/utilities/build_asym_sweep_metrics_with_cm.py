"""
Build per-cell `all_metrics_with_cm.csv` for an embedding sweep cohort.

Walks the per-cell metrics.json tree and aggregates into the long-form schema
consumed by plot_pca_*.py / plot_dropcorr_*.py. Does NOT rely on
combined_summary.csv (which gets overwritten by partial re-runs).

Layout (post unified-reducer + simplified-rev refactors):
    <cohort>/embedding_classification/<reducer_path>/{fwd,rev}/<part>/<emb>/<clf>[/C_X]/
    <cohort>/embedding_asymmetry_classification/<feature_type>/<reducer_path>/{fwd,rev}/...

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
        --cohort-mode p_all_hc_all
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def _row_from_block(block, partition, emb, clf, lr_C, strategy, scope):
    if not block:
        return None
    row = {
        "partition": partition,
        "embedding": emb,
        "classifier": clf,
        "lr_C": lr_C,
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


def _iter_cell_dirs(emb_d):
    """Yield (cell_dir, classifier, lr_C) for each cell under emb_d.

    Logistic cells: emb_d/logistic/C_<X>/ → ('logistic', float(X)).
    XGB cells:      emb_d/xgb/            → ('xgb', None).
    """
    for clf_d in sorted(emb_d.iterdir()):
        if not clf_d.is_dir():
            continue
        clf = clf_d.name
        if clf == "logistic":
            for c_d in sorted(clf_d.iterdir()):
                if not c_d.is_dir() or not c_d.name.startswith("C_"):
                    continue
                try:
                    lr_C = float(c_d.name[2:])
                except ValueError:
                    continue
                yield c_d, clf, lr_C
        elif clf == "xgb":
            yield clf_d, clf, None


def _scan_cells(variant_dir):
    """Walk the cell tree under variant_dir and return rows per scope."""
    rows = []
    for bucket in ("fwd", "rev"):
        top = variant_dir / bucket
        if not top.is_dir():
            continue
        for partition_d in sorted(top.iterdir()):
            if not partition_d.is_dir():
                continue
            for emb_d in sorted(partition_d.iterdir()):
                if not emb_d.is_dir():
                    continue
                for cell_d, clf, lr_C in _iter_cell_dirs(emb_d):
                    for (b, fname, key, scope, strategy) in EXTRACTORS:
                        if b != bucket:
                            continue
                        json_path = cell_d / fname
                        if not json_path.exists():
                            continue
                        try:
                            d = json.loads(json_path.read_text())
                        except (ValueError, json.JSONDecodeError):
                            continue
                        block = d.get(key)
                        row = _row_from_block(block, partition_d.name,
                                               emb_d.name, clf, lr_C,
                                               strategy, scope)
                        if row is None:
                            continue
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
    col_order = (["partition", "embedding", "classifier", "lr_C",
                  "strategy", "scope"]
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


def _iter_reducer_dirs(class_root):
    """Yield reducer dirs under a classification root — any directory whose
    parent path is fwd-or-rev-bearing. Walks arbitrarily deep so visit/photo
    variants nested under a reducer (e.g. no_drop/visit_all,
    pca/n_components_100/visit_all_photo_all) are picked up.

    Skips paths containing any underscore-prefixed segment (_summary, etc.).
    """
    if not class_root.is_dir():
        return
    seen = set()
    for marker_name in ("fwd", "rev"):
        for marker in class_root.rglob(marker_name):
            if marker.is_dir():
                seen.add(marker.parent)
    for reducer in sorted(seen):
        rel_parts = reducer.relative_to(class_root).parts
        if any(p.startswith("_") for p in rel_parts):
            continue
        yield reducer


def walk_root(root, label):
    if not root.exists():
        return 0, 0
    n_files, n_rows = 0, 0
    if root.name == "embedding_asymmetry_classification":
        # feature_type pivots before reducer
        candidates = []
        for ft in sorted(root.iterdir()):
            if ft.is_dir() and not ft.name.startswith("_"):
                candidates.extend(_iter_reducer_dirs(ft))
    else:
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
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort-mode", default="default",
                    choices=["default", "p_first_hc_all", "p_all_hc_all"])
    args = p.parse_args()
    if args.cohort_mode == "p_first_hc_all":
        cohort_dir = "p_first_hc_all"
    elif args.cohort_mode == "p_all_hc_all":
        cohort_dir = "p_all_hc_all"
    else:
        cohort_dir = "p_first_hc_strict"
    base = PROJECT_ROOT / "workspace" / "arms_analysis" / cohort_dir

    total_files, total_rows = 0, 0
    for sub in ("embedding_classification", "embedding_asymmetry_classification"):
        f, r = walk_root(base / sub, sub)
        total_files += f
        total_rows += r
    print(f"\nTOTAL: {total_files} files, {total_rows} rows")


if __name__ == "__main__":
    main()
