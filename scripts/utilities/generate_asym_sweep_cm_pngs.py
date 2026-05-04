"""
Generate per-cell confusion-matrix PNGs for the dropcorr / PCA sweep cells in
embedding_asymmetry_classification/<reducer>/<variant>/{fwd,rev/{ensemble,single}}/...

The original `no_drop` run rendered these PNGs but the sweep runs (drop_*, pca_*)
only saved metrics.json — leaving cells with `confusion_matrix` data but no
visualization. This script walks the cell tree, reads each metrics.json, and
renders the missing CM PNGs in a layout matching no_drop:
    fwd/.../forward_cm_full.png        ← metrics_full_cohort
    fwd/.../forward_cm_matched.png     ← metrics_matched_subset
    rev/.../ensemble/cm_full.png       ← metrics_full
    rev/.../ensemble/cm_matched_oof.png ← metrics_matched_oof
    rev/.../ensemble/cm_unmatched.png  ← metrics_unmatched
    rev/.../single/cm_matched_train.png ← metrics_matched_train
    rev/.../single/cm_unmatched.png    ← metrics_unmatched

Run:
    conda run -n Alz_face_main_analysis \
        python scripts/utilities/generate_asym_sweep_cm_pngs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cohort_base(cohort_mode):
    cohort_dir = "p_first_hc_all" if cohort_mode == "p_first_hc_all" else "p_first_hc_strict"
    return PROJECT_ROOT / "workspace" / "arms_analysis" / cohort_dir


def asym_root(cohort_mode):
    return cohort_base(cohort_mode) / "embedding_asymmetry_classification"


def orig_root(cohort_mode):
    return cohort_base(cohort_mode) / "embedding_classification"


ROOT = asym_root("default")  # legacy; main() picks both roots

# (json filename, key in JSON, output PNG filename, scope label for title)
FWD_TARGETS = [
    ("forward_matched_metrics.json", "metrics_full_cohort",
     "forward_cm_full.png", "forward_full"),
    ("forward_matched_metrics.json", "metrics_matched_subset",
     "forward_cm_matched.png", "forward_matched"),
]
REV_ENSEMBLE_TARGETS = [
    ("metrics.json", "metrics_full",
     "cm_full.png", "reverse_ensemble_full"),
    ("metrics.json", "metrics_matched_oof",
     "cm_matched_oof.png", "reverse_ensemble_matched_oof"),
    ("metrics.json", "metrics_unmatched",
     "cm_unmatched.png", "reverse_ensemble_unmatched"),
]
REV_SINGLE_TARGETS = [
    ("metrics.json", "metrics_matched_train",
     "cm_matched_train.png", "reverse_single_matched_train"),
    ("metrics.json", "metrics_unmatched",
     "cm_unmatched.png", "reverse_single_unmatched"),
]


def render_cm(cm, partition, emb, clf, scope, n, auc, out_png):
    """Render a 2×2 confusion-matrix heatmap. cm is [[TN, FP], [FN, TP]]."""
    cm = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    # 2x2 cells with text — use white text on dark background, black otherwise
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred HC", "Pred AD"])
    ax.set_yticklabels(["True HC", "True AD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    auc_str = f"  AUC={auc:.3f}" if auc is not None else ""
    title = f"{partition} / {emb} / {clf}\n{scope}  n={n}{auc_str}"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def render_one_cell(json_path, key, out_png, scope, partition, emb, clf):
    if out_png.exists():
        return False  # don't overwrite
    if not json_path.exists():
        return False
    try:
        m = json.loads(json_path.read_text())
        block = m.get(key)
    except (ValueError, json.JSONDecodeError):
        return False
    if not block:
        return False
    cm = block.get("confusion_matrix")
    if cm is None or len(cm) != 2 or len(cm[0]) != 2:
        return False
    n = block.get("n")
    auc = block.get("auc")
    render_cm(cm, partition, emb, clf, scope, n, auc, out_png)
    return True


def render_under_variant_dir(variant_dir):
    """Render all CM PNGs under a directory containing fwd/ and rev/ subtrees."""
    n = 0
    fwd_root = variant_dir / "fwd"
    if fwd_root.exists():
        for partition_d in fwd_root.iterdir():
            if not partition_d.is_dir():
                continue
            for emb_d in partition_d.iterdir():
                if not emb_d.is_dir():
                    continue
                for clf_d in emb_d.iterdir():
                    if not clf_d.is_dir():
                        continue
                    for jname, key, png, scope in FWD_TARGETS:
                        if render_one_cell(clf_d / jname, key, clf_d / png,
                                            scope, partition_d.name,
                                            emb_d.name, clf_d.name):
                            n += 1
    rev_root = variant_dir / "rev"
    if rev_root.exists():
        for partition_d in rev_root.iterdir():
            if not partition_d.is_dir():
                continue
            for emb_d in partition_d.iterdir():
                if not emb_d.is_dir():
                    continue
                for clf_d in emb_d.iterdir():
                    if not clf_d.is_dir():
                        continue
                    for sub, targets in [
                        ("ensemble", REV_ENSEMBLE_TARGETS),
                        ("single", REV_SINGLE_TARGETS),
                    ]:
                        sub_d = clf_d / sub
                        if not sub_d.is_dir():
                            continue
                        for jname, key, png, scope in targets:
                            if render_one_cell(sub_d / jname, key,
                                                sub_d / png, scope,
                                                partition_d.name,
                                                emb_d.name, clf_d.name):
                                n += 1
    return n


def walk_root(root, has_variant_layer, args):
    """has_variant_layer=True: <reducer>/<variant>/{fwd,rev}/...
    has_variant_layer=False: <reducer>/{fwd,rev}/...   (baseline 'original')"""
    if not root.exists():
        return 0
    n_total = 0
    for reducer in sorted(root.iterdir()):
        if not reducer.is_dir() or reducer.name.startswith("_"):
            continue
        if reducer.name.startswith("no_drop") and not args.include_no_drop:
            continue
        if not reducer.name.startswith(("drop_", "pca_", "no_drop")):
            continue
        n_red = 0
        if has_variant_layer:
            for variant_dir in sorted(reducer.iterdir()):
                if not variant_dir.is_dir():
                    continue
                n_red += render_under_variant_dir(variant_dir)
        else:
            n_red += render_under_variant_dir(reducer)
        if n_red > 0:
            label = root.name + "/" + reducer.name
            print(f"  {label}: rendered {n_red} PNGs")
            n_total += n_red
    return n_total


def main():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort-mode", default="default",
                    choices=["default", "p_first_hc_all"])
    p.add_argument("--include-no-drop", action="store_true",
                    help="Also render PNGs in no_drop/ subtree (default skips "
                         "since legacy strict no_drop already has them).")
    args = p.parse_args()

    asym = asym_root(args.cohort_mode)
    orig = orig_root(args.cohort_mode)
    if not asym.exists() and not orig.exists():
        print(f"Neither root exists under cohort_mode={args.cohort_mode}",
              file=sys.stderr)
        sys.exit(1)

    n_asym = walk_root(asym, has_variant_layer=True, args=args)
    n_orig = walk_root(orig, has_variant_layer=False, args=args)
    print(f"\nTOTAL: {n_asym} (asymmetry) + {n_orig} (original) = "
          f"{n_asym + n_orig} PNGs rendered")


if __name__ == "__main__":
    main()
