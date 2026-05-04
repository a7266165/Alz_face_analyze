"""
Post-process run_embedding_classification outputs — confusion matrices + metric tables.

Walks workspace/arms_analysis/embedding_classification/<partition>/<embedding>/<classifier>/
and reads any *_metrics.json that exist (compatible with a still-running sweep).
For each metrics block (forward × {full, matched_subset};
reverse × {matched_oof, full_ensemble}), writes a confusion matrix PNG next to
the JSON. Also emits a flattened CSV + markdown summary.

Usage:
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_fwd_rev_metrics.py
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_fwd_rev_metrics.py \\
        --partition ad_vs_hc --embedding arcface
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
DEFAULT_ROOT = ARMS_ROOT / "p_first_hc_strict" / "embedding_classification" / "no_drop"
ROOT = DEFAULT_ROOT  # set by main() when --root is passed
SUMMARY = ROOT / "_summary"

# Each entry: (json_filename, [(metric_block_key, cm_filename, scope_label,
# pos_label, neg_label), ...])
# Each entry: (relative_path_to_json, [(metric_block_key, cm_filename, scope_label), ...])
JSON_LAYOUT = {
    "forward_matched_metrics.json": [
        ("metrics_matched_subset", "forward_cm_matched.png",
         "forward_matched"),
        ("metrics_full_cohort", "forward_cm_full.png",
         "forward_full"),
    ],
    # Method A — ensemble (10 fold-models)
    "ensemble/metrics.json": [
        ("metrics_matched_oof", "ensemble/cm_matched_oof.png",
         "reverse_ensemble_matched_oof"),
        ("metrics_full", "ensemble/cm_full.png",
         "reverse_ensemble_full"),
        ("metrics_unmatched", "ensemble/cm_unmatched.png",
         "reverse_ensemble_unmatched"),
    ],
    # Method B — single model trained on all matched
    "single/metrics.json": [
        ("metrics_matched_train", "single/cm_matched_train.png",
         "reverse_single_matched_train"),
        ("metrics_unmatched", "single/cm_unmatched.png",
         "reverse_single_unmatched"),
    ],
}

# Per-partition group label (for CM x/y tick labels)
LABELS = {
    "ad_vs_hc":  ("HC",  "AD"),
    "ad_vs_nad": ("NAD", "AD"),
    "ad_vs_acs": ("ACS", "AD"),
    "mmse_hilo": ("MMSE-low", "MMSE-high"),
    "casi_hilo": ("CASI-low", "CASI-high"),
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CM plot
# ============================================================

def plot_cm(cm, title, out_path, neg_label="0", pos_label="1"):
    """cm = [[tn, fp], [fn, tp]] (matches compute_clf_metrics output).

    Text color picked per cell from the actual rendered cell luminance:
    look up RGB from the colormap at the cell's normalized value, compute
    YIQ luminance, and use white when luminance < 0.5 (dark cell), else
    black. This is robust to any colormap and value range.
    """
    cm = np.asarray(cm, dtype=float)
    cmap = plt.get_cmap("Blues")
    vmin = float(cm.min())
    vmax = float(cm.max())
    rng = vmax - vmin if vmax > vmin else 1.0

    fig, ax = plt.subplots(figsize=(4.68, 3.4))
    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {neg_label}", f"Pred {pos_label}"])
    ax.set_yticklabels([f"True {neg_label}", f"True {pos_label}"])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            r, g, b, _ = cmap((v - vmin) / rng)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color=text_color, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Manual margins so CM + colorbar block sits centered horizontally.
    fig.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Metrics flattening
# ============================================================

METRIC_KEYS = ["n", "n_pos", "n_neg", "auc", "auc_ci_low", "auc_ci_high",
               "balacc", "mcc", "f1", "sens", "spec"]


def flatten_metrics(m, partition, embedding, classifier, strategy, scope,
                    extra=None):
    cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0]
    fn, tp = cm[1]
    row = {
        "partition": partition, "embedding": embedding,
        "classifier": classifier, "strategy": strategy, "scope": scope,
        **{k: m.get(k) for k in METRIC_KEYS},
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if extra:
        row.update(extra)
    return row


# ============================================================
# Walk + emit
# ============================================================

def iter_cells(partition_filter=None, embedding_filter=None,
               classifier_filter=None):
    """Yield (partition, embedding, classifier, fwd_dir, rev_dir).

    Layout: embedding_classification/{fwd,rev}/<partition>/<embedding>/<classifier>/.
    A cell exists if either fwd or rev side has a matching dir.
    """
    if not ROOT.exists():
        return
    fwd_root, rev_root = ROOT / "fwd", ROOT / "rev"
    seen = set()

    def _scan(bucket_root):
        if not bucket_root.exists():
            return
        for part_dir in sorted(bucket_root.iterdir()):
            if not part_dir.is_dir():
                continue
            if partition_filter and part_dir.name != partition_filter:
                continue
            for emb_dir in sorted(part_dir.iterdir()):
                if not emb_dir.is_dir():
                    continue
                if embedding_filter and emb_dir.name != embedding_filter:
                    continue
                for clf_dir in sorted(emb_dir.iterdir()):
                    if not clf_dir.is_dir():
                        continue
                    if classifier_filter and clf_dir.name != classifier_filter:
                        continue
                    yield part_dir.name, emb_dir.name, clf_dir.name

    for combo in _scan(fwd_root):
        seen.add(combo)
    for combo in _scan(rev_root):
        seen.add(combo)

    for partition, embedding, classifier in sorted(seen):
        fwd_dir = fwd_root / partition / embedding / classifier
        rev_dir = rev_root / partition / embedding / classifier
        yield partition, embedding, classifier, fwd_dir, rev_dir


def process_cell(partition, embedding, classifier, fwd_dir, rev_dir):
    rows = []
    neg_lbl, pos_lbl = LABELS.get(partition, ("0", "1"))

    for json_name, blocks in JSON_LAYOUT.items():
        cell_dir = fwd_dir if json_name.startswith("forward_") else rev_dir
        json_path = cell_dir / json_name
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
        strategy = payload.get("strategy", "?")

        for block_key, cm_filename, scope in blocks:
            m = payload.get(block_key)
            if not m:
                continue
            cm = m.get("confusion_matrix")
            if cm is None:
                continue
            cm_path = cell_dir / cm_filename
            title = (f"{partition} / {embedding} / {classifier}\n"
                     f"{scope}  n={m.get('n', '?')}  "
                     f"AUC={m.get('auc', float('nan')):.3f}")
            plot_cm(cm, title, cm_path, neg_label=neg_lbl, pos_label=pos_lbl)
            extra = {}
            if scope == "forward_matched":
                pw = payload.get("paired_wilcoxon") or {}
                extra = {
                    "wilcoxon_W": pw.get("W"),
                    "wilcoxon_p": pw.get("p"),
                    "n_pairs": pw.get("n_pairs"),
                    "mean_diff": pw.get("mean_diff"),
                }
            rows.append(flatten_metrics(m, partition, embedding, classifier,
                                          strategy, scope, extra=extra))

    return rows


# ============================================================
# Markdown report
# ============================================================

def _df_to_md(df):
    """Pure-Python markdown table — no tabulate dependency."""
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(
            "" if pd.isna(v) else str(v) for v in r.tolist()
        ) + " |")
    return "\n".join([head, sep, *rows])


def write_markdown(rows, out_path):
    df = pd.DataFrame(rows)
    if df.empty:
        out_path.write_text("# embedding_classification — no cells found yet\n",
                            encoding="utf-8")
        return

    cols = [
        "embedding", "classifier", "scope",
        "n", "n_pos", "n_neg", "auc", "auc_ci_low", "auc_ci_high",
        "balacc", "mcc", "f1", "sens", "spec",
        "TN", "FP", "FN", "TP",
        "wilcoxon_W", "wilcoxon_p", "n_pairs",
    ]
    cols = [c for c in cols if c in df.columns]

    lines = [
        "# embedding_classification — metrics summary",
        "",
        f"Source: `{ROOT.relative_to(PROJECT_ROOT)}`",
        f"Cells found: {df[['partition','embedding','classifier']].drop_duplicates().shape[0]}",
        "",
    ]
    for partition in sorted(df["partition"].unique()):
        sub = df[df["partition"] == partition].copy()
        sub = sub.sort_values(["embedding", "classifier", "strategy", "scope"])
        lines.append(f"## {partition}")
        lines.append("")
        # truncate floats for readability
        sub_disp = sub[cols].copy()
        for c in sub_disp.select_dtypes(include="float").columns:
            sub_disp[c] = sub_disp[c].apply(
                lambda x: "" if pd.isna(x) else (
                    f"{x:.3g}" if c in ("wilcoxon_p", "wilcoxon_W") else f"{x:.3f}"
                )
            )
        lines.append(_df_to_md(sub_disp))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--classifier", default=None)
    parser.add_argument(
        "--root", default=str(DEFAULT_ROOT),
        help="Output root dir to scan. Default: embedding_classification/. "
             "For asymmetry sweeps pass e.g. "
             "embedding_asymmetry_absolute_relative_differences_classification/.",
    )
    args = parser.parse_args()

    global ROOT, SUMMARY
    ROOT = Path(args.root) if Path(args.root).is_absolute() else \
        ARMS_ROOT / args.root
    SUMMARY = ROOT / "_summary"
    SUMMARY.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scanning: {ROOT}")

    all_rows = []
    n_cells = 0
    for partition, embedding, classifier, fwd_dir, rev_dir in iter_cells(
        args.partition, args.embedding, args.classifier
    ):
        rows = process_cell(partition, embedding, classifier, fwd_dir, rev_dir)
        if rows:
            n_cells += 1
            all_rows.extend(rows)
            logger.info(f"  {partition}/{embedding}/{classifier}: "
                        f"{len(rows)} metric blocks")

    if not all_rows:
        logger.warning(f"No metric JSON found under {ROOT}")
        return

    df = pd.DataFrame(all_rows)
    csv_path = SUMMARY / "all_metrics_with_cm.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path} ({len(df)} rows from {n_cells} cells)")

    md_path = SUMMARY / "all_metrics_report.md"
    write_markdown(all_rows, md_path)
    logger.info(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
