"""
Split reverse outputs into ensemble/ + single/ subfolders by training method.

Old layout (per cell):
    rev/<part>/<emb>/<clf>/
        reverse_full_metrics.json
        reverse_full_scores.csv          (columns: ID, y_true, score_ensemble, score_single, in_matched)
        reverse_cm_full_ensemble.png
        reverse_cm_matched_oof.png
        reverse_cm_unmatched_ensemble.png
        reverse_cm_matched_single_train.png
        reverse_cm_unmatched_single.png
        matched_cohort.csv  matched_pairs.csv

New layout:
    rev/<part>/<emb>/<clf>/
        matched_cohort.csv  matched_pairs.csv
        ensemble/                       ← method A (10 fold-models)
            scores.csv                  (ID, y_true, y_score, in_matched)
            metrics.json                (matched_oof + full + unmatched)
            cm_matched_oof.png  cm_full.png  cm_unmatched.png
        single/                         ← method B (1 model trained on all matched)
            scores.csv
            metrics.json                (matched_train + unmatched)
            cm_matched_train.png  cm_unmatched.png

Idempotent — safe to re-run. Deletes legacy files only after successful split.

Usage:
    conda run -n Alz_face_main_analysis python scripts/utilities/reorg_reverse_split.py
    conda run -n Alz_face_main_analysis python scripts/utilities/reorg_reverse_split.py --dry-run
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REV_ROOT = (PROJECT_ROOT / "workspace" / "arms_analysis"
            / "p_first_hc_strict" / "embedding_classification" / "rev")

LEGACY_FILES = [
    "reverse_full_metrics.json",
    "reverse_full_scores.csv",
    "reverse_cm_full_ensemble.png",
    "reverse_cm_matched_oof.png",
    "reverse_cm_unmatched_ensemble.png",
    "reverse_cm_matched_single_train.png",
    "reverse_cm_unmatched_single.png",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def iter_cells():
    if not REV_ROOT.exists():
        return
    for part_dir in sorted(REV_ROOT.iterdir()):
        if not part_dir.is_dir():
            continue
        for emb_dir in sorted(part_dir.iterdir()):
            if not emb_dir.is_dir():
                continue
            for clf_dir in sorted(emb_dir.iterdir()):
                if not clf_dir.is_dir():
                    continue
                yield clf_dir


def split_cell(cell_dir, dry_run=False):
    legacy_json = cell_dir / "reverse_full_metrics.json"
    legacy_csv = cell_dir / "reverse_full_scores.csv"
    if not legacy_json.exists() or not legacy_csv.exists():
        return False  # already split or missing

    payload = json.loads(legacy_json.read_text(encoding="utf-8"))
    df = pd.read_csv(legacy_csv)

    common = {k: v for k, v in payload.items()
              if k in ("partition", "embedding", "classifier", "strategy",
                       "k_folds_used", "n_dropped_no_emb_matched",
                       "n_dropped_no_emb_full", "n_unmatched")}

    # Method A — ensemble
    ens_dir = cell_dir / "ensemble"
    ens_payload = {
        **common, "method": "ensemble (10 fold-models)",
        "metrics_matched_oof": payload.get("metrics_matched_oof"),
        "metrics_full": payload.get("metrics_full_ensemble"),
        "metrics_unmatched": payload.get("metrics_unmatched_ensemble"),
    }
    ens_scores = (df[["ID", "y_true", "score_ensemble", "in_matched"]]
                  .rename(columns={"score_ensemble": "y_score"}))

    # Method B — single
    sgl_dir = cell_dir / "single"
    sgl_payload = {
        **common, "method": "single (1 model on all matched)",
        "metrics_matched_train": payload.get("metrics_matched_single_train"),
        "metrics_unmatched": payload.get("metrics_unmatched_single"),
    }
    sgl_scores = (df[["ID", "y_true", "score_single", "in_matched"]]
                  .rename(columns={"score_single": "y_score"}))

    if dry_run:
        logger.info(f"  would write {ens_dir}/{{scores.csv,metrics.json}}")
        logger.info(f"  would write {sgl_dir}/{{scores.csv,metrics.json}}")
        return True

    ens_dir.mkdir(parents=True, exist_ok=True)
    ens_scores.to_csv(ens_dir / "scores.csv", index=False)
    (ens_dir / "metrics.json").write_text(
        json.dumps(ens_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    sgl_dir.mkdir(parents=True, exist_ok=True)
    sgl_scores.to_csv(sgl_dir / "scores.csv", index=False)
    (sgl_dir / "metrics.json").write_text(
        json.dumps(sgl_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Remove legacy files
    for fname in LEGACY_FILES:
        f = cell_dir / fname
        if f.exists():
            f.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_done = 0
    n_skip = 0
    for cell in iter_cells():
        ok = split_cell(cell, dry_run=args.dry_run)
        rel = cell.relative_to(REV_ROOT)
        if ok:
            n_done += 1
            logger.info(f"  {'[dry] ' if args.dry_run else ''}split {rel}")
        else:
            n_skip += 1
            logger.debug(f"  skip {rel} (no legacy files)")

    logger.info(f"Done: {n_done} cells split, {n_skip} skipped. "
                f"{'(dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
