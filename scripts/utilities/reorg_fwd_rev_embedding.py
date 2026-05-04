"""
Re-organize embedding_classification outputs into separate fwd/ and rev/ trees.

Old layout:
    embedding_classification/<partition>/<embedding>/<classifier>/
        matched_cohort.csv  matched_pairs.csv
        forward_*  reverse_*

New layout:
    embedding_classification/
      fwd/<partition>/<embedding>/<classifier>/
          matched_cohort.csv  matched_pairs.csv  forward_*
      rev/<partition>/<embedding>/<classifier>/
          matched_cohort.csv  matched_pairs.csv  reverse_*

Shared cohort files (matched_*) are copied into both subtrees so each is
self-contained. Idempotent — safe to re-run.

Usage:
    conda run -n Alz_face_main_analysis python scripts/utilities/reorg_embedding_classification.py
    conda run -n Alz_face_main_analysis python scripts/utilities/reorg_embedding_classification.py --dry-run
"""
import argparse
import logging
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "workspace" / "arms_analysis" / "p_first_hc_strict" / "embedding_classification"

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def iter_old_cells():
    """Yield (partition, embedding, classifier, cell_dir) for legacy layout."""
    for part_dir in sorted(ROOT.iterdir()):
        if not part_dir.is_dir() or part_dir.name in ("_summary", "fwd", "rev"):
            continue
        if part_dir.name not in PARTITIONS:
            continue
        for emb_dir in sorted(part_dir.iterdir()):
            if not emb_dir.is_dir():
                continue
            for clf_dir in sorted(emb_dir.iterdir()):
                if not clf_dir.is_dir():
                    continue
                yield part_dir.name, emb_dir.name, clf_dir.name, clf_dir


def reorg_one(partition, embedding, classifier, src_dir, dry_run=False):
    fwd_dst = ROOT / "fwd" / partition / embedding / classifier
    rev_dst = ROOT / "rev" / partition / embedding / classifier
    if not dry_run:
        fwd_dst.mkdir(parents=True, exist_ok=True)
        rev_dst.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in src_dir.iterdir():
        if f.is_dir():
            continue
        name = f.name
        if name.startswith("forward_"):
            dst = fwd_dst / name
            action = "mv"
            target = dst
        elif name.startswith("reverse_"):
            dst = rev_dst / name
            action = "mv"
            target = dst
        elif name.startswith("matched_"):
            # copy to both subtrees, then delete original
            action = "cp×2"
            target = (fwd_dst / name, rev_dst / name)
        else:
            logger.warning(f"  unknown file (left in place): {f}")
            continue

        logger.debug(f"  {action} {f.name} → {target}")
        if dry_run:
            moved += 1
            continue

        if action == "mv":
            shutil.move(str(f), str(dst))
        else:
            shutil.copy2(str(f), str(target[0]))
            shutil.copy2(str(f), str(target[1]))
            f.unlink()
        moved += 1

    # Remove now-empty cell dir + bubble up if parent is empty
    if not dry_run:
        try:
            src_dir.rmdir()
            if not any(src_dir.parent.iterdir()):
                src_dir.parent.rmdir()
                if not any(src_dir.parent.parent.iterdir()):
                    src_dir.parent.parent.rmdir()
        except OSError:
            pass

    return moved


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print actions, don't touch files.")
    args = parser.parse_args()

    if not ROOT.exists():
        logger.error(f"Missing root dir: {ROOT}")
        return

    cells = list(iter_old_cells())
    if not cells:
        logger.info("No legacy cells found — already reorganized.")
        return

    logger.info(f"Found {len(cells)} legacy cells. dry_run={args.dry_run}")
    total = 0
    for partition, embedding, classifier, src_dir in cells:
        n = reorg_one(partition, embedding, classifier, src_dir,
                       dry_run=args.dry_run)
        logger.info(f"  {partition}/{embedding}/{classifier}: {n} files")
        total += n

    logger.info(f"Done: {total} files {'would be' if args.dry_run else ''} "
                f"reorganized into {ROOT}/fwd and {ROOT}/rev")


if __name__ == "__main__":
    main()
