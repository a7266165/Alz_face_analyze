"""One-time migration: restructure classification output to 10-variable pipeline order.

OLD layout (current on disk):
  match_acs_first/1by1matched/<emb>/<bg>/<feat>/<cohort>/<reducer>/<partition>/<fwd|rev>/<clf>/<param>/
  match_randomly/<emb>/<bg>/<feat>/<cohort>/<reducer>/<partition>/<fwd|rev>/<clf>/<param>/

NEW layout:
  <visit>/<cdr_mmse>/<bg>/<emb>/<feat>/mean/<reducer>/<clf>/<param>/<fwd|rev>/1by1matched/match_acs_first/<partition>/

Cohort name split:
  p_first_cdr05_hc_all_cdrall_or_mmseall
  → visit: P_first_HC_all
  → cdr_mmse: P_cdr05_HC_cdrall_mmseall

Usage:
  python scripts/embedding/migrate_dir_layout.py --dry-run
  python scripts/embedding/migrate_dir_layout.py
"""

import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSIFICATION_BASE = PROJECT_ROOT / "workspace" / "embedding" / "analysis" / "classification"

DIRECTIONS = ("fwd", "rev")

COHORT_RE = re.compile(
    r"^p_(?P<pv>first|all)_(?P<pc>cdr05|cdrall)"
    r"_hc_(?P<hv>first|all)_(?P<hc>cdr0|cdrall)_or_(?P<hm>mmse26|mmseall)$"
)


def split_cohort(cohort_name):
    m = COHORT_RE.match(cohort_name)
    if m is None:
        return None, None
    visit_dir = f"P_{m['pv']}_HC_{m['hv']}"
    cdr_dir = f"P_{m['pc']}_HC_{m['hc']}_{m['hm']}"
    return visit_dir, cdr_dir


def find_cells_under(root):
    """Find all cell directories (contain JSON files) under root.

    Returns list of (cell_path, relative_parts_as_tuple).
    """
    cells = []
    seen = set()
    for json_file in root.rglob("*.json"):
        cell_dir = json_file.parent
        if cell_dir in seen:
            continue
        seen.add(cell_dir)
        cells.append(cell_dir)
    return cells


def parse_old_cell_path(cell_dir, root, match_strategy, eval_method):
    """Parse old layout: <emb>/<bg>/<feat>/<cohort>/<reducer...>/<partition>/<fwd|rev>/<clf>/<param>/

    Returns dict with all components, or None if unparseable.
    """
    try:
        rel = cell_dir.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)

    dir_idx = None
    for i, p in enumerate(parts):
        if p in DIRECTIONS:
            dir_idx = i
            break
    if dir_idx is None or dir_idx < 5:
        return None

    emb = parts[0]
    bg = parts[1]
    feat = parts[2]
    cohort = parts[3]
    reducer_parts = parts[4:dir_idx - 1]
    partition = parts[dir_idx - 1]
    direction = parts[dir_idx]
    clf_parts = parts[dir_idx + 1:]

    visit_dir, cdr_dir = split_cohort(cohort)
    if visit_dir is None:
        return None

    reducer_rel = "/".join(reducer_parts) if reducer_parts else "no_drop"
    clf_rel = "/".join(clf_parts) if clf_parts else ""

    return {
        "emb": emb, "bg": bg, "feat": feat,
        "visit_dir": visit_dir, "cdr_dir": cdr_dir,
        "reducer_rel": reducer_rel, "partition": partition,
        "direction": direction, "clf_rel": clf_rel,
        "match_strategy": match_strategy,
        "eval_method": eval_method,
    }


def build_new_path(base, info):
    return (base
            / info["visit_dir"] / info["cdr_dir"]
            / info["bg"] / info["emb"] / info["feat"]
            / "mean" / info["reducer_rel"]
            / info["clf_rel"] / info["direction"]
            / info["eval_method"] / info["match_strategy"]
            / info["partition"])


def plan_moves(base):
    moves = []

    # match_acs_first/1by1matched/<emb>/...
    acs_dir = base / "match_acs_first"
    if acs_dir.is_dir():
        for eval_dir in sorted(acs_dir.iterdir()):
            if not eval_dir.is_dir():
                continue
            eval_method = eval_dir.name
            cells = find_cells_under(eval_dir)
            for cell in cells:
                info = parse_old_cell_path(cell, eval_dir, "match_acs_first", eval_method)
                if info is None:
                    continue
                new_path = build_new_path(base, info)
                if cell != new_path:
                    moves.append((cell, new_path))

    # match_randomly/<emb>/... (no eval_method layer)
    rand_dir = base / "match_randomly"
    if rand_dir.is_dir():
        cells = find_cells_under(rand_dir)
        for cell in cells:
            info = parse_old_cell_path(cell, rand_dir, "match_randomly", "1by1matched")
            if info is None:
                continue
            new_path = build_new_path(base, info)
            if cell != new_path:
                moves.append((cell, new_path))

    return moves


def execute(moves, dry_run=True):
    for src, dst in moves:
        if dry_run:
            print(f"  {src}")
            print(f"    -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def cleanup_empty(path, stop_at):
    for d in sorted(path.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()) and d != stop_at:
            d.rmdir()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moves = plan_moves(CLASSIFICATION_BASE)
    print(f"{len(moves)} moves planned")
    if moves:
        execute(moves, dry_run=args.dry_run)
    if not args.dry_run and moves:
        cleanup_empty(CLASSIFICATION_BASE, CLASSIFICATION_BASE)
        print("Cleaned up empty directories")

    action = "would move" if args.dry_run else "moved"
    print(f"Total: {action} {len(moves)} cell directories")
    if args.dry_run:
        print("(use without --dry-run to execute)")


if __name__ == "__main__":
    main()
