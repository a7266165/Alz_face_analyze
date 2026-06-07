"""回填既有 sidecar 缺漏的 abstract。

  python -m scripts.literature_monitor.enrich_abstracts --dry-run
  python -m scripts.literature_monitor.enrich_abstracts --apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: E402

from src.literature_monitor.curate import fill_missing_abstracts  # noqa: E402
from src.literature_monitor.state import iter_sidecars  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="實際抓取並寫回（預設 dry-run）")
    ap.add_argument("--max-papers", type=int, default=None, help="上限（測試用）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wr = PROJECT_ROOT / "references" / "waiting_review"
    targets = [(p, m) for p, m in iter_sidecars(wr) if not (m.get("abstract") or "").strip()]
    if args.max_papers:
        targets = targets[:args.max_papers]

    by_source: dict[str, int] = {}
    for _, m in targets:
        by_source[m.get("source", "?")] = by_source.get(m.get("source", "?"), 0) + 1
    print(f"Empty-abstract records: {len(targets)} total")
    for source, n in by_source.items():
        print(f"  {source}: {n}")

    if not args.apply:
        print("\n(dry-run; pass --apply to actually fetch and write)")
        return 0

    filled = fill_missing_abstracts(targets)
    print(f"\nFilled abstracts: {filled}/{len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
