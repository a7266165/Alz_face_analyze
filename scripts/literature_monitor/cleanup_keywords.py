"""多階段關鍵字噪音清理（標題 block → abstract positive → 補抓 → 再 positive → 歸檔）。

  python -m scripts.literature_monitor.cleanup_keywords            # dry-run
  python -m scripts.literature_monitor.cleanup_keywords --apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: E402

from src.literature_monitor.curate import pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="實際補抓 abstract 並歸檔噪音（預設 dry-run）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    pipeline(PROJECT_ROOT / "references" / "waiting_review", apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
