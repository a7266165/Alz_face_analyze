"""補抓 abstract 過濾後倖存者的缺漏 PDF。

  python -m scripts.literature_monitor.download_pdfs --apply
  python -m scripts.literature_monitor.download_pdfs --apply --topic emotion
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: E402

from src.literature_monitor.download import download_missing, missing_pdf_targets  # noqa: E402
from src.literature_monitor.queries import TOPICS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="實際下載（預設 dry-run）")
    ap.add_argument("--topic", choices=list(TOPICS) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="只試前 N 篇（測試用）")
    ap.add_argument("--retry-no-oa", action="store_true", help="重試先前標記 no_oa 的")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wr = PROJECT_ROOT / "references" / "waiting_review"
    topics = None if args.topic == "all" else [args.topic]
    targets = missing_pdf_targets(wr, topics=topics, retry_no_oa=args.retry_no_oa)
    if args.limit:
        targets = targets[:args.limit]

    print(f"Candidates needing PDF: {len(targets)}")
    if not args.apply:
        for j, m in targets[:10]:
            print(f"  {j.relative_to(PROJECT_ROOT)} :: {m.get('title', '')[:80]}")
        print("\n(dry-run; pass --apply to actually fetch)")
        return 0

    ok, no_oa = download_missing(targets)
    print(f"\nDone: {ok} downloaded, {no_oa} no OA available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
