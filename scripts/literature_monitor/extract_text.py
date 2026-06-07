"""把 references/ 下的 PDF 抽成 .txt sidecar（pymupdf）。

  python -m scripts.literature_monitor.extract_text --apply
  python -m scripts.literature_monitor.extract_text --apply --waiting-review --overwrite
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: E402

from src.literature_monitor.download import LOW_TEXT_THRESHOLD, extract_pdf_text  # noqa: E402
from src.literature_monitor.queries import TOPICS  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", choices=list(TOPICS) + ["all"], default="all")
    ap.add_argument("--waiting-review", action="store_true", help="改抽 waiting_review/ 而非 references/")
    ap.add_argument("--apply", action="store_true", help="實際寫 .txt（預設 dry-run）")
    ap.add_argument("--overwrite", action="store_true", help="覆寫既有 .txt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = PROJECT_ROOT / ("references/waiting_review" if args.waiting_review else "references")
    topics = list(TOPICS) if args.topic == "all" else [args.topic]
    pattern = "*/*.pdf" if args.waiting_review else "*.pdf"
    targets = [pdf for t in topics for pdf in (base / t).glob(pattern)]

    if not targets:
        print("No PDFs found.")
        return 0
    print(f"Found {len(targets)} PDFs.")
    if not args.apply:
        for p in targets[:10]:
            print(f"  {p.relative_to(PROJECT_ROOT)}")
        print("\n(dry-run; pass --apply to actually extract)")
        return 0

    ok = skipped = failed = 0
    low_text: list[tuple[Path, float]] = []
    for pdf in targets:
        txt_path = pdf.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            text, stats = extract_pdf_text(pdf)
        except Exception as e:
            logger.warning("Failed %s: %s", pdf.name, e)
            failed += 1
            continue
        txt_path.write_text(text, encoding="utf-8")
        ok += 1
        if stats["chars_per_page"] < LOW_TEXT_THRESHOLD:
            low_text.append((pdf, stats["chars_per_page"]))

    print(f"\nDone: extracted={ok} skipped={skipped} failed={failed}")
    if low_text:
        print(f"\n{len(low_text)} PDFs 每頁字數偏低（疑掃描檔需 OCR）：")
        for p, cpp in sorted(low_text, key=lambda x: x[1]):
            print(f"  {cpp:6.1f} chars/page  {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
