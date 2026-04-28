"""Extract clean text sidecars (.txt) from PDFs in references/.

For each <basename>.pdf, writes <basename>.txt next to it via pymupdf.
Reports PDFs with very low text-per-page (likely scans needing OCR — those
need tesseract + ocrmypdf as a separate step).

Usage:
    python -m scripts.literature_monitor.extract_text --apply
    python -m scripts.literature_monitor.extract_text --apply --topic emotion
    python -m scripts.literature_monitor.extract_text --apply --waiting-review
    python -m scripts.literature_monitor.extract_text --apply --overwrite
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Heuristic: PDF with < this many chars per page is probably a scan
LOW_TEXT_THRESHOLD = 200


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("project root not found")


def _clean_text(raw: str) -> str:
    # Fix hyphenated line breaks: "intro-\nduction" -> "introduction"
    s = re.sub(r"-\n(?=[a-z])", "", raw)
    # Collapse 3+ blank lines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Strip trailing whitespace per line
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip() + "\n"


def extract_pdf_text(pdf_path: Path) -> tuple[str, dict]:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    raw = "\n\n".join(pages)
    text = _clean_text(raw)
    return text, {
        "page_count": len(pages),
        "char_count": len(text),
        "chars_per_page": len(text) / max(1, len(pages)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", choices=["embedding", "asymmetry", "emotion", "age", "all"],
                        default="all")
    parser.add_argument("--waiting-review", action="store_true",
                        help="extract from waiting_review/ instead of references/")
    parser.add_argument("--apply", action="store_true",
                        help="actually write .txt files (default is dry-run)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing .txt sidecars")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo = _project_root()
    base = repo / ("references/waiting_review" if args.waiting_review else "references")
    topics = ["embedding", "asymmetry", "emotion", "age"] if args.topic == "all" else [args.topic]

    targets: list[Path] = []
    for topic in topics:
        if args.waiting_review:
            targets.extend((base / topic).glob("*/*.pdf"))
        else:
            targets.extend((base / topic).glob("*.pdf"))

    if not targets:
        print("No PDFs found.")
        return 0

    print(f"Found {len(targets)} PDFs.")
    if not args.apply:
        for p in targets[:10]:
            print(f"  {p.relative_to(repo)}")
        if len(targets) > 10:
            print(f"  ... ({len(targets) - 10} more)")
        print("\n(dry-run; pass --apply to actually extract)")
        return 0

    ok = 0
    skipped = 0
    failed = 0
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
        if (ok + skipped + failed) % 20 == 0:
            print(f"  progress: {ok + skipped + failed}/{len(targets)} "
                  f"(extracted={ok} skipped={skipped} failed={failed})")

    print(f"\nDone: extracted={ok} skipped={skipped} failed={failed}")
    if low_text:
        print(f"\n{len(low_text)} PDFs have low text-per-page (likely scans needing OCR):")
        for p, cpp in sorted(low_text, key=lambda x: x[1]):
            print(f"  {cpp:6.1f} chars/page  {p.relative_to(repo)}")
        print("\nFor OCR: install tesseract + ocrmypdf system-wide and run "
              "`ocrmypdf <pdf> <pdf>` to add a text layer, then re-run this with --overwrite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
