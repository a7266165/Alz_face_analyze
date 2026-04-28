"""Download PDFs for survivors of the abstract-stage filter.

Walks references/waiting_review/<topic>/<DATE>/*.json and, for any record
without a companion .pdf, attempts the standard fallback chain (arXiv ->
S2 openAccessPdf -> Unpaywall -> OpenAlex). Updates the JSON's pdf_status
field accordingly.

Usage:
    python -m scripts.literature_monitor.download_pdfs --dry-run
    python -m scripts.literature_monitor.download_pdfs --apply
    python -m scripts.literature_monitor.download_pdfs --apply --topic emotion
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.literature_monitor.download import _candidate_pdf_urls, _download_pdf
from scripts.literature_monitor.sources import PaperRecord

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("project root not found")


def _record_from_meta(meta: dict) -> PaperRecord:
    """Reconstruct a PaperRecord shape good enough for download.py helpers."""
    return PaperRecord(
        title=meta.get("title", ""),
        authors=meta.get("authors", []) or [],
        year=meta.get("year"),
        abstract=meta.get("abstract", ""),
        doi=meta.get("doi"),
        arxiv_id=meta.get("arxiv_id"),
        venue=meta.get("venue"),
        citation_count=meta.get("citation_count"),
        source=meta.get("source", ""),
        pdf_url=meta.get("pdf_url"),
        relevance_score=meta.get("relevance_score"),
        extra=meta.get("extra", {}) or {},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="actually download (default is dry-run)")
    parser.add_argument("--topic", choices=["embedding", "asymmetry", "emotion", "age", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="only attempt N downloads (for testing)")
    parser.add_argument("--retry-no-oa", action="store_true",
                        help="retry papers previously marked no_oa")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo = _project_root()
    base = repo / "references" / "waiting_review"

    targets: list[tuple[Path, dict]] = []
    topics = ["embedding", "asymmetry", "emotion", "age"] if args.topic == "all" else [args.topic]
    for topic in topics:
        for json_path in (base / topic).glob("*/*.json"):
            companion_pdf = json_path.with_suffix(".pdf")
            if companion_pdf.exists():
                continue
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            status = meta.get("pdf_status")
            if status == "no_oa" and not args.retry_no_oa:
                # Already known to have no OA — skip unless user wants retry
                continue
            targets.append((json_path, meta))

    if args.limit:
        targets = targets[:args.limit]

    print(f"Candidates needing PDF: {len(targets)}")
    if not args.apply:
        for j, m in targets[:10]:
            print(f"  {j.relative_to(repo)} :: {m.get('title','')[:80]}")
        if len(targets) > 10:
            print(f"  ... ({len(targets) - 10} more)")
        print("\n(dry-run; pass --apply to actually fetch)")
        return 0

    ok = 0
    no_oa = 0
    for json_path, meta in targets:
        rec = _record_from_meta(meta)
        urls = _candidate_pdf_urls(rec)
        candidate_pdf = json_path.with_suffix(".pdf")
        success = False
        for url in urls:
            if _download_pdf(url, candidate_pdf):
                success = True
                break
        if success:
            ok += 1
            meta["pdf_status"] = "ok"
        else:
            no_oa += 1
            meta["pdf_status"] = "no_oa"
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        if (ok + no_oa) % 20 == 0:
            print(f"  progress: {ok + no_oa}/{len(targets)} (ok={ok} no_oa={no_oa})")

    print(f"\nDone: {ok} downloaded, {no_oa} no OA available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
