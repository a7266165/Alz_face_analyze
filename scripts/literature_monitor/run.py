"""文獻監測 CLI。

  python -m scripts.literature_monitor.run --slot 0 --auto-push
  python -m scripts.literature_monitor.run --topic embedding --dry-run --max-per-source 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: E402

from src.literature_monitor.queries import TOPICS  # noqa: E402
from src.literature_monitor.runner import rebuild_aliases, run_slot  # noqa: E402
from src.literature_monitor.state import write_reference_index  # noqa: E402


def _setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Alz Face 文獻監測")
    ap.add_argument("--slot", type=int, default=None, help="0-9，依 SLOT_PLAN")
    ap.add_argument("--topic", choices=list(TOPICS) + ["all"], default=None,
                    help="手動單主題；覆寫 slot 的主題清單")
    ap.add_argument("--max-per-source", type=int, default=25)
    ap.add_argument("--year-from", type=int, default=2018)
    ap.add_argument("--dry-run", action="store_true", help="只搜尋，不下載/存檔/推送")
    ap.add_argument("--auto-push", action="store_true", help="run 後 git pull/commit/push")
    ap.add_argument("--no-push", action="store_true", help="強制關閉推送")
    ap.add_argument("--rebuild-index", action="store_true", help="重建 references/_indexed.json 後結束")
    ap.add_argument("--rebuild-aliases", action="store_true", help="從 sidecar 回填 _state.json 別名後結束")
    ap.add_argument("--batch", type=int, default=1, help="連跑 N 次 sweep；某次 0 新增即提早停")
    ap.add_argument("--query-idx", type=int, default=None, help="覆寫 slot 的 query index")
    ap.add_argument("--no-pdf", action="store_true", help="只存 metadata，PDF 留給 download_pdfs.py")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(
        PROJECT_ROOT / "references" / "waiting_review" / "_logs" / f"{datetime.now():%Y%m%d}.log",
        args.verbose,
    )

    if args.rebuild_index:
        out = write_reference_index(PROJECT_ROOT / "references")
        print(f"wrote {out.relative_to(PROJECT_ROOT)}")
        return 0
    if args.rebuild_aliases:
        print(f"added {rebuild_aliases(PROJECT_ROOT)} aliases to _state.json")
        return 0
    if args.slot is None and args.topic is None:
        ap.error("either --slot or --topic is required")

    push = args.auto_push and not args.no_push

    def _sweep() -> dict:
        if args.slot is not None:
            override = [args.topic] if (args.topic and args.topic != "all") else None
            return run_slot(args.slot, PROJECT_ROOT, max_per_source=args.max_per_source,
                            year_from=args.year_from, dry_run=args.dry_run, push=push,
                            override_topics=override, override_query_idx=args.query_idx,
                            download_pdf=not args.no_pdf)
        topics = list(TOPICS) if args.topic == "all" else [args.topic]
        return run_slot(999, PROJECT_ROOT, max_per_source=args.max_per_source,
                        year_from=args.year_from, dry_run=args.dry_run, push=push,
                        override_topics=topics,
                        override_sources=["arxiv", "s2", "openalex", "pubmed"],
                        override_query_idx=args.query_idx if args.query_idx is not None else 0,
                        download_pdf=not args.no_pdf)

    for n in range(max(1, args.batch)):
        if args.batch > 1:
            print(f"\n=== Sweep {n + 1}/{args.batch} ===", flush=True)
        result = _sweep()
        if args.batch > 1 and sum((result.get("counts") or {}).values()) == 0:
            print(f"\nSweep {n + 1} 0 new — stopping early", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
