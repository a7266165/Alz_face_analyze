"""Entry point for the literature monitor.

Usage examples:
    python -m scripts.literature_monitor.run --slot 0 --auto-push
    python -m scripts.literature_monitor.run --slot 0 --no-push
    python -m scripts.literature_monitor.run --topic embedding --dry-run --max-per-source 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make sure the project root is on sys.path so absolute imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.literature_monitor import digest as digest_mod
from scripts.literature_monitor import sources as src_mod
from scripts.literature_monitor.download import save_record
from scripts.literature_monitor.git_ops import auto_push
from scripts.literature_monitor.queries import (
    SLOT_PLAN,
    TOPICS,
    TOPIC_QUERIES,
    query_for,
)
from scripts.literature_monitor.state import StateStore, write_reference_index


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("project root with pyproject.toml not found")


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers)


def run_slot(
    slot: int,
    repo_root: Path,
    *,
    max_per_source: int = 25,
    year_from: int = 2018,
    dry_run: bool = False,
    push: bool = False,
    override_topics: list[str] | None = None,
    override_sources: list[str] | None = None,
    override_query_idx: int | None = None,
) -> dict:
    plan = SLOT_PLAN.get(slot, {"topics": [], "sources": [], "query_idx": 0})
    digest_only = plan.get("digest_only", False)
    if digest_only:
        topics, sources, query_idx = [], [], -1
    else:
        topics = override_topics if override_topics is not None else plan["topics"]
        sources = override_sources if override_sources is not None else plan["sources"]
        query_idx = override_query_idx if override_query_idx is not None else plan["query_idx"]

    waiting_review = repo_root / "references" / "waiting_review"
    references_dir = repo_root / "references"
    state = StateStore(waiting_review, references_dir)

    new_per_topic: dict[str, list[tuple[src_mod.PaperRecord, Path | None]]] = {t: [] for t in TOPICS}
    skipped_total = 0
    written_paths: list[str] = []

    for topic in topics:
        q = query_for(topic, query_idx)
        for source in sources:
            cursor = state.get_cursor(source, topic, q)
            logging.info("[slot %d] %s x %s @offset=%d :: %s",
                         slot, topic, source, cursor, q)
            try:
                records = src_mod.search(
                    source, q, max_results=max_per_source,
                    year_from=year_from, offset=cursor,
                )
            except Exception as e:
                logging.warning("search %s/%s failed: %s", source, topic, e)
                continue
            # Cursor management: advance by what we got; reset if exhausted
            if not records:
                if cursor > 0:
                    logging.info("  exhausted at offset %d; resetting cursor", cursor)
                    state.reset_cursor(source, topic, q)
                # else: zero results from offset 0, leave cursor at 0
            else:
                state.advance_cursor(source, topic, q, len(records))
            for rec in records:
                if not rec.title or rec.title == "":
                    continue
                pid = rec.primary_id()
                aliases = rec.all_ids()
                if state.is_seen(pid, aliases) or state.is_existing_reference(rec.title):
                    skipped_total += 1
                    continue
                if dry_run:
                    logging.info("DRY RUN: would save %s (%s)", rec.title[:80], pid)
                    new_per_topic[topic].append((rec, None))
                    continue
                day_dir = waiting_review / topic / _today()
                pdf_path, json_path = save_record(rec, day_dir, download_pdf=True)
                state.mark_seen(
                    pid,
                    topic,
                    pdf_path=str(pdf_path.relative_to(repo_root)) if pdf_path else None,
                    pdf_status="ok" if pdf_path else "no_oa",
                    all_ids=aliases,
                )
                new_per_topic[topic].append((rec, pdf_path))
                written_paths.append(str(json_path.relative_to(repo_root)))
                if pdf_path:
                    written_paths.append(str(pdf_path.relative_to(repo_root)))
        state.update_last_run(topic)

    # Persist state + per-slot digest
    if not dry_run:
        state.save()
        written_paths.append(str(state.state_path.relative_to(repo_root)))
        digest_path = digest_mod.append_slot_digest(
            waiting_review / "_digests",
            slot,
            [item for items in new_per_topic.values() for item in items],
            skipped_total,
            repo_root=repo_root,
        )
        written_paths.append(str(digest_path.relative_to(repo_root)))

    # Slot 9: also produce daily summary
    if plan.get("digest_only") and not dry_run:
        summary = digest_mod.write_daily_summary(
            waiting_review / "_digests",
            waiting_review,
        )
        written_paths.append(str(summary.relative_to(repo_root)))

    # Build summary line
    counts = {t: len(v) for t, v in new_per_topic.items()}
    summary_line = (
        f"[slot {slot}] "
        + " ".join(f"{t}={counts[t]}" for t in TOPICS)
        + f" (skipped {skipped_total})"
    )
    print(summary_line)

    pushed_sha = None
    if push and not dry_run and written_paths:
        commit_msg = _build_commit_msg(slot, new_per_topic)
        pushed_sha = auto_push(repo_root, written_paths, commit_msg)
        if pushed_sha:
            print(f"pushed: {pushed_sha[:8]}")
        else:
            print("push: nothing to commit or push failed")

    return {
        "counts": counts,
        "skipped": skipped_total,
        "pushed_sha": pushed_sha,
        "written_paths": written_paths,
    }


def _build_commit_msg(slot: int, new_per_topic: dict) -> str:
    parts = [f"+{len(v)} {t}" for t, v in new_per_topic.items() if v]
    head = f"chore(lit-monitor): slot {slot}" + (f" — {', '.join(parts)}" if parts else " — no new") + " [auto]"
    body_lines: list[str] = []
    for t, items in new_per_topic.items():
        if not items:
            continue
        body_lines.append(f"\n{t}:")
        for rec, _ in items[:5]:
            body_lines.append(f"  - {rec.primary_id()} — {rec.title[:90]}")
        if len(items) > 5:
            body_lines.append(f"  ... ({len(items) - 5} more)")
    return head + ("\n" + "\n".join(body_lines) if body_lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Alz Face literature monitor")
    parser.add_argument("--slot", type=int, default=None, help="0-9 slot per SLOT_PLAN")
    parser.add_argument("--topic", choices=list(TOPICS) + ["all"], default=None,
                        help="manual single-topic run; overrides slot's topic list")
    parser.add_argument("--max-per-source", type=int, default=25)
    parser.add_argument("--year-from", type=int, default=2018)
    parser.add_argument("--dry-run", action="store_true", help="search only; no download/state/push")
    parser.add_argument("--auto-push", action="store_true", help="git pull/commit/push after run")
    parser.add_argument("--no-push", action="store_true", help="explicitly disable push")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="rebuild references/_indexed.json from existing PDFs and exit")
    parser.add_argument("--rebuild-aliases", action="store_true",
                        help="back-fill _state.json aliases from existing waiting_review JSON sidecars and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = _project_root()
    log_path = repo_root / "references" / "waiting_review" / "_logs" / f"{_today()}.log"
    _setup_logging(log_path, verbose=args.verbose)

    if args.rebuild_index:
        out = write_reference_index(repo_root / "references")
        print(f"wrote {out.relative_to(repo_root)}")
        return 0

    if args.rebuild_aliases:
        import json as _json
        wr = repo_root / "references" / "waiting_review"
        state = StateStore(wr, repo_root / "references")
        added = 0
        for sidecar in wr.glob("*/*/*.json"):
            try:
                meta = _json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue
            primary = meta.get("primary_id") or ""
            if not primary:
                continue
            ids = []
            if meta.get("arxiv_id"):
                ids.append(f"arxiv:{meta['arxiv_id']}")
            if meta.get("doi"):
                ids.append(f"doi:{meta['doi'].lower()}")
            pmid = (meta.get("extra") or {}).get("pmid")
            if pmid:
                ids.append(f"pmid:{pmid}")
            title = meta.get("title", "")
            if title:
                import re as _re
                norm = _re.sub(r"\s+", " ", title.strip().lower())[:120]
                ids.append(f"title:{norm}")
            for alias in ids:
                if alias != primary and alias not in state._state["aliases"]:
                    state._state["aliases"][alias] = primary
                    added += 1
        state.save()
        print(f"added {added} aliases to _state.json")
        return 0

    if args.slot is None and args.topic is None:
        parser.error("either --slot or --topic is required")

    push = args.auto_push and not args.no_push

    if args.slot is not None:
        override = None
        if args.topic and args.topic != "all":
            override = [args.topic]
        run_slot(
            args.slot,
            repo_root,
            max_per_source=args.max_per_source,
            year_from=args.year_from,
            dry_run=args.dry_run,
            push=push,
            override_topics=override,
        )
    else:
        # Manual --topic without slot: search across all sources, broad query
        topics = list(TOPICS) if args.topic == "all" else [args.topic]
        run_slot(
            slot=999,  # synthetic slot id, not in SLOT_PLAN
            repo_root=repo_root,
            max_per_source=args.max_per_source,
            year_from=args.year_from,
            dry_run=args.dry_run,
            push=push,
            override_topics=topics,
            override_sources=["arxiv", "s2", "openalex", "pubmed"],
            override_query_idx=0,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
