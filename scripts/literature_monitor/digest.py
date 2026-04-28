"""Markdown digest writers for the literature monitor.

- `append_slot_digest(...)` after each slot run records what was newly added.
- `write_daily_summary(...)` (slot 9) aggregates the day's results.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from scripts.literature_monitor.sources import PaperRecord


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def append_slot_digest(
    digest_dir: Path,
    slot: int,
    new_records: Iterable[tuple[PaperRecord, Path | None]],
    skipped_count: int,
    repo_root: Path | None = None,
) -> Path:
    digest_dir.mkdir(parents=True, exist_ok=True)
    out = digest_dir / f"{_today()}.md"
    is_new_file = not out.exists()
    lines: list[str] = []
    if is_new_file:
        lines.append(f"# Literature Monitor Digest — {datetime.now().strftime('%Y-%m-%d')}\n")
    lines.append(f"\n## slot {slot} ({datetime.now().strftime('%H:%M:%S')})")
    new_list = list(new_records)
    lines.append(f"- new: {len(new_list)}, skipped (dup): {skipped_count}")
    for rec, pdf_path in new_list:
        if pdf_path:
            link = (
                pdf_path.relative_to(repo_root).as_posix()
                if repo_root else pdf_path.as_posix()
            )
            pdf_str = f" [PDF](/{link})"
        else:
            pdf_str = " *(no OA PDF)*"
        venue = rec.venue or rec.source
        lines.append(
            f"- **{rec.title}**{pdf_str}\n"
            f"  - {', '.join(rec.authors[:3])}{'...' if len(rec.authors) > 3 else ''}"
            f" · {rec.year} · {venue} · src={rec.source}"
            f"{' · cites=' + str(rec.citation_count) if rec.citation_count else ''}"
        )
    with out.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def write_daily_summary(
    digest_dir: Path,
    waiting_review_dir: Path,
) -> Path:
    """Aggregate today's per-topic counts and top-5 most-cited papers.

    Reads JSON metadata sidecars in `waiting_review_dir/<topic>/<TODAY>/*.json`.
    """
    today = _today()
    out = digest_dir / f"{today}_summary.md"
    by_topic: dict[str, list[dict]] = {}
    for topic_dir in sorted(p for p in waiting_review_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        day_dir = topic_dir / today
        if not day_dir.exists():
            continue
        items = []
        for j in day_dir.glob("*.json"):
            try:
                items.append(json.loads(j.read_text(encoding="utf-8")))
            except Exception:
                continue
        by_topic[topic_dir.name] = items

    lines = [f"# Daily Summary — {datetime.now().strftime('%Y-%m-%d')}\n"]
    total = sum(len(v) for v in by_topic.values())
    lines.append(f"- **total new today**: {total}")
    for topic, items in by_topic.items():
        lines.append(f"- {topic}: {len(items)}")
    lines.append("\n## Top 5 by citation count\n")
    flat = [it for items in by_topic.values() for it in items]
    flat.sort(key=lambda x: (x.get("citation_count") or 0), reverse=True)
    for it in flat[:5]:
        lines.append(
            f"- **{it.get('title')}** ({it.get('year')}, {it.get('venue') or it.get('source')})"
            f" — cites={it.get('citation_count') or 'n/a'}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
