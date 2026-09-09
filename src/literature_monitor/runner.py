"""每次 sweep 的編排 + git 自動推送 + 索引/別名重建。

CLI 入口在 scripts/literature_monitor/run.py。
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from . import digest, sources
from .download import save_record
from .queries import SLOT_PLAN, TOPICS, query_for
from .sources import PaperRecord
from .state import StateStore, iter_sidecars
from src.config import LIT_QUEUE_DIR, REFERENCES_DIR


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


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
    download_pdf: bool = True,
) -> dict:
    """跑一個 slot：topic × source 搜尋 → 去重 → 存檔 → digest →（選擇性）git push。"""
    plan = SLOT_PLAN.get(slot, {"topics": [], "sources": [], "query_idx": 0})
    if plan.get("digest_only"):
        topics, srcs, query_idx = [], [], -1
    else:
        topics = override_topics if override_topics is not None else plan["topics"]
        srcs = override_sources if override_sources is not None else plan["sources"]
        query_idx = override_query_idx if override_query_idx is not None else plan["query_idx"]

    waiting_review = LIT_QUEUE_DIR
    state = StateStore(waiting_review, REFERENCES_DIR)

    new_per_topic: dict[str, list[tuple[PaperRecord, Path | None]]] = {t: [] for t in TOPICS}
    skipped_total = 0
    written_paths: list[str] = []

    for topic in topics:
        q = query_for(topic, query_idx)
        for source in srcs:
            cursor = state.get_cursor(source, topic, q)
            logging.info("[slot %d] %s x %s @offset=%d :: %s", slot, topic, source, cursor, q)
            try:
                records = sources.search(source, q, max_results=max_per_source,
                                         year_from=year_from, offset=cursor)
            except Exception as e:
                logging.warning("search %s/%s failed: %s", source, topic, e)
                continue
            # cursor：有結果就前進，取盡則歸零
            if not records:
                if cursor > 0:
                    logging.info("  exhausted at offset %d; resetting cursor", cursor)
                    state.reset_cursor(source, topic, q)
            else:
                state.advance_cursor(source, topic, q, len(records))
            for rec in records:
                if not rec.title:
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
                pdf_path, json_path = save_record(rec, day_dir, download_pdf=download_pdf)
                state.mark_seen(
                    pid, topic,
                    pdf_path=str(pdf_path.relative_to(waiting_review)) if pdf_path else None,
                    pdf_status="ok" if pdf_path else "no_oa",
                    all_ids=aliases,
                )
                new_per_topic[topic].append((rec, pdf_path))
                written_paths.append(str(json_path))
                if pdf_path:
                    written_paths.append(str(pdf_path))
        state.update_last_run(topic)

    if not dry_run:
        state.save()
        written_paths.append(str(state.state_path))
        digest_path = digest.append_slot_digest(
            waiting_review / "_digests", slot,
            [item for items in new_per_topic.values() for item in items],
            skipped_total, repo_root=waiting_review,  # 連結相對於佇列根（已搬離 repo）
        )
        written_paths.append(str(digest_path))

    if plan.get("digest_only") and not dry_run:
        summary = digest.write_daily_summary(waiting_review / "_digests", waiting_review)
        written_paths.append(str(summary))

    counts = {t: len(v) for t, v in new_per_topic.items()}
    print(f"[slot {slot}] " + " ".join(f"{t}={counts[t]}" for t in TOPICS)
          + f" (skipped {skipped_total})")

    pushed_sha = None
    if push and not dry_run and written_paths:
        # 佇列已搬離 repo（D:\Alz 佈局）：只有佇列目錄自己是 git repo 時才推送
        if (waiting_review / ".git").exists():
            pushed_sha = auto_push(waiting_review, written_paths, _build_commit_msg(slot, new_per_topic))
        else:
            logging.info("auto_push skipped: %s is not a git repo", waiting_review)
        print(f"pushed: {pushed_sha[:8]}" if pushed_sha else "push: nothing to commit or failed")

    return {"counts": counts, "skipped": skipped_total,
            "pushed_sha": pushed_sha, "written_paths": written_paths}


def _build_commit_msg(slot: int, new_per_topic: dict) -> str:
    parts = [f"+{len(v)} {t}" for t, v in new_per_topic.items() if v]
    head = f"chore(lit-monitor): slot {slot}" + (f" — {', '.join(parts)}" if parts else " — no new") + " [auto]"
    body: list[str] = []
    for t, items in new_per_topic.items():
        if not items:
            continue
        body.append(f"\n{t}:")
        body += [f"  - {rec.primary_id()} — {rec.title[:90]}" for rec, _ in items[:5]]
        if len(items) > 5:
            body.append(f"  ... ({len(items) - 5} more)")
    return head + ("\n" + "\n".join(body) if body else "")


def rebuild_aliases(repo_root: Path) -> int:
    """從既有 sidecar 回填 _state.json 的別名（用 PaperRecord.all_ids）。回新增筆數。"""
    wr = LIT_QUEUE_DIR
    state = StateStore(wr, REFERENCES_DIR)
    aliases = state.state_dict["aliases"]
    added = 0
    for _, meta in iter_sidecars(wr):
        primary = meta.get("primary_id") or ""
        if not primary:
            continue
        for alias in PaperRecord.from_dict(meta).all_ids():
            if alias != primary and alias not in aliases:
                aliases[alias] = primary
                added += 1
    state.save()
    return added


# ---------------------------------------------------------------------------
# git 自動推送（commit-first-then-rebase）
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    logging.debug("git: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def auto_push(repo_root: Path, paths: list[str], commit_msg: str) -> str | None:
    """stage→commit→rebase origin/main→push，回 SHA 或 None。"""
    for p in paths:
        _run(["git", "add", "--", p], cwd=repo_root, check=False)
    staged = _run(["git", "diff", "--cached", "--name-only"], cwd=repo_root, check=False).stdout.strip()
    if not staged:
        logging.info("no staged changes to commit")
        _run(["git", "fetch", "origin", "main"], cwd=repo_root, check=False)
        return None
    try:
        _run(["git", "commit", "-m", commit_msg], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        logging.error("git commit failed: %s", e.stderr)
        return None
    try:
        _run(["git", "fetch", "origin", "main"], cwd=repo_root)
        _run(["git", "rebase", "origin/main"], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        logging.warning("rebase failed; aborting and skipping push: %s", e.stderr)
        _run(["git", "rebase", "--abort"], cwd=repo_root, check=False)
        return None
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    try:
        _run(["git", "push", "origin", "main"], cwd=repo_root)
        return sha
    except subprocess.CalledProcessError as e:
        logging.error("git push failed: %s", e.stderr)
        return None
