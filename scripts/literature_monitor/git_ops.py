"""Git automation for the literature monitor.

Wraps `git pull --rebase`, `add`, `commit`, `push` with conflict tolerance.
Used by `run.py --auto-push`.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("git: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _has_changes(cwd: Path) -> bool:
    r = _run(["git", "status", "--porcelain"], cwd=cwd, check=False)
    return bool(r.stdout.strip())


def auto_push(repo_root: Path, paths: list[str], commit_msg: str) -> str | None:
    """Pull --rebase, stage `paths`, commit, push. Returns commit SHA or None.

    On rebase conflict: stash, retry; if still failing, abort and return None.
    """
    try:
        _run(["git", "pull", "--rebase", "origin", "main"], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        logger.warning("git pull --rebase failed: %s", e.stderr)
        # Try recovery: stash, rebase abort, pull again, pop
        try:
            _run(["git", "rebase", "--abort"], cwd=repo_root, check=False)
            _run(["git", "stash"], cwd=repo_root, check=False)
            _run(["git", "pull", "--rebase", "origin", "main"], cwd=repo_root)
            _run(["git", "stash", "pop"], cwd=repo_root, check=False)
        except subprocess.CalledProcessError as e2:
            logger.error("git rebase recovery failed: %s", e2.stderr)
            return None

    for p in paths:
        _run(["git", "add", "--", p], cwd=repo_root, check=False)

    if not _has_changes(repo_root):
        logger.info("no changes to commit")
        return None

    try:
        _run(["git", "commit", "-m", commit_msg], cwd=repo_root)
        sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        _run(["git", "push", "origin", "main"], cwd=repo_root)
        return sha
    except subprocess.CalledProcessError as e:
        logger.error("git commit/push failed: %s", e.stderr)
        return None
