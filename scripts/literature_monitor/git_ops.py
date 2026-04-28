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
    """Stage `paths`, commit locally, rebase onto remote, push. Returns SHA or None.

    Pattern: commit-first-then-rebase. Pull --rebase requires a clean tree, so
    we commit our changes first (creating a tree-clean state), then rebase the
    new commit on top of remote main, then push.
    """
    # Stage everything first
    for p in paths:
        _run(["git", "add", "--", p], cwd=repo_root, check=False)

    # If nothing was staged (after dedup runs leave _state.json idempotent), bail
    staged = _run(
        ["git", "diff", "--cached", "--name-only"], cwd=repo_root, check=False
    ).stdout.strip()
    if not staged:
        logger.info("no staged changes to commit")
        # Still try to fetch/sync so next run starts clean
        _run(["git", "fetch", "origin", "main"], cwd=repo_root, check=False)
        return None

    try:
        _run(["git", "commit", "-m", commit_msg], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        logger.error("git commit failed: %s", e.stderr)
        return None

    # Rebase onto latest remote main
    try:
        _run(["git", "fetch", "origin", "main"], cwd=repo_root)
        _run(["git", "rebase", "origin/main"], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        logger.warning("rebase failed; aborting and skipping push: %s", e.stderr)
        _run(["git", "rebase", "--abort"], cwd=repo_root, check=False)
        return None

    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    try:
        _run(["git", "push", "origin", "main"], cwd=repo_root)
        return sha
    except subprocess.CalledProcessError as e:
        logger.error("git push failed: %s", e.stderr)
        return None
