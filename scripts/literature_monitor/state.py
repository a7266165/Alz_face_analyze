"""Dedup state for the literature monitor.

`_state.json` records all paper IDs that have been seen by any prior run, so
subsequent runs do not re-download them. Existing `references/<topic>/*.pdf`
files are also cross-checked via a sidecar index `references/_indexed.json`.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

STATE_FILE = "_state.json"
INDEXED_FILE = "_indexed.json"


class StateStore:
    def __init__(self, waiting_review_dir: Path, references_dir: Path):
        self.waiting_review_dir = waiting_review_dir
        self.references_dir = references_dir
        self.state_path = waiting_review_dir / STATE_FILE
        self.indexed_path = references_dir / INDEXED_FILE
        self._state: dict = {"seen_ids": {}, "last_run": {}}
        self._existing_titles: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to load %s: %s; starting empty", self.state_path, e)
                self._state = {"seen_ids": {}, "last_run": {}}
        # Indexed reference titles (from existing references/<topic>/*.pdf)
        if self.indexed_path.exists():
            try:
                idx = json.loads(self.indexed_path.read_text(encoding="utf-8"))
                self._existing_titles = {_norm_title(t) for t in idx.get("titles", [])}
            except Exception as e:
                logger.warning("Failed to load %s: %s", self.indexed_path, e)

    def is_seen(self, primary_id: str) -> bool:
        return primary_id in self._state["seen_ids"]

    def is_existing_reference(self, title: str) -> bool:
        return _norm_title(title) in self._existing_titles

    def mark_seen(
        self,
        primary_id: str,
        topic: str,
        pdf_path: str | None = None,
        pdf_status: str = "ok",
    ) -> None:
        self._state["seen_ids"][primary_id] = {
            "first_seen": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "topic": topic,
            "pdf_path": pdf_path,
            "pdf_status": pdf_status,
        }

    def update_last_run(self, topic: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._state["last_run"][topic] = ts

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    @property
    def state_dict(self) -> dict:
        return self._state


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


# ---------------------------------------------------------------------------
# Builder for references/_indexed.json (one-shot, scans existing references PDFs)
# ---------------------------------------------------------------------------
def build_reference_index(references_dir: Path) -> dict:
    """Scan references/<topic>/*.pdf and produce an index of titles.

    Title is best-effort extracted from filename (stem). This is used for
    coarse dedup against existing references.
    """
    titles: list[str] = []
    paths: list[str] = []
    for sub in references_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_") or sub.name == "waiting_review":
            continue
        for pdf in sub.glob("*.pdf"):
            stem = pdf.stem
            # Try to strip leading author/year-style prefix to get a rough title
            cleaned = re.sub(r"^[A-Za-z]+_\d{4}_", "", stem)
            cleaned = re.sub(r"^\d{4}-", "", cleaned)
            cleaned = cleaned.replace("_", " ")
            titles.append(cleaned)
            paths.append(str(pdf.relative_to(references_dir)))
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "titles": titles,
        "paths": paths,
    }


def write_reference_index(references_dir: Path) -> Path:
    idx = build_reference_index(references_dir)
    out = references_dir / INDEXED_FILE
    out.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
