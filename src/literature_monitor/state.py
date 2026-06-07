"""去重狀態 + sidecar JSON 讀寫。

_state.json 記所有看過的 paper id（避免重抓），並用 _indexed.json 對既有
references/<topic>/*.pdf 做粗比對；另記 (source, topic, query) 的分頁 cursor。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = "_state.json"
INDEXED_FILE = "_indexed.json"


def _cursor_key(source: str, topic: str, query: str) -> str:
    qhash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
    return f"{source}|{topic}|{qhash}"


class StateStore:
    def __init__(self, waiting_review_dir: Path, references_dir: Path):
        self.waiting_review_dir = waiting_review_dir
        self.references_dir = references_dir
        self.state_path = waiting_review_dir / STATE_FILE
        self.indexed_path = references_dir / INDEXED_FILE
        self._state: dict = {
            "seen_ids": {}, "aliases": {}, "cursors": {}, "last_run": {},
        }
        self._existing_titles: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to load %s: %s; starting empty", self.state_path, e)
                self._state = {"seen_ids": {}, "aliases": {}, "last_run": {}}
        # Backward-compat for older state files
        self._state.setdefault("aliases", {})
        self._state.setdefault("seen_ids", {})
        self._state.setdefault("last_run", {})
        self._state.setdefault("cursors", {})
        # Indexed reference titles (from existing references/<topic>/*.pdf)
        if self.indexed_path.exists():
            try:
                idx = json.loads(self.indexed_path.read_text(encoding="utf-8"))
                self._existing_titles = {_norm_title(t) for t in idx.get("titles", [])}
            except Exception as e:
                logger.warning("Failed to load %s: %s", self.indexed_path, e)

    def is_seen(self, primary_id: str, all_ids: list[str] | None = None) -> bool:
        """True if primary_id OR any of `all_ids` matches a known canonical or alias."""
        if primary_id in self._state["seen_ids"]:
            return True
        if all_ids is None:
            return False
        for alias in all_ids:
            if alias in self._state["aliases"]:
                return True
            if alias in self._state["seen_ids"]:
                return True
        return False

    def is_existing_reference(self, title: str) -> bool:
        return _norm_title(title) in self._existing_titles

    def mark_seen(
        self,
        primary_id: str,
        topic: str,
        pdf_path: str | None = None,
        pdf_status: str = "ok",
        all_ids: list[str] | None = None,
    ) -> None:
        self._state["seen_ids"][primary_id] = {
            "first_seen": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "topic": topic,
            "pdf_path": pdf_path,
            "pdf_status": pdf_status,
        }
        # Register all known IDs of this record as aliases pointing to primary_id
        if all_ids:
            for alias in all_ids:
                if alias != primary_id:
                    self._state["aliases"][alias] = primary_id

    def update_last_run(self, topic: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._state["last_run"][topic] = ts

    def get_cursor(self, source: str, topic: str, query: str) -> int:
        return self._state["cursors"].get(_cursor_key(source, topic, query), 0)

    def advance_cursor(self, source: str, topic: str, query: str, increment: int) -> None:
        key = _cursor_key(source, topic, query)
        self._state["cursors"][key] = self._state["cursors"].get(key, 0) + increment

    def reset_cursor(self, source: str, topic: str, query: str) -> None:
        self._state["cursors"][_cursor_key(source, topic, query)] = 0

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
# references/_indexed.json 建置 + sidecar JSON 讀寫
# ---------------------------------------------------------------------------
def build_reference_index(references_dir: Path) -> dict:
    """掃 references/<topic>/*.pdf，從檔名粗抽 title 建索引（供粗比對去重）。"""
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


def iter_sidecars(waiting_review_dir: Path):
    """走訪 waiting_review/<topic>/<date>/*.json，yield (path, meta)；壞檔略過。"""
    for path in waiting_review_dir.glob("*/*/*.json"):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue


def write_meta(path: Path, meta: dict) -> None:
    """寫回 JSON sidecar。"""
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
