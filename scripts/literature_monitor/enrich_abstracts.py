"""Back-fill missing abstracts in existing waiting_review JSON sidecars.

Walks references/waiting_review/<topic>/<DATE>/*.json. For each record with
an empty abstract, refetches abstract from the appropriate API based on
`source` field and writes it back into the JSON.

Usage:
    python -m scripts.literature_monitor.enrich_abstracts --dry-run
    python -m scripts.literature_monitor.enrich_abstracts --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.literature_monitor.sources import (
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    _decode_inverted_index,
    _get,
    _pubmed_fetch_abstracts,
)

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("project root not found")


def fetch_openalex_abstract(openalex_id: str) -> str:
    """openalex_id is a URL like 'https://openalex.org/W123456789'."""
    if not openalex_id:
        return ""
    work_id = openalex_id.rsplit("/", 1)[-1]
    try:
        time.sleep(0.2)
        r = _get(
            f"https://api.openalex.org/works/{work_id}",
            params={"mailto": "a1234567891934@gmail.com"},
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        return _decode_inverted_index(r.json().get("abstract_inverted_index"))
    except Exception as e:
        logger.warning("openalex fetch %s failed: %s", work_id, e)
        return ""


def fetch_arxiv_abstract(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    try:
        time.sleep(3)  # arxiv courtesy
        r = _get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id},
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            summary = entry.find("a:summary", ns)
            if summary is not None and summary.text:
                return summary.text.strip()
    except Exception as e:
        logger.warning("arxiv fetch %s failed: %s", arxiv_id, e)
    return ""


def fetch_s2_abstract(doi: str | None, arxiv_id: str | None, s2_id: str | None) -> str:
    """Try S2 by paperId, then by DOI, then by arxiv. Uses S2_API_KEY if set."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    candidates = []
    if s2_id:
        candidates.append(s2_id)
    if doi:
        candidates.append(f"DOI:{doi}")
    if arxiv_id:
        candidates.append(f"arXiv:{arxiv_id}")

    for cand in candidates:
        try:
            time.sleep(1.1)
            r = _get(
                f"https://api.semanticscholar.org/graph/v1/paper/{cand}",
                params={"fields": "abstract"},
                headers=headers,
            )
            abst = (r.json() or {}).get("abstract") or ""
            if abst:
                return abst.strip()
        except Exception as e:
            logger.debug("s2 fetch %s failed: %s", cand, e)
    return ""


def main(apply: bool, max_papers: int | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = _project_root()
    base = repo_root / "references" / "waiting_review"

    targets: list[tuple[Path, dict]] = []
    for j in base.glob("*/*/*.json"):
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (meta.get("abstract") or "").strip():
            continue
        targets.append((j, meta))

    if max_papers:
        targets = targets[:max_papers]

    by_source: dict[str, list[tuple[Path, dict]]] = {}
    for path, meta in targets:
        by_source.setdefault(meta.get("source", "?"), []).append((path, meta))

    print(f"Empty-abstract records: {len(targets)} total")
    for source, items in by_source.items():
        print(f"  {source}: {len(items)}")

    if not apply:
        print("\n(dry-run; pass --apply to actually fetch and write)")
        return 0

    # PubMed: batch via efetch (max 200 IDs per call)
    pubmed_items = by_source.get("pubmed", [])
    if pubmed_items:
        print(f"\nFetching {len(pubmed_items)} PubMed abstracts via efetch...")
        for batch_start in range(0, len(pubmed_items), 100):
            batch = pubmed_items[batch_start:batch_start + 100]
            pmids = [m.get("extra", {}).get("pmid") for _, m in batch if m.get("extra", {}).get("pmid")]
            if not pmids:
                continue
            abstracts = _pubmed_fetch_abstracts(
                pmids, headers={"User-Agent": DEFAULT_USER_AGENT}
            )
            for path, meta in batch:
                pmid = meta.get("extra", {}).get("pmid")
                if pmid and abstracts.get(pmid):
                    meta["abstract"] = abstracts[pmid]
                    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
            print(f"  pubmed batch {batch_start}+: wrote {sum(1 for p,m in batch if m.get('abstract'))}")

    # OpenAlex: one-by-one
    oa_items = by_source.get("openalex", [])
    if oa_items:
        print(f"\nFetching {len(oa_items)} OpenAlex abstracts...")
        ok = 0
        for path, meta in oa_items:
            oa_id = (meta.get("extra") or {}).get("openalex_id")
            abst = fetch_openalex_abstract(oa_id)
            if abst:
                meta["abstract"] = abst
                path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                ok += 1
        print(f"  openalex: {ok}/{len(oa_items)} filled")

    # arXiv: one-by-one (skip — should already have abstract from initial fetch)
    arxiv_items = by_source.get("arxiv", [])
    if arxiv_items:
        print(f"\nFetching {len(arxiv_items)} arXiv abstracts...")
        ok = 0
        for path, meta in arxiv_items:
            abst = fetch_arxiv_abstract(meta.get("arxiv_id"))
            if abst:
                meta["abstract"] = abst
                path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                ok += 1
        print(f"  arxiv: {ok}/{len(arxiv_items)} filled")

    # S2: one-by-one (rare since we already store from initial search)
    s2_items = by_source.get("s2", [])
    if s2_items:
        print(f"\nFetching {len(s2_items)} S2 abstracts...")
        ok = 0
        for path, meta in s2_items:
            abst = fetch_s2_abstract(
                meta.get("doi"), meta.get("arxiv_id"),
                (meta.get("extra") or {}).get("s2_paperId"),
            )
            if abst:
                meta["abstract"] = abst
                path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                ok += 1
        print(f"  s2: {ok}/{len(s2_items)} filled")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="cap how many records to enrich (for testing)")
    args = parser.parse_args()
    sys.exit(main(apply=args.apply, max_papers=args.max_papers))
