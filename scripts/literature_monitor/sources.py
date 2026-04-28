"""API clients for paper metadata sources.

All clients return a list of `PaperRecord` objects with a unified schema, so
downstream code (download / state / digest) does not care which source it
came from.

Supported sources:
- arxiv     : arXiv (uses `arxiv` python package)
- s2        : Semantic Scholar Graph API
- pubmed    : NCBI E-utilities (esearch + esummary)
- openalex  : OpenAlex
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "AlzFaceLitMonitor/0.1 (mailto:a1234567891934@gmail.com)"
)
DEFAULT_TIMEOUT = 30


@dataclass
class PaperRecord:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    doi: str | None
    arxiv_id: str | None
    venue: str | None
    citation_count: int | None
    source: str
    pdf_url: str | None
    relevance_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def primary_id(self) -> str:
        """Stable canonical ID for dedup."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        # Fallback: title-based hash
        norm = re.sub(r"\s+", " ", self.title.strip().lower())
        return f"title:{norm[:120]}"


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search_arxiv(query: str, max_results: int = 25, year_from: int = 2018) -> list[PaperRecord]:
    """Direct hit to arXiv's Atom API (no `arxiv` package — avoids sgmllib3k chain)."""
    import xml.etree.ElementTree as ET

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(3)  # arXiv courtesy delay
        r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        logger.warning("arxiv search failed: %s", e)
        return []

    out: list[PaperRecord] = []
    try:
        root = ET.fromstring(r.text)
        for entry in root.findall("a:entry", _ARXIV_NS):
            title_el = entry.find("a:title", _ARXIV_NS)
            summary_el = entry.find("a:summary", _ARXIV_NS)
            published_el = entry.find("a:published", _ARXIV_NS)
            id_el = entry.find("a:id", _ARXIV_NS)
            doi_el = entry.find("arxiv:doi", _ARXIV_NS)
            if title_el is None or id_el is None or published_el is None:
                continue
            year_str = (published_el.text or "")[:4]
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year < year_from:
                continue
            authors = [
                (a.find("a:name", _ARXIV_NS).text or "").strip()
                for a in entry.findall("a:author", _ARXIV_NS)
                if a.find("a:name", _ARXIV_NS) is not None
            ]
            arxiv_id = (id_el.text or "").rsplit("/", 1)[-1].split("v")[0]
            pdf_url = None
            for link in entry.findall("a:link", _ARXIV_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            out.append(
                PaperRecord(
                    title=(title_el.text or "").strip(),
                    authors=authors,
                    year=year,
                    abstract=(summary_el.text or "").strip() if summary_el is not None else "",
                    doi=(doi_el.text or "").strip() if doi_el is not None else None,
                    arxiv_id=arxiv_id,
                    venue="arXiv",
                    citation_count=None,
                    source="arxiv",
                    pdf_url=pdf_url,
                )
            )
    except ET.ParseError as e:
        logger.warning("arxiv XML parse failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------
S2_FIELDS = ",".join([
    "title",
    "authors.name",
    "year",
    "abstract",
    "externalIds",
    "venue",
    "citationCount",
    "openAccessPdf",
])


def search_semantic_scholar(
    query: str, max_results: int = 25, year_from: int = 2018
) -> list[PaperRecord]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": S2_FIELDS,
        "year": f"{year_from}-",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(1.1)  # 1 req/sec for unauthenticated
        r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 429:
            logger.warning("S2 rate-limited; sleeping 10s and retrying once")
            time.sleep(10)
            r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("Semantic Scholar search failed: %s", e)
        return []

    out: list[PaperRecord] = []
    for p in data.get("data", []):
        ext = p.get("externalIds") or {}
        oa = (p.get("openAccessPdf") or {}).get("url")
        out.append(
            PaperRecord(
                title=(p.get("title") or "").strip(),
                authors=[a["name"] for a in (p.get("authors") or []) if a.get("name")],
                year=p.get("year"),
                abstract=(p.get("abstract") or "").strip(),
                doi=ext.get("DOI"),
                arxiv_id=ext.get("ArXiv"),
                venue=p.get("venue") or None,
                citation_count=p.get("citationCount"),
                source="s2",
                pdf_url=oa,
                extra={"s2_paperId": p.get("paperId")},
            )
        )
    return out


# ---------------------------------------------------------------------------
# PubMed (E-utilities)
# ---------------------------------------------------------------------------
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _pubmed_translate(query: str) -> str:
    # PubMed accepts AND/OR/parens & quoted phrases natively; keep query as-is.
    return query


def search_pubmed(query: str, max_results: int = 25, year_from: int = 2018) -> list[PaperRecord]:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(0.4)
        r = requests.get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": _pubmed_translate(query) + f" AND ({year_from}:3000[dp])",
                "retmax": max_results,
                "retmode": "json",
            },
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            return []
        time.sleep(0.4)
        s = requests.get(
            f"{PUBMED_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        s.raise_for_status()
        result = s.json().get("result", {})
    except Exception as e:
        logger.warning("PubMed search failed: %s", e)
        return []

    out: list[PaperRecord] = []
    for pmid in ids:
        rec = result.get(pmid)
        if not rec:
            continue
        doi = None
        for a in rec.get("articleids", []):
            if a.get("idtype") == "doi":
                doi = a.get("value")
        year_str = (rec.get("pubdate") or "").split(" ")[0]
        try:
            year = int(year_str[:4])
        except (ValueError, TypeError):
            year = None
        authors = [a.get("name", "") for a in rec.get("authors", []) if a.get("name")]
        out.append(
            PaperRecord(
                title=(rec.get("title") or "").strip(),
                authors=authors,
                year=year,
                abstract="",  # PubMed esummary does not return abstract; efetch needed
                doi=doi,
                arxiv_id=None,
                venue=rec.get("fulljournalname") or rec.get("source"),
                citation_count=None,
                source="pubmed",
                pdf_url=None,  # try via Unpaywall in download.py
                extra={"pmid": pmid},
            )
        )
    return out


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def search_openalex(query: str, max_results: int = 25, year_from: int = 2018) -> list[PaperRecord]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "filter": f"from_publication_date:{year_from}-01-01",
        "mailto": "a1234567891934@gmail.com",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(0.2)
        r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("OpenAlex search failed: %s", e)
        return []

    out: list[PaperRecord] = []
    for w in data.get("results", []):
        ids = w.get("ids", {}) or {}
        doi = (ids.get("doi") or "").replace("https://doi.org/", "") or None
        # OpenAlex stores abstract as inverted index — skip reconstruction here.
        oa = (w.get("best_oa_location") or {}).get("pdf_url")
        out.append(
            PaperRecord(
                title=(w.get("title") or "").strip(),
                authors=[
                    (a.get("author") or {}).get("display_name", "")
                    for a in (w.get("authorships") or [])
                ],
                year=w.get("publication_year"),
                abstract="",
                doi=doi,
                arxiv_id=None,
                venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                citation_count=w.get("cited_by_count"),
                source="openalex",
                pdf_url=oa,
                extra={"openalex_id": w.get("id")},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
SEARCHERS = {
    "arxiv": search_arxiv,
    "s2": search_semantic_scholar,
    "pubmed": search_pubmed,
    "openalex": search_openalex,
}


def search(source: str, query: str, max_results: int = 25, year_from: int = 2018) -> list[PaperRecord]:
    fn = SEARCHERS.get(source)
    if fn is None:
        raise ValueError(f"unknown source: {source}")
    return fn(query, max_results=max_results, year_from=year_from)
