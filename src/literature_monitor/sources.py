"""論文來源 API client：各來源都回統一的 PaperRecord，下游不分來源。

來源：arxiv / s2（Semantic Scholar）/ pubmed / openalex。
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "AlzFaceLitMonitor/0.1 (mailto:a1234567891934@gmail.com)"
)
DEFAULT_TIMEOUT = 30


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 / 5xx / network errors. Do NOT retry on 4xx other than 429."""
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status is None:
            return False
        return status == 429 or 500 <= status < 600
    return False


def _get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_attempts: int = 4,
    max_wait: int = 30,
) -> requests.Response:
    """GET，對 429/5xx/網路錯誤指數退避重試；共用 rate-limit 的端點（如未認證 S2）可調大 max_attempts/max_wait。"""

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
    def _do() -> requests.Response:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            logger.info("HTTP %s on %s; will retry with backoff", r.status_code, url)
        r.raise_for_status()
        return r

    return _do()


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
        """Stable canonical ID for dedup. Prefers arxiv > doi > title."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        norm = re.sub(r"\s+", " ", self.title.strip().lower())
        return f"title:{norm[:120]}"

    def all_ids(self) -> list[str]:
        """此 record 的所有已知 ID，供跨來源 dedup 別名（不同來源填的 id 欄不同，全登記才能互相命中）。"""
        ids: list[str] = []
        if self.arxiv_id:
            ids.append(f"arxiv:{self.arxiv_id}")
        if self.doi:
            ids.append(f"doi:{self.doi.lower()}")
        pmid = (self.extra or {}).get("pmid")
        if pmid:
            ids.append(f"pmid:{pmid}")
        # Always include title-hash as last-resort alias
        norm = re.sub(r"\s+", " ", self.title.strip().lower())
        if norm:
            ids.append(f"title:{norm[:120]}")
        return ids

    @classmethod
    def from_dict(cls, meta: dict) -> "PaperRecord":
        """從 JSON sidecar 還原 record（供 dedup / 下載輔助用）。"""
        return cls(
            title=meta.get("title", ""),
            authors=meta.get("authors") or [],
            year=meta.get("year"),
            abstract=meta.get("abstract", ""),
            doi=meta.get("doi"),
            arxiv_id=meta.get("arxiv_id"),
            venue=meta.get("venue"),
            citation_count=meta.get("citation_count"),
            source=meta.get("source", ""),
            pdf_url=meta.get("pdf_url"),
            relevance_score=meta.get("relevance_score"),
            extra=meta.get("extra") or {},
        )


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _decode_inverted_index(idx: dict | None) -> str:
    """OpenAlex stores abstracts as inverted index: {word: [positions]}.
    Reconstruct linear text in original word order.
    """
    if not idx:
        return ""
    try:
        max_pos = max((max(positions) for positions in idx.values() if positions), default=-1)
    except ValueError:
        return ""
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in idx.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return " ".join(w for w in words if w)


def search_arxiv(query: str, max_results: int = 25, year_from: int = 2018,
                 offset: int = 0) -> list[PaperRecord]:
    """Direct hit to arXiv's Atom API (no `arxiv` package — avoids sgmllib3k chain)."""
    import xml.etree.ElementTree as ET

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": offset,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(3)  # arXiv courtesy delay
        r = _get(url, params=params, headers=headers)
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
    query: str, max_results: int = 25, year_from: int = 2018,
    offset: int = 0,
) -> list[PaperRecord]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "offset": offset,
        "fields": S2_FIELDS,
        "year": f"{year_from}-",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
        # With key: own 1 RPS lane, normal retry budget
        sleep_s, max_attempts, max_wait = 1.1, 4, 30
    else:
        # Unauthenticated: shared global pool, 429 storms common.
        # Slow inter-call cadence to 2s and use 6 attempts up to 60s wait
        # (~2 min total budget) to ride out transient saturation.
        sleep_s, max_attempts, max_wait = 2.0, 6, 60
    try:
        time.sleep(sleep_s)
        r = _get(url, params=params, headers=headers,
                 max_attempts=max_attempts, max_wait=max_wait)
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


def _pubmed_fetch_abstracts(ids: list[str], headers: dict) -> dict[str, str]:
    """Fetch abstract text for each PMID via efetch. Returns {pmid: abstract}."""
    import xml.etree.ElementTree as ET

    if not ids:
        return {}
    out: dict[str, str] = {}
    try:
        time.sleep(0.4)
        r = _get(
            f"{PUBMED_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "rettype": "abstract",
                "retmode": "xml",
            },
            headers=headers,
        )
        root = ET.fromstring(r.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()
            parts: list[str] = []
            for abst in article.findall(".//Abstract/AbstractText"):
                # Some have Label="BACKGROUND/METHODS/RESULTS/CONCLUSIONS"
                label = abst.get("Label")
                text = "".join(abst.itertext()).strip()
                if not text:
                    continue
                parts.append(f"{label}: {text}" if label else text)
            out[pmid] = " ".join(parts)
    except Exception as e:
        logger.warning("PubMed efetch abstracts failed: %s", e)
    return out


def search_pubmed(query: str, max_results: int = 25, year_from: int = 2018,
                  offset: int = 0) -> list[PaperRecord]:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(0.4)
        r = _get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": _pubmed_translate(query) + f" AND ({year_from}:3000[dp])",
                "retmax": max_results,
                "retstart": offset,
                "retmode": "json",
                "sort": "pub_date",
            },
            headers=headers,
        )
        ids = r.json().get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            return []
        time.sleep(0.4)
        s = _get(
            f"{PUBMED_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            headers=headers,
        )
        result = s.json().get("result", {})
    except Exception as e:
        logger.warning("PubMed search failed: %s", e)
        return []

    abstracts = _pubmed_fetch_abstracts(ids, headers)

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
                abstract=abstracts.get(pmid, ""),
                doi=doi,
                arxiv_id=None,
                venue=rec.get("fulljournalname") or rec.get("source"),
                citation_count=None,
                source="pubmed",
                pdf_url=None,
                extra={"pmid": pmid},
            )
        )
    return out


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def search_openalex(query: str, max_results: int = 25, year_from: int = 2018,
                    offset: int = 0) -> list[PaperRecord]:
    url = "https://api.openalex.org/works"
    page = (offset // max_results) + 1 if max_results else 1
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "page": page,
        "sort": "publication_date:desc",
        "filter": f"from_publication_date:{year_from}-01-01",
        "mailto": "a1234567891934@gmail.com",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        time.sleep(0.2)
        r = _get(url, params=params, headers=headers)
        data = r.json()
    except Exception as e:
        logger.warning("OpenAlex search failed: %s", e)
        return []

    out: list[PaperRecord] = []
    for w in data.get("results", []):
        ids = w.get("ids", {}) or {}
        doi = (ids.get("doi") or "").replace("https://doi.org/", "") or None
        oa = (w.get("best_oa_location") or {}).get("pdf_url")
        abstract = _decode_inverted_index(w.get("abstract_inverted_index"))
        out.append(
            PaperRecord(
                title=(w.get("title") or "").strip(),
                authors=[
                    (a.get("author") or {}).get("display_name", "")
                    for a in (w.get("authorships") or [])
                ],
                year=w.get("publication_year"),
                abstract=abstract,
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


def search(source: str, query: str, max_results: int = 25, year_from: int = 2018,
           offset: int = 0) -> list[PaperRecord]:
    fn = SEARCHERS.get(source)
    if fn is None:
        raise ValueError(f"unknown source: {source}")
    return fn(query, max_results=max_results, year_from=year_from, offset=offset)


# ---------------------------------------------------------------------------
# 單篇 abstract 補抓（供 curate.fill_missing_abstracts 用）
# ---------------------------------------------------------------------------
def fetch_openalex_abstract(openalex_id: str) -> str:
    """openalex_id 形如 'https://openalex.org/W123'。"""
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
    import xml.etree.ElementTree as ET
    try:
        time.sleep(3)  # arxiv courtesy
        r = _get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id},
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
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
    """依序試 S2 paperId → DOI → arXiv；有 S2_API_KEY 則帶上。"""
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
