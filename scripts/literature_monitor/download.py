"""PDF download for the literature monitor.

Tries fallback URLs in order: arxiv -> S2 openAccessPdf -> Unpaywall ->
OpenAlex best_oa_location. Saves PDF + JSON sidecar, or just JSON if no
open-access PDF is available.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterable

import requests

from scripts.literature_monitor.sources import (
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    PaperRecord,
)

logger = logging.getLogger(__name__)

UNPAYWALL_EMAIL = "a1234567891934@gmail.com"


def filename_for(rec: PaperRecord) -> str:
    """`{firstAuthorLastName}_{year}_{shortTitle}_{paperID}` (no extension)."""
    last = "Anon"
    if rec.authors:
        first = rec.authors[0]
        # "Smith, John" or "John Smith"
        if "," in first:
            last = first.split(",", 1)[0].strip()
        else:
            last = first.split()[-1] if first.split() else "Anon"
    last = re.sub(r"[^A-Za-z]", "", last) or "Anon"
    year = str(rec.year or "0000")
    title = re.sub(r"[^A-Za-z0-9]+", "_", rec.title)[:80].strip("_") or "untitled"
    pid = rec.primary_id().split(":", 1)[-1].replace("/", "_").replace(":", "_")[:40]
    return f"{last}_{year}_{title}_{pid}"


def _candidate_pdf_urls(rec: PaperRecord) -> list[str]:
    urls: list[str] = []
    if rec.arxiv_id:
        urls.append(f"https://arxiv.org/pdf/{rec.arxiv_id}.pdf")
    if rec.pdf_url:
        urls.append(rec.pdf_url)
    if rec.doi:
        try:
            time.sleep(0.5)
            r = requests.get(
                f"https://api.unpaywall.org/v2/{rec.doi}",
                params={"email": UNPAYWALL_EMAIL},
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
            )
            if r.status_code == 200:
                d = r.json()
                best = d.get("best_oa_location") or {}
                if best.get("url_for_pdf"):
                    urls.append(best["url_for_pdf"])
                for loc in d.get("oa_locations") or []:
                    if loc.get("url_for_pdf") and loc["url_for_pdf"] not in urls:
                        urls.append(loc["url_for_pdf"])
        except Exception as e:
            logger.debug("Unpaywall lookup failed for %s: %s", rec.doi, e)
    return urls


def _download_pdf(url: str, dest: Path) -> bool:
    try:
        time.sleep(2)
        with requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/pdf,*/*"},
            timeout=60,
            stream=True,
            allow_redirects=True,
        ) as r:
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200:
                logger.debug("HTTP %s on %s", r.status_code, url)
                return False
            if "pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                logger.debug("Non-PDF content-type %s on %s", ctype, url)
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
            if dest.stat().st_size < 10_000:
                logger.debug("Downloaded file <10kB, treating as failure: %s", dest)
                dest.unlink(missing_ok=True)
                return False
            return True
    except Exception as e:
        logger.debug("Download exception %s: %s", url, e)
        return False


def save_record(
    rec: PaperRecord,
    target_dir: Path,
    download_pdf: bool = True,
) -> tuple[Path | None, Path]:
    """Save metadata JSON (always) and PDF (if open access available).

    Returns: (pdf_path_or_None, json_path)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    base = filename_for(rec)
    json_path = target_dir / f"{base}.json"
    pdf_path: Path | None = None

    if download_pdf:
        urls = _candidate_pdf_urls(rec)
        for u in urls:
            candidate = target_dir / f"{base}.pdf"
            if _download_pdf(u, candidate):
                pdf_path = candidate
                break

    # Always write JSON sidecar
    payload = rec.to_dict()
    payload["pdf_status"] = "ok" if pdf_path else "no_oa"
    payload["primary_id"] = rec.primary_id()
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return pdf_path, json_path
