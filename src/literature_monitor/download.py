"""PDF 下載與文字抽取。

下載依序試 arxiv → S2 openAccessPdf → Unpaywall → OpenAlex；存 PDF + JSON
sidecar（無 OA 則只存 JSON）。另含 backfill（補抓缺 PDF）與 pdf→txt。
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests

from .sources import DEFAULT_TIMEOUT, DEFAULT_USER_AGENT, PaperRecord
from .state import iter_sidecars, write_meta

logger = logging.getLogger(__name__)

UNPAYWALL_EMAIL = "a1234567891934@gmail.com"


def filename_for(rec: PaperRecord) -> str:
    """檔名 `{姓}_{年}_{短標題}_{paperID}`（無副檔名）。"""
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
    pid_raw = rec.primary_id().split(":", 1)[-1]
    pid = re.sub(r"[^A-Za-z0-9]+", "_", pid_raw)[:40].strip("_") or "noid"
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
            # Validate magic bytes — publishers sometimes serve recaptcha
            # HTML with Content-Type: application/pdf
            with dest.open("rb") as f:
                head = f.read(8)
            if not head.startswith(b"%PDF"):
                logger.debug("Not a real PDF (magic bytes %r) on %s", head, url)
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
    """一律存 JSON sidecar，有 OA PDF 才存 PDF。回 (pdf_path 或 None, json_path)。"""
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

    payload = rec.to_dict()
    payload["pdf_status"] = "deferred" if not download_pdf else ("ok" if pdf_path else "no_oa")
    payload["primary_id"] = rec.primary_id()
    write_meta(json_path, payload)
    return pdf_path, json_path


# ---------------------------------------------------------------------------
# Backfill：補抓缺 PDF 的 sidecar（download_pdfs CLI 用）
# ---------------------------------------------------------------------------
def missing_pdf_targets(waiting_review_dir: Path, *, topics: list[str] | None = None,
                        retry_no_oa: bool = False) -> list[tuple[Path, dict]]:
    """缺 PDF 的 (path, meta) 清單；no_oa 預設略過（除非 retry_no_oa）。"""
    out = []
    for json_path, meta in iter_sidecars(waiting_review_dir):
        if topics and json_path.parent.parent.name not in topics:
            continue
        if json_path.with_suffix(".pdf").exists():
            continue
        if meta.get("pdf_status") == "no_oa" and not retry_no_oa:
            continue
        out.append((json_path, meta))
    return out


def download_missing(targets: list[tuple[Path, dict]]) -> tuple[int, int]:
    """對每個 (path, meta) 試下載鏈並更新 pdf_status。回 (ok, no_oa)。"""
    ok = no_oa = 0
    for json_path, meta in targets:
        rec = PaperRecord.from_dict(meta)
        got = any(_download_pdf(u, json_path.with_suffix(".pdf")) for u in _candidate_pdf_urls(rec))
        meta["pdf_status"] = "ok" if got else "no_oa"
        write_meta(json_path, meta)
        ok, no_oa = (ok + 1, no_oa) if got else (ok, no_oa + 1)
    return ok, no_oa


# ---------------------------------------------------------------------------
# PDF → 純文字（extract_text CLI 用）
# ---------------------------------------------------------------------------
LOW_TEXT_THRESHOLD = 200  # 每頁字數低於此，多半是掃描檔需 OCR


def _clean_text(raw: str) -> str:
    s = re.sub(r"-\n(?=[a-z])", "", raw)       # 接合斷行連字號
    s = re.sub(r"\n{3,}", "\n\n", s)            # 收斂多重空行
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip() + "\n"


def extract_pdf_text(pdf_path: Path) -> tuple[str, dict]:
    """pymupdf 抽文字，回 (text, {page_count, char_count, chars_per_page})。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    text = _clean_text("\n\n".join(pages))
    return text, {
        "page_count": len(pages),
        "char_count": len(text),
        "chars_per_page": len(text) / max(1, len(pages)),
    }
