'''
SECTION 0 — High-level goal / TODO
- Crawl the AASB standards listing pages
- Open each standard page
- Find the "Download PDF" link
- Download each PDF and save locally
- Write a manifest and a summary report for auditing/debugging
'''

# SECTION 1 — Imports (stdlib + third-party)
import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# SECTION 2 — Configuration (URLs, output locations, HTTP headers)
BASE_URL = "https://standards.aasb.gov.au"
LIST_URL = f"{BASE_URL}/accounting-standards"

RAW_PDF_DIR = Path("data/raw_pdfs")
REPORT_DIR = Path("data/reports")

# Ensure output folders exist before attempting any downloads/logging.
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = REPORT_DIR / "download_manifest.jsonl"
SUMMARY_PATH = REPORT_DIR / "download_report.json"

# Basic UA header to reduce chance of being blocked by default bot filters.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AASB-RAG-Downloader/1.0)"
}

# Regex to extract "AASB <number>" from link text (e.g., "AASB 15").
# The negative lookahead avoids matching "AASB 123-" style patterns.
AASB_NUM_RE = re.compile(r"\bAASB\s+(\d{1,4})\b(?!-)", re.IGNORECASE)


# SECTION 3 — Logging & Resilient HTTP Helpers
def log_manifest(record: dict) -> None:
    """
    Append one JSON record per event to a JSONL manifest.
    This acts like an audit trail of what happened for each standard.
    """
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_get_text(session: requests.Session, url: str, retries: int = 3) -> str:
    """
    Fetch HTML/text content with simple retry + backoff.
    Raises the last error if all retries fail.
    """
    last_err = None
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err  # type: ignore


def safe_get_bytes(session: requests.Session, url: str, retries: int = 3) -> bytes:
    """
    Fetch binary content (PDF bytes) with retry + backoff.
    Raises the last error if all retries fail.
    """
    last_err = None
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err  # type: ignore


def format_doc_id(num: int) -> str:
    """
    Standardize filenames / identifiers so downstream steps can reliably reference them.
    Example: 15 -> "AASB_015"
    """
    return f"AASB_{num:03d}"


# SECTION 4 — Scraping Helpers (extract standard pages, paginate, locate PDF link)
def extract_standard_links(html: str, page_url: str) -> dict[int, str]:
    """
    Parse a listing page and return a mapping:
        { standard_number -> absolute_standard_page_url }

    Notes:
    - Uses anchor text to detect "AASB <num>"
    - Filters to URLs that look like actual standard pages ("/aasb-")
    - Keeps first occurrence of each standard number (dedup)
    """
    soup = BeautifulSoup(html, "html.parser")
    standards: dict[int, str] = {}

    for a in soup.select("a[href]"):
        text = a.get_text(" ", strip=True)
        m = AASB_NUM_RE.search(text)
        if not m:
            continue

        href = a.get("href", "").strip()
        if not href:
            continue

        # Only keeping actual standard pages
        if "/aasb-" not in href:
            continue

        num = int(m.group(1))

        # Keep only first occurrence
        if num not in standards:
            standards[num] = urljoin(page_url, href)

    return standards


def find_next_page(html: str, page_url: str) -> str | None:
    """
    Find a "Next" pagination link and return the absolute URL.
    Returns None if a next page is not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a[href]"):
        if "next" in a.get_text(" ", strip=True).lower():
            href = a.get("href", "").strip()
            if href:
                return urljoin(page_url, href)
    return None


def find_pdf_link(standard_html: str, standard_url: str) -> str | None:
    """
    On an individual standard page, find a link labeled "Download PDF"
    and return the absolute URL to the PDF.
    Returns None if not found.
    """
    soup = BeautifulSoup(standard_html, "html.parser")
    for a in soup.select("a[href]"):
        label = a.get_text(" ", strip=True).lower()
        if "download pdf" in label:
            href = a.get("href", "").strip()
            if href:
                return urljoin(standard_url, href)
    return None


# SECTION 5 — Main Orchestration (crawl -> collect -> download -> report)
def download_current_standards() -> None:
    """
    End-to-end pipeline:
      1) Crawl listing pages to collect standard URLs
      2) For each standard: open page, find PDF link, download PDF
      3) Write per-item manifest records and a final summary report
    """
    session = requests.Session()

    standards: dict[int, str] = {}
    visited_pages = set()
    page_url: str | None = LIST_URL

    # Step 1: Crawl the AASB list
    while page_url and page_url not in visited_pages:
        visited_pages.add(page_url)
        html = safe_get_text(session, page_url)

        standards.update(extract_standard_links(html, page_url))
        page_url = find_next_page(html, page_url)

        time.sleep(0.2)  # polite crawling

    # Step 2: Download PDFs (skips already-downloaded PDFs by size > 0)
    downloaded = 0
    skipped = 0
    failed = 0

    for num in tqdm(sorted(standards.keys()), desc="Downloading AASB PDFs"):
        standard_url = standards[num]
        doc_id = format_doc_id(num)
        out_path = RAW_PDF_DIR / f"{doc_id}.pdf"

        # Idempotency: don't re-download if the file already exists and looks non-empty.
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            log_manifest({
                "status": "skipped_existing",
                "doc_id": doc_id,
                "standard_url": standard_url,
                "file": str(out_path),
            })
            continue

        try:
            # Load the standard page, then locate the PDF download URL.
            standard_html = safe_get_text(session, standard_url)
            pdf_url = find_pdf_link(standard_html, standard_url)

            if not pdf_url:
                failed += 1
                log_manifest({
                    "status": "no_pdf_link_found",
                    "doc_id": doc_id,
                    "standard_url": standard_url,
                })
                continue

            # Download raw bytes and write to disk.
            pdf_bytes = safe_get_bytes(session, pdf_url)
            out_path.write_bytes(pdf_bytes)

            downloaded += 1
            log_manifest({
                "status": "downloaded",
                "doc_id": doc_id,
                "standard_url": standard_url,
                "pdf_url": pdf_url,
                "file": str(out_path),
                "bytes": len(pdf_bytes),
            })

        except Exception as e:
            # Any unexpected error gets recorded with enough context to debug later.
            failed += 1
            log_manifest({
                "status": "error",
                "doc_id": doc_id,
                "standard_url": standard_url,
                "error": str(e),
            })

        time.sleep(0.2)

    # Step 3: Write a single JSON summary report for quick status checking.
    summary = {
        "total_standards_found": len(standards),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "output_dir": str(RAW_PDF_DIR),
        "manifest": str(MANIFEST_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# SECTION 6 — Script Entry Point
if __name__ == "__main__":
    download_current_standards()


# SECTION SUMMARY — What this code does (by section)
# 0) Goal / TODO:
#    Defines the intent: crawl the AASB standards site and download PDFs with reporting.
#
# 1) Imports:
#    Uses stdlib for paths/JSON/regex/time/URL joining, plus requests (HTTP),
#    BeautifulSoup (HTML parsing), and tqdm (progress bar).
#
# 2) Configuration:
#    Sets the base URLs and output directories, ensures folders exist,
#    defines where manifest/summary reports are written, and sets a User-Agent header.
#
# 3) Logging & Resilient HTTP:
#    - log_manifest(): writes JSONL records for each action/outcome (audit trail).
#    - safe_get_text()/safe_get_bytes(): wraps GET requests with retries and backoff.
#    - format_doc_id(): standardizes naming (AASB_###) to keep file references consistent.
#
# 4) Scraping helpers:
#    - extract_standard_links(): parses listing pages and collects {AASB_num -> standard_page_url}.
#    - find_next_page(): follows pagination via a "Next" link until exhausted.
#    - find_pdf_link(): on a standard page, finds the "Download PDF" link and returns its URL.
#
# 5) Main pipeline:
#    - Crawls listing pages to build a set of standard URLs (deduplicated by standard number).
#    - For each standard: checks if already downloaded; otherwise fetches and saves the PDF.
#    - Records every skip/download/failure event in a manifest for traceability.
#    - Writes a summary JSON report with totals and output locations.
#
# 6) Entry point:
#    Allows the file to run as a script, executing the end-to-end download pipeline.
