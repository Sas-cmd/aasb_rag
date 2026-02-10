'''
To do
 - Crawl the AASB list
 - Open each link
 - Download each pdf
 - Save files 
 - Report downloaded files 
'''

import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# Configuration

BASE_URL = "https://standards.aasb.gov.au"
LIST_URL = f"{BASE_URL}/accounting-standards"

RAW_PDF_DIR = Path("data/raw_pdfs")
REPORT_DIR = Path("data/reports")

RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = REPORT_DIR / "download_manifest.jsonl"
SUMMARY_PATH = REPORT_DIR / "download_report.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AASB-RAG-Downloader/1.0)"
}


AASB_NUM_RE = re.compile(r"\bAASB\s+(\d{1,4})\b(?!-)", re.IGNORECASE)


# error catch

def log_manifest(record: dict) -> None:
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_get_text(session: requests.Session, url: str, retries: int = 3) -> str:
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
    return f"AASB_{num:03d}"



# Scraping 
def extract_standard_links(html: str, page_url: str) -> dict[int, str]:
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
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a[href]"):
        if "next" in a.get_text(" ", strip=True).lower():
            href = a.get("href", "").strip()
            if href:
                return urljoin(page_url, href)
    return None


def find_pdf_link(standard_html: str, standard_url: str) -> str | None:
    soup = BeautifulSoup(standard_html, "html.parser")
    for a in soup.select("a[href]"):
        label = a.get_text(" ", strip=True).lower()
        if "download pdf" in label:
            href = a.get("href", "").strip()
            if href:
                return urljoin(standard_url, href)
    return None



# Main 

def download_current_standards() -> None:
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

    # Step 2: Download PDFs
    downloaded = 0
    skipped = 0
    failed = 0

    for num in tqdm(sorted(standards.keys()), desc="Downloading AASB PDFs"):
        standard_url = standards[num]
        doc_id = format_doc_id(num)
        out_path = RAW_PDF_DIR / f"{doc_id}.pdf"

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
            failed += 1
            log_manifest({
                "status": "error",
                "doc_id": doc_id,
                "standard_url": standard_url,
                "error": str(e),
            })

        time.sleep(0.2)

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


if __name__ == "__main__":
    download_current_standards()
