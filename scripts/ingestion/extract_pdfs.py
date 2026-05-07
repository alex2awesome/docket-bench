"""Download PDFs and extract text for any external comment source.

Reads comments.csv.gz, downloads each attachment_url, extracts text via PyMuPDF
(falls back to pdfminer or OCR via ai_corpus.utils.pdf_parsing), and writes
the extracted text back into comment_text.

Usage:
    python extract_pdfs.py --source state_ag_letters
    python extract_pdfs.py --source fec_fosers --workers 8
    python extract_pdfs.py --source sec_comments --max-comments 100  # test
    python extract_pdfs.py --source fs_cara --workers 16             # big one
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Add project root for ai_corpus import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "ai_corpus").exists():
        PROJECT_ROOT = parent
        break
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ai_corpus.utils import pdf_parsing
except ImportError:
    pdf_parsing = None
    print("WARNING: ai_corpus.utils.pdf_parsing not available; falling back to pypdf", file=sys.stderr)

from common import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source-specific User-Agents and download conventions
SOURCE_HEADERS = {
    "sec_comments": {"User-Agent": "regulations-demo-research research@example.edu",
                     "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"},
    "fdic_comments": {"User-Agent": "Mozilla/5.0"},
    "fs_cara": {"User-Agent": "stanford-regulations-research/1.0 research@example.edu"},
    "state_ag_letters": {"User-Agent": "Mozilla/5.0"},
    "fed_reserve": {"User-Agent": "Mozilla/5.0 (compatible; stanford-regulations-research/1.0)"},
    "fec_fosers": {"User-Agent": "Mozilla/5.0"},
    "ferc_elibrary": {"User-Agent": "stanford-regulations-research/1.0"},
    "nyu_ag_actions": {"User-Agent": "Mozilla/5.0"},
}
DEFAULT_HEADERS = {"User-Agent": "stanford-regulations-research/1.0"}

# Per-source throttle (seconds between requests, per worker)
SOURCE_THROTTLE = {
    "sec_comments": 1.0,  # SEC asks for polite throttling
    "fec_fosers": 1.5,     # ALB/WAF trips
    "fdic_comments": 0.3,
    "fs_cara": 0.3,
    "fed_reserve": 0.4,
    "ferc_elibrary": 1.0,
    "fcc_ecfs": 3.6,  # Full 1000 req/hr budget
    "default": 0.3,
}


def safe_filename(url: str, max_len: int = 100) -> str:
    """Generate a safe filename from a URL using sha256 hash."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    # Take last segment of URL as suffix hint
    last = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0][:60]
    last = "".join(c if c.isalnum() or c in ".-_" else "_" for c in last)
    return f"{h}_{last}"


def download_pdf(url: str, dest: Path, headers: dict, timeout: int = 60) -> bool:
    """Download a single PDF with retries. Returns True if file exists after."""
    if dest.exists() and dest.stat().st_size > 100:
        return True
    # ecfsapi.fcc.gov is often slow — use tighter timeout to fail fast
    if "ecfsapi.fcc.gov" in url.lower():
        timeout = 20
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True,
                             allow_redirects=True)
            if r.status_code != 200:
                if attempt == 2:
                    logger.debug("HTTP %d for %s", r.status_code, url)
                    return False
                time.sleep(2 * (attempt + 1))
                continue
            ct = r.headers.get("content-type", "").lower()
            # Accept PDFs OR octet-stream OR HTML pages that may contain inline PDFs
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            # Verify it's actually a PDF (starts with %PDF)
            with open(dest, "rb") as f:
                head = f.read(8)
            if not head.startswith(b"%PDF"):
                # Not a PDF — discard
                dest.unlink(missing_ok=True)
                return False
            return True
        except requests.RequestException as e:
            if attempt == 2:
                logger.debug("download error %s: %s", url, e)
                return False
            time.sleep(2 * (attempt + 1))
    return False


def extract_pdf_text(pdf_path: Path, max_pages: int = 100,
                     per_page_timeout: float = 5.0) -> str:
    """Extract text from a PDF. Uses ai_corpus.utils.pdf_parsing if available,
    falls back to pypdf or PyMuPDF directly."""
    if pdf_parsing is not None:
        try:
            pages, stats = pdf_parsing.extract_pages(
                str(pdf_path), parser="auto",
                per_page_timeout=per_page_timeout,
                max_pages=max_pages,
                capture_pages=True,
            )
            return "\n".join(pages).strip()
        except Exception as e:
            logger.debug("pdf_parsing failed for %s: %s", pdf_path, e)
    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        n = min(len(reader.pages), max_pages)
        return "\n".join((reader.pages[i].extract_text() or "") for i in range(n)).strip()
    except Exception as e:
        logger.debug("pypdf failed for %s: %s", pdf_path, e)
    # Fallback: PyMuPDF directly
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = min(doc.page_count, max_pages)
        return "\n".join(doc.load_page(i).get_text() for i in range(n)).strip()
    except Exception as e:
        logger.debug("fitz failed for %s: %s", pdf_path, e)
    return ""


def resolve_cara_pdf_url(detail_url: str, headers: dict) -> str | None:
    """CARA Letter detail page → DownloadCommentFile URL."""
    import re
    try:
        r = requests.get(detail_url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        # Look for DownloadCommentFile link with IsLetter=True (the actual letter PDF)
        m = re.search(r'(/Public/DownloadCommentFile\?LetterId=\d+&(?:amp;)?IsLetter=True)',
                      r.text, re.IGNORECASE)
        if m:
            path = m.group(1).replace("&amp;", "&")
            return "https://cara.fs2c.usda.gov" + path
        # Fallback: any DownloadCommentFile link
        m = re.search(r'(/Public/DownloadCommentFile\?[^"\']+)', r.text)
        if m:
            path = m.group(1).replace("&amp;", "&")
            return "https://cara.fs2c.usda.gov" + path
    except Exception:
        pass
    return None


def download_ferc_pdf(filelist_url: str, dest: Path) -> bool:
    """FERC: POST to DownloadP8File with accession, extract PDF from ZIP."""
    import re
    from io import BytesIO
    import zipfile
    m = re.search(r"accession_number=([A-Z0-9-]+)", filelist_url, re.IGNORECASE)
    if not m:
        return False
    accession = m.group(1)
    try:
        r = requests.post(
            "https://elibrary.ferc.gov/eLibrarywebapi/api/File/DownloadP8File",
            json={"accession": accession, "FileType": "main"},
            headers={"Content-Type": "application/json",
                     "User-Agent": "stanford-regulations-research/1.0"},
            timeout=60,
        )
        if r.status_code != 200 or not r.content.startswith(b"PK"):
            return False
        # Extract first .pdf/.PDF from the ZIP
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            pdf_names = [n for n in z.namelist() if n.lower().endswith(".pdf")]
            if not pdf_names:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(pdf_names[0]) as zf, open(dest, "wb") as f:
                f.write(zf.read())
        with open(dest, "rb") as f:
            if not f.read(8).startswith(b"%PDF"):
                dest.unlink(missing_ok=True)
                return False
        return True
    except Exception as e:
        logger.debug("FERC download failed for %s: %s", accession, e)
        return False


def resolve_nyu_pdf_url(action_url: str, headers: dict) -> str | None:
    """NYU action detail page → first PDF link on page."""
    try:
        r = requests.get(action_url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        import re
        # Look for PDF links in the action detail page (also allow .PDF, query strings)
        patterns = [
            r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
            r'href=["\']([^"\']+\.PDF[^"\']*)["\']',
            r'(?:src|href)=["\']([^"\']*(?:letter|comment|filing)[^"\']*\.(?:pdf|PDF)[^"\']*)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, r.text)
            if m:
                url = m.group(1)
                if not url.startswith("http"):
                    url = "https://stateimpactcenter.org" + (url if url.startswith("/") else "/" + url)
                return url
    except Exception:
        pass
    return None


# FCC API key for resolving document URLs
_fcc_api_key = None
def _get_fcc_key() -> str:
    global _fcc_api_key
    if _fcc_api_key is None:
        import os
        for p in [os.path.expanduser("~/.fcc-key")]:
            if os.path.exists(p):
                _fcc_api_key = open(p).read().strip()
                break
        if _fcc_api_key is None:
            _fcc_api_key = os.environ.get("FCC_API_KEY", "")
    return _fcc_api_key


def resolve_fcc_pdf_url(old_url: str) -> str | None:
    """Look up the real FCC PDF URL from the API via id_submission."""
    import re
    m = re.search(r"/document/(\d+)", old_url)
    if not m:
        return None
    fid = m.group(1)
    key = _get_fcc_key()
    if not key:
        return None
    for attempt in range(2):
        try:
            r = requests.get(f"https://publicapi.fcc.gov/ecfs/filings/{fid}",
                             params={"api_key": key}, timeout=15)
            if r.status_code == 429:
                # Rate-limited: short backoff, don't loop forever
                time.sleep(30)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            docs = data.get("documents") or data.get("attachments") or []
            # Only return files.fcc.gov URLs — ecfsapi.fcc.gov is server-broken
            # (HTTP/2 INTERNAL_ERROR on all external clients). docs.fcc.gov has
            # only press releases, not comments.
            for doc in docs:
                src = doc.get("src", "")
                if src and "files.fcc.gov" in src.lower():
                    return src
            return None
        except Exception:
            time.sleep(2)
    return None


def extract_html_text(html_path: Path) -> str:
    """Extract text from an HTML file (for SEC .htm comments)."""
    try:
        from bs4 import BeautifulSoup
        html = html_path.read_bytes().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""


def download_html_or_pdf(url: str, dest: Path, headers: dict) -> tuple[bool, str]:
    """Download and return (ok, 'pdf' or 'html' or '')."""
    if dest.exists() and dest.stat().st_size > 100:
        with open(dest, "rb") as f:
            head = f.read(8)
        return True, ("pdf" if head.startswith(b"%PDF") else "html")
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)
            if r.status_code != 200:
                if attempt == 1:
                    return False, ""
                time.sleep(2)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            with open(dest, "rb") as f:
                head = f.read(8)
            if head.startswith(b"%PDF"):
                return True, "pdf"
            # Accept HTML as a fallback
            if b"<html" in head.lower() or b"<!doc" in head.lower():
                return True, "html"
            dest.unlink(missing_ok=True)
            return False, ""
        except requests.RequestException:
            if attempt == 1:
                return False, ""
            time.sleep(2)
    return False, ""


# FEC session singleton — reused across workers via a module-level client
_fec_session = None
_fec_session_lock = None

def _get_fec_session():
    global _fec_session, _fec_session_lock
    import threading
    if _fec_session_lock is None:
        _fec_session_lock = threading.Lock()
    with _fec_session_lock:
        if _fec_session is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://sers.fec.gov/fosers/",
            })
            # Bootstrap — fetch home page to get JSESSIONID
            try:
                s.get("https://sers.fec.gov/fosers/", timeout=30)
            except Exception:
                pass
            _fec_session = s
        return _fec_session


def download_fec_pdf(url: str, dest: Path) -> bool:
    """FEC requires session cookies + referer. Use shared session."""
    if dest.exists() and dest.stat().st_size > 100:
        return True
    s = _get_fec_session()
    for attempt in range(3):
        try:
            r = s.get(url, timeout=60, stream=True, allow_redirects=True)
            if r.status_code == 403 and attempt < 2:
                # Re-bootstrap session
                time.sleep(3)
                try:
                    s.get("https://sers.fec.gov/fosers/", timeout=30)
                except Exception:
                    pass
                continue
            if r.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            with open(dest, "rb") as f:
                head = f.read(8)
            if not head.startswith(b"%PDF"):
                dest.unlink(missing_ok=True)
                return False
            return True
        except requests.RequestException:
            time.sleep(2)
    return False


def process_one(comment_id: str, url: str, raw_dir: Path, headers: dict,
                throttle: float, max_pages: int,
                source: str = "") -> tuple[str, str, str]:
    """Download + extract one PDF. Returns (comment_id, text, status)."""
    if not url or not isinstance(url, str) or url == "nan":
        return comment_id, "", "no_url"
    # Use first URL if multiple are semi-colon separated
    first_url = url.split(";")[0].strip()
    if not first_url:
        return comment_id, "", "no_url"

    # NYU AG: URLs are HTML action pages; follow to find embedded PDF
    if source == "nyu_ag_actions" and ".pdf" not in first_url.lower():
        resolved = resolve_nyu_pdf_url(first_url, headers)
        if not resolved:
            time.sleep(throttle)
            return comment_id, "", "nyu_no_pdf_link"
        first_url = resolved

    # FCC: resolve legacy /ecfs/document/{id}/{n} URLs to real files.fcc.gov URLs
    if source == "fcc_ecfs" and "/ecfs/document/" in first_url:
        resolved = resolve_fcc_pdf_url(first_url)
        if not resolved:
            time.sleep(throttle)
            return comment_id, "", "fcc_no_doc_src"
        first_url = resolved

    # CARA: resolve Public/Letter/{id} detail pages to actual PDF download URL
    if source == "fs_cara" and "/Public/Letter/" in first_url:
        resolved = resolve_cara_pdf_url(first_url, headers)
        if not resolved:
            time.sleep(throttle)
            return comment_id, "", "cara_no_pdf_link"
        first_url = resolved

    fname = safe_filename(first_url) + ".pdf"
    dest = raw_dir / fname

    # Source-specific download method
    if source == "fec_fosers":
        ok = download_fec_pdf(first_url, dest)
        kind = "pdf" if ok else ""
    elif source == "ferc_elibrary":
        ok = download_ferc_pdf(first_url, dest)
        kind = "pdf" if ok else ""
    elif source == "sec_comments":
        # SEC: many URLs are .htm (HTML pages) — accept both PDF and HTML
        ok, kind = download_html_or_pdf(first_url, dest, headers)
    else:
        ok = download_pdf(first_url, dest, headers)
        kind = "pdf" if ok else ""

    if not ok:
        time.sleep(throttle)
        return comment_id, "", "download_failed"

    if kind == "html":
        text = extract_html_text(dest)
    else:
        text = extract_pdf_text(dest, max_pages=max_pages)
    time.sleep(throttle)
    if not text or len(text) < 30:
        return comment_id, text, "empty_or_short"
    return comment_id, text, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="External source name (e.g., state_ag_letters)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent download workers (default: 4)")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="Limit number of comments to process (for testing)")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Max pages to extract per PDF (default: 100)")
    parser.add_argument("--per-page-timeout", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip comments that already have non-empty comment_text (default: true)")
    parser.add_argument("--reextract", action="store_true",
                        help="Re-extract text from already-downloaded PDFs (no re-download)")
    args = parser.parse_args()

    output_dir = get_output_dir(args.source)
    comments_path = output_dir / "comments.csv.gz"
    raw_dir = output_dir / "raw" / "pdfs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not comments_path.exists():
        logger.error("No comments file at %s", comments_path)
        return

    df = pd.read_csv(comments_path, dtype=str)
    logger.info("Loaded %d comments from %s", len(df), comments_path)

    headers = SOURCE_HEADERS.get(args.source, DEFAULT_HEADERS)
    throttle = SOURCE_THROTTLE.get(args.source, SOURCE_THROTTLE["default"])

    # Determine which rows need processing
    df["comment_text"] = df["comment_text"].fillna("")
    if args.skip_existing:
        needs_text = df["comment_text"].apply(lambda x: len(str(x)) < 30)
    else:
        needs_text = pd.Series(True, index=df.index)
    work_df = df[needs_text & df["attachment_urls"].notna() &
                 df["attachment_urls"].apply(lambda x: bool(str(x).strip()) and str(x) != "nan")]

    if args.max_comments:
        work_df = work_df.head(args.max_comments)

    logger.info("Will process %d of %d comments (workers=%d, throttle=%.2fs)",
                len(work_df), len(df), args.workers, throttle)
    if work_df.empty:
        logger.info("Nothing to do")
        return

    text_map: dict[str, str] = {}
    statuses: dict[str, int] = {}
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, row["comment_id"], row["attachment_urls"],
                      raw_dir, headers, throttle, args.max_pages,
                      args.source): row["comment_id"]
            for _, row in work_df.iterrows()
        }
        for fut in as_completed(futures):
            try:
                cid, text, status = fut.result()
            except Exception as e:
                logger.debug("worker error: %s", e)
                continue
            if text:
                text_map[cid] = text
            statuses[status] = statuses.get(status, 0) + 1
            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.time() - t0)
                logger.info("  %d/%d done (%.1f/s, statuses=%s, with_text=%d)",
                            completed, len(work_df), rate, statuses, len(text_map))
            # Incremental save every 500 rows to guard against crashes
            if completed % 500 == 0 and text_map:
                try:
                    def _clean(s):
                        if not isinstance(s, str):
                            return s
                        return "".join("?" if 0xD800 <= ord(c) <= 0xDFFF else c for c in s)
                    clean_map = {k: _clean(v) for k, v in text_map.items()}
                    df_copy = df.copy()
                    df_copy.loc[df_copy["comment_id"].isin(clean_map), "comment_text"] = \
                        df_copy["comment_id"].map(clean_map)
                    df_copy["comment_text"] = df_copy["comment_text"].apply(_clean)
                    df_copy.to_csv(comments_path, index=False, compression="gzip")
                    logger.debug("  checkpoint saved at %d", completed)
                except Exception as e:
                    logger.warning("  checkpoint save failed: %s", e)

    # Write back into comments.csv.gz
    if text_map:
        # Sanitize: strip surrogate characters that can't be encoded to UTF-8
        def _clean(s):
            if not isinstance(s, str):
                return s
            # Surrogates (U+D800..U+DFFF) can't be encoded to UTF-8
            return "".join("?" if 0xD800 <= ord(c) <= 0xDFFF else c for c in s)
        clean_map = {k: _clean(v) for k, v in text_map.items()}
        df.loc[df["comment_id"].isin(clean_map), "comment_text"] = df["comment_id"].map(clean_map)
        # Also sanitize all existing comment_text
        df["comment_text"] = df["comment_text"].apply(_clean)
        df.to_csv(comments_path, index=False, compression="gzip")
        logger.info("Saved updated comments to %s", comments_path)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs. Statuses: %s. With text: %d",
                elapsed, statuses, len(text_map))


if __name__ == "__main__":
    main()
