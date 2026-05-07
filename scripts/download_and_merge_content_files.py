#!/usr/bin/env python3
"""Download content payloads referenced by bulk CSV exports, then merge with inline text.

Key flags:
  --recursive             Process all CSVs under the given directory tree.
  --skip-pdf-when-text-prefix  Skip PDF download when inline text already exists.
  --cleanup-ocr-pdfs      Delete intermediate OCR PDF files after parsing.
  --no-retry-download-errors   Don't retry rows that previously failed to download.
  --pdf-parse-page-timeout N   Per-page timeout (seconds) for PDF parsing.
  --pdf-parser {fitz,tesseract}  PDF parsing backend (default: fitz).
  --csv-filter SUBSTRING  Only process CSVs whose filename contains SUBSTRING
                           (e.g. 'public_submission' to skip rules/notices).
  --skip-if-all-text-exists / --no-skip-if-all-text-exists
                           Skip CSVs that already have a corresponding _all_text.csv
                           (default: enabled).
  --skip-merge             Disable the post-download merge step.

Example usage:

  # All CSVs:
  python data/bulk_downloads/scripts/download_and_merge_content_files.py \\
      --recursive --skip-pdf-when-text-prefix --cleanup-ocr-pdfs \\
      --no-retry-download-errors --pdf-parse-page-timeout 10 --pdf-parser fitz

  # Only public_submission CSVs:
  python data/bulk_downloads/scripts/download_and_merge_content_files.py \\
      --recursive --skip-pdf-when-text-prefix --cleanup-ocr-pdfs \\
      --no-retry-download-errors --pdf-parse-page-timeout 10 --pdf-parser fitz \\
      --csv-filter public_submission
"""
from __future__ import annotations

import argparse
import atexit
import csv
import gc
import json
import logging
import os
import random
import re
import signal
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


csv.field_size_limit(sys.maxsize)
from curl_cffi import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


ANCHOR_DIRS = (".git", "ai_corpus")


def find_repo_root(current: Path) -> Path:
    for candidate in [current.parent] + list(current.parents):
        if any((candidate / anchor).exists() for anchor in ANCHOR_DIRS):
            return candidate
    return current.parent


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = find_repo_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from ai_corpus.utils import pdf_parsing
except ImportError:  # pragma: no cover - optional dependency
    pdf_parsing = None

PDF_PARSER_CHOICES = ("auto", "pdfminer", "fitz")
ERROR_LOG_BASENAME = "content_download_errors.csv"
ACTIVE_PROCESSING_MARKERS: Set[Path] = set()
PDF_PARSE_GC_INTERVAL = 1000
_PDF_PARSE_COUNTER = 0
PAGE_MARKER_RE = re.compile(r"<<PAGE\s+(?P<num>\d+)\s*>>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download files listed under the 'Content Files' column for each CSV row "
            "contained inside a bulk download directory."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=(
            "Directory containing CSV exports directly or organized inside agency subdirectories."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively under --base-dir instead of only one level deep.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="downloaded_content",
        help="Name of the subdirectory (per agency folder) to store downloaded files.",
    )
    parser.add_argument(
        "--lock-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize on CSV files so only one process handles a CSV at a time (use --no-lock-csv to disable).",
    )
    parser.add_argument(
        "--lock-individual-download-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize on individual downloads so multiple processes do not fetch the same file "
            "(use --no-lock-individual-download-file to disable)."
        ),
    )
    parser.add_argument(
        "--max-rows-per-csv",
        type=int,
        default=None,
        help="Optional limit of rows to process per CSV file before moving to the next.",
    )
    parser.set_defaults(extract_pdf_text=True)
    parser.add_argument(
        "--skip-pdf-text",
        action="store_false",
        dest="extract_pdf_text",
        help="Disable PDF text extraction after downloads complete.",
    )
    parser.add_argument(
        "--skip-pdf-parsing",
        action="store_false",
        dest="extract_pdf_text",
        help="Alias for --skip-pdf-text; disables PDF extraction.",
    )
    parser.add_argument(
        "--skip-pdf-download",
        action="store_true",
        help="Skip downloading PDF files entirely (also skips parsing).",
    )
    parser.add_argument(
        "--pdf-parser",
        choices=PDF_PARSER_CHOICES,
        default="auto",
        help="PDF extraction backend to use (requires ai_corpus.utils.pdf_parsing).",
    )
    parser.add_argument(
        "--pdf-parse-timeout",
        type=float,
        default=20.0,
        help=(
            "Maximum seconds to spend parsing a single PDF before aborting that extraction "
            "(set to 0 or a negative value to disable the limit)."
        ),
    )
    parser.add_argument(
        "--pdf-parse-page-timeout",
        type=float,
        default=None,
        help="Maximum seconds to spend parsing each individual PDF page.",
    )
    parser.add_argument(
        "--pdf-parse-max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to extract per PDF (default: all pages).",
    )
    parser.add_argument(
        "--pdf-parse-max-pages-public-submission",
        type=int,
        default=None,
        help=(
            "Maximum pages to extract when processing public_submission CSVs "
            "(overrides --pdf-parse-max-pages for those files)."
        ),
    )
    parser.add_argument(
        "--pdf-progress-threshold",
        type=int,
        default=20,
        help="Show a per-PDF progress bar when the document has at least this many pages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only download a handful of files per CSV for quick debugging.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
        help="Maximum number of files to download per CSV when --debug is enabled.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of download retries per file before failing.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout (seconds) for each GET request.",
    )
    parser.add_argument(
        "--sleep-base",
        type=float,
        default=1.5,
        help="Base delay (seconds) for exponential backoff during retries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--impersonate",
        default="chrome110",
        help="curl_cffi impersonation profile to apply to HTTP requests.",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        help="Custom User-Agent header for download requests.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue processing even if a download ultimately fails (errors are logged).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for shuffling CSV work items.",
    )
    parser.add_argument(
        "--skip-pdf-when-html",
        action="store_true",
        help="Do not extract PDF text for rows that already contain an HTML content file.",
    )
    parser.add_argument(
        "--skip-pdf-when-text-prefix",
        action="store_true",
        help=(
            "Skip downloading PDFs that share a filename prefix with .htm/.html/.txt attachments."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit detailed per-row logs to diagnose stalls.",
    )
    parser.add_argument(
        "--max-download-seconds",
        type=float,
        default=None,
        help="Abort a single file download if it takes longer than this many seconds (best-effort).",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        dest="max_download_seconds",
        help="Alias for --max-download-seconds.",
    )
    parser.add_argument(
        "--cleanup-ocr-pdfs",
        action="store_true",
        help=(
            "Delete any content.pdf files under --base-dir that already have matching "
            "content.pdf.txt payloads before downloading."
        ),
    )
    parser.add_argument(
        "--verify-pdf-pages",
        action="store_true",
        help=(
            "Re-parse PDFs unless the existing text already contains page markers for every PDF page. "
            "Ignores partial markers and enforces a full crawl/resume."
        ),
    )
    parser.add_argument(
        "--verify-pdf-pages-filter",
        type=str,
        default=None,
        help=(
            "Limit --verify-pdf-pages enforcement to CSV filenames containing this substring "
            "(e.g. 'public_submission')."
        ),
    )
    parser.add_argument(
        "--verify-pdf-pages-min-marker",
        type=int,
        default=49,
        help=(
            "Only force verify-mode re-downloads when an existing text file already contains "
            "a <<PAGE n>> marker at or above this threshold. Set to 0 to reprocess every match."
        ),
    )
    parser.set_defaults(retry_download_errors=True)
    parser.add_argument(
        "--no-retry-download-errors",
        action="store_false",
        dest="retry_download_errors",
        help=(
            "Skip files previously recorded in content_download_errors.csv instead of retrying them."
        ),
    )
    parser.add_argument(
        "--cookie",
        action="append",
        default=[],
        metavar="NAME=VALUE[; NAME2=VALUE2...]",
        help=(
            "Optional cookie header(s) to include in download requests. "
            "Repeatable; entries are split on ';' to extract individual cookies."
        ),
    )
    parser.add_argument(
        "--cookie-domain",
        default=".regulations.gov",
        help="Domain to associate with cookies supplied via --cookie.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merge step that produces _all_text.csv after downloading.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip ALL downloads and only run the merge step. "
             "Use this to re-merge with existing downloaded files.",
    )
    parser.add_argument(
        "--skip-if-all-text-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip downloading for a CSV if its _all_text.csv already exists (default: on; use --no-skip-if-all-text-exists to disable).",
    )
    parser.add_argument(
        "--csv-filter",
        default=None,
        help="Only process CSVs whose filename contains this substring (e.g. 'public_submission').",
    )
    parser.add_argument(
        "--csv-exclude",
        default=None,
        help="Skip CSVs whose filename contains this substring (e.g. 'public_submission').",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair mode: find stub rows (<--repair-min-words words) in existing "
             "_all_text.csv.gz files and re-merge canonical_text from downloaded "
             "content files. Does not re-download anything.",
    )
    parser.add_argument(
        "--repair-min-words",
        type=int,
        default=400,
        help="Word count threshold for stub detection in repair mode (default: 400).",
    )
    parser.add_argument(
        "--repair-dry-run",
        action="store_true",
        help="In repair mode, report what would be fixed without modifying files.",
    )
    return parser.parse_args()


DOWNLOAD_DIR_NAME = "downloaded_content"


def _resolve_doc_id(doc_dir: Path) -> str:
    """Return the document ID for a download directory, preferring metadata.json."""
    metadata_path = doc_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if isinstance(metadata, dict):
                for key in ("Document ID", "DocumentId", "document_id", "doc_id", "DocumentID"):
                    value = metadata.get(key)
                    if value:
                        value_str = str(value).strip()
                        if value_str:
                            return value_str
        except json.JSONDecodeError:
            pass
    return doc_dir.name


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")


def _read_html_as_text(path: Path) -> str:
    """Read an HTML file and return plain text with tags stripped."""
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = _HTML_TAG_RE.sub("", raw)
    text = text.strip()
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text


def _collect_download_records(download_root: Path, csv_name: str) -> List[Dict[str, str]]:
    """Collect downloaded text file records for a single CSV's download directory."""
    csv_dir = download_root / csv_name
    if not csv_dir.is_dir():
        return []
    records: List[Dict[str, str]] = []
    for doc_dir in sorted(p for p in csv_dir.iterdir() if p.is_dir()):
        doc_id = _resolve_doc_id(doc_dir)
        for file_path in sorted(doc_dir.iterdir()):
            if not file_path.is_file():
                continue
            name = file_path.name
            if name == "metadata.json":
                continue
            if name.endswith((".processing", ".partial", ".stats.json")):
                continue
            records.append({"doc_id": doc_id, "file_name": name, "file_path": str(file_path)})
    return records


def merge_all_text(csv_path: Path, output_suffix: str = "_all_text.csv.gz") -> Optional[Path]:
    """Merge inline CSV comments with downloaded .txt attachments into an _all_text.csv."""
    import pandas as pd

    output_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}")
    download_root = csv_path.parent / DOWNLOAD_DIR_NAME
    csv_name = csv_path.stem

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
    except Exception as exc:
        logging.warning("Could not read %s for merge: %s", csv_path, exc)
        return None

    if "Document ID" not in df.columns and "DocumentId" in df.columns:
        df = df.rename(columns={"DocumentId": "Document ID"})
    if "Document ID" not in df.columns:
        logging.warning("Skipping merge for %s: no Document ID column", csv_path)
        return None

    comment_cols = [
        col for col in df.columns
        if ("comment" in col.lower()) and ("comment on" not in col.lower())
    ]
    inline_series = (
        df[comment_cols]
        .fillna("")
        .agg(lambda row: "\n\n".join([t.strip() for t in row if str(t).strip()]), axis=1)
        if comment_cols
        else pd.Series("", index=df.index)
    )

    records = _collect_download_records(download_root, csv_name)
    doc_text_map: Dict[str, List[str]] = {}
    docs_with_txt: set = set()
    for rec in records:
        if not rec["file_name"].endswith(".txt"):
            continue
        fp = Path(rec["file_path"])
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            doc_text_map.setdefault(rec["doc_id"], []).append(text)
            docs_with_txt.add(rec["doc_id"])
    # Also read .htm/.html files for docs that don't already have .txt
    for rec in records:
        fn = rec["file_name"]
        if not (fn.endswith(".htm") or fn.endswith(".html")):
            continue
        if rec["doc_id"] in docs_with_txt:
            continue
        fp = Path(rec["file_path"])
        text = _read_html_as_text(fp)
        if text:
            doc_text_map.setdefault(rec["doc_id"], []).append(text)

    def assemble(row) -> str:
        pieces: List[str] = []
        inline_val = str(row["_inline_text"]).strip()
        if inline_val:
            pieces.append(inline_val)
        for extra in doc_text_map.get(row.get("Document ID", ""), []):
            cleaned = str(extra).strip()
            if cleaned:
                pieces.append(cleaned)
        if not pieces:
            return ""
        return "\n\n".join(
            f"<<COMMENT {idx}>>\n{chunk}" for idx, chunk in enumerate(pieces, start=1)
        )

    enriched = (
        df.assign(_inline_text=inline_series)
        .assign(canonical_text=lambda pdf: pdf.apply(assemble, axis=1))
        .drop(columns="_inline_text")
    )
    enriched.to_csv(output_path, index=False)
    logging.info("Wrote merged %s (%d rows)", output_path, len(enriched))
    return output_path


def sanitize(value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or fallback


def normalized_prefix_key(file_name: str) -> str:
    """
    Produce a normalized prefix identifier by stripping recognized suffixes (.pdf/.htm/.html/.txt)
    so related files (different extensions) can be compared.
    """

    prefix = file_name.lower()
    while True:
        for suffix in (".txt", ".html", ".htm", ".pdf"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        else:
            break
    return prefix


def split_content_files(cell_value: Optional[str]) -> List[str]:
    if not cell_value:
        return []
    # The CSV already quotes the field, so a simple split works here.
    return [item.strip() for item in cell_value.split(",") if item.strip()]


def apply_cli_cookies(session: requests.Session, args: argparse.Namespace) -> None:
    """
    Parse --cookie arguments (semicolon-delimited name/value pairs) and add them to the session.
    """

    if not args.cookie:
        return

    for raw_cookie in args.cookie:
        pairs = [item.strip() for item in raw_cookie.split(";") if item.strip()]
        for pair in pairs:
            if "=" not in pair:
                logging.warning("Ignoring malformed cookie entry: %s", pair)
                continue
            name, value = pair.split("=", 1)
            session.cookies.set(name.strip(), value.strip(), domain=args.cookie_domain)


def iter_csv_files(base_dir: Path, *, recursive: bool = False) -> Iterable[Path]:
    """
    Yield CSV files located directly under ``base_dir`` or inside optional recursive subdirectories.
    """

    if not base_dir.exists():
        return []

    def _should_skip(path: Path) -> bool:
        name = path.name
        if name.endswith(("_1.csv", "_1.csv.gz")):
            return True
        # Skip derived CSVs — only process the original base CSVs.
        if "_all_text" in name:
            return True
        if "__" in name:  # e.g. __claims.csv.gz, __dedup_mapper.csv.gz
            return True
        if "_chunked" in name or "_labels" in name or "_matches" in name or "_score_" in name:
            return True
        return False

    if recursive:
        for csv_path in sorted(base_dir.rglob("*.csv.gz")):
            if _should_skip(csv_path):
                continue
            yield csv_path
        for csv_path in sorted(base_dir.rglob("*.csv")):
            if _should_skip(csv_path):
                continue
            # Skip if a .csv.gz version exists (prefer compressed)
            if csv_path.with_name(csv_path.name + ".gz").exists():
                continue
            yield csv_path
        return

    root_csvs = sorted(
        list(base_dir.glob("*.csv.gz")) + list(base_dir.glob("*.csv"))
    )
    seen_stems: set = set()
    for csv_path in root_csvs:
        if _should_skip(csv_path):
            continue
        stem = csv_path.stem.replace(".csv", "")  # handle .csv.gz double suffix
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        yield csv_path

    for sub_dir in sorted(base_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        sub_csvs = sorted(
            list(sub_dir.glob("*.csv.gz")) + list(sub_dir.glob("*.csv"))
        )
        seen_stems = set()
        for csv_path in sub_csvs:
            if _should_skip(csv_path):
                continue
            stem = csv_path.stem.replace(".csv", "")
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            yield csv_path


def delete_ocrd_content_pdfs(base_dir: Path, *, target_suffix: str = ".pdf") -> int:
    """
    Remove PDFs ending with ``target_suffix`` when a ``.txt`` OCR payload already exists.

    Returns the number of deleted files so callers can report how much space was freed.
    """

    deleted = 0
    if not base_dir.exists():
        logging.warning("Base directory %s does not exist; nothing to clean.", base_dir)
        return deleted

    search_pattern = f"*{target_suffix}" if not target_suffix.startswith("*") else target_suffix
    for pdf_path in sorted(base_dir.rglob(search_pattern)):
        text_path = pdf_path.with_suffix(pdf_path.suffix + ".txt")
        partial_marker = pdf_path.with_suffix(pdf_path.suffix + ".partial")
        if not text_path.exists():
            continue
        if partial_marker.exists():
            logging.debug(
                "Skipping deletion for %s (partial parse marker present).", pdf_path
            )
            continue
        try:
            pdf_path.unlink()
            deleted += 1
            logging.debug("Deleted %s (text already exists).", pdf_path)
        except FileNotFoundError:
            continue
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.warning("Failed to delete %s: %s", pdf_path, exc)
    return deleted


def looks_like_html(url_or_name: str) -> bool:
    suffix = Path(url_or_name.split("?")[0]).suffix.lower()
    return suffix in {".html", ".htm"}


def log_verbose(args: argparse.Namespace, message: str, *fmt: object) -> None:
    if args.verbose:
        logging.info("[verbose] " + message, *fmt)


def log_download_error(error_log_path: Path, source_file: Path, download_file: Path, reason: str) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = error_log_path.exists()
    with error_log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["source_file", "download_file", "error_reason"])
        writer.writerow([str(source_file), str(download_file), reason])


def load_logged_failures(error_log_path: Path) -> Set[Tuple[str, str]]:
    failures: Set[Tuple[str, str]] = set()
    if not error_log_path.exists():
        return failures
    with error_log_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source = row.get("source_file")
            dest = row.get("download_file")
            if source and dest:
                failures.add((source, dest))
    return failures


def max_page_marker_in_text(text_path: Path) -> int:
    if not text_path.exists():
        return 0
    try:
        data = text_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    pages = [int(match.group("num")) for match in PAGE_MARKER_RE.finditer(data)]
    if pages:
        return max(pages)
    return 1 if data.strip() else 0


def should_force_verify_download(
    *,
    verify_enabled: bool,
    text_exists: bool,
    partial_exists: bool,
    marker_max: int,
    min_marker: Optional[int],
) -> bool:
    if not verify_enabled:
        return False
    if partial_exists or not text_exists:
        return True
    threshold = min_marker or 0
    if threshold <= 0:
        return True
    return marker_max >= threshold


def acquire_processing_marker(dest_path: Path) -> Optional[Path]:
    """
    Create a best-effort marker to signal other processes that this file is in-flight.
    Returns the marker path on success, or None if one already exists.
    """

    marker_path = dest_path.with_suffix(dest_path.suffix + ".processing")
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None
    except OSError as exc:  # pragma: no cover - best effort logging
        logging.warning("Unable to create processing marker %s: %s", marker_path, exc)
        return None
    ACTIVE_PROCESSING_MARKERS.add(marker_path)
    return marker_path


def release_processing_marker(marker_path: Optional[Path]) -> None:
    if marker_path is None:
        return
    try:
        marker_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - best effort logging
        logging.debug("Unable to remove processing marker %s: %s", marker_path, exc)
    finally:
        if marker_path in ACTIVE_PROCESSING_MARKERS:
            ACTIVE_PROCESSING_MARKERS.discard(marker_path)


def cleanup_processing_markers() -> None:
    while ACTIVE_PROCESSING_MARKERS:
        marker_path = ACTIVE_PROCESSING_MARKERS.pop()
        try:
            marker_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _handle_exit_signal(signum, _frame):
    logging.warning("Received signal %s; cleaning up processing markers.", signum)
    cleanup_processing_markers()
    raise SystemExit(128 + signum)


atexit.register(cleanup_processing_markers)
signal.signal(signal.SIGINT, _handle_exit_signal)
signal.signal(signal.SIGTERM, _handle_exit_signal)


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def write_metadata(per_row_dir: Path, row: Dict[str, str]) -> None:
    per_row_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = per_row_dir / "metadata.json"
    if metadata_path.exists():
        return
    sanitized_row = {str(key): value for key, value in row.items() if key is not None}
    metadata_path.write_text(
        json.dumps(sanitized_row, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def download_with_retries(
    session: requests.Session,
    url: str,
    dest_path: Path,
    *,
    retries: int,
    timeout: float,
    max_download_seconds: Optional[float],
    sleep_base: float,
    impersonate: Optional[str],
) -> Tuple[bool, Optional[str]]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logging.debug("Skipping existing file %s", dest_path)
        return True, None

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        download_started = time.perf_counter()
        try:
            logging.debug("GET %s (attempt %s/%s)", url, attempt, retries)
            request_kwargs: Dict[str, object] = {"timeout": timeout, "stream": True}
            if impersonate:
                request_kwargs["impersonate"] = impersonate
            response = session.get(url, **request_kwargs)
            try:
                response.raise_for_status()
                with dest_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            fh.write(chunk)
                        if (
                            max_download_seconds is not None
                            and time.perf_counter() - download_started > max_download_seconds
                        ):
                            raise TimeoutError(
                                f"Download exceeded {max_download_seconds} seconds"
                            )
            finally:
                response.close()
            return True, None
        except Exception as exc:  # noqa: PERF203 - fine for retry loop
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                break
            sleep_time = min(sleep_base * (2 ** (attempt - 1)), 60.0)
            logging.warning(
                "Error downloading %s (attempt %s/%s): %s. Retrying in %.1fs",
                url,
                attempt,
                retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
    return False, last_error or "Unknown error"


def repair_stubs(
    base_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Download missing attachments and re-merge canonical_text for stub rows.

    Finds rows in _all_text.csv.gz where canonical_text has fewer than
    args.repair_min_words words. For each stub:
      1. Check if downloaded content files already exist on disk.
      2. If not, download attachment/content URLs from the source CSV row.
      3. Parse any downloaded PDFs to .txt.
      4. Re-assemble canonical_text from all available text files.

    Returns the total number of rows repaired.
    """
    import pandas as pd

    min_word_threshold = args.repair_min_words
    csv_filter = args.csv_filter
    csv_exclude = args.csv_exclude
    dry_run = args.repair_dry_run

    all_text_files = sorted(base_dir.glob("*_all_text.csv.gz"))
    total_repaired = 0
    total_stubs = 0
    total_downloaded = 0

    # Set up a session for downloading
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    apply_cli_cookies(session, args)

    for all_text_path in all_text_files:
        fname = all_text_path.name
        if csv_filter and csv_filter not in fname:
            continue
        if csv_exclude and csv_exclude in fname:
            continue

        try:
            df = pd.read_csv(all_text_path, dtype=str, keep_default_na=False).fillna("")
        except Exception as exc:
            logging.warning("Could not read %s: %s", all_text_path, exc)
            continue

        if "canonical_text" not in df.columns or "Document ID" not in df.columns:
            continue

        # Find stub rows
        word_counts = df["canonical_text"].str.split().str.len().fillna(0).astype(int)
        stub_mask = word_counts < min_word_threshold
        n_stubs = stub_mask.sum()
        if n_stubs == 0:
            continue

        total_stubs += n_stubs
        logging.info("%s: %d stubs (<%d words)", fname, n_stubs, min_word_threshold)

        # Find the source CSV and download directory
        csv_stem = fname.replace("_all_text.csv.gz", "").replace("_all_text.csv", "")
        agency_dir = all_text_path.parent
        download_root = agency_dir / DOWNLOAD_DIR_NAME

        # Also find the source CSV (without _all_text suffix) to get URLs
        source_csv = agency_dir / f"{csv_stem}.csv"
        source_df = None
        if source_csv.exists():
            try:
                source_df = pd.read_csv(
                    source_csv, dtype=str, keep_default_na=False
                ).fillna("")
                if "Document ID" in source_df.columns:
                    source_df = source_df.set_index("Document ID", drop=False)
            except Exception:
                source_df = None

        # Step 1: Download missing attachments for stub rows
        n_downloaded = 0
        for idx in df.index[stub_mask]:
            doc_id = str(df.at[idx, "Document ID"])
            if not doc_id:
                continue

            doc_dir_name = sanitize(doc_id, fallback=f"row_{idx:05d}")
            per_row_dir = download_root / csv_stem / doc_dir_name

            # Check if we already have text content files
            existing_text_files = []
            if per_row_dir.exists():
                for f in per_row_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in (".txt", ".htm", ".html"):
                        if f.name != "metadata.json" and not f.name.endswith(".stats.json"):
                            existing_text_files.append(f)

            if existing_text_files:
                continue  # Already have content, just needs re-merge

            # Get URLs from the _all_text row or source CSV
            content_urls = split_content_files(df.at[idx, "Content Files"] if "Content Files" in df.columns else "")
            attachment_urls = split_content_files(df.at[idx, "Attachment Files"] if "Attachment Files" in df.columns else "")

            # Fall back to source CSV if _all_text doesn't have URLs
            if not content_urls and not attachment_urls and source_df is not None:
                if doc_id in source_df.index:
                    src_row = source_df.loc[doc_id]
                    if isinstance(src_row, pd.DataFrame):
                        src_row = src_row.iloc[0]
                    content_urls = split_content_files(src_row.get("Content Files", ""))
                    attachment_urls = split_content_files(src_row.get("Attachment Files", ""))

            combined_urls = (
                [(url, "content") for url in content_urls]
                + [(url, "attachment") for url in attachment_urls]
            )
            if not combined_urls:
                continue

            if dry_run:
                logging.info("  [dry-run] Would download %d files for %s", len(combined_urls), doc_id)
                continue

            # Deduplicate by prefix, preferring textual over PDF
            url_entries = []
            for url_idx, (url, url_kind) in enumerate(combined_urls, start=1):
                suffix = Path(url.split("?")[0]).name or f"file_{url_idx}"
                base_name = sanitize(suffix, fallback=f"{url_kind}_{url_idx}")
                prefix_key = normalized_prefix_key(base_name)
                extension = Path(base_name).suffix.lower()
                url_entries.append({
                    "url": url, "base_name": base_name,
                    "prefix_key": prefix_key,
                    "is_pdf": extension == ".pdf",
                    "is_textual": extension in {".htm", ".html", ".txt"},
                    "order": url_idx,
                })
            # Sort: textual first, then PDFs
            url_entries.sort(key=lambda e: (e["is_pdf"], e["order"]))

            # Pre-scan: identify prefixes that have textual URLs available
            prefixes_with_textual_urls = {
                e["prefix_key"] for e in url_entries if e["is_textual"]
            }

            successful_textual_prefixes: Set[str] = set()
            for entry in url_entries:
                # Skip PDFs when a textual URL with same prefix exists in the list
                if entry["is_pdf"] and entry["prefix_key"] in prefixes_with_textual_urls:
                    continue
                # Also skip if we already downloaded a textual file with same prefix
                if entry["is_pdf"] and entry["prefix_key"] in successful_textual_prefixes:
                    continue

                dest_path = per_row_dir / entry["base_name"]
                if dest_path.exists():
                    if entry["is_textual"]:
                        successful_textual_prefixes.add(entry["prefix_key"])
                    continue

                ok, err = download_with_retries(
                    session, entry["url"], dest_path,
                    retries=args.retries, timeout=args.timeout,
                    max_download_seconds=args.max_download_seconds,
                    sleep_base=args.sleep_base,
                    impersonate=args.impersonate,
                )
                if ok:
                    n_downloaded += 1
                    if entry["is_textual"]:
                        successful_textual_prefixes.add(entry["prefix_key"])
                    # Parse PDFs
                    if entry["is_pdf"] and args.extract_pdf_text:
                        maybe_extract_pdf_text(
                            dest_path,
                            parser_choice=args.pdf_parser,
                            parse_timeout=args.pdf_parse_timeout,
                            per_page_timeout=args.pdf_parse_page_timeout,
                            progress_threshold=args.pdf_progress_threshold,
                            max_pages=args.pdf_parse_max_pages,
                            verify_pdf_pages=False,
                        )
                else:
                    logging.warning("  Failed to download %s for %s: %s", entry["url"], doc_id, err)

            # Write metadata
            write_metadata(per_row_dir, dict(df.iloc[idx]))

        if n_downloaded > 0:
            total_downloaded += n_downloaded
            logging.info("%s: downloaded %d new files", fname, n_downloaded)

        # Step 2: Re-collect all downloaded records and re-merge
        records = _collect_download_records(download_root, csv_stem)

        doc_text_map: Dict[str, List[str]] = {}
        docs_with_txt: set = set()
        for rec in records:
            if not rec["file_name"].endswith(".txt"):
                continue
            fp = Path(rec["file_path"])
            if not fp.exists():
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                doc_text_map.setdefault(rec["doc_id"], []).append(text)
                docs_with_txt.add(rec["doc_id"])
        for rec in records:
            fn = rec["file_name"]
            if not (fn.endswith(".htm") or fn.endswith(".html")):
                continue
            if rec["doc_id"] in docs_with_txt:
                continue
            fp = Path(rec["file_path"])
            text = _read_html_as_text(fp)
            if text:
                doc_text_map.setdefault(rec["doc_id"], []).append(text)

        comment_cols = [
            col for col in df.columns
            if ("comment" in col.lower()) and ("comment on" not in col.lower())
        ]

        n_repaired = 0
        for idx in df.index[stub_mask]:
            doc_id = df.at[idx, "Document ID"]
            extra_texts = doc_text_map.get(doc_id, [])
            if not extra_texts:
                continue

            pieces: List[str] = []
            if comment_cols:
                inline = "\n\n".join(
                    str(df.at[idx, col]).strip()
                    for col in comment_cols
                    if str(df.at[idx, col]).strip()
                )
                if inline:
                    pieces.append(inline)
            for extra in extra_texts:
                cleaned = str(extra).strip()
                if cleaned:
                    pieces.append(cleaned)

            if not pieces:
                continue

            new_text = "\n\n".join(
                f"<<COMMENT {i}>>\n{chunk}" for i, chunk in enumerate(pieces, start=1)
            )
            new_word_count = len(new_text.split())
            old_word_count = int(word_counts[idx])

            if new_word_count > old_word_count:
                if not dry_run:
                    df.at[idx, "canonical_text"] = new_text
                n_repaired += 1
                if n_repaired <= 5:
                    logging.info(
                        "  %s: %s %d -> %d words",
                        fname, doc_id, old_word_count, new_word_count,
                    )

        if n_repaired > 0:
            total_repaired += n_repaired
            logging.info(
                "%s: repaired %d/%d stubs",
                fname, n_repaired, n_stubs,
            )
            if not dry_run:
                df.to_csv(all_text_path, index=False)
                logging.info("  Wrote %s", all_text_path)
        else:
            logging.debug(
                "%s: %d stubs, no content found to repair",
                fname, n_stubs,
            )

    logging.info(
        "Repair complete: %d rows repaired, %d files downloaded, %d total stubs",
        total_repaired, total_downloaded, total_stubs,
    )
    return total_repaired


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pdfminer_logger = logging.getLogger("pdfminer")
    pdfminer_logger.setLevel(logging.ERROR)
    pdfminer_logger.propagate = False
    logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)

    if not args.base_dir.exists():
        logging.error("Base directory %s does not exist", args.base_dir)
        return 1

    if args.repair:
        # Repair mode: fix stub rows in existing _all_text files
        if args.recursive:
            # Search all agency subdirs
            agency_dirs = sorted(
                d for d in args.base_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
                and d.name not in ("scripts", "__pycache__")
            )
            total = 0
            for agency_dir in agency_dirs:
                for subdir in sorted(agency_dir.iterdir()):
                    if not subdir.is_dir():
                        continue
                    n = repair_stubs(subdir, args)
                    total += n
            logging.info("Total repaired across all directories: %d", total)
        else:
            repair_stubs(args.base_dir, args)
        return 0

    if args.cleanup_ocr_pdfs:
        removed = delete_ocrd_content_pdfs(args.base_dir)
        logging.info(
            "Deleted %s existing content.pdf files that already had OCR text.",
            removed,
        )

    scripts_dir = args.base_dir / "scripts"
    error_log_path = scripts_dir / ERROR_LOG_BASENAME
    logged_failures: Set[Tuple[str, str]] = set()
    if not args.retry_download_errors:
        logged_failures = load_logged_failures(error_log_path)
        logging.info(
            "Skipping %s previously failed downloads (disable with --retry-download-errors).",
            len(logged_failures),
        )
    csv_paths = list(iter_csv_files(args.base_dir, recursive=args.recursive))
    if args.seed is not None:
        random.seed(args.seed)
        logging.info("Shuffling CSV tasks with seed %s.", args.seed)
    else:
        logging.info("Shuffling CSV tasks with an implicit random seed.")
    random.shuffle(csv_paths)
    if not csv_paths:
        logging.warning("No CSV files found under %s", args.base_dir)
        return 0

    logging.info("Found %s CSV files under %s", len(csv_paths), args.base_dir)

    if tqdm is None:
        logging.info("Install tqdm for progress bars (pip install tqdm).")
        csv_iterator = csv_paths
        colour_cycle = None
    else:
        csv_iterator = tqdm(
            csv_paths,
            desc="CSV files",
            unit="csv",
            colour="cyan",
            leave=True,
        )
        colour_cycle = cycle(["magenta", "yellow", "green", "blue", "white"])

    def create_session() -> requests.Session:
        session_obj = requests.Session()
        session_obj.headers.update(
            {
                "User-Agent": args.user_agent,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "application/pdf;q=0.9,image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.regulations.gov/",
            }
        )
        apply_cli_cookies(session_obj, args)
        return session_obj

    session = create_session()
    try:
        processed_files = 0
        processed_rows = 0
        downloaded_files = 0
        failed_downloads: List[str] = []

        for index, csv_path in enumerate(csv_iterator, start=1):
            csv_marker: Optional[Path] = None
            if args.lock_csv:
                csv_marker = acquire_processing_marker(csv_path)
                if csv_marker is None:
                    logging.info("Skipping CSV %s (processing marker present).", csv_path)
                    continue
            try:
                agency_dir = csv_path.parent
                csv_name = csv_path.stem
                if args.csv_filter and args.csv_filter.lower() not in csv_name.lower():
                    continue
                if args.csv_exclude and args.csv_exclude.lower() in csv_name.lower():
                    continue
                all_text_path_gz = csv_path.with_name(f"{csv_name}_all_text.csv.gz")
                all_text_path_csv = csv_path.with_name(f"{csv_name}_all_text.csv")
                if args.skip_if_all_text_exists and (all_text_path_gz.exists() or all_text_path_csv.exists()):
                    logging.info("Skipping %s (_all_text already exists)", csv_path)
                    continue
                csv_is_public_submission = "public_submission" in csv_name.lower()
                csv_pdf_max_pages = (
                    args.pdf_parse_max_pages_public_submission
                    if csv_is_public_submission and args.pdf_parse_max_pages_public_submission
                    else args.pdf_parse_max_pages
                )
                csv_verify_pages = args.verify_pdf_pages and (
                    not args.verify_pdf_pages_filter
                    or args.verify_pdf_pages_filter.lower() in csv_name.lower()
                )
                logging.info("Processing CSV %s/%s: %s", index, len(csv_paths), csv_path)
                if args.merge_only:
                    # Skip all downloads, just merge
                    if not args.skip_merge:
                        try:
                            merge_all_text(csv_path)
                        except Exception as exc:
                            logging.warning("Merge failed for %s: %s", csv_path, exc)
                    processed_files += 1
                    continue
                per_csv_downloads = 0
                per_csv_rows = 0
                row_bar = None
                row_total = None
                if tqdm is not None:
                    try:
                        row_total = count_csv_rows(csv_path)
                        if args.max_rows_per_csv:
                            row_total = min(row_total, args.max_rows_per_csv)
                    except Exception as exc:  # pragma: no cover - file read issues
                        logging.debug("Could not count rows for %s: %s", csv_path, exc)
                    row_colour = next(colour_cycle) if colour_cycle else None
                    row_bar = tqdm(
                        total=row_total,
                        desc=f"{agency_dir.name}/{csv_name}",
                        unit="row",
                        colour=row_colour,
                        leave=False,
                        position=1,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    )
                with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    for idx, row in enumerate(reader, start=1):
                        if args.max_rows_per_csv and per_csv_rows >= args.max_rows_per_csv:
                            logging.debug(
                                "Per-CSV row cap (%s) reached for %s",
                                args.max_rows_per_csv,
                                csv_path,
                            )
                            break
                        if args.max_rows_per_csv and idx > args.max_rows_per_csv:
                            logging.debug(
                                "Per-CSV row index cap (%s) reached for %s",
                                args.max_rows_per_csv,
                                csv_path,
                            )
                            break
                        if args.debug and per_csv_downloads >= args.debug_limit:
                            logging.debug(
                                "Debug limit reached for %s (limit=%s)",
                                csv_path,
                                args.debug_limit,
                            )
                            break
                        content_urls = split_content_files(row.get("Content Files"))
                        attachment_urls = split_content_files(row.get("Attachment Files"))
                        combined_urls = (
                            [(url, "content") for url in content_urls]
                            + [(url, "attachment") for url in attachment_urls]
                        )
                        if not combined_urls:
                            if row_bar is not None:
                                row_bar.update(1)
                            continue
                        row_has_html = any(looks_like_html(url) for url, _ in combined_urls)
                        doc_id = row.get("Document ID") or row.get("DocumentId") or ""
                        doc_dir_name = sanitize(doc_id, fallback=f"row_{idx:05d}")
                        per_row_dir = (
                            agency_dir
                            / args.output_dir_name
                            / csv_name
                            / doc_dir_name
                        )
                        log_verbose(
                            args,
                            "Row %s doc=%s urls=%s has_html=%s",
                            idx,
                            doc_id or doc_dir_name,
                            len(combined_urls),
                            row_has_html,
                        )
                        write_metadata(per_row_dir, row)
                        url_entries = []
                        successful_textual_prefixes: Set[str] = set()
                        for url_idx, (url, url_kind) in enumerate(combined_urls, start=1):
                            suffix = Path(url.split("?")[0]).name or f"file_{url_idx}"
                            base_name = sanitize(suffix, fallback=f"{url_kind}_{url_idx}")
                            prefix_key = normalized_prefix_key(base_name)
                            extension = Path(base_name).suffix.lower()
                            url_entries.append(
                                {
                                    "url": url,
                                    "url_kind": url_kind,
                                    "suffix": suffix,
                                    "base_name": base_name,
                                    "prefix_key": prefix_key,
                                    "is_pdf": extension == ".pdf",
                                    "is_textual": extension in {".htm", ".html", ".txt"},
                                    "order": url_idx,
                                }
                            )
                        url_entries.sort(key=lambda entry: (entry["is_pdf"], entry["order"]))
                        used_names: Dict[str, int] = {}
                        for url_idx, entry in enumerate(url_entries, start=1):
                            url = entry["url"]
                            url_kind = entry["url_kind"]
                            suffix = entry["suffix"]
                            base_name = entry["base_name"]
                            prefix_key = entry["prefix_key"]
                            counter = used_names.get(base_name, 0)
                            if counter:
                                file_name = f"{base_name}_{counter+1}"
                            else:
                                file_name = base_name
                            used_names[base_name] = counter + 1
                            dest_path = per_row_dir / file_name
                            error_key = (str(csv_path), str(dest_path))
                            is_pdf = entry["is_pdf"]
                            is_textual = entry["is_textual"]
                            if args.skip_pdf_download and is_pdf:
                                log_verbose(
                                    args,
                                    "  skipping PDF download for %s (flag enabled)",
                                    dest_path,
                                )
                                continue
                            text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                            partial_marker = dest_path.with_suffix(dest_path.suffix + ".partial")
                            text_payload_exists = text_path.exists()
                            partial_exists = partial_marker.exists()
                            text_marker_max = 0
                            if csv_verify_pages and text_payload_exists:
                                text_marker_max = max_page_marker_in_text(text_path)
                            force_verify_download = should_force_verify_download(
                                verify_enabled=csv_verify_pages,
                                text_exists=text_payload_exists,
                                partial_exists=partial_exists,
                                marker_max=text_marker_max,
                                min_marker=args.verify_pdf_pages_min_marker,
                            )
                            if (
                                not force_verify_download
                                and args.skip_pdf_when_text_prefix
                                and is_pdf
                                and prefix_key in successful_textual_prefixes
                            ):
                                log_verbose(
                                    args,
                                    "  skipping PDF download for %s (text/HTML counterpart present)",
                                    dest_path,
                                )
                                continue
                            if (
                                not force_verify_download
                                and is_pdf
                                and text_payload_exists
                                and not partial_exists
                            ):
                                if csv_verify_pages and args.verify_pdf_pages_min_marker:
                                    log_verbose(
                                        args,
                                        "  skipping verify for %s (max page marker %s < threshold %s).",
                                        dest_path,
                                        text_marker_max,
                                        args.verify_pdf_pages_min_marker,
                                    )
                                else:
                                    log_verbose(
                                        args,
                                        "  skipping download for %s (text payload present)",
                                        dest_path,
                                    )
                                continue
                            if not args.retry_download_errors and error_key in logged_failures:
                                log_verbose(
                                    args,
                                    "  skipping %s (previous failure logged)",
                                    dest_path,
                                )
                                continue
                            processing_marker: Optional[Path] = None
                            if args.lock_individual_download_file:
                                processing_marker = acquire_processing_marker(dest_path)
                                if processing_marker is None:
                                    log_verbose(
                                        args,
                                        "  skipping %s (processing marker present)",
                                        dest_path,
                                    )
                                    continue
                            try:
                                log_verbose(
                                    args,
                                    "  fetching url #%s (%s) -> %s",
                                    url_idx,
                                    suffix,
                                    dest_path,
                                )
                                download_success, error_detail = download_with_retries(
                                    session,
                                    url,
                                    dest_path,
                                    retries=args.retries,
                                    timeout=args.timeout,
                                    max_download_seconds=args.max_download_seconds,
                                    sleep_base=args.sleep_base,
                                    impersonate=args.impersonate,
                                )
                                if not download_success:
                                    reason = error_detail or "Failed to download"
                                    error_msg = (
                                        f"{csv_path} | row {idx} | {doc_id or doc_dir_name} | {url}: {reason}"
                                    )
                                    failed_downloads.append(error_msg)
                                    logging.error(error_msg)
                                    log_download_error(error_log_path, csv_path, dest_path, reason)
                                    if dest_path.exists():
                                        try:
                                            dest_path.unlink()
                                        except OSError:
                                            pass
                                    used_names[base_name] -= 1
                                    if used_names[base_name] <= 0:
                                        used_names.pop(base_name, None)
                                    continue
                                if (
                                    download_success
                                    and args.skip_pdf_when_text_prefix
                                    and is_textual
                                ):
                                    successful_textual_prefixes.add(prefix_key)
                                log_verbose(
                                    args,
                                    "  downloaded %s (%s bytes)",
                                    dest_path.name,
                                    dest_path.stat().st_size if dest_path.exists() else "n/a",
                                )
                                if (
                                    args.extract_pdf_text
                                    and not (args.skip_pdf_when_html and row_has_html)
                                    and not args.skip_pdf_download
                                    and is_pdf
                                ):
                                    parsed_full_pdf = maybe_extract_pdf_text(
                                        dest_path,
                                        parser_choice=args.pdf_parser,
                                        parse_timeout=args.pdf_parse_timeout,
                                        per_page_timeout=args.pdf_parse_page_timeout,
                                        progress_threshold=args.pdf_progress_threshold,
                                        max_pages=csv_pdf_max_pages,
                                        verify_pdf_pages=force_verify_download,
                                    )
                                    if args.cleanup_ocr_pdfs and parsed_full_pdf:
                                        text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                                        if text_path.exists():
                                            try:
                                                dest_path.unlink()
                                                log_verbose(
                                                    args,
                                                    "  deleted %s after OCR (cleanup enabled)",
                                                    dest_path.name,
                                                )
                                            except OSError as exc:
                                                logging.warning(
                                                    "Failed to delete %s after OCR: %s",
                                                    dest_path,
                                                    exc,
                                                )
                                elif args.skip_pdf_when_html and is_pdf:
                                    log_verbose(
                                        args,
                                        "  skipped PDF extraction for %s (HTML present in row)",
                                        dest_path.name,
                                    )
                            finally:
                                if args.lock_individual_download_file:
                                    release_processing_marker(processing_marker)
                                downloaded_files += 1
                                per_csv_downloads += 1
                                if args.debug and per_csv_downloads >= args.debug_limit:
                                    logging.debug(
                                        "Debug limit reached for %s (limit=%s)",
                                        csv_path,
                                        args.debug_limit,
                                    )
                                    break
                        per_csv_rows += 1
                        processed_rows += 1
                        if row_bar is not None:
                            row_bar.update(1)
                processed_files += 1
                if row_bar is not None:
                    row_bar.close()
                logging.info(
                    "Finished %s (%s rows with content, %s files downloaded)",
                    csv_path.name,
                    per_csv_rows,
                    per_csv_downloads,
                )
                # Merge inline comments + downloaded text into _all_text.csv.
                if not args.skip_merge:
                    try:
                        merge_all_text(csv_path)
                    except Exception as exc:
                        logging.warning("Merge failed for %s: %s", csv_path, exc)
            finally:
                if args.lock_csv:
                    release_processing_marker(csv_marker)
                session.close()
                session = create_session()
                gc.collect()

        logging.info(
            "Done. Processed %s CSV files, %s rows with content, downloaded %s files.",
            processed_files,
            processed_rows,
            downloaded_files,
        )
        if failed_downloads:
            logging.warning(
                "Encountered %s failed downloads. See log summary below.",
                len(failed_downloads),
            )
            for entry in failed_downloads[:20]:
                logging.warning("FAILED: %s", entry)
            if len(failed_downloads) > 20:
                logging.warning("... %s additional failures omitted from log", len(failed_downloads) - 20)
            return 1 if not args.allow_failures else 0
        return 0
    finally:
        session.close()


def maybe_extract_pdf_text(
    dest_path: Path,
    *,
    parser_choice: str,
    parse_timeout: Optional[float],
    per_page_timeout: Optional[float],
    progress_threshold: Optional[int],
    max_pages: Optional[int],
    verify_pdf_pages: bool,
) -> bool:
    if dest_path.suffix.lower() != ".pdf":
        return True
    if pdf_parsing is None:
        logging.warning(
            "Skipping PDF text extraction for %s: ai_corpus.utils.pdf_parsing unavailable",
            dest_path,
        )
        return False

    text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
    stats_path = dest_path.with_suffix(dest_path.suffix + ".stats.json")
    partial_marker = dest_path.with_suffix(dest_path.suffix + ".partial")
    existing_stats: Optional[Dict[str, object]] = None
    if stats_path.exists():
        try:
            existing_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            existing_stats = None
    if (
        not verify_pdf_pages
        and text_path.exists()
        and stats_path.exists()
        and not partial_marker.exists()
        and existing_stats
        and not existing_stats.get("truncated_to_pages")
    ):
        logging.debug("PDF text already extracted for %s", dest_path)
        return True
    existing_pages = 0
    if existing_stats:
        try:
            existing_pages = int(existing_stats.get("total_pages", 0))  # type: ignore[arg-type]
        except Exception:
            existing_pages = 0
    partial_info: Optional[Dict[str, object]] = None
    resume_mode = False
    resume_start_page = 1
    remaining_limit = max_pages
    if not verify_pdf_pages and partial_marker.exists():
        try:
            partial_info = json.loads(partial_marker.read_text(encoding="utf-8"))
        except Exception:
            partial_info = None
            logging.debug("Failed to read partial marker for %s; ignoring resume.", dest_path)
        else:
            logging.debug("Detected partial marker for %s: %s", dest_path, partial_info)
    elif verify_pdf_pages and partial_marker.exists():
        logging.debug(
            "Ignoring partial marker for %s due to verification mode.",
            dest_path,
        )

    if not verify_pdf_pages:
        def _iter_partial_entries(info: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
            if not info:
                return []
            if info.get("reason") == "partial":
                details = info.get("details")
                if isinstance(details, list):
                    return [entry for entry in details if isinstance(entry, dict)]
                return []
            return [info]

        truncated_entry = next(
            (entry for entry in _iter_partial_entries(partial_info) if entry.get("reason") == "pdf_parse_max_pages"),
            None,
        )
        if truncated_entry and existing_pages > 0:
            resumed_limit = truncated_entry.get("max_pages")
            resume_mode = False
            if max_pages is None or (isinstance(max_pages, int) and max_pages > existing_pages):
                resume_mode = True
                resume_start_page = existing_pages + 1
                if max_pages is None:
                    remaining_limit = None
                else:
                    remaining_limit = max_pages - existing_pages
                    if remaining_limit is not None and remaining_limit <= 0:
                        logging.info(
                            "Skipping resume for %s (already reached max pages %s).",
                            dest_path,
                            max_pages,
                        )
                        return True
            if resume_mode and not text_path.exists():
                logging.warning(
                    "Cannot resume %s (expected text file missing). Rebuilding from scratch.",
                    dest_path,
                )
                resume_mode = False
                resume_start_page = 1
                remaining_limit = max_pages
        else:
            resume_mode = False
            resume_start_page = 1
            remaining_limit = max_pages
        if resume_mode:
            logging.info(
                "Resuming PDF %s from page %s (previously captured %s pages).",
                dest_path,
                resume_start_page,
                existing_pages,
            )
        elif partial_marker.exists():
            logging.debug(
                "Partial marker for %s present but resume was not triggered (existing_pages=%s, max_pages=%s).",
                dest_path,
                existing_pages,
                max_pages,
            )

    page_count: Optional[int] = None
    progress_bar = None
    progress_callback = None
    if tqdm is not None and progress_threshold and progress_threshold > 0:
        try:
            page_count = pdf_parsing.get_pdf_page_count(str(dest_path))
        except Exception:
            page_count = None
        show_progress = page_count is not None and page_count >= progress_threshold
        if show_progress:
            progress_bar = tqdm(
                total=page_count,
                desc=f"Parse {dest_path.name}",
                unit="page",
                colour="cyan",
                leave=False,
                position=2,
            )

            def _progress_callback(event: Dict[str, object]) -> None:
                if progress_bar is None:
                    return
                event_type = event.get("type")
                stage = event.get("stage")
                if event_type == "stage":
                    total = event.get("total")
                    try:
                        progress_bar.reset(total=total)
                    except Exception:
                        if total:
                            progress_bar.total = total
                    if stage:
                        progress_bar.set_description(f"Parse {dest_path.name} [{stage}]")
                    progress_bar.n = 0
                    progress_bar.refresh()
                elif event_type == "page":
                    total = event.get("total")
                    page_number = event.get("page")
                    if total and progress_bar.total != total:
                        try:
                            progress_bar.reset(total=total)
                        except Exception:
                            progress_bar.total = total
                    if isinstance(page_number, int) and page_number >= 0:
                        if page_number > progress_bar.n:
                            progress_bar.n = page_number
                            progress_bar.refresh()

            progress_callback = _progress_callback

    if verify_pdf_pages and page_count is None and pdf_parsing is not None:
        try:
            page_count = pdf_parsing.get_pdf_page_count(str(dest_path))
        except Exception:
            page_count = None

    if verify_pdf_pages:
        resume_mode = False
        resume_start_page = 1
        remaining_limit = max_pages
        expected_pages = page_count
        if expected_pages and text_path.exists():
            recorded_pages = max_page_marker_in_text(text_path)
            if recorded_pages >= expected_pages:
                logging.info(
                    "Verified existing text for %s (%s/%s pages).",
                    dest_path,
                    recorded_pages,
                    expected_pages,
                )
                if partial_marker.exists():
                    try:
                        partial_marker.unlink()
                    except OSError:
                        pass
                return True
            logging.info(
                "Existing text for %s only has %s/%s pages; re-parsing.",
                dest_path,
                recorded_pages,
                expected_pages,
            )
            if partial_marker.exists():
                try:
                    partial_marker.unlink()
                except OSError:
                    pass
        existing_pages = 0
        partial_info = None

    parse_started = time.perf_counter()
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path_handle = None

    def _write_page(page_number: int, page_text: str) -> None:
        if text_path_handle is None:
            raise RuntimeError("Text path handle unavailable for PDF streaming")
        rendered = f"<<PAGE {page_number}>>\n{page_text.strip()}\n\n"
        text_path_handle.write(rendered)

    stats_run_pages_target = remaining_limit
    try:
        mode = "a" if resume_mode else "w"
        with text_path.open(mode, encoding="utf-8") as handle:
            text_path_handle = handle
            if resume_mode and handle.tell() > 0:
                handle.write("\n")
            deadline = None
            if parse_timeout and parse_timeout > 0:
                deadline = parse_started + parse_timeout
            if stats_run_pages_target is not None and stats_run_pages_target <= 0:
                logging.debug(
                    "No remaining pages required for %s (max_pages=%s).",
                    dest_path,
                    max_pages,
                )
                return True
            _pages_unused, stats = pdf_parsing.extract_pages(
                str(dest_path),
                parser=parser_choice,
                per_page_timeout=per_page_timeout,
                parse_deadline=deadline,
                progress_callback=progress_callback,
                total_pages=page_count,
                max_pages=stats_run_pages_target,
                page_callback=_write_page,
                capture_pages=False,
                start_page=resume_start_page,
            )
    except TimeoutError as exc:
        logging.warning("PDF parsing timeout for %s: %s", dest_path, exc)
        if text_path.exists():
            try:
                text_path.unlink()
            except OSError:
                pass
        return False
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to parse PDF %s: %s", dest_path, exc)
        if text_path.exists():
            try:
                text_path.unlink()
            except OSError:
                pass
        return False
    finally:
        if progress_bar is not None:
            progress_bar.close()

    elapsed = time.perf_counter() - parse_started
    stats.setdefault("elapsed_seconds", elapsed)
    truncated = False
    run_pages = stats.get("total_pages") or 0
    total_emitted = existing_pages + run_pages if resume_mode else run_pages
    if remaining_limit is not None and remaining_limit > 0 and run_pages >= remaining_limit:
        stats["truncated_to_pages"] = max_pages
        truncated = True
    elif "truncated_to_pages" in stats:
        stats.pop("truncated_to_pages", None)

    timed_out = bool(stats.get("timed_out"))
    if timed_out:
        logging.warning(
            "PDF parsing timed out for %s after %.1fs (%s pages extracted; stage=%s).",
            dest_path,
            stats.get("elapsed_seconds", parse_timeout) or (parse_timeout or 0),
            total_emitted,
            stats.get("timed_out_stage") or "unknown",
        )

    if resume_mode and existing_stats:
        additive_keys = [
            "initial_blanks",
            "pymupdf_filled",
            "pymupdf_attempted",
            "pymupdf_failed",
            "ocr_filled",
            "ocr_attempted",
            "ocr_failed",
            "remaining_blanks",
        ]
        merged_stats: Dict[str, object] = dict(existing_stats)
        for key in additive_keys:
            merged_stats[key] = merged_stats.get(key, 0) + stats.get(key, 0)  # type: ignore[assignment]
        merged_stats["total_pages"] = total_emitted
        merged_stats["elapsed_seconds"] = (
            merged_stats.get("elapsed_seconds", 0.0)  # type: ignore[arg-type]
            + stats.get("elapsed_seconds", 0.0)
        )
        merged_stats["timed_out"] = stats.get("timed_out", 0)
        merged_stats["timed_out_stage"] = stats.get("timed_out_stage", "")
        if "timed_out_reason" in stats:
            merged_stats["timed_out_reason"] = stats.get("timed_out_reason")
        stats = merged_stats  # type: ignore[assignment]
        if not truncated and "truncated_to_pages" in stats:
            stats.pop("truncated_to_pages", None)

    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    partial_payloads = []
    if truncated:
        partial_payloads.append(
            {
                "reason": "pdf_parse_max_pages",
                "max_pages": max_pages,
                "total_pages": stats.get("total_pages"),
            }
        )
        logging.info(
            "Limited extraction for %s to %s pages (total %s).",
            dest_path,
            max_pages,
            stats.get("total_pages"),
        )
    if timed_out:
            partial_payloads.append(
                {
                    "reason": "pdf_parse_timeout",
                    "timeout_seconds": parse_timeout if parse_timeout and parse_timeout > 0 else None,
                    "elapsed_seconds": stats.get("elapsed_seconds"),
                    "timed_out_stage": stats.get("timed_out_stage"),
                    "timed_out_reason": stats.get("timed_out_reason"),
                    "pages_extracted": total_emitted,
                }
            )
    if partial_payloads:
        if len(partial_payloads) == 1:
            marker_payload = partial_payloads[0]
        else:
            marker_payload = {"reason": "partial", "details": partial_payloads}
        partial_marker.write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    elif partial_marker.exists():
        try:
            partial_marker.unlink()
        except OSError:
            pass
    global _PDF_PARSE_COUNTER
    _PDF_PARSE_COUNTER += 1
    if PDF_PARSE_GC_INTERVAL and _PDF_PARSE_COUNTER % PDF_PARSE_GC_INTERVAL == 0:
        logging.debug(
            "Triggering garbage collection after %s parsed PDFs.",
            _PDF_PARSE_COUNTER,
        )
        gc.collect()
    return not (truncated or timed_out)


if __name__ == "__main__":
    sys.exit(main())
