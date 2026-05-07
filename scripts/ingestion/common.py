"""Common utilities for external source ingestion scripts."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
BULK_DIR = SCRIPTS_DIR.parent
EXTERNAL_DIR = BULK_DIR / "external_sources"


def get_output_dir(source: str) -> Path:
    """Return the output directory for a given external source."""
    path = EXTERNAL_DIR / source
    path.mkdir(parents=True, exist_ok=True)
    return path


REQUIRED_COMMENT_COLS = [
    "source", "comment_id", "docket_id", "agency_id",
    "submitter_name", "submitter_org", "posted_date",
    "comment_text", "attachment_urls", "raw_metadata",
]


def save_comments(df: pd.DataFrame, source: str, append: bool = False):
    """Save comments to external_sources/{source}/comments.csv.gz with standard columns."""
    out = get_output_dir(source) / "comments.csv.gz"
    for col in REQUIRED_COMMENT_COLS:
        if col not in df.columns:
            df[col] = None
    df["source"] = source

    if append and out.exists():
        try:
            existing = pd.read_csv(out, dtype=str)
            df = pd.concat([existing, df[REQUIRED_COMMENT_COLS].astype(str)],
                           ignore_index=True).drop_duplicates(subset=["comment_id"])
        except Exception as e:
            logger.warning("Append read failed, overwriting: %s", e)

    df[REQUIRED_COMMENT_COLS].to_csv(out, index=False, compression="gzip")
    logger.info("Saved %d comments to %s", len(df), out)


def append_comments(df: pd.DataFrame, source: str):
    """Shortcut: save_comments with append=True."""
    save_comments(df, source, append=True)


def done_file_path(source: str, name: str = "done_units.txt") -> Path:
    """Return path to a checkpoint file tracking completed work units."""
    return get_output_dir(source) / name


def load_done_units(source: str, name: str = "done_units.txt") -> set[str]:
    """Load the set of already-completed units (docket IDs, RINs, etc.)."""
    p = done_file_path(source, name)
    if not p.exists():
        return set()
    with open(p) as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(source: str, unit: str, name: str = "done_units.txt"):
    """Append a unit to the done-units checkpoint file.

    Atomic-enough for single-process use; uses line-append so partial writes
    are self-contained.
    """
    p = done_file_path(source, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(unit + "\n")


def save_dockets(df: pd.DataFrame, source: str):
    """Save docket metadata."""
    out = get_output_dir(source) / "dockets.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    logger.info("Saved %d dockets to %s", len(df), out)


def save_metadata(source: str, metadata: dict):
    """Save source metadata (fetch timestamp, counts, version)."""
    out = get_output_dir(source) / "metadata.json"
    metadata["fetch_timestamp"] = pd.Timestamp.now().isoformat()
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", out)


def http_get(url: str, session: Optional[requests.Session] = None,
             headers: Optional[dict] = None, params: Optional[dict] = None,
             max_retries: int = 3, backoff: float = 1.0, timeout: int = 30) -> requests.Response:
    """HTTP GET with retries."""
    s = session or requests.Session()
    default_headers = {"User-Agent": "regulations-demo-research/1.0"}
    if headers:
        default_headers.update(headers)

    for attempt in range(max_retries):
        try:
            resp = s.get(url, headers=default_headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                logger.warning("Rate limited, sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("Request failed (%s), retrying in %.1fs", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def hash_content(text: str) -> str:
    """Generate a stable hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
