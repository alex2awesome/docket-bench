"""Scrape state AG letters and formal comments from attorneysgeneral.org (Nolette DB).

This is the canonical index of multistate Attorney General letters and formal
comments on federal rulemakings, maintained by Paul Nolette at Marquette.

The site uses a Ninja Tables (WordPress) plugin with AJAX-loaded data. The
initial HTML is just a container — rows are fetched via a POST to
/wp-admin/admin-ajax.php with a nonce extracted from the landing page.

Covers 2012-present. HTML scraping; no API key needed. Requires browser-like
headers to bypass the site's ModSecurity rules.

Usage:
    python fetch_state_ag_letters.py
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, save_comments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://attorneysgeneral.org"
LANDING_URL = f"{BASE}/letters-and-formal-comments/searchable-list-of-multistate-letters-and-formal-comments-2017-present/"
AJAX_URL = f"{BASE}/wp-admin/admin-ajax.php"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_nonce_and_table_id() -> tuple[str, int]:
    """Fetch the landing page and extract the Ninja Tables nonce + table_id."""
    logger.info("Fetching landing page for nonce")
    resp = requests.get(LANDING_URL, headers=BROWSER_HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    nonce_match = re.search(r'nonce"\s*:\s*"([a-f0-9]+)"', html)
    table_match = re.search(r'table_id"\s*:\s*"?(\d+)"?', html)
    if not nonce_match or not table_match:
        raise RuntimeError("Could not extract Ninja Tables nonce/table_id from landing page")
    return nonce_match.group(1), int(table_match.group(1))


def fetch_all_rows() -> list[dict]:
    """Fetch all rows via the Ninja Tables AJAX endpoint."""
    nonce, table_id = fetch_nonce_and_table_id()
    logger.info("Fetching rows via AJAX (table_id=%d)", table_id)

    session = requests.Session()
    session.headers.update({
        **BROWSER_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": BASE,
        "Referer": LANDING_URL,
    })
    data = {
        "action": "wp_ajax_ninja_tables_public_action",
        "table_id": table_id,
        "target_action": "get-all-data",
        "default_sorting": "old_first",
        "ninja_table_public_nonce": nonce,
        "skip_rows": 0,
    }
    resp = session.post(AJAX_URL, data=data, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected AJAX payload type: {type(payload).__name__}")
    logger.info("  Got %d rows", len(payload))
    return payload


def _strip_html(s: str) -> str:
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)


def _extract_href(html_field: str) -> str:
    if not html_field:
        return ""
    m = re.search(r'href="([^"]+)"', html_field)
    return m.group(1) if m else ""


def normalize(rows: list[dict]) -> pd.DataFrame:
    """Normalize Ninja Tables rows to the common comments schema."""
    records = []
    for i, row in enumerate(rows):
        value = row.get("value", {}) if isinstance(row, dict) else {}
        letter_url = _extract_href(value.get("copyofletter", ""))
        records.append({
            "comment_id": f"ag_letter_{i}",
            "docket_id": value.get("recipient", ""),
            "agency_id": value.get("recipient", ""),
            "submitter_name": "State Attorneys General (multistate)",
            "submitter_org": value.get("agsinvolved", ""),
            "posted_date": value.get("datefiled", ""),
            "comment_text": _strip_html(value.get("specificissue", "")),
            "attachment_urls": letter_url,
            "raw_metadata": json.dumps({
                "type": value.get("type"),
                "year": value.get("year"),
                "general_issue": value.get("generalissue"),
                "specific_issue": _strip_html(value.get("specificissue", "")),
                "recipient": value.get("recipient"),
                "ags_involved": value.get("agsinvolved"),
                "partisan": value.get("partisan"),
            }),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()

    output_dir = get_output_dir("state_ag_letters")

    rows = fetch_all_rows()
    if not rows:
        logger.error("No entries found. Site structure may have changed.")
        return

    # Filter by year
    def _in_range(row):
        try:
            y = int((row.get("value", {}) or {}).get("year", 0))
            return args.start_year <= y <= args.end_year
        except (ValueError, TypeError):
            return True
    rows = [r for r in rows if _in_range(r)]
    logger.info("After year filter %d-%d: %d rows", args.start_year, args.end_year, len(rows))

    # Save raw JSON for reference
    (output_dir / "raw_rows.json").write_text(json.dumps(rows, indent=2))

    df = normalize(rows)
    save_comments(df, "state_ag_letters")

    save_metadata("state_ag_letters", {
        "source_url": LANDING_URL,
        "n_letters": len(df),
        "years": f"{args.start_year}-{args.end_year}",
        "notes": (
            "Nolette multistate AG letters database. Fetched via Ninja Tables "
            "WordPress plugin AJAX endpoint (/wp-admin/admin-ajax.php) with "
            "dynamically extracted nonce. Covers 2012-present (the URL says "
            "2017-present but the table actually contains earlier entries). "
            "comment_text is a short specific_issue blurb; full letter is at "
            "attachment_urls (usually a PDF). Requires browser-like headers "
            "to bypass site's ModSecurity rules."
        ),
    })


if __name__ == "__main__":
    main()
