"""Download state AG actions from NYU State Impact Center database.

The State Energy & Environmental Impact Center at NYU Law maintains a
database of state AG actions on energy/environmental/climate matters,
including rulemaking comment letters. Covers 2017-present, updated through 2026.

URL: https://stateimpactcenter.org/ag-work/ag-actions
Downloadable as CSV.

Usage:
    python fetch_nyu_ag_actions.py
"""

import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, save_comments, http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://stateimpactcenter.org/ag-work/ag-actions"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def parse_page(html: str) -> list[dict]:
    """Parse a single results page into action dicts."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    # NYU site uses <article> elements per action
    for item in soup.find_all(["article", "div"], class_=re.compile(r"action|entry|item|card|post")):
        title_el = item.find(["h2", "h3", "h4", "a"])
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if not title or len(title) < 5:
            continue
        date_el = item.find(class_=re.compile(r"date"))
        desc_el = item.find(class_=re.compile(r"desc|summary|excerpt|content"))
        link_el = item.find("a", href=True)
        rows.append({
            "title": title,
            "date": date_el.get_text(strip=True) if date_el else "",
            "description": desc_el.get_text(strip=True) if desc_el else "",
            "url": link_el["href"] if link_el else "",
        })
    return rows


def fetch_ag_actions() -> pd.DataFrame:
    """Fetch all AG actions across all pages."""
    logger.info("Fetching AG actions from %s", BASE_URL)

    # Get page 1 to discover total page count
    resp = requests.get(BASE_URL, headers=BROWSER_HEADERS, timeout=60)
    resp.raise_for_status()

    # Find max page from pagination links
    page_nums = re.findall(r"/ag-actions/page/(\d+)", resp.text)
    max_page = max([int(p) for p in page_nums]) if page_nums else 1
    logger.info("Discovered %d pages of results", max_page)

    all_rows = parse_page(resp.text)
    logger.info("  Page 1: %d entries", len(all_rows))

    # Iterate through remaining pages
    for page in range(2, max_page + 1):
        url = f"{BASE_URL}/page/{page}"
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=60)
            r.raise_for_status()
        except Exception as e:
            logger.warning("Page %d failed: %s", page, e)
            continue
        rows = parse_page(r.text)
        all_rows.extend(rows)
        if page % 10 == 0:
            logger.info("  Page %d/%d: %d total entries", page, max_page, len(all_rows))
        time.sleep(0.3)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["title", "date"])
    logger.info("Total unique entries: %d", len(df))
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NYU AG actions to common comments schema."""
    import json
    records = []
    for i, row in df.iterrows():
        # Adapt column names (may vary depending on source format)
        title = str(row.get("title", row.get("Title", row.get("Action", ""))))
        date = str(row.get("date", row.get("Date", row.get("Date Filed", ""))))
        desc = str(row.get("description", row.get("Description", row.get("Summary", ""))))
        url = str(row.get("url", row.get("URL", row.get("Link", ""))))
        states = str(row.get("states", row.get("States", row.get("Lead State(s)", ""))))
        agency = str(row.get("agency", row.get("Agency", row.get("Federal Agency", ""))))
        action_type = str(row.get("type", row.get("Type", row.get("Action Type", ""))))

        records.append({
            "comment_id": f"nyu_ag_{i}",
            "docket_id": agency,
            "agency_id": agency,
            "submitter_name": "State Attorneys General",
            "submitter_org": states,
            "posted_date": date,
            "comment_text": f"{title}. {desc}",
            "attachment_urls": url,
            "raw_metadata": json.dumps({
                "title": title,
                "action_type": action_type,
                "states": states,
                "federal_agency": agency,
                "source": "NYU State Impact Center",
            }),
        })
    return pd.DataFrame(records)


def main():
    output_dir = get_output_dir("nyu_ag_actions")

    df = fetch_ag_actions()
    if df.empty:
        logger.error("No data retrieved")
        return

    # Save raw data
    df.to_csv(output_dir / "raw_ag_actions.csv.gz", index=False, compression="gzip")
    logger.info("Saved %d raw entries", len(df))

    # Normalize and save as comments
    comments = normalize(df)
    save_comments(comments, "nyu_ag_actions")

    save_metadata("nyu_ag_actions", {
        "source_url": BASE_URL,
        "n_actions": len(df),
        "notes": (
            "State AG actions from NYU State Energy & Environmental Impact Center. "
            "Covers 2017-present, energy/environmental/climate topics. "
            "Supplements the Nolette database (frozen at 2019) for post-2019 AG activity."
        ),
    })


if __name__ == "__main__":
    main()
