"""FCC ECFS scraper via camoufox UI — bypasses 10K API cap.

The FCC ECFS UI (https://www.fcc.gov/ecfs/search/filings?proceedings.name=XX-YY)
is a React SPA. We drive it via camoufox to extract all filings for a given
docket, including bulk/mass-campaign submissions that the API-based approach
may miss beyond its 10K-per-query hard cap.

Usage:
    python fetch_fcc_ui.py --docket 16-42
    python fetch_fcc_ui.py --docket 17-108 --max-pages 5  # testing
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://www.fcc.gov/ecfs/search/filings"


def scrape_docket(docket: str, max_pages: int = 10000, headless: bool = True):
    from camoufox.sync_api import Camoufox
    output_dir = get_output_dir("fcc_ui")
    done = load_done_units("fcc_ui")

    with Camoufox(headless=headless, humanize=False) as browser:
        page = browser.new_page()
        url = f"{BASE}?proceedings.name={docket}&sort=date_received,ASC&limit=500"
        logger.info("Loading: %s", url)
        page.goto(url, wait_until="networkidle", timeout=120000)
        time.sleep(8)  # React render

        all_filings = []
        seen = set()
        page_num = 0

        while page_num < max_pages:
            # Wait for filing items to load
            try:
                page.wait_for_selector('[data-testid*="filing"], .filing-item, article a[href*="/filing/"]', timeout=20000)
            except Exception:
                # Try alternate selectors
                items_count = page.evaluate("document.querySelectorAll('a[href*=\"/filing/\"]').length")
                if items_count == 0:
                    break

            filings = page.evaluate(r"""
                () => {
                    const items = [];
                    // Filing links lead to /ecfs/filing/{id}
                    const links = document.querySelectorAll('a[href*="/ecfs/filing/"]');
                    const seen = new Set();
                    for (const a of links) {
                        const m = a.href.match(/\/filing\/(\d+)/);
                        if (!m) continue;
                        const id = m[1];
                        if (seen.has(id)) continue;
                        seen.add(id);
                        // Try to find parent card with more info
                        const card = a.closest('article, li, [class*="card"], [class*="item"], [class*="result"]');
                        const text = card ? card.textContent.trim().slice(0, 500) : a.textContent.trim();
                        items.push({id, url: a.href, text});
                    }
                    return items;
                }
            """)

            new_count = 0
            for f in filings:
                if f["id"] not in seen:
                    seen.add(f["id"])
                    all_filings.append(f)
                    new_count += 1

            if new_count == 0:
                break

            logger.info("  page %d: %d new filings (total %d)", page_num, new_count, len(all_filings))

            # Try clicking next page
            next_ok = page.evaluate(r"""
                () => {
                    const btn = document.querySelector('button[aria-label*="next" i]:not([disabled]), a[aria-label*="next" i]:not(.disabled), .pagination-next:not(.disabled) a');
                    if (btn) { btn.click(); return true; }
                    return false;
                }
            """)
            if not next_ok:
                break
            page_num += 1
            time.sleep(3)

        # Save
        if all_filings:
            records = []
            for f in all_filings:
                records.append({
                    "source": "fcc_ui",
                    "comment_id": f"fcc_ui_{f['id']}",
                    "docket_id": docket,
                    "agency_id": "FCC",
                    "submitter_name": "",
                    "submitter_org": "",
                    "posted_date": "",
                    "comment_text": "",
                    "attachment_urls": f["url"],
                    "raw_metadata": json.dumps({"row_text": f["text"][:500]}),
                })
            df = pd.DataFrame(records)
            append_comments(df, "fcc_ui")
            logger.info("Saved %d filings for %s", len(records), docket)
        mark_done("fcc_ui", docket)

    save_metadata("fcc_ui", {
        "source_url": BASE,
        "last_docket": docket,
        "method": "camoufox (React SPA pagination, bypasses 10K API cap)",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", required=True, help="FCC docket (e.g., 16-42)")
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    scrape_docket(args.docket, args.max_pages, headless=not args.headful)


if __name__ == "__main__":
    main()
