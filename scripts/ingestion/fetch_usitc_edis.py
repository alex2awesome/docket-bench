"""Scrape USITC EDIS (Electronic Document Information System) for public comments.

USITC posts ALL filings (including public-interest comments and docket comments)
in Section 337 investigations, antidumping/CVD, and rulemaking in EDIS at
https://edis.usitc.gov/. Bot-protected (403 without browser); use camoufox.

Usage:
    python fetch_usitc_edis.py
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

BASE = "https://edis.usitc.gov"


def scrape_investigations(headless: bool = True, max_pages: int = 100):
    from camoufox.sync_api import Camoufox
    output_dir = get_output_dir("usitc_edis")
    done = load_done_units("usitc_edis")

    with Camoufox(headless=headless, humanize=False) as browser:
        page = browser.new_page()

        # Navigate to search
        logger.info("Opening EDIS search...")
        page.goto(f"{BASE}/search", wait_until="networkidle", timeout=60000)
        time.sleep(5)

        # Use search filters for public-interest comments / rulemaking
        # EDIS categorizes filings by "Document Type" (e.g., "Public Interest Comments")
        # Extract all investigations visible
        total_comments = 0
        pages_processed = 0
        while pages_processed < max_pages:
            try:
                page.wait_for_selector("table tr, [class*='investigation']", timeout=15000)
            except Exception:
                break

            rows = page.evaluate(r"""
                () => {
                    const rows = document.querySelectorAll('table tbody tr, [class*="investigation-item"]');
                    return Array.from(rows).map(r => {
                        const links = r.querySelectorAll('a[href]');
                        return {
                            text: r.textContent.trim().slice(0, 400),
                            links: Array.from(links).map(a => ({href: a.href, text: a.textContent.trim().slice(0, 100)})),
                        };
                    });
                }
            """)

            for row in rows:
                # Try to find investigation number + filing link
                inv_match = re.search(r"\b(33[1-7]-[A-Z]+-\d+|\d{3}-[A-Z]+-\d+)\b", row["text"])
                inv_id = inv_match.group(1) if inv_match else None
                if not inv_id:
                    continue
                if inv_id in done:
                    continue

                # Get first PDF link
                pdf_link = next((l["href"] for l in row["links"] if ".pdf" in l["href"].lower()), None)
                filing_link = next((l["href"] for l in row["links"] if l["href"] != page.url), None)

                records = [{
                    "source": "usitc_edis",
                    "comment_id": f"edis_{inv_id}",
                    "docket_id": inv_id,
                    "agency_id": "USITC",
                    "submitter_name": "",
                    "submitter_org": "",
                    "posted_date": "",
                    "comment_text": "",
                    "attachment_urls": pdf_link or filing_link or "",
                    "raw_metadata": json.dumps({"row_text": row["text"][:500]}),
                }]
                df = pd.DataFrame(records)
                append_comments(df, "usitc_edis")
                total_comments += 1
                mark_done("usitc_edis", inv_id)

            # Paginate
            next_ok = page.evaluate("""
                () => {
                    const btn = document.querySelector('button[aria-label*="next" i]:not([disabled]), a[aria-label*="next" i]:not(.disabled)');
                    if (btn) { btn.click(); return true; }
                    return false;
                }
            """)
            if not next_ok:
                break
            pages_processed += 1
            time.sleep(2)

    save_metadata("usitc_edis", {
        "source_url": BASE,
        "n_comments": total_comments,
    })
    logger.info("USITC: %d comments", total_comments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    scrape_investigations(headless=not args.headful)
