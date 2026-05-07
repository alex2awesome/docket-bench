"""Scrape USACE district public notice pages via camoufox.

Each USACE district maintains its own public-notice page. The district sites
return 403 to plain requests; camoufox bypasses bot detection.

Usage:
    python fetch_usace_districts.py
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

DISTRICTS = [
    "mvn", "mvp", "mvs", "spn", "spk", "poa", "nae", "nan", "nab",
    "nws", "swg", "sac", "saw", "saj", "sam", "lrl", "lrd", "sas",
    "swf", "swt", "swl", "spa", "nwk", "lrb", "lrh", "lrp", "mvk",
    "spd", "nwp", "mvm", "lre", "nwo", "nao", "lrc", "mvr", "nws",
]


def scrape_district(page, district: str) -> list[dict]:
    """Try multiple URL patterns for a district's public notices page."""
    urls = [
        f"https://www.{district}.usace.army.mil/Missions/Regulatory/Public-Notices/",
        f"https://www.{district}.usace.army.mil/Missions/Regulatory-Branch/Public-Notices/",
        f"https://www.{district}.usace.army.mil/Media/Public-Notices/",
        f"https://www.{district}.usace.army.mil/Missions/Regulatory/",
    ]
    for url in urls:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)
            # Check if page has content
            body_len = page.evaluate("document.body.innerText.length")
            if body_len > 3000:
                break
        except Exception:
            continue
    else:
        return []

    # Extract notices
    notices = page.evaluate(r"""
        () => {
            const out = [];
            const articles = document.querySelectorAll('article, .card, .notice-item, tr, li');
            const seen_urls = new Set();
            for (const item of articles) {
                const link = item.querySelector('a[href]');
                if (!link) continue;
                const href = link.href;
                if (seen_urls.has(href)) continue;
                // Filter out nav
                const text = item.textContent.trim();
                if (text.length < 30 || text.length > 1500) continue;
                // Look for notice-like patterns (permit numbers, dates)
                const has_signal = /\b\d{4}-\d+|\b\d{2}-\d{4}-\d+|SPA-|LRN-|NAE-|SAJ-|Public Notice|Permit Application|File Number/.test(text);
                if (!has_signal) continue;
                seen_urls.add(href);
                out.push({
                    url: href,
                    text: text.slice(0, 1000),
                });
            }
            return out;
        }
    """)
    return notices


def main():
    from camoufox.sync_api import Camoufox
    parser = argparse.ArgumentParser()
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()

    done = load_done_units("usace_districts")
    total = 0
    with Camoufox(headless=not args.headful, humanize=False) as browser:
        page = browser.new_page()
        for d in DISTRICTS:
            if d in done:
                continue
            try:
                notices = scrape_district(page, d)
            except Exception as e:
                logger.error("  %s failed: %s", d, e)
                continue
            logger.info("  %s: %d notices", d, len(notices))
            if notices:
                records = []
                for i, n in enumerate(notices):
                    records.append({
                        "source": "usace_districts",
                        "comment_id": f"usace_{d}_{i}",
                        "docket_id": d,
                        "agency_id": "USACE",
                        "submitter_name": "",
                        "submitter_org": "",
                        "posted_date": "",
                        "comment_text": n["text"],
                        "attachment_urls": n["url"],
                        "raw_metadata": json.dumps({"district": d}),
                    })
                df = pd.DataFrame(records)
                append_comments(df, "usace_districts")
                total += len(records)
            mark_done("usace_districts", d)
            time.sleep(1)

    save_metadata("usace_districts", {
        "districts": DISTRICTS,
        "n_notices": total,
        "method": "camoufox (bypasses 403 bot-detection on district sites)",
    })


if __name__ == "__main__":
    main()
