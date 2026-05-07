"""Scrape CFTC public comments via camoufox.

CFTC hosts comments at https://comments.cftc.gov/ (NOT regulations.gov).
Cloudflare-protected; camoufox bypasses bot detection.

Archive URL pattern:
  /PublicComments/ReleasesWithComments.aspx?Type=ListAll&Year=YYYY  (list releases per year)
  /PublicComments/CommentList.aspx?id=N                             (comments for release N)

Covers 2007-present.

Usage:
    python fetch_cftc.py                              # all years 2007-2026
    python fetch_cftc.py --years 2020 2021 2022      # specific years
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

BASE = "https://comments.cftc.gov/PublicComments"


def scrape_all(years: list[int], headless: bool = True):
    from camoufox.sync_api import Camoufox
    output_dir = get_output_dir("cftc_comments")
    done = load_done_units("cftc_comments")

    with Camoufox(headless=headless, humanize=False) as browser:
        page = browser.new_page()

        # Step 1: Enumerate all releases across years
        all_releases = []
        for year in years:
            logger.info("Year %d: fetching releases...", year)
            page.goto(f"{BASE}/ReleasesWithComments.aspx?Type=ListAll&Year={year}",
                      wait_until="networkidle", timeout=60000)
            time.sleep(3)
            links = page.evaluate(r"""
                () => Array.from(document.querySelectorAll('a[href*="CommentList.aspx?id="]'))
                    .map(a => {
                        const m = a.href.match(/id=(\d+)/);
                        const id = m ? m[1] : null;
                        const row = a.closest('tr, li, div');
                        const rowText = row ? row.textContent.trim().slice(0, 300) : '';
                        return { id, href: a.href, text: rowText };
                    })
                    .filter(r => r.id)
            """)
            # Dedupe
            seen_ids = set(r["id"] for r in all_releases)
            for l in links:
                if l["id"] not in seen_ids:
                    all_releases.append({**l, "year": year})
                    seen_ids.add(l["id"])
            logger.info("  year %d: %d new releases", year, len(links))

        logger.info("Total releases: %d", len(all_releases))

        # Step 2: Scrape each release's comment list
        total_comments = 0
        for i, rel in enumerate(all_releases):
            rel_id = rel["id"]
            if rel_id in done:
                continue
            try:
                page.goto(f"{BASE}/CommentList.aspx?id={rel_id}",
                          wait_until="networkidle", timeout=60000)
                time.sleep(2)
                comments = page.evaluate(r"""
                    () => {
                        const out = [];
                        const links = document.querySelectorAll('a[href*="CommentDetail"], a[href*=".pdf"]');
                        const seen = new Set();
                        for (const a of links) {
                            if (seen.has(a.href)) continue;
                            seen.add(a.href);
                            const row = a.closest('tr, li');
                            const rowCells = row ? row.querySelectorAll('td') : [];
                            const cellTexts = Array.from(rowCells).map(c => c.textContent.trim());
                            out.push({
                                url: a.href,
                                text: a.textContent.trim().slice(0, 200),
                                cells: cellTexts,
                            });
                        }
                        return out;
                    }
                """)

                if comments:
                    records = []
                    for c in comments:
                        submitter = c["cells"][0] if c["cells"] else c["text"]
                        org = c["cells"][1] if len(c["cells"]) > 1 else ""
                        date = c["cells"][2] if len(c["cells"]) > 2 else ""
                        records.append({
                            "source": "cftc_comments",
                            "comment_id": f"cftc_{rel_id}_{len(records)}",
                            "docket_id": rel_id,
                            "agency_id": "CFTC",
                            "submitter_name": submitter[:200],
                            "submitter_org": org[:200],
                            "posted_date": date,
                            "comment_text": "",
                            "attachment_urls": c["url"],
                            "raw_metadata": json.dumps({"release_info": rel.get("text", ""), "year": rel["year"]}),
                        })
                    df = pd.DataFrame(records)
                    append_comments(df, "cftc_comments")
                    total_comments += len(records)
                    logger.info("  release %s (year %d): %d comments (total %d)",
                                rel_id, rel["year"], len(records), total_comments)

                mark_done("cftc_comments", rel_id)
                time.sleep(0.8)
            except Exception as e:
                logger.error("  release %s failed: %s", rel_id, e)

    save_metadata("cftc_comments", {
        "source_url": BASE,
        "method": "camoufox",
        "n_releases": len(all_releases),
        "n_comments": total_comments,
        "years": years,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2007, 2027)))
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    scrape_all(args.years, headless=not args.headful)
