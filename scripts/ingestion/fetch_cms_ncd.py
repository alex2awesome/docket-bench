"""Scrape CMS National Coverage Determination (NCD) public comments.

CMS NCD comments are posted on the Medicare Coverage Database (cms.gov), not
regulations.gov. Each NCA (National Coverage Analysis) has a dedicated page
with comment letters.

URL pattern:
  Index: https://www.cms.gov/medicare-coverage-database/reports/national-coverage-ncacal-status-report.aspx
  Detail: https://www.cms.gov/medicare-coverage-database/view/nca.aspx?ncaid=N

Uses camoufox because the page is ASP.NET-rendered.

Usage:
    python fetch_cms_ncd.py
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

BASE = "https://www.cms.gov/medicare-coverage-database"


def scrape_all(headless: bool = True):
    from camoufox.sync_api import Camoufox
    output_dir = get_output_dir("cms_ncd")
    done = load_done_units("cms_ncd")

    with Camoufox(headless=headless, humanize=False) as browser:
        page = browser.new_page()

        # Step 1: Enumerate NCAs
        logger.info("Fetching NCA index...")
        page.goto(f"{BASE}/reports/national-coverage-ncacal-status-report.aspx?ncacaldoctype=all",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)

        ncas = page.evaluate(r"""
            () => {
                const links = Array.from(document.querySelectorAll('a[href*="ncaid="]'));
                const uniq = new Map();
                for (const a of links) {
                    const m = a.href.match(/ncaid=(\d+)/);
                    if (m && !uniq.has(m[1])) {
                        uniq.set(m[1], {
                            ncaid: m[1],
                            url: a.href,
                            text: a.textContent.trim().slice(0, 200),
                        });
                    }
                }
                return Array.from(uniq.values());
            }
        """)
        logger.info("Found %d unique NCAs", len(ncas))

        # Step 2: For each NCA, visit and extract public comments
        total = 0
        for nca in ncas:
            if nca["ncaid"] in done:
                continue
            try:
                # Detail page shows "View Public Comments" link
                page.goto(nca["url"], wait_until="networkidle", timeout=45000)
                time.sleep(2)

                # Find public comments link
                comments_url = page.evaluate(r"""
                    () => {
                        const a = document.querySelector('a[href*="public-comments"], a[href*="PublicComments"]');
                        return a ? a.href : null;
                    }
                """)
                if not comments_url:
                    mark_done("cms_ncd", nca["ncaid"])
                    continue

                # Visit public comments page
                page.goto(comments_url, wait_until="networkidle", timeout=45000)
                time.sleep(2)

                comments = page.evaluate(r"""
                    () => {
                        const out = [];
                        // Look for tables with commenter rows
                        const rows = document.querySelectorAll('table tr');
                        for (const row of rows) {
                            const cells = row.querySelectorAll('td');
                            if (cells.length < 2) continue;
                            const link = row.querySelector('a[href]');
                            const text = row.textContent.trim();
                            if (text.length < 10 || text.length > 800) continue;
                            out.push({
                                submitter: cells[0] ? cells[0].textContent.trim().slice(0, 200) : '',
                                org: cells[1] ? cells[1].textContent.trim().slice(0, 200) : '',
                                date: cells[2] ? cells[2].textContent.trim() : '',
                                url: link ? link.href : '',
                            });
                        }
                        // Also look for comment letter PDFs directly
                        const pdfs = document.querySelectorAll('a[href*=".pdf"]');
                        for (const pdf of pdfs) {
                            const text = pdf.textContent.trim();
                            if (text.length > 3 && text.length < 200) {
                                out.push({
                                    submitter: text.slice(0, 200),
                                    org: '',
                                    date: '',
                                    url: pdf.href,
                                });
                            }
                        }
                        return out;
                    }
                """)

                if comments:
                    records = []
                    for i, c in enumerate(comments):
                        records.append({
                            "source": "cms_ncd",
                            "comment_id": f"cms_{nca['ncaid']}_{i}",
                            "docket_id": nca["ncaid"],
                            "agency_id": "CMS",
                            "submitter_name": c["submitter"],
                            "submitter_org": c["org"],
                            "posted_date": c["date"],
                            "comment_text": "",
                            "attachment_urls": c["url"],
                            "raw_metadata": json.dumps({"nca_title": nca["text"]}),
                        })
                    df = pd.DataFrame(records)
                    append_comments(df, "cms_ncd")
                    total += len(records)
                    logger.info("  NCA %s: %d comments", nca["ncaid"], len(records))

                mark_done("cms_ncd", nca["ncaid"])
                time.sleep(0.5)
            except Exception as e:
                logger.error("  NCA %s failed: %s", nca["ncaid"], e)

    save_metadata("cms_ncd", {
        "source_url": BASE,
        "n_ncas": len(ncas),
        "n_comments": total,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    scrape_all(headless=not args.headful)
