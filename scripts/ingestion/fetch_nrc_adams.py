"""Scrape NRC ADAMS public documents.

NRC ADAMS Public Search at https://adams.nrc.gov/wba/ is a JS search interface.
Fill in "keywords" and submit to find public comment documents.

Usage:
    python fetch_nrc_adams.py --keyword "public comment rulemaking"
    python fetch_nrc_adams.py --keyword "10 CFR 50"
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

BASE = "https://adams.nrc.gov/wba/"


def search_and_extract(page, keyword: str) -> list[dict]:
    """Execute a keyword search and extract results."""
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    time.sleep(5)

    # Fill keyword and submit
    page.evaluate(f"""
        () => {{
            const input = document.querySelector('input#keywords');
            if (input) {{
                input.value = {json.dumps(keyword)};
                input.dispatchEvent(new Event('input', {{bubbles: true}}));
            }}
            const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Search' && b.type === 'submit');
            if (btn) btn.click();
        }}
    """)
    time.sleep(8)  # wait for results

    # Extract results from the results page (after redirect to adams-search.nrc.gov)
    all_docs = []
    page_num = 0
    max_pages = 500
    seen_acc = set()
    time.sleep(5)  # let results render

    while page_num < max_pages:
        # Results page parses body text for accession numbers (they're in plain text, not table)
        docs = page.evaluate(r"""
            () => {
                const out = [];
                const text = document.body.innerText;
                // Accession IDs like ML24360A120 followed by title + dates
                const matches = [...text.matchAll(/(ML[A-Z0-9]{7,10})\s+(.+?)\s+(\d{4}-\d{2}-\d{2}[^\n]*?)(?:\s+(\d{4}-\d{2}-\d{2}))?(?=\s*\n|$)/g)];
                for (const m of matches) {
                    out.push({
                        accession: m[1],
                        title: m[2].trim().slice(0, 300),
                        posted: m[3].trim(),
                        document_date: (m[4] || '').trim(),
                    });
                }
                // Fallback: any ML accession numbers we missed
                const allIds = [...new Set(text.match(/ML[A-Z0-9]{7,10}/g) || [])];
                const seen = new Set(out.map(d => d.accession));
                for (const id of allIds) {
                    if (!seen.has(id)) out.push({accession: id, title: '', posted: '', document_date: ''});
                }
                return out;
            }
        """)

        new_count = 0
        for d in docs:
            if d["accession"] not in seen_acc:
                seen_acc.add(d["accession"])
                all_docs.append(d)
                new_count += 1

        if new_count == 0:
            break

        logger.info("  page %d: %d new docs (total %d)", page_num, new_count, len(all_docs))

        # Click next page - try ARIA labels and text-content matches
        next_ok = page.evaluate(r"""
            () => {
                // Try various pagination button patterns
                const candidates = [
                    'button[aria-label*="next" i]:not([disabled])',
                    'button[title*="Next" i]:not([disabled])',
                    'button[aria-label*="Next page" i]:not([disabled])',
                    '.ui-paginator-next:not(.ui-state-disabled)',
                ];
                for (const sel of candidates) {
                    const btn = document.querySelector(sel);
                    if (btn) { btn.click(); return true; }
                }
                // Try text-based next button
                const all_btns = Array.from(document.querySelectorAll('button, a'));
                const next_btn = all_btns.find(b => /^next$|next\s*>|>/i.test(b.textContent.trim()) && !b.disabled);
                if (next_btn) { next_btn.click(); return true; }
                return false;
            }
        """)
        if not next_ok:
            break
        page_num += 1
        time.sleep(3)

    return all_docs


def main():
    from camoufox.sync_api import Camoufox
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="public comment",
                        help="Search keyword for NRC ADAMS")
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()

    done = load_done_units("nrc_adams")
    with Camoufox(headless=not args.headful, humanize=False) as browser:
        page = browser.new_page()
        try:
            docs = search_and_extract(page, args.keyword)
        except Exception as e:
            logger.error("Search failed: %s", e)
            docs = []

        logger.info("Found %d docs", len(docs))
        if docs:
            records = []
            for d in docs:
                if d["accession"] in done:
                    continue
                records.append({
                    "source": "nrc_adams",
                    "comment_id": f"adams_{d['accession']}",
                    "docket_id": "",
                    "agency_id": "NRC",
                    "submitter_name": d.get("title", "")[:200],
                    "submitter_org": "",
                    "posted_date": d.get("document_date", ""),
                    "comment_text": "",
                    "attachment_urls": f"https://adams.nrc.gov/wba/?Accession={d['accession']}",
                    "raw_metadata": json.dumps({
                        "title": d.get("title", ""),
                        "keyword": args.keyword,
                        "posted": d.get("posted", ""),
                    }),
                })
                mark_done("nrc_adams", d["accession"])
            if records:
                df = pd.DataFrame(records)
                append_comments(df, "nrc_adams")

    save_metadata("nrc_adams", {
        "source_url": BASE,
        "keyword": args.keyword,
        "n_docs": len(docs),
    })


if __name__ == "__main__":
    main()
