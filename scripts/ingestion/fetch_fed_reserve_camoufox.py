"""Fed Reserve scraper via camoufox (headless browser) — gets FULL submitter metadata.

The Fed Reserve comments portal (https://www.federalreserve.gov/apps/proposals/) is a
Blazor Server SPA. The server-side rendered HTML only shows the 10 NEWEST comments per
proposal — so the existing brute-force-ID approach captures comments but loses submitter
metadata for 98% of them.

With camoufox we can drive the Blazor SignalR connection and paginate through all
comments, capturing submitter name, org, and posted date for every one.

Usage:
    python fetch_fed_reserve_camoufox.py --proposals FR-2026-0002-01 FR-2025-0077-01
    python fetch_fed_reserve_camoufox.py --proposals-file discovered_proposals.txt
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

BASE = "https://www.federalreserve.gov/apps/proposals"


def scrape_proposal_comments(page, proposal_id: str, max_wait_sec: int = 30) -> list[dict]:
    """Navigate to a proposal's comments page, paginate through all, collect metadata."""
    url = f"{BASE}/{proposal_id}/comments"
    page.goto(url, wait_until="networkidle", timeout=60000)

    # Set items per page to max (500) if selector exists
    try:
        page.evaluate("""
            const sel = document.querySelector('select[name*="itemsPerPage"]');
            if (sel) { sel.value = '500'; sel.dispatchEvent(new Event('change')); }
        """)
        page.wait_for_timeout(2000)
    except Exception:
        pass

    # Find total pages
    all_rows = []
    seen_ids = set()
    page_num = 0
    max_pages = 2000  # safety (large proposals can have 700+ pages)

    while page_num < max_pages:
        # Wait for comment rows to render
        try:
            page.wait_for_selector(".comment-row", timeout=15000)
        except Exception:
            break

        # Extract all visible rows
        rows = page.evaluate(r"""
            () => {
                const rows = document.querySelectorAll('.comment-row');
                return Array.from(rows).map(r => {
                    const link = r.querySelector('a[href*="comments/"]');
                    const href = link ? link.getAttribute('href') : '';
                    const text = link ? link.textContent.trim() : '';
                    // Posted date is in a div without class that contains "Posted |"
                    const divs = r.querySelectorAll('div');
                    let postedText = '';
                    for (const d of divs) {
                        const t = d.textContent || '';
                        if (t.includes('Posted')) { postedText = t.trim(); break; }
                    }
                    const id_m = href.match(/comments\/([A-Z0-9-]+)/);
                    return {
                        id: id_m ? id_m[1] : '',
                        display: text,
                        posted: postedText,
                        href: href,
                    };
                });
            }
        """)

        new_count = 0
        for r in rows:
            if r["id"] and r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_rows.append(r)
                new_count += 1

        if new_count == 0:
            break

        # Capture first row ID to detect page change
        first_id_before = rows[0]["id"] if rows else None

        # Try clicking "Next" button (button with aria-label="Next page")
        next_clicked = page.evaluate("""
            () => {
                const next = document.querySelector('button[aria-label="Next page"]:not([disabled])');
                if (next) { next.click(); return true; }
                return false;
            }
        """)
        if not next_clicked:
            break
        page_num += 1

        # Wait for page to actually change (first row should have different ID)
        for _ in range(30):  # up to 15 seconds
            page.wait_for_timeout(500)
            new_first = page.evaluate("""
                () => {
                    const r = document.querySelector('.comment-row a[href*="comments/"]');
                    const m = r ? r.getAttribute('href').match(/comments\\/([A-Z0-9-]+)/) : null;
                    return m ? m[1] : null;
                }
            """)
            if new_first and new_first != first_id_before:
                break

    return all_rows


def parse_display(display: str) -> tuple[str, str, str]:
    """Parse '{Org}, {Name} (PDF)' → (org, name, format)."""
    if not display:
        return "", "", ""
    fmt_m = re.search(r"\(([^)]+)\)\s*$", display)
    fmt = fmt_m.group(1) if fmt_m else ""
    core = re.sub(r"\s*\([^)]+\)\s*$", "", display).strip()
    if core.lower() == "anonymous":
        return "", "", fmt
    tmp = core.replace(", et. al.,", "§§§ET§§§")
    parts = tmp.rsplit(",", 1)
    if len(parts) == 2:
        org = parts[0].replace("§§§ET§§§", ", et. al.,").strip()
        name = parts[1].replace("§§§ET§§§", ", et. al.,").strip()
        return org, name, fmt
    return "", core, fmt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", nargs="+", help="Proposal IDs")
    parser.add_argument("--proposals-file", help="File with one proposal ID per line")
    parser.add_argument("--headful", action="store_true", help="Show browser (debug)")
    args = parser.parse_args()

    proposals = []
    if args.proposals:
        proposals = args.proposals
    elif args.proposals_file:
        proposals = [l.strip() for l in Path(args.proposals_file).read_text().splitlines() if l.strip()]
    else:
        # Auto-discover from existing dockets.csv.gz
        d = get_output_dir("fed_reserve")
        dockets = d / "dockets.csv.gz"
        if dockets.exists():
            df = pd.read_csv(dockets, dtype=str)
            # Only proposals with >10 comments (too-small ones are fine with existing method)
            df["n_comments"] = df["n_comments"].fillna("0").astype(int)
            proposals = df[df["n_comments"] > 10]["proposal_id"].tolist()
            logger.info("Auto-discovered %d proposals with >10 comments", len(proposals))

    if not proposals:
        logger.error("No proposals provided")
        return

    done = load_done_units("fed_reserve_camoufox")
    logger.info("%d proposals, %d done, %d to process",
                len(proposals), len(done), len(proposals) - len(done))

    from camoufox.sync_api import Camoufox
    with Camoufox(headless=not args.headful, humanize=False, geoip=True) as browser:
        page = browser.new_page()
        for pid in proposals:
            if pid in done:
                continue
            try:
                rows = scrape_proposal_comments(page, pid)
                logger.info("  %s: %d comments captured", pid, len(rows))
                if rows:
                    # Normalize to standard schema
                    records = []
                    for r in rows:
                        org, name, fmt = parse_display(r["display"])
                        records.append({
                            "source": "fed_reserve_camoufox",
                            "comment_id": r["id"],
                            "docket_id": pid,
                            "agency_id": "FRS",
                            "submitter_name": name,
                            "submitter_org": org,
                            "posted_date": r.get("posted", "").replace("Posted | ", "").strip(),
                            "comment_text": "",
                            "attachment_urls": f"{BASE}/comments/{r['id']}",
                            "raw_metadata": json.dumps({"display": r["display"], "format": fmt}),
                        })
                    df = pd.DataFrame(records)
                    append_comments(df, "fed_reserve_camoufox")
                mark_done("fed_reserve_camoufox", pid)
            except Exception as e:
                logger.error("  %s failed: %s", pid, e)

    save_metadata("fed_reserve_camoufox", {
        "source_url": BASE,
        "method": "camoufox (Blazor SPA scraping)",
        "n_proposals": len(proposals),
        "notes": "Drives Blazor SignalR pagination to capture ALL comments with submitter metadata, not just first 10 per proposal.",
    })


if __name__ == "__main__":
    main()
