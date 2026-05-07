"""Scrape FDIC public comments.

FDIC posts comments at RIN-specific pages:
    https://www.fdic.gov/federal-register-publications/comments-rin-XXXX-XXXX

Comments are typically PDFs linked from the RIN page with submitter names.

Usage:
    python fetch_fdic_comments.py --rins 3064-AG04 3064-AF29
    python fetch_fdic_comments.py --list-all
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, http_get, save_comments, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FDIC_BASE = "https://www.fdic.gov"
FDIC_LISTING = f"{FDIC_BASE}/federal-register-publications"


def list_all_rins(start_year: int = 2016, end_year: int = 2026) -> list[tuple[str, str]]:
    """Scrape the FDIC federal register publications listing for comment page URLs.

    Returns a list of (rin, full_slug) tuples. The slug includes a date suffix
    (e.g. "3064-ag20-december-19-2025") that is required for the per-RIN URL.

    Scans:
      1. The landing page /federal-register-publications (active only)
      2. Each year's index /federal-register-publications/{YYYY}-federal-register-publications
    """
    slugs = set()
    urls_to_scan = [FDIC_LISTING] + [
        f"{FDIC_BASE}/federal-register-publications/{y}-federal-register-publications"
        for y in range(start_year, end_year + 1)
    ]
    for url in urls_to_scan:
        try:
            resp = http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        except Exception as e:
            logger.debug("skip %s: %s", url, e)
            continue
        # Match new-format comment pages (comments-rin-*, comments-omb-*, comments-N)
        for m in re.finditer(
            r'/federal-register-publications/(comments-(?:rin|omb)-[a-z0-9-]+)',
            resp.text,
            re.IGNORECASE,
        ):
            slugs.add(m.group(1).lower())

        # Also find old-format links (/resources/regulations/...) that have "Read Comments"
        # Follow their redirects to discover the new-site URLs
        import time
        for a_match in re.finditer(
            r'href="(/resources/regulations/federal-register-publications/\d{4}/[^"]+\.html)"',
            resp.text,
        ):
            old_url = FDIC_BASE + a_match.group(1)
            try:
                import requests as _req
                r = _req.head(old_url, headers={"User-Agent": "Mozilla/5.0"},
                              allow_redirects=True, timeout=15)
                # The redirect target might be a comment page
                final = r.url
                for pattern in [
                    r'/federal-register-publications/(comments-[a-z0-9-]+)',
                    r'/federal-register-publications/(fdic-federal-register-citations-[a-z0-9-]+)',
                ]:
                    m2 = re.search(pattern, final)
                    if m2:
                        slugs.add(m2.group(1).lower())
                time.sleep(0.3)
            except Exception:
                pass

    # For each slug, also add the base (undated) version
    base_slugs = set()
    for slug in slugs:
        rm = re.match(r"(comments-(?:rin|omb)-\d{4}-[a-z]{2}\d{2})", slug)
        if rm:
            base_slugs.add(rm.group(1))
    slugs.update(base_slugs)

    results = []
    for slug in sorted(slugs):
        rm = re.match(r"comments-(?:rin|omb)-(\d{4}-[a-z0-9]{2,6})", slug)
        if rm:
            results.append((rm.group(1).upper(), slug))
        else:
            results.append((slug, slug))
    logger.info("Found %d comment pages (%d with base slugs) across %d-%d",
                len(results), len(base_slugs), start_year, end_year)
    return results


def fetch_comments_for_rin(rin: str, slug: str | None = None) -> list[dict]:
    """Scrape a single RIN page.

    If `slug` is provided (e.g., "comments-rin-3064-ag20-december-19-2025"),
    uses that URL directly. Otherwise falls back to the minimal slug which
    typically redirects to the latest comment page.
    """
    if slug:
        url = f"{FDIC_BASE}/federal-register-publications/{slug}"
    else:
        url = f"{FDIC_BASE}/federal-register-publications/comments-rin-{rin.lower()}"
    logger.info("Fetching %s", url)
    try:
        resp = http_get(url, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        logger.warning("Failed: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    # FDIC comment links come in several formats:
    # New site: /federal-register-publications/{commenter-slug} or /media/{id}
    # Old site: /resources/regulations/federal-register-publications/YYYY/...-c-NNN.pdf
    # Skip navigation links (search, archive, yearly index pages).
    skip_patterns = re.compile(
        r"search|archive|publication-archive|\d{4}-federal-register|"
        r"comments-rin-|comments-omb-|comments-\d|govinfo\.gov|"
        r"fdic-federal-register-citations",
        re.IGNORECASE,
    )
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        anchor_text = a.get_text(strip=True)
        # Must be a link under /federal-register-publications/, /media/,
        # or /resources/regulations/ (old-format PDFs)
        if not ("/federal-register-publications/" in href or "/media/" in href
                or "/resources/regulations/" in href):
            continue
        # Skip navigation and self-referential links
        if skip_patterns.search(href):
            continue
        # Skip very short anchor text (icons, arrows, etc.)
        if not anchor_text or len(anchor_text) < 3:
            continue
        abs_url = href if href.startswith("http") else FDIC_BASE + href
        if abs_url in seen:
            continue
        seen.add(abs_url)
        records.append({
            "rin": rin,
            "slug": slug or "",
            "submitter": anchor_text,
            "pdf_url": abs_url,
            "filename": href.rsplit("/", 1)[-1],
        })

    logger.info("Found %d comments for %s", len(records), rin)
    return records


def normalize(rows: list[dict]) -> pd.DataFrame:
    import json
    records = []
    for i, row in enumerate(rows):
        records.append({
            "comment_id": f"fdic_{row.get('rin', 'unknown')}_{i}",
            "docket_id": row.get("rin", ""),
            "agency_id": "FDIC",
            "submitter_name": row.get("submitter", ""),
            "submitter_org": "",
            "posted_date": "",
            "comment_text": "",
            "attachment_urls": row.get("pdf_url", ""),
            "raw_metadata": json.dumps({
                "filename": row.get("filename"),
                "slug": row.get("slug"),
            }),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rins", nargs="+", help="List of RINs (e.g., 3064-AG04)")
    parser.add_argument("--list-all", action="store_true", help="Scrape all RINs from landing page")
    args = parser.parse_args()

    # Build list of (rin, slug) tuples
    if args.list_all:
        rin_slugs = list_all_rins()
    elif args.rins:
        rin_slugs = [(r.upper(), None) for r in args.rins]
    else:
        logger.error("Provide --rins or --list-all")
        return

    done = load_done_units("fdic_comments")
    logger.info("Total RIN pages: %d, already done: %d", len(rin_slugs), len(done))

    for rin, slug in rin_slugs:
        unit = slug or rin
        if unit in done:
            continue
        try:
            rows = fetch_comments_for_rin(rin, slug=slug)
        except Exception as e:
            logger.warning("Failed %s: %s", unit, e)
            continue
        if rows:
            df = normalize(rows)
            append_comments(df, "fdic_comments")
        mark_done("fdic_comments", unit)

    save_metadata("fdic_comments", {
        "rins": sorted({r for r, _ in rin_slugs}),
        "n_rin_pages": len(rin_slugs),
        "notes": (
            "FDIC comments scraped from fdic.gov/federal-register-publications/. "
            "Per-RIN resume via done_units.txt checkpoint."
        ),
    })


if __name__ == "__main__":
    main()
