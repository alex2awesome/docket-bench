"""Scrape PCAOB rulemaking comment letters.

PCAOB posts comment letters for each rulemaking docket at:
  https://pcaobus.org/about/rules-rulemaking/rulemaking-dockets/docket-{NNN}/comment-letters

Index of all dockets:
  https://pcaobus.org/about/rules-rulemaking/rulemaking-dockets

Usage:
    python fetch_pcaob.py
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://pcaobus.org"
INDEX = f"{BASE}/about/rules-rulemaking/rulemaking-dockets"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_dockets() -> list[dict]:
    """Get all docket pages."""
    r = requests.get(INDEX, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    dockets = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.match(r"/about/rules-rulemaking/rulemaking-dockets/(docket-\d+[^/]*)", href)
        if m:
            dockets.append({"slug": m.group(1), "url": BASE + href, "title": a.get_text(strip=True)[:200]})
    # Dedupe
    seen = set()
    uniq = []
    for d in dockets:
        if d["slug"] not in seen:
            seen.add(d["slug"])
            uniq.append(d)
    logger.info("Found %d dockets", len(uniq))
    return uniq


def fetch_docket_comments(docket_url: str) -> list[dict]:
    """Fetch comment letters from a docket's comment-letters subpage."""
    letters_url = docket_url.rstrip("/") + "/comment-letters"
    try:
        r = requests.get(letters_url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            # Fallback: maybe letters are on main docket page
            r = requests.get(docket_url, headers=HEADERS, timeout=30)
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    letters = []
    # PCAOB typically lists commenter name + date + PDF link
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        # Look for PDF/doc links
        if href.lower().endswith(".pdf") or "assets.pcaobus.org" in href.lower():
            full = href if href.startswith("http") else BASE + href
            # Filter out navigation links
            if any(skip in text.lower() for skip in ["print", "download all", "view all"]):
                continue
            if len(text) < 3 or len(text) > 300:
                continue
            letters.append({"pdf_url": full, "submitter": text})
    return letters


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    dockets = fetch_dockets()
    done = load_done_units("pcaob")
    total = 0
    for d in dockets:
        if d["slug"] in done:
            continue
        letters = fetch_docket_comments(d["url"])
        if letters:
            records = []
            for i, l in enumerate(letters):
                records.append({
                    "source": "pcaob",
                    "comment_id": f"pcaob_{d['slug']}_{i}",
                    "docket_id": d["slug"],
                    "agency_id": "PCAOB",
                    "submitter_name": l["submitter"],
                    "submitter_org": "",
                    "posted_date": "",
                    "comment_text": "",
                    "attachment_urls": l["pdf_url"],
                    "raw_metadata": json.dumps({"rule_title": d["title"], "rule_url": d["url"]}),
                })
            df = pd.DataFrame(records)
            append_comments(df, "pcaob")
            total += len(records)
            logger.info("  %s: %d letters", d["slug"], len(records))
        mark_done("pcaob", d["slug"])
        time.sleep(0.5)

    save_metadata("pcaob", {
        "source_url": INDEX,
        "n_dockets": len(dockets),
        "n_comments": total,
    })


if __name__ == "__main__":
    main()
