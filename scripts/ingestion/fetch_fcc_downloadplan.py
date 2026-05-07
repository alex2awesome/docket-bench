"""FCC scraper using the downloadplan API — the correct approach.

The publicapi.fcc.gov `type=downloadplan` endpoint returns pre-computed
time buckets with doc counts and suggested API calls. Each bucket is sized
to stay under the 10K offset cap. We just follow each suggested URL.

This is how Handan-Nader et al. Nader likely scraped 24M comments — the downloadplan
gives you every filing without hitting any caps.

Usage:
    python fetch_fcc_downloadplan.py --dockets-file fcc_capped_dockets.txt
    python fetch_fcc_downloadplan.py --docket 14-28
    python fetch_fcc_downloadplan.py --all-dockets  # enumerate + fetch everything
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API = "https://publicapi.fcc.gov/ecfs/filings"


def get_api_key() -> str:
    for p in [Path.home() / ".fcc-key"]:
        if p.exists():
            return p.read_text().strip()
    import os
    return os.environ.get("FCC_API_KEY", "")


def get_download_plan(key: str, docket: str) -> list[dict]:
    """Get the download plan buckets for a docket."""
    for attempt in range(3):
        try:
            r = requests.get(API, params={"api_key": key, "proceedings.name": docket, "type": "downloadplan"},
                             timeout=120)
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            if r.status_code != 200:
                logger.warning("downloadplan for %s: HTTP %d", docket, r.status_code)
                return []
            text = r.text.strip()
            if not text or not text.startswith("{"):
                time.sleep(5)
                continue
            d = r.json()
            buckets = d.get("download_plan", {}).get("buckets", [])
            return buckets
        except (requests.RequestException, ValueError) as e:
            logger.warning("downloadplan for %s attempt %d: %s", docket, attempt, e)
            time.sleep(5 * (attempt + 1))
    return []


def fetch_bucket(key: str, docket: str, bucket: dict) -> list[dict]:
    """Fetch all filings in a downloadplan bucket, paginating as needed."""
    date_range = bucket["key_as_string"]
    doc_count = bucket["doc_count"]

    all_filings = []
    offset = 0
    # Use date_submission (what downloadplan uses) not date_received
    while offset < doc_count + 500:
        params = {
            "api_key": key,
            "proceedings.name": docket,
            "limit": 500,
            "offset": offset,
            "sort": "date_submission,ASC",
            "date_submission": date_range,
        }
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, timeout=60)
                if r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("  429 rate limit, sleeping %ds", wait)
                    time.sleep(wait)
                    continue
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        else:
            break

        if r.status_code != 200:
            break
        try:
            text = r.text.strip()
            if not text or not text.startswith("{"):
                break
            filings = r.json().get("filing", [])
        except (ValueError, AttributeError):
            break
        if not filings:
            break
        all_filings.extend(filings)
        if len(filings) < 500:
            break
        offset += 500
        time.sleep(0.3)

    return all_filings


def normalize_filings(filings: list[dict], docket: str) -> pd.DataFrame:
    records = []
    for f in filings:
        filers = f.get("filers") or []
        filer_name = filers[0].get("name", "") if filers else ""
        docs = f.get("documents") or []
        records.append({
            "source": "fcc_downloadplan",
            "comment_id": f"fcc_dp_{f.get('id_submission', '')}",
            "docket_id": docket,
            "agency_id": "FCC",
            "submitter_name": filer_name,
            "submitter_org": "",
            "posted_date": f.get("date_received", ""),
            "comment_text": f.get("text_data", "") or "",
            "attachment_urls": ";".join(d.get("src", "") for d in docs if d.get("src")),
            "raw_metadata": json.dumps({
                "id_submission": f.get("id_submission"),
                "express_comment": f.get("express_comment"),
                "submission_type": (f.get("submissiontype") or {}).get("short", ""),
            }),
        })
    return pd.DataFrame(records)


def scrape_docket(key: str, docket: str):
    """Scrape a docket using the downloadplan approach."""
    buckets = get_download_plan(key, docket)
    if not buckets:
        logger.info("  %s: no download plan (may not have filings)", docket)
        return 0

    total_expected = sum(b["doc_count"] for b in buckets)
    logger.info("  %s: %d buckets, %d expected docs", docket, len(buckets), total_expected)

    total = 0
    for i, bucket in enumerate(buckets):
        unit = f"{docket}_bucket_{i}"
        done = load_done_units("fcc_downloadplan")
        if unit in done:
            continue

        filings = fetch_bucket(key, docket, bucket)
        if filings:
            df = normalize_filings(filings, docket)
            append_comments(df, "fcc_downloadplan")
            total += len(filings)
            logger.info("    bucket %d/%d: %d/%d filings (total %d)",
                        i + 1, len(buckets), len(filings), bucket["doc_count"], total)

        mark_done("fcc_downloadplan", unit)

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket")
    parser.add_argument("--dockets-file", help="File with docket list")
    parser.add_argument("--all-dockets", action="store_true",
                        help="Enumerate all proceedings and fetch each")
    args = parser.parse_args()

    key = get_api_key()
    if not key:
        logger.error("No FCC API key found")
        return

    if args.docket:
        dockets = [args.docket]
    elif args.dockets_file:
        dockets = [l.strip() for l in Path(args.dockets_file).read_text().splitlines() if l.strip()]
    elif args.all_dockets:
        # Use existing enumerated list
        p = Path(__file__).resolve().parent / "fcc_all_dockets.txt"
        if p.exists():
            dockets = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        else:
            logger.error("No dockets file found")
            return
    else:
        logger.error("Provide --docket, --dockets-file, or --all-dockets")
        return

    done = load_done_units("fcc_downloadplan")
    remaining = [d for d in dockets if f"{d}_COMPLETE" not in done]
    logger.info("Dockets: %d total, %d remaining", len(dockets), len(remaining))

    grand_total = 0
    for i, docket in enumerate(remaining):
        logger.info("=== [%d/%d] %s ===", i + 1, len(remaining), docket)
        count = scrape_docket(key, docket)
        grand_total += count
        mark_done("fcc_downloadplan", f"{docket}_COMPLETE")

    save_metadata("fcc_downloadplan", {
        "source_url": API,
        "method": "publicapi.fcc.gov downloadplan (pre-computed buckets, no 10K cap)",
        "n_dockets": len(dockets),
        "n_filings": grand_total,
    })
    logger.info("Grand total: %d filings", grand_total)


if __name__ == "__main__":
    main()
