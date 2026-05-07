"""FCC scraper using publicapi.fcc.gov with hourly date_submission shards.

Strategy:
1. Get downloadplan buckets (tells us which months have data + counts)
2. For each bucket with >10K docs, shard to daily
3. For each day with >10K, shard to hourly
4. Paginate fully within each window (up to 10K offset cap)
5. For any hourly window still >10K, shard to 15-minute intervals

Uses publicapi (reliable) with API key. Rate limit: 1000 req/hr.

Usage:
    python fetch_fcc_publicapi_hourly.py --docket 14-28
    python fetch_fcc_publicapi_hourly.py --dockets-file /tmp/fcc_gap_dockets.txt
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API = "https://publicapi.fcc.gov/ecfs/filings"
BATCH = 500


def get_api_key():
    for p in [Path.home() / ".fcc-key"]:
        if p.exists():
            return p.read_text().strip()
    import os
    return os.environ.get("FCC_API_KEY", "")


def api_get(key, params, max_retries=5):
    """Rate-limit-aware GET."""
    for attempt in range(max_retries):
        try:
            r = requests.get(API, params={**params, "api_key": key}, timeout=60)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.debug("429, sleeping %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return []
            text = r.text.strip()
            if not text.startswith("{"):
                return []
            return r.json().get("filing", [])
        except Exception:
            time.sleep(5 * (attempt + 1))
    return []


def fetch_window(key, docket, start_ts, end_ts):
    """Fetch ALL filings in a time window, paginating up to 10K."""
    date_param = f"[gte]{start_ts}[lte]{end_ts}"
    all_filings = []
    offset = 0
    while offset < 10000:
        filings = api_get(key, {
            "proceedings.name": docket,
            "date_submission": date_param,
            "sort": "date_submission,ASC",
            "limit": BATCH,
            "offset": offset,
        })
        if not filings:
            break
        all_filings.extend(filings)
        if len(filings) < BATCH:
            break
        offset += BATCH
        time.sleep(0.3)
    return all_filings


def save_filings(filings, docket):
    if not filings:
        return
    records = []
    for f in filings:
        filers = f.get("filers") or []
        docs = f.get("documents") or []
        records.append({
            "source": "fcc_pubapi_hourly",
            "comment_id": f"fcc_ph_{f.get('id_submission', '')}",
            "docket_id": docket,
            "agency_id": "FCC",
            "submitter_name": filers[0].get("name", "") if filers else "",
            "submitter_org": "",
            "posted_date": f.get("date_received", ""),
            "comment_text": f.get("text_data", "") or "",
            "attachment_urls": ";".join(d.get("src", "") for d in docs if d.get("src")),
            "raw_metadata": json.dumps({
                "id_submission": f.get("id_submission"),
                "express_comment": f.get("express_comment"),
                "date_submission": f.get("date_submission"),
            }),
        })
    df = pd.DataFrame(records)
    append_comments(df, "fcc_pubapi_hourly")


def scrape_docket(key, docket):
    """Scrape a docket using downloadplan + adaptive sharding."""
    done = load_done_units("fcc_pubapi_hourly")

    # Get downloadplan
    r = requests.get(API, params={"api_key": key, "proceedings.name": docket, "type": "downloadplan"}, timeout=120)
    if r.status_code != 200:
        logger.warning("%s: downloadplan HTTP %d", docket, r.status_code)
        return 0
    try:
        buckets = r.json().get("download_plan", {}).get("buckets", [])
    except Exception:
        return 0

    total_expected = sum(b["doc_count"] for b in buckets)
    logger.info("%s: %d buckets, %d expected total", docket, len(buckets), total_expected)

    total = 0
    seen = set()

    for bi, bucket in enumerate(buckets):
        bucket_key = f"{docket}_b{bi}"
        if bucket_key in done:
            continue

        count = bucket["doc_count"]
        key_str = bucket["key_as_string"]
        # Parse dates from key_as_string: [gte]YYYY-MM-DDT...[lte]YYYY-MM-DDT...
        import re
        dates = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", key_str)
        if len(dates) < 2:
            continue
        bucket_start = datetime.fromisoformat(dates[0].replace("Z", "+00:00").replace("+00:00", ""))
        bucket_end = datetime.fromisoformat(dates[1].replace("Z", "+00:00").replace("+00:00", ""))

        if count <= 10000:
            # Small bucket: fetch directly
            filings = fetch_window(key, docket, dates[0], dates[1])
            new = [f for f in filings if f.get("id_submission") not in seen]
            for f in new: seen.add(f["id_submission"])
            save_filings(new, docket)
            total += len(new)
            logger.info("  bucket %d/%d (%s): %d filings (total %d)",
                        bi+1, len(buckets), dates[0][:10], len(new), total)
        else:
            # Large bucket: shard by day
            logger.info("  bucket %d/%d (%s): %d expected, sharding by day...",
                        bi+1, len(buckets), dates[0][:10], count)
            cursor = bucket_start
            while cursor < bucket_end:
                day_end = min(cursor + timedelta(days=1), bucket_end)
                day_start_s = cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                day_end_s = day_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                day_key = f"{docket}_d{cursor.strftime('%Y%m%d')}"

                if day_key not in done:
                    filings = fetch_window(key, docket, day_start_s, day_end_s)

                    if len(filings) >= 9900:
                        # Day is capped: shard by hour
                        logger.info("    %s: %d (capped), sharding by hour...", cursor.strftime("%Y-%m-%d"), len(filings))
                        filings = []  # discard, re-fetch by hour
                        for hour in range(24):
                            h_start = cursor + timedelta(hours=hour)
                            h_end = min(h_start + timedelta(hours=1), day_end)
                            h_filings = fetch_window(key, docket,
                                                     h_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                                                     h_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
                            filings.extend(h_filings)

                            if len(h_filings) >= 9900:
                                # Hour capped: shard by 15 min
                                logger.info("      %s %02d:00: %d (capped), sharding by 15min...",
                                            cursor.strftime("%Y-%m-%d"), hour, len(h_filings))
                                for q in range(4):
                                    q_start = h_start + timedelta(minutes=q*15)
                                    q_end = q_start + timedelta(minutes=15)
                                    q_filings = fetch_window(key, docket,
                                                             q_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                                                             q_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
                                    filings.extend(q_filings)

                    new = [f for f in filings if f.get("id_submission") not in seen]
                    for f in new: seen.add(f["id_submission"])
                    save_filings(new, docket)
                    total += len(new)
                    if new:
                        logger.info("    %s: %d new (total %d)", cursor.strftime("%Y-%m-%d"), len(new), total)
                    mark_done("fcc_pubapi_hourly", day_key)

                cursor = day_end
                time.sleep(0.1)

        mark_done("fcc_pubapi_hourly", bucket_key)

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket")
    parser.add_argument("--dockets-file", help="File with gap dockets (JSON)")
    args = parser.parse_args()

    key = get_api_key()

    if args.docket:
        dockets = [{"docket": args.docket}]
    elif args.dockets_file:
        dockets = json.loads(Path(args.dockets_file).read_text())
    else:
        logger.error("Provide --docket or --dockets-file")
        return

    done = load_done_units("fcc_pubapi_hourly")
    grand_total = 0
    for i, d in enumerate(dockets):
        docket = d["docket"] if isinstance(d, dict) else d
        if f"{docket}_COMPLETE" in done:
            continue
        logger.info("=== [%d/%d] %s ===", i+1, len(dockets), docket)
        count = scrape_docket(key, docket)
        grand_total += count
        mark_done("fcc_pubapi_hourly", f"{docket}_COMPLETE")
        logger.info("  %s: %d filings", docket, count)

    save_metadata("fcc_pubapi_hourly", {
        "method": "publicapi.fcc.gov with date_submission hourly shards + downloadplan buckets",
        "n_filings": grand_total,
    })
    logger.info("Grand total: %d", grand_total)


if __name__ == "__main__":
    main()
