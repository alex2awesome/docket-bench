"""FCC ECFS scraper with hourly date shards via camoufox.

Bypasses the 10K offset cap by using the ecfsapi.fcc.gov internal API
(no API key, no rate limit) with hourly date windows. For peak days like
2017-07-12 (Day of Action), even daily shards hit the 10K cap.

ecfsapi.fcc.gov requires browser-level TLS (direct requests library fails).
Uses camoufox browser context for API calls.

Date format: [gte]2017-07-12T00:00:00.000Z[lte]2017-07-12T01:00:00.000Z

Usage:
    python fetch_fcc_hourly.py --docket 17-108 --start 2017-04-01 --end 2017-12-15
    python fetch_fcc_hourly.py --docket 16-42 --start 2016-01-01 --end 2017-12-31
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API = "https://ecfsapi.fcc.gov/filings"
BATCH_SIZE = 500  # max per request


def fetch_window(context, docket: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch all filings in a time window, paginating up to the 10K cap.
    If we hit 10K, return what we have (caller will split the window)."""
    start_s = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_s = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    date_param = quote(f"[gte]{start_s}[lte]{end_s}")

    all_filings = []
    offset = 0
    while offset < 10000:
        url = (f"{API}?proceedings.name={docket}&limit={BATCH_SIZE}&offset={offset}"
               f"&sort=date_submission,ASC&date_submission={date_param}")
        try:
            r = context.request.get(url, timeout=60000)
            if r.status != 200:
                break
            d = r.json()
        except Exception:
            break
        batch = d.get("filing", [])
        if not batch:
            break
        all_filings.extend(batch)
        if len(batch) < BATCH_SIZE:
            break
        offset += BATCH_SIZE
    return all_filings


def scrape_docket(docket: str, start_date: str, end_date: str,
                  shard_hours: int = 6, headless: bool = True):
    """Scrape an FCC docket with adaptive hourly sharding."""
    from camoufox.sync_api import Camoufox

    output_dir = get_output_dir("fcc_hourly")
    done = load_done_units("fcc_hourly")

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    with Camoufox(headless=headless, humanize=False) as browser:
        context = browser.new_context()
        page = context.new_page()
        # Prime cookies
        page.goto("https://www.fcc.gov/ecfs", wait_until="networkidle", timeout=60000)
        time.sleep(2)

        total = 0
        seen_ids = set()
        cursor = start

        while cursor < end:
            window_end = min(cursor + timedelta(hours=shard_hours), end)
            window_key = f"{docket}_{cursor.isoformat()}"

            if window_key in done:
                cursor = window_end
                continue

            filings = fetch_window(context, docket, cursor, window_end)

            # If we hit 10K, recursively split the window down to 1-minute resolution
            if len(filings) >= 9900:
                window_minutes = (window_end - cursor).total_seconds() / 60
                if window_minutes > 1:
                    half_delta = timedelta(minutes=max(1, window_minutes / 2))
                    logger.info("  %s: %d filings in %.0f-min window — splitting",
                                cursor.isoformat()[:16], len(filings), window_minutes)

                    def _recursive_fetch(rc_start, rc_end, depth=0):
                        nonlocal total
                        rc_filings = fetch_window(context, docket, rc_start, rc_end)
                        if len(rc_filings) >= 9900:
                            rc_mins = (rc_end - rc_start).total_seconds() / 60
                            if rc_mins > 1 and depth < 15:
                                mid = rc_start + timedelta(minutes=rc_mins / 2)
                                _recursive_fetch(rc_start, mid, depth + 1)
                                _recursive_fetch(mid, rc_end, depth + 1)
                                return
                            else:
                                logger.warning("  %s: >10K in 1-min window at depth %d",
                                               rc_start.isoformat()[:19], depth)
                        new = [f for f in rc_filings if f.get("id_submission") not in seen_ids]
                        for f in new:
                            seen_ids.add(f["id_submission"])
                        if new:
                            _save_filings(new, docket)
                            total += len(new)
                        time.sleep(0.1)

                    _recursive_fetch(cursor, window_end)
                    mark_done("fcc_hourly", window_key)
                    cursor = window_end
                    continue
                else:
                    logger.warning("  %s: >10K in 1-min window", cursor.isoformat()[:19])

            # Normal case: save filings from this window
            new = [f for f in filings if f.get("id_submission") not in seen_ids]
            for f in new:
                seen_ids.add(f["id_submission"])
            if new:
                _save_filings(new, docket)
                total += len(new)

            mark_done("fcc_hourly", window_key)

            if len(filings) > 0:
                logger.info("  %s → %s: %d filings (%d new, %d total)",
                            cursor.strftime("%Y-%m-%d %H:%M"), window_end.strftime("%H:%M"),
                            len(filings), len(new), total)

            cursor = window_end
            time.sleep(0.1)

    save_metadata("fcc_hourly", {
        "docket": docket,
        "start_date": start_date,
        "end_date": end_date,
        "shard_hours": shard_hours,
        "total_filings": total,
        "method": "ecfsapi.fcc.gov via camoufox, hourly date shards (no API key, no rate limit)",
    })
    logger.info("Done. %d total filings for %s", total, docket)


def _save_filings(filings: list[dict], docket: str):
    """Normalize and append a batch of filings."""
    records = []
    for f in filings:
        filers = f.get("filers") or []
        filer_name = filers[0].get("name", "") if filers else ""
        docs = f.get("documents") or []
        records.append({
            "source": "fcc_hourly",
            "comment_id": f"fcc_h_{f.get('id_submission', '')}",
            "docket_id": docket,
            "agency_id": "FCC",
            "submitter_name": filer_name,
            "submitter_org": "",
            "posted_date": f.get("date_submission", ""),
            "comment_text": f.get("text_data", "") or "",
            "attachment_urls": ";".join(d.get("src", "") for d in docs if d.get("src")),
            "raw_metadata": json.dumps({
                "id_submission": f.get("id_submission"),
                "express_comment": f.get("express_comment"),
                "submission_type": (f.get("submissiontype") or {}).get("short", ""),
            }),
        })
    df = pd.DataFrame(records)
    append_comments(df, "fcc_hourly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket (e.g., 17-108)")
    parser.add_argument("--dockets-file", help="File with one docket per line")
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--shard-hours", type=int, default=6)
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()

    if args.dockets_file:
        dockets = [l.strip() for l in Path(args.dockets_file).read_text().splitlines() if l.strip()]
    elif args.docket:
        dockets = [args.docket]
    else:
        logger.error("Provide --docket or --dockets-file")
        return

    done = load_done_units("fcc_hourly")
    # Skip dockets we've fully completed (marked as docket-level done)
    remaining = [d for d in dockets if f"{d}_COMPLETE" not in done]
    logger.info("Total dockets: %d, remaining: %d", len(dockets), len(remaining))

    for i, docket in enumerate(remaining):
        logger.info("=== [%d/%d] Docket %s ===", i + 1, len(remaining), docket)
        try:
            scrape_docket(docket, args.start, args.end,
                          shard_hours=args.shard_hours, headless=not args.headful)
            mark_done("fcc_hourly", f"{docket}_COMPLETE")
        except Exception as e:
            logger.error("  docket %s failed: %s", docket, e)


if __name__ == "__main__":
    main()
