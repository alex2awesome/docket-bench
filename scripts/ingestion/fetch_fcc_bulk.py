"""FCC bulk scraper via ecfsapi.fcc.gov (no API key, no rate limit).

For each docket: paginate via offset up to 10K. If a docket exceeds 10K,
fall back to hourly date sharding (like fetch_fcc_hourly.py).

Much faster than hourly-sharding everything since most dockets have <10K total.

Usage:
    python fetch_fcc_bulk.py --dockets-file fcc_all_dockets.txt
    python fetch_fcc_bulk.py --docket 16-142
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
BATCH = 500


def _save_batch(filings: list[dict], docket: str):
    records = []
    for f in filings:
        filers = f.get("filers") or []
        filer_name = filers[0].get("name", "") if filers else ""
        docs = f.get("documents") or []
        records.append({
            "source": "fcc_bulk",
            "comment_id": f"fcc_b_{f.get('id_submission', '')}",
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
    if records:
        df = pd.DataFrame(records)
        append_comments(df, "fcc_bulk")


def fetch_simple(context, docket: str) -> int:
    """Simple offset pagination. Returns count fetched."""
    total = 0
    offset = 0
    while offset < 10000:
        url = f"{API}?proceedings.name={docket}&limit={BATCH}&offset={offset}&sort=date_received,ASC"
        try:
            r = context.request.get(url, timeout=60000)
            if r.status != 200:
                break
            filings = r.json().get("filing", [])
        except Exception:
            break
        if not filings:
            break
        _save_batch(filings, docket)
        total += len(filings)
        if len(filings) < BATCH:
            break
        offset += BATCH
        time.sleep(0.05)

    hit_cap = offset >= 9500 and total >= 9500
    return total, hit_cap


def fetch_hourly(context, docket: str, start: str = "2010-01-01", end: str = "2026-12-31") -> int:
    """Hourly date sharding for dockets that exceed 10K."""
    total = 0
    seen = set()
    cursor = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    while cursor < end_dt:
        window_end = min(cursor + timedelta(hours=6), end_dt)
        start_s = cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_s = window_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        date_param = quote(f"[gte]{start_s}[lte]{end_s}")

        offset = 0
        window_filings = []
        while offset < 10000:
            url = f"{API}?proceedings.name={docket}&limit={BATCH}&offset={offset}&sort=date_received,ASC&date_received={date_param}"
            try:
                r = context.request.get(url, timeout=60000)
                if r.status != 200:
                    break
                batch = r.json().get("filing", [])
            except Exception:
                break
            if not batch:
                break
            window_filings.extend(batch)
            if len(batch) < BATCH:
                break
            offset += BATCH

        # Dedupe + save
        new = [f for f in window_filings if f.get("id_submission") not in seen]
        for f in new:
            seen.add(f["id_submission"])
        if new:
            _save_batch(new, docket)
            total += len(new)

        cursor = window_end
        time.sleep(0.02)

    return total


def main():
    from camoufox.sync_api import Camoufox

    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket")
    parser.add_argument("--dockets-file", help="File with docket list")
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()

    if args.dockets_file:
        dockets = [l.strip() for l in Path(args.dockets_file).read_text().splitlines() if l.strip()]
    elif args.docket:
        dockets = [args.docket]
    else:
        logger.error("Provide --docket or --dockets-file")
        return

    done = load_done_units("fcc_bulk")
    remaining = [d for d in dockets if d not in done]
    logger.info("Total: %d, done: %d, remaining: %d", len(dockets), len(done), len(remaining))

    with Camoufox(headless=not args.headful, humanize=False) as browser:
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.fcc.gov/ecfs", wait_until="networkidle", timeout=60000)
        time.sleep(2)

        total_all = 0
        hourly_dockets = []
        for i, docket in enumerate(remaining):
            count, hit_cap = fetch_simple(context, docket)
            if hit_cap:
                logger.info("[%d/%d] %s: %d filings (HIT CAP — will do hourly)", i+1, len(remaining), docket, count)
                hourly_dockets.append(docket)
            elif count > 0:
                logger.info("[%d/%d] %s: %d filings", i+1, len(remaining), docket, count)
            total_all += count
            mark_done("fcc_bulk", docket)

            if (i + 1) % 50 == 0:
                logger.info("  progress: %d/%d dockets, %d total filings", i+1, len(remaining), total_all)

        # Phase 2: hourly sharding for capped dockets
        for docket in hourly_dockets:
            logger.info("Hourly sharding for %s", docket)
            extra = fetch_hourly(context, docket)
            total_all += extra
            logger.info("  %s: +%d from hourly (total now %d)", docket, extra, total_all)

    save_metadata("fcc_bulk", {
        "n_dockets": len(dockets),
        "n_filings": total_all,
        "hourly_dockets": hourly_dockets,
        "method": "ecfsapi.fcc.gov via camoufox — simple pagination then hourly fallback",
    })
    logger.info("Done. %d total filings across %d dockets", total_all, len(dockets))


if __name__ == "__main__":
    main()
