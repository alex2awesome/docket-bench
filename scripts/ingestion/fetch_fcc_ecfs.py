"""Download FCC comments from the ECFS (Electronic Comment Filing System) public API.

FCC ECFS at https://publicapi.fcc.gov/ecfs/ provides a public JSON API for
filings. Gateway is api.data.gov, so any api.data.gov key works.

Get an API key at https://api.data.gov/signup/ (the FCC ECFS help page just
redirects here). No approval wait — the key is issued immediately.

Set the key via environment variable:
    export FCC_API_KEY="your-key-here"
    # Or put it in ~/.fcc-key

Key facts (verified 2026-04-04 via live probes):
  * Base URL: https://publicapi.fcc.gov/ecfs
  * Only two read endpoints: /filings and /proceedings
  * Response envelope: {"filing": [...], "aggregations": {...}}  (singular "filing")
  * Per-filing primary key: id_submission (store as string)
  * Filter param for docket: proceedings.name=17-108 (repeat for multiple)
  * Date range syntax: date_received=[gte]YYYY-MM-DD[lte]YYYY-MM-DD
  * HARD 10,000-record ceiling per filter combination (offset+limit capped)
    → for large dockets we shard by month/day.
  * Alternative: type=downloadplan bypasses the 10k ceiling for whole-docket pulls.
  * Rate limit: 1000 req/hour with a registered key (10/hour with DEMO_KEY).
    X-Ratelimit-Remaining header tells you how many remain in the rolling window.
  * PII (contact_email, address) is STRIPPED from filings post-2019.
    For docket 17-108 specifically, the HuggingFace slnader/fcc-comments
    dataset has the pre-strip data.
  * A filing can belong to multiple dockets (proceedings[] is a list).
  * text_data is populated ONLY for express comments (express_comment=1).
    Standard filings store content in documents[].src (PDF/DOCX URLs).

Usage:
    python fetch_fcc_ecfs.py --docket 17-108
    python fetch_fcc_ecfs.py --docket 21-450 --start-date 2021-01-01 --end-date 2022-12-31
    python fetch_fcc_ecfs.py --dockets-file dockets.txt
    python fetch_fcc_ecfs.py --docket 17-108 --downloadplan   # single-shot, no sharding
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, save_comments, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ECFS_API_BASE = "https://publicapi.fcc.gov/ecfs"
DEFAULT_LIMIT = 100
OFFSET_CEILING = 10000  # API hard cap per query
DEFAULT_THROTTLE = 0.25  # seconds between requests


def get_api_key() -> str:
    """Load FCC API key from env or file."""
    key = os.environ.get("FCC_API_KEY")
    if key:
        return key
    for path in [
        Path.home() / ".fcc-key",
        Path.home() / ".fcc-key.txt",
    ]:
        if path.exists():
            return path.read_text().strip()
    raise RuntimeError(
        "No FCC API key found. Sign up at https://api.data.gov/signup/ "
        "and set FCC_API_KEY env var or save to ~/.fcc-key"
    )


def _ecfs_get(api_key: str, params: dict, max_retries: int = 5) -> dict:
    """Single GET to /filings with rate-limit handling."""
    params = {**params, "api_key": api_key}
    url = f"{ECFS_API_BASE}/filings"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("Request error %s, retry in %ds", e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            # Rate-limited; sleep until the next window
            wait = 60 * (attempt + 1)
            logger.warning("429 rate-limit, sleeping %ds", wait)
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            wait = 2 ** attempt
            logger.warning("HTTP %d, retry in %ds", resp.status_code, wait)
            time.sleep(wait)
            continue

        resp.raise_for_status()

        remaining = resp.headers.get("X-Ratelimit-Remaining")
        if remaining is not None and int(remaining) < 5:
            logger.warning("Rate-limit nearly exhausted (remaining=%s)", remaining)

        return resp.json()

    raise RuntimeError("exhausted retries")


def _extract_filings(payload: dict) -> list[dict]:
    """Extract filings from the API response envelope.

    Current API uses {"filing": [...]} (singular); legacy was {"filings": [...]}.
    """
    if not isinstance(payload, dict):
        return []
    return payload.get("filing") or payload.get("filings") or []


def fetch_filings_window(
    api_key: str,
    docket: str,
    start_date: str,
    end_date: str,
    throttle: float = DEFAULT_THROTTLE,
) -> list[dict]:
    """Fetch all filings for a docket within a date window, paginating via offset.

    Uses sort=date_received,ASC for stable pagination. Caller is responsible
    for choosing a window small enough that results fit under the 10k ceiling.

    NOTE: The FCC API treats `[lte]YYYY-MM-DD` as "≤ midnight of that day",
    so a single-day query (`[gte]X[lte]X`) matches only timestamps at exactly
    00:00:00. We bump the upper bound by 1 day to get inclusive end-of-day.
    """
    # Bump end_date by 1 day to make the window end-inclusive
    from datetime import date as _date, timedelta as _td
    end_inclusive = (_date.fromisoformat(end_date) + _td(days=1)).isoformat()

    all_filings = []
    offset = 0
    while True:
        params = {
            "proceedings.name": docket,
            "limit": DEFAULT_LIMIT,
            "offset": offset,
            "sort": "date_received,ASC",
            "date_received": f"[gte]{start_date}[lte]{end_inclusive}",
        }
        payload = _ecfs_get(api_key, params)
        batch = _extract_filings(payload)
        if not batch:
            break
        all_filings.extend(batch)
        if len(batch) < DEFAULT_LIMIT:
            break
        offset += DEFAULT_LIMIT
        if offset >= OFFSET_CEILING:
            logger.warning(
                "  docket=%s window %s..%s hit 10k ceiling — sub-window needed",
                docket, start_date, end_date,
            )
            break
        time.sleep(throttle)
    return all_filings


def fetch_docket_with_sharding(
    api_key: str,
    docket: str,
    start_date: str = "1992-01-01",
    end_date: str | None = None,
    shard_days: int = 30,
    throttle: float = DEFAULT_THROTTLE,
) -> list[dict]:
    """Fetch a whole docket by date-sharding to stay under the 10k-per-query cap.

    If any shard itself returns >= 10k results, the shard is recursively halved
    until it fits or until a 1-day resolution is reached.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    all_filings = []
    seen_ids = set()

    def _fetch(start: date, end: date):
        start_s = start.isoformat()
        end_s = end.isoformat()
        logger.info("  shard %s..%s", start_s, end_s)
        batch = fetch_filings_window(api_key, docket, start_s, end_s, throttle=throttle)
        # Detect ceiling-hit: exactly at or above 10k → shard smaller
        if len(batch) >= OFFSET_CEILING:
            # Halve the range
            delta = (end - start).days
            if delta <= 1:
                logger.warning("    cannot split further, %d filings in 1 day", len(batch))
            else:
                mid = start + timedelta(days=delta // 2)
                _fetch(start, mid)
                _fetch(mid + timedelta(days=1), end)
                return
        # Deduplicate (overlap at shard boundaries is unlikely but possible)
        for f in batch:
            fid = f.get("id_submission") or f.get("id")
            if fid and fid not in seen_ids:
                seen_ids.add(fid)
                all_filings.append(f)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Advance in `shard_days` increments
    cursor = start
    while cursor <= end:
        shard_end = min(cursor + timedelta(days=shard_days - 1), end)
        _fetch(cursor, shard_end)
        cursor = shard_end + timedelta(days=1)

    return all_filings


def fetch_docket_downloadplan(api_key: str, docket: str) -> list[dict]:
    """Fetch whole docket via type=downloadplan (bypasses 10k ceiling).

    Streams a single large response. Use for medium-sized dockets where you
    want to avoid sharding complexity.
    """
    logger.info("Downloadplan for docket %s", docket)
    params = {
        "proceedings.name": docket,
        "type": "downloadplan",
    }
    payload = _ecfs_get(api_key, params)
    return _extract_filings(payload)


def normalize(filings: list[dict]) -> pd.DataFrame:
    """Normalize FCC filings to the common comments schema."""
    records = []
    for f in filings:
        filers = f.get("filers") or []
        filer_name = filers[0].get("name", "") if filers else ""

        proceedings = f.get("proceedings") or []
        docket_id = ";".join(p.get("name", "") for p in proceedings if p.get("name"))

        docs = f.get("documents") or []
        attachments = ";".join(d.get("src", "") for d in docs if d.get("src"))

        submission_type = f.get("submissiontype") or {}
        st_short = submission_type.get("short") if isinstance(submission_type, dict) else ""

        records.append({
            "comment_id": str(f.get("id_submission") or f.get("id") or ""),
            "docket_id": docket_id,
            "agency_id": "FCC",
            "submitter_name": filer_name,
            "submitter_org": "",  # Not reliably distinguished from filer_name in API
            "posted_date": f.get("date_received") or f.get("date_submission") or "",
            "comment_text": f.get("text_data") or "",
            "attachment_urls": attachments,
            "raw_metadata": json.dumps({
                "id_submission": f.get("id_submission"),
                "submissiontype": st_short,
                "express_comment": f.get("express_comment"),
                "date_received": f.get("date_received"),
                "date_disseminated": f.get("date_disseminated"),
                "exparte_or_late_filed": f.get("exparte_or_late_filed"),
                "proceedings": [p.get("name") for p in proceedings],
                "authors": [a.get("name") for a in (f.get("authors") or [])],
                "lawfirms": [lf.get("name") for lf in (f.get("lawfirms") or [])],
                "n_documents": len(docs),
            }),
        })
    return pd.DataFrame(records)


def enumerate_proceedings(api_key: str, start_date: str = "2016-01-01",
                          end_date: str | None = None) -> list[dict]:
    """Enumerate all FCC proceedings created in the given date range.

    Paginates via offset; the 10k cap applies so we shard by year if needed.
    """
    if end_date is None:
        end_date = date.today().isoformat()
    all_procs = []
    url = f"{ECFS_API_BASE}/proceedings"

    def _fetch_window(s: str, e: str):
        offset = 0
        while True:
            params = {
                "api_key": api_key,
                "limit": 100,
                "offset": offset,
                "sort": "date_proceeding_created,ASC",
                "date_proceeding_created": f"[gte]{s}[lte]{e}",
            }
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            batch = r.json().get("proceeding", [])
            if not batch:
                break
            all_procs.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
            if offset >= OFFSET_CEILING:
                # Split the window in half and recurse
                from datetime import date as _d, timedelta as _td
                sd = _d.fromisoformat(s)
                ed = _d.fromisoformat(e)
                if (ed - sd).days <= 1:
                    logger.warning("  can't split below 1 day, %d procs", len(all_procs))
                    break
                mid = sd + _td(days=(ed - sd).days // 2)
                # Truncate the last cap-overflow batch — we'll re-enumerate via halves
                all_procs[:] = [p for p in all_procs if not (s <= p.get("date_proceeding_created", "")[:10] <= e)]
                _fetch_window(s, mid.isoformat())
                _fetch_window((mid + _td(days=1)).isoformat(), e)
                return
            time.sleep(0.25)

    _fetch_window(start_date, end_date)
    # Dedupe on name
    seen = set()
    unique = []
    for p in all_procs:
        name = p.get("name")
        if name and name not in seen:
            seen.add(name)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="FCC docket number (e.g., 17-108)")
    parser.add_argument("--dockets-file", help="Text file with one docket per line")
    parser.add_argument("--enumerate-proceedings", action="store_true",
                        help="Auto-enumerate all proceedings in the date range, then fetch each")
    parser.add_argument("--start-date", default="2016-01-01",
                        help="Start date YYYY-MM-DD (default: 2016-01-01)")
    parser.add_argument("--end-date", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--shard-days", type=int, default=30,
                        help="Initial shard size in days for large dockets")
    parser.add_argument("--downloadplan", action="store_true",
                        help="Use type=downloadplan (bypasses 10k ceiling, single request)")
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE)
    parser.add_argument("--max-dockets", type=int, default=None,
                        help="Stop after processing N dockets (for testing)")
    args = parser.parse_args()

    try:
        api_key = get_api_key()
    except RuntimeError as e:
        logger.error(str(e))
        return

    output_dir = get_output_dir("fcc_ecfs")

    dockets: list[str] = []
    if args.enumerate_proceedings:
        logger.info("Enumerating all FCC proceedings %s..%s", args.start_date, args.end_date or "today")
        procs = enumerate_proceedings(api_key, args.start_date, args.end_date)
        dockets = [p.get("name") for p in procs if p.get("name")]
        logger.info("Found %d proceedings", len(dockets))
        (output_dir / "enumerated_dockets.txt").write_text("\n".join(dockets))
    elif args.docket:
        dockets = [args.docket]
    elif args.dockets_file:
        with open(args.dockets_file) as f:
            dockets = [line.strip() for line in f if line.strip()]

    if not dockets:
        logger.error("No dockets provided. Use --docket, --dockets-file, or --enumerate-proceedings")
        return

    if args.max_dockets:
        dockets = dockets[: args.max_dockets]

    done = load_done_units("fcc_ecfs")
    logger.info("Total dockets: %d, already done: %d, to process: %d",
                len(dockets), len(done), len(dockets) - len(done))

    for i, docket in enumerate(dockets, 1):
        if docket in done:
            continue
        logger.info("=== [%d/%d] docket %s ===", i, len(dockets), docket)
        try:
            if args.downloadplan:
                filings = fetch_docket_downloadplan(api_key, docket)
            else:
                filings = fetch_docket_with_sharding(
                    api_key, docket,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    shard_days=args.shard_days,
                    throttle=args.throttle,
                )
        except Exception as e:
            logger.error("  failed: %s", e)
            continue

        logger.info("  got %d filings", len(filings))
        if filings:
            df = normalize(filings)
            append_comments(df, "fcc_ecfs")
        mark_done("fcc_ecfs", docket)

    save_metadata("fcc_ecfs", {
        "source_url": ECFS_API_BASE,
        "n_dockets": len(dockets),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "notes": (
            "FCC ECFS filings fetched via publicapi.fcc.gov/ecfs. "
            "Per-docket resume via done_units.txt checkpoint. "
            "Enumerated via /ecfs/proceedings?date_proceeding_created=..."
        ),
    })


if __name__ == "__main__":
    main()
