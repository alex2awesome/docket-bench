"""Fetch USACE public notices and permit metadata.

IMPORTANT LIMITATION: USACE does NOT expose individual public comments on
permit applications or public notices. The RRS Public Notice Module (new in
January 2025) only accepts comment submissions — it has no read API for them.
This script collects what IS publicly available: public notice metadata and
pending permit applications.

Architecture: two separate public endpoints.

1. RRS Public Notice API (https://rrs.usace.army.mil/api/publicnotice/)
   - POST getPublicNotices      → all active notices (JSON)
   - GET  getPublicNotice?id=N  → single notice
   - POST exportPublicNotices   → CSV export (base64 in fileContents)
   - POST getPublicNoticeFilterCounts → state/district distribution
   - No auth, no rate limit headers seen.

2. ORM-Public API (https://permits.ops.usace.army.mil/orm-public-api/)
   - GET permits/search?q=vtype:pending  → pending individual permits (~1,200)
   - GET permits/search?q=vtype:jds      → jurisdictional determinations
   - GET get_orm_as_of_date              → nightly data refresh timestamp
   - No per-comment data.

This script collects category 1 (active RRS notices) and category 2 (pending permits).
All data is metadata only — there is no comment text.

Usage:
    python fetch_usace_rrs.py                     # fetch everything
    python fetch_usace_rrs.py --notices-only
    python fetch_usace_rrs.py --permits-only
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_dockets, save_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RRS_API = "https://rrs.usace.army.mil/api/publicnotice"
ORM_API = "https://permits.ops.usace.army.mil/orm-public-api"
HEADERS = {
    "User-Agent": "stanford-regulations-research/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def fetch_rrs_notices() -> list[dict]:
    """Fetch all active RRS public notices."""
    logger.info("POSTing to %s/getPublicNotices", RRS_API)
    resp = requests.post(
        f"{RRS_API}/getPublicNotices",
        json={},
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    # Actual shape: {status, msg, data: {publicNotices: [...], totalFilteredCount: N}}
    notices = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            notices = data.get("publicNotices", []) or []
        elif isinstance(data, list):
            notices = data
        elif "publicNotices" in payload:
            notices = payload.get("publicNotices") or []
    elif isinstance(payload, list):
        notices = payload
    logger.info("  Got %d active public notices", len(notices))
    return notices


def fetch_rrs_filter_counts() -> dict:
    """Get distribution of notices by state/district (for metadata)."""
    resp = requests.post(
        f"{RRS_API}/getPublicNoticeFilterCounts",
        json={},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", payload) if isinstance(payload, dict) else {}


def fetch_rrs_csv() -> bytes | None:
    """Download RRS CSV export (same data as getPublicNotices in CSV form)."""
    try:
        resp = requests.post(
            f"{RRS_API}/exportPublicNotices",
            json={},
            headers=HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        b64 = data.get("fileContents") if isinstance(data, dict) else None
        if not b64:
            return None
        return base64.b64decode(b64)
    except Exception as e:
        logger.warning("CSV export failed: %s", e)
        return None


def _orm_features(payload: dict) -> list[dict]:
    """Extract features list from ORM response.

    Shape: {total: N, results: {type: 'FeatureCollection', features: [...]}}
    """
    if not isinstance(payload, dict):
        return []
    results = payload.get("results") or payload
    if isinstance(results, dict):
        return results.get("features", []) or []
    return []


def fetch_orm_pending(max_records: int = 10000) -> list[dict]:
    """Fetch pending individual permit applications from ORM-public."""
    logger.info("GET %s/permits/search?vtype=pending", ORM_API)
    resp = requests.get(
        f"{ORM_API}/permits/search",
        params={"max": max_records, "sa": "true", "q": "vtype:pending"},
        headers={"User-Agent": HEADERS["User-Agent"]},
        timeout=120,
    )
    resp.raise_for_status()
    features = _orm_features(resp.json())
    records = []
    for feat in features:
        props = dict(feat.get("properties", {}) or {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if coords and isinstance(coords, list) and len(coords) >= 2:
            props["longitude"] = coords[0]
            props["latitude"] = coords[1]
        records.append(props)
    logger.info("  Got %d pending permits", len(records))
    return records


def fetch_orm_jds(max_records: int = 10000) -> list[dict]:
    """Fetch jurisdictional determinations."""
    logger.info("GET %s/permits/search?vtype=jds", ORM_API)
    resp = requests.get(
        f"{ORM_API}/permits/search",
        params={"max": max_records, "sa": "true", "q": "vtype:jds"},
        headers={"User-Agent": HEADERS["User-Agent"]},
        timeout=120,
    )
    resp.raise_for_status()
    features = _orm_features(resp.json())
    records = [dict(f.get("properties", {}) or {}) for f in features]
    logger.info("  Got %d jurisdictional determinations", len(records))
    return records


def fetch_orm_as_of_date() -> str | None:
    try:
        resp = requests.get(
            f"{ORM_API}/get_orm_as_of_date",
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notices-only", action="store_true")
    parser.add_argument("--permits-only", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("usace_rrs")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    fetch_notices = not args.permits_only
    fetch_permits = not args.notices_only

    n_notices = 0
    n_permits = 0
    n_jds = 0
    filter_counts = None

    # --- RRS Public Notices ---
    if fetch_notices:
        try:
            notices = fetch_rrs_notices()
            n_notices = len(notices)
            (raw_dir / "public_notices.json").write_text(json.dumps(notices, indent=2))

            # Save CSV export too (authoritative snapshot)
            csv_bytes = fetch_rrs_csv()
            if csv_bytes:
                (raw_dir / "public_notices_export.csv").write_bytes(csv_bytes)

            # Normalize to dockets CSV
            if notices:
                df = pd.DataFrame(notices)
                # Rename known columns to match our common docket schema loosely
                rename_map = {
                    "publicNoticeID": "docket_id",
                    "daNumber": "da_number",
                    "projectName": "title",
                    "districtCode": "district_code",
                    "startDate": "posted_date",
                    "commentsEndDate": "comments_end_date",
                }
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                df["source"] = "usace_rrs"
                df["agency_id"] = "USACE"
                save_dockets(df, "usace_rrs")

            filter_counts = fetch_rrs_filter_counts()
            (raw_dir / "filter_counts.json").write_text(json.dumps(filter_counts, indent=2))
        except Exception as e:
            logger.error("RRS fetch failed: %s", e)

    # --- ORM-Public permits ---
    if fetch_permits:
        try:
            pending = fetch_orm_pending()
            n_permits = len(pending)
            (raw_dir / "orm_pending_permits.json").write_text(json.dumps(pending, indent=2))
            if pending:
                df_pending = pd.DataFrame(pending)
                df_pending.to_csv(output_dir / "orm_pending_permits.csv.gz",
                                  index=False, compression="gzip")
        except Exception as e:
            logger.error("ORM pending fetch failed: %s", e)

        try:
            jds = fetch_orm_jds()
            n_jds = len(jds)
            (raw_dir / "orm_jds.json").write_text(json.dumps(jds, indent=2))
            if jds:
                df_jds = pd.DataFrame(jds)
                df_jds.to_csv(output_dir / "orm_jds.csv.gz",
                              index=False, compression="gzip")
        except Exception as e:
            logger.error("ORM JDs fetch failed: %s", e)

    orm_as_of = fetch_orm_as_of_date() if fetch_permits else None

    save_metadata("usace_rrs", {
        "sources": {
            "rrs_public_notices": RRS_API,
            "orm_public": ORM_API,
        },
        "n_active_public_notices": n_notices,
        "n_pending_permits": n_permits,
        "n_jurisdictional_determinations": n_jds,
        "orm_data_as_of": orm_as_of,
        "filter_counts": filter_counts,
        "notes": (
            "USACE does NOT expose individual public comments. The RRS Public "
            "Notice Module accepts comment submissions via addPublicNoticeComment "
            "but provides no read API. ORM-public contains permit application "
            "metadata only. Per-comment data is only obtainable via FOIA to the "
            "relevant district office. USACE rulemakings (WOTUS, Nationwide "
            "Permit reissuances) are docketed on Regulations.gov in the usual way "
            "— RRS is for permit application public notices only."
        ),
    })


if __name__ == "__main__":
    main()
