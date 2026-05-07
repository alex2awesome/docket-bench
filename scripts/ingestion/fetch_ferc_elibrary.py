"""Download FERC rulemaking comments from the eLibrary JSON API.

FERC (Federal Energy Regulatory Commission) has its own filing system
separate from regulations.gov. All public comments on rulemakings are
filed through eLibrary.

API base: https://elibrary.ferc.gov/eLibrarywebapi/api/
No authentication required. POST JSON endpoints.

Key endpoints:
  - Search/AdvancedSearch — search filings by docket, date, category
  - Docket/GetSingleDocketSheet — list filings within a docket

Docket numbering: RM{YY}-{N} for rulemakings (e.g., RM22-14).

Usage:
    python fetch_ferc_elibrary.py                          # all RM dockets 2016-2026
    python fetch_ferc_elibrary.py --docket RM22-14         # single docket
    python fetch_ferc_elibrary.py --start-year 2020 --end-year 2026
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

API_BASE = "https://elibrary.ferc.gov/eLibrarywebapi/api"
HEADERS = {
    "User-Agent": "stanford-regulations-research/1.0",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
THROTTLE = 1.0


def search_rm_dockets(start_year: int = 2016, end_year: int = 2026) -> list[str]:
    """Enumerate RM (rulemaking) dockets by probing RM{YY}-{N}.

    The eLibrary search API doesn't support wildcard docket searches,
    so we probe each possible docket ID and keep those with >0 hits.
    """
    logger.info("Probing RM dockets %d-%d", start_year, end_year)
    dockets = []

    for year in range(start_year, end_year + 1):
        yy = year % 100
        consecutive_misses = 0
        for n in range(1, 60):
            docket = f"RM{yy}-{n}"
            payload = {
                "searchText": "*",
                "searchFullText": False,
                "searchDescription": True,
                "docketSearches": [{"docketNumber": docket, "subDocketNumbers": []}],
                "categories": [],
                "libraries": [],
                "classTypes": [],
                "availability": None,
                "affiliations": [],
                "eFiling": False,
                "allDates": True,
                "dateSearches": [],
                "resultsPerPage": 1,
                "curPage": 0,
                "sortBy": "",
                "groupBy": "NONE",
                "idolResultID": "",
            }
            try:
                resp = requests.post(f"{API_BASE}/Search/AdvancedSearch",
                                     json=payload, headers=HEADERS, timeout=30)
                data = resp.json()
                hits = data.get("totalHits", 0)
            except Exception:
                hits = 0

            if hits > 0:
                dockets.append(docket)
                consecutive_misses = 0
                logger.info("  %s: %d filings", docket, hits)
            else:
                consecutive_misses += 1
                if consecutive_misses >= 15:
                    break
            time.sleep(0.3)

    logger.info("Found %d RM dockets", len(dockets))
    return dockets


def fetch_docket_comments(docket: str) -> list[dict]:
    """Fetch all comment/submittal filings for a single RM docket via AdvancedSearch."""
    all_filings = []
    page = 0
    while True:
        payload = {
            "searchText": "*",
            "searchFullText": False,
            "searchDescription": True,
            "docketSearches": [{"docketNumber": docket, "subDocketNumbers": []}],
            "categories": ["Submittal"],
            "libraries": [],
            "classTypes": [],
            "availability": None,
            "affiliations": [],
            "eFiling": False,
            "allDates": True,
            "dateSearches": [],
            "resultsPerPage": 200,
            "curPage": page,
            "sortBy": "",
            "groupBy": "NONE",
            "idolResultID": "",
        }
        try:
            resp = requests.post(f"{API_BASE}/Search/AdvancedSearch",
                                 json=payload, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Docket %s page %d failed: %s", docket, page, e)
            break

        hits = data.get("searchHits", [])
        if not hits:
            break

        for item in hits:
            accession = item.get("acesssionNumber", "")  # note: API typo "acesssion"
            affiliations = item.get("affiliations", [])
            org_name = ""
            if affiliations and isinstance(affiliations, list):
                first = affiliations[0]
                if isinstance(first, dict):
                    org_name = first.get("organization", first.get("name", ""))
                elif isinstance(first, str):
                    org_name = first

            all_filings.append({
                "comment_id": f"ferc_{accession}",
                "docket_id": docket,
                "agency_id": "FERC",
                "submitter_name": item.get("description", ""),
                "submitter_org": str(org_name),
                "posted_date": item.get("filedDate", ""),
                "comment_text": item.get("summary", ""),
                "attachment_urls": f"https://elibrary.ferc.gov/eLibrary/filelist?accession_number={accession}",
                "raw_metadata": json.dumps({
                    "accession_no": accession,
                    "category": item.get("category", ""),
                    "class_types": item.get("classTypes", []),
                    "docket_numbers": item.get("docketNumbers", []),
                }),
            })

        if len(hits) < 200:
            break
        page += 1
        time.sleep(THROTTLE)

    return all_filings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docket", help="Single RM docket (e.g., RM22-14)")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()

    output_dir = get_output_dir("ferc_elibrary")

    if args.docket:
        dockets = [args.docket]
    else:
        dockets = search_rm_dockets(args.start_year, args.end_year)
        (output_dir / "enumerated_dockets.txt").write_text("\n".join(dockets))

    done = load_done_units("ferc_elibrary")
    logger.info("Dockets: %d total, %d done, %d to process",
                len(dockets), len(done), len(dockets) - len(done))

    for docket in dockets:
        if docket in done:
            continue
        try:
            filings = fetch_docket_comments(docket)
            logger.info("  %s: %d comment filings", docket, len(filings))
        except Exception as e:
            logger.error("  %s failed: %s", docket, e)
            continue

        if filings:
            df = pd.DataFrame(filings)
            append_comments(df, "ferc_elibrary")
        mark_done("ferc_elibrary", docket)
        time.sleep(THROTTLE)

    save_metadata("ferc_elibrary", {
        "source_url": "https://elibrary.ferc.gov/",
        "n_dockets": len(dockets),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "notes": (
            "FERC eLibrary rulemaking comments. RM-prefixed dockets searched via "
            "the JSON REST API. Each filing includes accession number for PDF download. "
            "Per-docket resume via done_units.txt."
        ),
    })


if __name__ == "__main__":
    main()
