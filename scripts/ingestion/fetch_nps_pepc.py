"""Scrape NPS PEPC (Planning, Environment and Public Comment).

https://parkplanning.nps.gov

IMPORTANT LIMITATION: NPS PEPC does NOT expose individual per-commenter letters.
Like BLM ePlanning, comments only appear as bundled PDF documents such as
"Public Comments Correspondence Table" posted by staff.

This script enumerates projects and documents, optionally filtering for
comment-bundle PDFs whose titles contain keywords like "comment",
"correspondence", "scoping report", or "response".

URL structure (all ColdFusion .cfm):
    /publicHome.cfm                        - projects with open comment periods
    /parks.cfm                             - master list of parks
    /parkHome.cfm?parkID={N}               - all projects/docs for a park
    /documentsList.cfm?projectID={N}       - documents for a project
    /document.cfm?projectID={N}&documentID={N}  - single document page
    /showFile.cfm?sfid={N}&projectID={N}   - attached file download

Usage:
    python fetch_nps_pepc.py                     # enumerate all parks and projects
    python fetch_nps_pepc.py --park-ids 65 158   # specific parks
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, http_get, save_dockets, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PEPC_BASE = "https://parkplanning.nps.gov"
COMMENT_KEYWORDS = re.compile(r"comment|correspondence|scoping report|response", re.IGNORECASE)


def list_all_parks() -> list[dict]:
    """Scrape the master parks list."""
    logger.info("Fetching %s/parks.cfm", PEPC_BASE)
    resp = http_get(f"{PEPC_BASE}/parks.cfm")
    soup = BeautifulSoup(resp.text, "html.parser")
    parks = []
    for a in soup.find_all("a", href=re.compile(r"parkHome\.cfm\?parkID=\d+")):
        m = re.search(r"parkID=(\d+)", a["href"])
        if m:
            parks.append({
                "park_id": m.group(1),
                "park_name": a.get_text(strip=True),
            })
    # Dedupe
    seen = set()
    unique = []
    for p in parks:
        if p["park_id"] not in seen:
            seen.add(p["park_id"])
            unique.append(p)
    logger.info("Found %d unique parks", len(unique))
    return unique


def fetch_park_projects(park_id: str) -> list[dict]:
    """Fetch all projects and documents for a park."""
    url = f"{PEPC_BASE}/parkHome.cfm?parkID={park_id}"
    try:
        resp = http_get(url)
    except Exception as e:
        logger.warning("Failed park %s: %s", park_id, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    projects = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        pm = re.search(r"[Pp]roject[Ii][Dd]=(\d+)", href)
        dm = re.search(r"[Dd]ocument[Ii][Dd]=(\d+)", href)
        if pm:
            pid = pm.group(1)
            if pid not in projects:
                projects[pid] = {
                    "park_id": park_id,
                    "project_id": pid,
                    "title": "",
                    "documents": [],
                }
            if not dm and not projects[pid]["title"]:
                projects[pid]["title"] = a.get_text(strip=True)
            if dm:
                projects[pid]["documents"].append({
                    "document_id": dm.group(1),
                    "title": a.get_text(strip=True),
                    "url": PEPC_BASE + "/" + href.lstrip("/"),
                })

    return list(projects.values())


def find_comment_docs(projects: list[dict]) -> list[dict]:
    """Filter documents to those that look like comment bundles."""
    comment_docs = []
    for p in projects:
        for doc in p.get("documents", []):
            if COMMENT_KEYWORDS.search(doc.get("title", "")):
                comment_docs.append({
                    **doc,
                    "park_id": p.get("park_id"),
                    "project_id": p.get("project_id"),
                    "project_title": p.get("title"),
                })
    return comment_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--park-ids", nargs="+", help="Specific park IDs (default: all)")
    parser.add_argument("--limit-parks", type=int, default=None, help="Only scan first N parks")
    args = parser.parse_args()

    if args.park_ids:
        parks = [{"park_id": p, "park_name": ""} for p in args.park_ids]
    else:
        parks = list_all_parks()
        if args.limit_parks:
            parks = parks[:args.limit_parks]

    done = load_done_units("nps_pepc")
    logger.info("Parks to scan: %d, already done: %d", len(parks), len(done))

    all_projects = []
    for p in parks:
        if p["park_id"] in done:
            continue
        projs = fetch_park_projects(p["park_id"])
        all_projects.extend(projs)
        mark_done("nps_pepc", p["park_id"])
        time.sleep(0.5)

    logger.info("Total projects: %d", len(all_projects))

    # Find comment bundle documents
    comment_docs = find_comment_docs(all_projects)
    logger.info("Comment bundle documents: %d", len(comment_docs))

    df_projects = pd.DataFrame([{
        "park_id": p["park_id"],
        "project_id": p["project_id"],
        "title": p["title"],
        "n_documents": len(p["documents"]),
    } for p in all_projects])
    save_dockets(df_projects, "nps_pepc")

    if comment_docs:
        df_docs = pd.DataFrame(comment_docs)
        df_docs.to_csv(get_output_dir("nps_pepc") / "comment_docs.csv.gz",
                       index=False, compression="gzip")
        logger.info("Saved %d comment documents", len(df_docs))

    save_metadata("nps_pepc", {
        "source_url": PEPC_BASE,
        "n_parks": len(parks),
        "n_projects": len(all_projects),
        "n_comment_docs": len(comment_docs),
        "notes": (
            "NPS PEPC does not expose individual comments. Only bundled staff PDFs "
            "(correspondence tables, summaries) are available via /showFile.cfm."
        ),
    })


if __name__ == "__main__":
    main()
