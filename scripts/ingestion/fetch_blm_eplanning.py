"""Scrape BLM ePlanning projects and comment documents.

IMPORTANT LIMITATION: BLM ePlanning does NOT expose individual per-commenter
letters. Submitted comments only appear indirectly as bundled PDF documents
attached by staff (e.g., "Public Comments Correspondence Table", "Scoping Report").

This script does two things:
1. Enumerates ALL BLM projects (67,835+ as of 2026) via the public JSON endpoint
2. Optionally fetches each project's document list to find comment-bundle PDFs

The project list endpoint:
    POST https://eplanning.blm.gov/searchresults/
    body: draw=1&start=0&length=50&download=true

This returns the entire project catalog in a single call.

Usage:
    python fetch_blm_eplanning.py                # just enumerate projects
    python fetch_blm_eplanning.py --fetch-docs   # also fetch document lists
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
from common import get_output_dir, save_metadata, save_dockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BLM_BASE = "https://eplanning.blm.gov"
SEARCH_ENDPOINT = f"{BLM_BASE}/searchresults/"


def fetch_all_projects() -> list[dict]:
    """Fetch the entire BLM project catalog via the search endpoint."""
    logger.info("POSTing to %s with download=true", SEARCH_ENDPOINT)
    data = {
        "draw": 1,
        "start": 0,
        "length": 50,
        "order_attribute": 0,
        "order_direction": "desc",
        "page": 1,
        "download": "true",
        "get_total_count": "false",
    }
    resp = requests.post(
        SEARCH_ENDPOINT,
        data=data,
        headers={"User-Agent": "regulations-demo-research/1.0"},
        timeout=300,
    )
    resp.raise_for_status()
    # BLM's response contains invalid JSON escape sequences (unescaped
    # backslashes in description fields). Clean them before parsing.
    import json
    import re
    raw = resp.content
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Replace any backslash not followed by a valid JSON escape char
        cleaned = re.sub(rb'\\(?![bfnrtu"/\\])', rb'\\\\', raw)
        payload = json.loads(cleaned)
    projects = payload.get("data", payload.get("results", []))
    logger.info("Got %d projects", len(projects))
    return projects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-docs", action="store_true",
                        help="Also fetch document lists per project (slow)")
    args = parser.parse_args()

    projects = fetch_all_projects()
    if not projects:
        logger.error("No projects returned")
        return

    # Save raw projects JSON
    output_dir = get_output_dir("blm_eplanning")
    (output_dir / "projects_raw.json").write_text(json.dumps(projects, indent=2))
    logger.info("Saved raw projects JSON")

    # Normalize to dockets schema
    df = pd.DataFrame(projects)
    save_dockets(df, "blm_eplanning")

    save_metadata("blm_eplanning", {
        "source_url": BLM_BASE,
        "n_projects": len(projects),
        "notes": (
            "BLM ePlanning exposes projects but NOT individual comments. "
            "Per-comment data requires staff-posted bundle PDFs from each project's Documents tab."
        ),
    })

    if args.fetch_docs:
        logger.warning("--fetch-docs not implemented yet; requires Power Pages auth handling")


if __name__ == "__main__":
    main()
