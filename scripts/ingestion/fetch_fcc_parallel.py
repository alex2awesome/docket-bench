"""Parallel FCC hourly scraper for capped dockets.

Reads docket date ranges from /tmp/fcc_capped_dockets.json and runs
multiple workers in parallel, each handling one docket at a time.

Usage:
    python fetch_fcc_parallel.py --workers 4
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from multiprocessing import Pool, current_process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT = str(Path(__file__).resolve().parent / "fetch_fcc_hourly.py")


def process_docket(args):
    """Run fetch_fcc_hourly.py for one docket with bounded date range."""
    docket, start, end = args
    # Pad dates by 7 days on each side for safety
    from datetime import datetime, timedelta
    s = (datetime.fromisoformat(start) - timedelta(days=7)).strftime("%Y-%m-%d")
    e = (datetime.fromisoformat(end) + timedelta(days=7)).strftime("%Y-%m-%d")

    worker = current_process().name
    logger.info("[%s] Starting %s (%s to %s)", worker, docket, s, e)

    result = subprocess.run(
        [sys.executable, SCRIPT, "--docket", docket, "--start", s, "--end", e, "--shard-hours", "24"],
        capture_output=True, text=True, timeout=7200,  # 2h max per docket
    )

    if result.returncode == 0:
        logger.info("[%s] Done %s", worker, docket)
    else:
        logger.error("[%s] Failed %s: %s", worker, docket, result.stderr[-500:] if result.stderr else "no stderr")

    return docket, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dockets-file", default="/tmp/fcc_capped_dockets.json")
    args = parser.parse_args()

    docket_info = json.loads(Path(args.dockets_file).read_text())

    # Skip 17-108 (Handan-Nader et al. dataset covers it) and already-completed dockets
    done = load_done_units("fcc_hourly")
    tasks = []
    for d in docket_info:
        docket = d["docket"]
        if docket == "17-108":
            logger.info("Skipping 17-108 (Handan-Nader et al. dataset covers it)")
            continue
        if f"{docket}_COMPLETE" in done:
            continue
        tasks.append((docket, d["start"], d["end"]))

    logger.info("Dockets to process: %d (workers=%d)", len(tasks), args.workers)

    with Pool(args.workers) as pool:
        results = pool.map(process_docket, tasks)

    succeeded = sum(1 for _, rc in results if rc == 0)
    logger.info("Done. %d/%d succeeded", succeeded, len(results))


if __name__ == "__main__":
    main()
