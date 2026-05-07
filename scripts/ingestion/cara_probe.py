"""Brute-force probe CARA project IDs to discover active reading rooms.

Probes integer IDs in a range and classifies each as:
  - "active": has a reading room with letters
  - "inactive": project exists but reading room not enabled
  - "not_found": ID doesn't exist

Usage:
    python cara_probe.py --start 10000 --end 70000 --step 1
    python cara_probe.py --start 10000 --end 70000 --step 10  # coarse scan first
"""

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CARA_BASE = "https://cara.fs2c.usda.gov"
HEADERS = {"User-Agent": "stanford-regulations-research/1.0"}
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "external_sources" / "fs_cara"


def probe_project(pid: str, session: requests.Session) -> dict:
    """Probe a single CARA project ID. Returns status dict."""
    url = f"{CARA_BASE}/Public/ReadingRoom"
    try:
        resp = session.get(url, params={"Project": pid}, headers=HEADERS, timeout=30)
    except Exception as e:
        return {"id": pid, "status": "error", "error": str(e)}

    text = resp.text
    if "wasn't found" in text or "wasn't found" in text:
        return {"id": pid, "status": "not_found"}
    if "not active" in text.lower() or "reading room is not active" in text.lower():
        # Extract project name if available
        name_match = re.search(r"<h[23][^>]*>([^<]+)</h[23]>", text)
        name = name_match.group(1).strip() if name_match else ""
        return {"id": pid, "status": "inactive", "name": name}

    # Active — extract letter count and name
    total_match = re.search(r"Total Letters:\s*(\d+)", text)
    total = int(total_match.group(1)) if total_match else 0
    name_match = re.search(r"<h[23][^>]*>([^<]+)</h[23]>", text)
    name = name_match.group(1).strip() if name_match else ""

    return {"id": pid, "status": "active", "name": name, "letters": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--end", type=int, default=70000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "cara_probe_results.csv"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)

    # Resume: load already-probed IDs
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(str(row["id"]))
        logger.info("Resuming: %d IDs already probed", len(done))

    # Open in append mode
    write_header = not out_path.exists() or len(done) == 0
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=["id", "status", "name", "letters"])
    if write_header:
        writer.writeheader()

    session = requests.Session()
    active_count = 0
    total_letters = 0

    ids_to_probe = range(args.start, args.end + 1, args.step)
    total = len(ids_to_probe)

    for i, pid in enumerate(ids_to_probe):
        pid_str = str(pid)
        if pid_str in done:
            continue

        result = probe_project(pid_str, session)
        writer.writerow({
            "id": result.get("id", pid_str),
            "status": result.get("status", "error"),
            "name": result.get("name", ""),
            "letters": result.get("letters", 0),
        })
        f.flush()

        if result["status"] == "active":
            active_count += 1
            letters = result.get("letters", 0)
            total_letters += letters
            logger.info("  ACTIVE: %s — %s (%d letters)", pid_str, result.get("name", ""), letters)

        if (i + 1) % 1000 == 0:
            logger.info("Progress: %d/%d probed, %d active (%d total letters)",
                        i + 1, total, active_count, total_letters)

        time.sleep(args.throttle)

    f.close()
    logger.info("Done. %d active projects found, %d total letters", active_count, total_letters)


if __name__ == "__main__":
    main()
