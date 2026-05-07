#!/usr/bin/env python3
"""Organize downloaded Regulations.gov CSV bundles into agency/year folders."""
from __future__ import annotations

import csv
import gzip
import logging
import re
import shutil
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DOWNLOAD_DIR = Path.home() / "Downloads"
CSV_PREFIXES = ("mm", "mn")
AGENCY_LIST_PATH = SCRIPT_DIR / "agency_list.csv"
REQUIRED_COLUMNS = ("Agency ID", "Posted Date", "Document Type")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_agency_id_map() -> dict[str, str]:
    """Build a mapping from regulations.gov Agency ID -> our folder name.

    Reads the ``agency_id`` column in agency_list.csv.  Only entries where
    ``agency_id`` differs from ``agency name`` produce a mapping entry.
    """
    mapping: dict[str, str] = {}
    if not AGENCY_LIST_PATH.exists():
        return mapping
    with AGENCY_LIST_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            agency_id = (row.get("agency_id") or "").strip()
            folder = (row.get("agency name") or "").strip().lower()
            if agency_id and folder:
                mapping[agency_id.lower()] = folder
    return mapping


def sanitize_doc_type(doc_type: str) -> str:
    clean = doc_type.strip().lower().replace("&", "and")
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    return clean or "unknown"


def parse_year(date_value: str) -> Optional[int]:
    if not date_value:
        return None
    match = re.search(r"\b(\d{4})\b", date_value)
    if not match:
        return None
    return int(match.group(1))


def load_first_row(csv_path: Path) -> Optional[dict]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = [col for col in REQUIRED_COLUMNS if col not in header]
            if missing:
                logging.warning("%s missing columns: %s", csv_path.name, ", ".join(missing))
                return None
            return next(reader, None)
    except FileNotFoundError:
        logging.error("File disappeared before processing: %s", csv_path)
    except Exception as exc:
        logging.error("Failed reading %s: %s", csv_path, exc)
    return None


def build_target_dir(agency_id: str, year: int, agency_id_map: dict[str, str]) -> Path:
    raw_slug = agency_id.strip().lower()
    agency_slug = agency_id_map.get(raw_slug, raw_slug)
    window_label = f"{agency_slug}_{year}_{year + 1}"
    return BASE_DIR / agency_slug / window_label


def ensure_unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    counter = 1
    while candidate.exists():
        stem, suffix = Path(filename).stem, Path(filename).suffix
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def process_csv(csv_path: Path, agency_id_map: dict[str, str]) -> None:
    row = load_first_row(csv_path)
    if not row:
        return
    agency_id = (row.get("Agency ID") or "").strip()
    doc_type = (row.get("Document Type") or "").strip()
    posted_date = (row.get("Posted Date") or "").strip()
    if not (agency_id and doc_type and posted_date):
        logging.warning("Skipping %s due to empty metadata fields.", csv_path.name)
        return
    year = parse_year(posted_date)
    if year is None:
        logging.warning("Could not parse year from '%s' in %s.", posted_date, csv_path.name)
        return
    target_dir = build_target_dir(agency_id, year, agency_id_map)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_doc_type(doc_type)}.csv"
    destination = ensure_unique_path(target_dir, filename)
    logging.info("Moving %s -> %s", csv_path.name, destination.relative_to(BASE_DIR))
    shutil.move(str(csv_path), str(destination))
    # Compress in place for storage efficiency
    gz_destination = Path(str(destination) + ".gz")
    with open(destination, "rb") as f_in, gzip.open(gz_destination, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    destination.unlink()
    logging.info("Compressed -> %s", gz_destination.relative_to(BASE_DIR))


def delete_numbered_duplicates(dry_run: bool = False) -> int:
    """Delete files matching *_1.csv.gz, *_2.csv.gz, etc. (duplicates from ensure_unique_path)."""
    pattern = re.compile(r"^.+_\d+\.csv\.gz$")
    deleted = 0
    for csv_path in sorted(BASE_DIR.rglob("*.csv.gz")):
        if csv_path.is_relative_to(SCRIPT_DIR):
            continue
        if pattern.match(csv_path.name):
            if dry_run:
                logging.info("[dry-run] Would delete %s", csv_path.relative_to(BASE_DIR))
            else:
                logging.info("Deleting duplicate %s", csv_path.relative_to(BASE_DIR))
                csv_path.unlink()
            deleted += 1
    return deleted


def main() -> None:
    setup_logging()

    import argparse as _ap

    parser = _ap.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dupes",
        action="store_true",
        help="Delete numbered duplicate CSVs (*_1.csv, *_2.csv, …) and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what --clean-dupes would delete without actually deleting.",
    )
    args = parser.parse_args()

    if args.clean_dupes or args.dry_run:
        count = delete_numbered_duplicates(dry_run=args.dry_run)
        label = "Would delete" if args.dry_run else "Deleted"
        logging.info("%s %d numbered duplicate file(s).", label, count)
        return

    if not DOWNLOAD_DIR.exists():
        logging.error("Download directory not found: %s", DOWNLOAD_DIR)
        return
    agency_id_map = load_agency_id_map()
    csv_files = sorted(f for prefix in CSV_PREFIXES for f in DOWNLOAD_DIR.glob(f"{prefix}*.csv"))
    if not csv_files:
        logging.info("No CSV files matching %s were found in %s.", [f"{p}*.csv" for p in CSV_PREFIXES], DOWNLOAD_DIR)
        return
    for csv_path in csv_files:
        process_csv(csv_path, agency_id_map)


if __name__ == "__main__":
    main()
