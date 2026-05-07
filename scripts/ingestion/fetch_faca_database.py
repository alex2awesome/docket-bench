"""Download the FACA (Federal Advisory Committee Act) database.

IMPORTANT: data.gov has a FACA entry but it only links to the HTML landing page
— it does NOT host the actual files. The authoritative source is facadatabase.gov
with opaque Salesforce CMS media IDs.

Verified URLs (as of 2026-04-04):
    https://www.facadatabase.gov/FACA/cms/delivery/media/<opaque_id>

Each FY bundle is a .zip.xzip file (plain ZIP despite the extension) containing
11 CSVs: Committee History, Meeting History, Member History, Report History,
Interest Area History, Sub-Committee tables.

No API key required. No d2d.gsa.gov programmatic export available.

Usage:
    python fetch_faca_database.py
    python fetch_faca_database.py --discover   # scrape FACADatasets page for new IDs
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACA_BASE = "https://www.facadatabase.gov/FACA/cms/delivery/media"

# Known bundle media IDs (verified 2026-04-04 via Internet Archive CDX)
KNOWN_BUNDLES = {
    "FACAData2018.zip.xzip": "MCKMDUNA5OURCBDK3ISET4MLCYHI",
    "FACAData2019.zip.xzip": "MCISIL2CTBSNATDJ74UGFH3QD7AI",
    "FACAData2020.zip.xzip": "MCG2YVBGAGY5G7PC2DH7ASLGQKHM",
    "FACAData2021.zip.xzip": "MCUR4RD5GDXVDL3AKQSPHK2VDSRU",
    "FACAData2022.zip.xzip": "MC5IDRUU23LNGIHGFPDM5LE6BSXA",
    "FACAData2023.zip.xzip": "MCGTZ5Q2MREJHIPEGVSF6NDWBACA",
}

# Standalone member lists (redundant with zips but included for completeness)
KNOWN_MEMBER_CSVS = {
    "FACAMembersList2019.csv": "MCLY6DCUFOVRDYFHHZJKTVCHCMYI",
    "FACAMembersList2020.csv": "MCFQZ7ETAIQ5D43LFYLKKKVBNOD4",
    "FACAMemberList2021.csv": "MCTABPL2EZJVBYPJZD4IZEZXIYTM",
    "FACAMemberList2022.csv": "MCTCDP7JKHPJBPBK5RT3TCUKOX6Q",
    "FACAMemberList2023.csv": "MC6A6F6MH2QJAM7F3N2QZTTM6MXU",
}


def download_bundle(filename: str, media_id: str, raw_dir: Path) -> Path:
    """Download a single bundle file."""
    url = f"{FACA_BASE}/{media_id}"
    out_path = raw_dir / filename
    if out_path.exists():
        logger.info("Already exists: %s", filename)
        return out_path
    logger.info("Downloading %s (id=%s)", filename, media_id)
    resp = http_get(url, timeout=300)
    out_path.write_bytes(resp.content)
    logger.info("Saved %s (%.1f MB)", filename, len(resp.content) / 1e6)
    return out_path


def extract_bundle(zip_path: Path, extract_dir: Path):
    """Extract a .zip.xzip bundle (plain ZIP inside)."""
    logger.info("Extracting %s", zip_path.name)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
        files = z.namelist()
        logger.info("  Extracted %d files", len(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-member-csvs", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("faca")
    raw_dir = output_dir / "raw"
    extract_dir = output_dir / "extracted"
    raw_dir.mkdir(exist_ok=True)
    extract_dir.mkdir(exist_ok=True)

    downloaded = []

    # Download bundles
    for filename, media_id in KNOWN_BUNDLES.items():
        try:
            path = download_bundle(filename, media_id, raw_dir)
            downloaded.append(str(path))
            if not args.skip_extract:
                extract_bundle(path, extract_dir)
        except Exception as e:
            logger.error("Failed %s: %s", filename, e)

    # Member CSVs
    if not args.skip_member_csvs:
        for filename, media_id in KNOWN_MEMBER_CSVS.items():
            try:
                path = download_bundle(filename, media_id, raw_dir)
                downloaded.append(str(path))
            except Exception as e:
                logger.error("Failed %s: %s", filename, e)

    save_metadata("faca", {
        "source_url": "https://www.facadatabase.gov/FACA/s/FACADatasets",
        "n_files": len(downloaded),
        "bundles": list(KNOWN_BUNDLES.keys()),
        "notes": (
            "FACA bundles downloaded directly from facadatabase.gov CMS. "
            "Each FY bundle contains 11 CSVs: Committee History, Meeting History, "
            "Member History, Report History, Interest Areas, Sub-Committee tables. "
            "FY2018-2023 available; FY2024+ not yet published as of 2026-04-04. "
            "Minutes and report URLs in the data point to external agency sites — "
            "secondary crawler needed to fetch actual documents."
        ),
    })

    # Summarize extracted CSVs
    if not args.skip_extract:
        csv_files = list(extract_dir.rglob("*.csv"))
        logger.info("Total extracted CSVs: %d", len(csv_files))
        for csv in sorted(csv_files)[:5]:
            try:
                df = pd.read_csv(csv, nrows=0)
                logger.info("  %s: %d cols", csv.name, len(df.columns))
            except Exception:
                pass


if __name__ == "__main__":
    main()
