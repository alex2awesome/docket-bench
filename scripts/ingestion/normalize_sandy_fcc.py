"""Normalize Handan-Nader et al.'s FCC docket 17-108 PostgreSQL dump into our schema.

The dump (8.6 GB fcc.pgsql) contains:
  - submissions: 24M+ submission metadata (id, type, date, email, city/state/zip, comment_id)
  - comments: 3.8M express comment texts
  - filers: filer names per submission
  - documents: attachment metadata
  - exact_duplicates, in_person_exparte, docs_cited, filers_cited — analysis tables

We extract submissions + comments + filers into our standard comments.csv.gz schema.
"""

import gzip
import json
import logging
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DUMP = Path("os.environ.get('DOCKET_BASE', '.')/data/bulk_downloads/external_sources/fcc_handan_nader_17_108/fcc.pgsql")
OUT = Path("os.environ.get('DOCKET_BASE', '.')/data/bulk_downloads/external_sources/fcc_handan_nader_17_108")


def parse_copy_block(f, columns: list[str]):
    """Parse a PostgreSQL COPY ... FROM stdin block. Yields dicts."""
    for line in f:
        if line.strip() == r"\.":
            return
        values = line.rstrip("\n").split("\t")
        if len(values) != len(columns):
            continue
        row = {}
        for col, val in zip(columns, values):
            row[col] = None if val == r"\N" else val
        yield row


def extract_table(dump_path: Path, table_name: str):
    """Extract rows from a specific COPY block. Returns (columns, generator)."""
    logger.info("Scanning for %s...", table_name)
    copy_re = re.compile(rf"^COPY public\.{re.escape(table_name)} \(([^)]+)\) FROM stdin")

    def _gen():
        with open(dump_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = copy_re.match(line)
                if m:
                    cols = [c.strip() for c in m.group(1).split(",")]
                    logger.info("  Found %s with columns: %s", table_name, cols)
                    for row in parse_copy_block(f, cols):
                        yield cols, row
                    return
    return _gen


def main():
    # Step 1: Load filers into a dict keyed by submission_id
    logger.info("Loading filers...")
    filers_by_sub = defaultdict(list)
    count = 0
    for cols, row in extract_table(DUMP, "filers")():
        sub_id = row.get("submission_id")
        if sub_id:
            filers_by_sub[sub_id].append(row.get("filer_name", ""))
        count += 1
        if count % 1_000_000 == 0:
            logger.info("  filers: %d loaded", count)
    logger.info("Total filers: %d across %d submissions", count, len(filers_by_sub))

    # Step 2: Load comments (plaintext) into dict keyed by comment_id
    logger.info("Loading comments (express text)...")
    comments_by_id = {}
    count = 0
    for cols, row in extract_table(DUMP, "comments")():
        cid = row.get("comment_id")
        if cid:
            comments_by_id[cid] = row.get("comment_text", "")
        count += 1
        if count % 1_000_000 == 0:
            logger.info("  comments: %d loaded", count)
    logger.info("Total comments: %d", len(comments_by_id))

    # Step 3: Stream through submissions, writing out the unified schema
    logger.info("Streaming submissions -> normalized comments.csv.gz...")
    out_path = OUT / "comments.csv.gz"
    fout = gzip.open(out_path, "wt", encoding="utf-8")
    # Write header
    fields = [
        "source", "comment_id", "docket_id", "agency_id",
        "submitter_name", "submitter_org", "posted_date",
        "comment_text", "attachment_urls", "raw_metadata",
    ]
    fout.write(",".join(fields) + "\n")

    import csv
    writer = csv.DictWriter(fout, fieldnames=fields, quoting=csv.QUOTE_ALL)

    total = 0
    written = 0
    for cols, row in extract_table(DUMP, "submissions")():
        total += 1
        sid = row.get("submission_id")
        cid = row.get("comment_id")
        filers = filers_by_sub.get(sid, [])
        text = comments_by_id.get(cid, "") if cid else ""

        raw_meta = {
            "submission_type": row.get("submission_type"),
            "express_comment": row.get("express_comment"),
            "email": row.get("contact_email"),
            "city": row.get("city"),
            "state": row.get("state"),
            "zip": row.get("zip_code"),
            "source_dataset": "slnader/fcc-comments",
        }
        writer.writerow({
            "source": "fcc_handan_nader_17_108",
            "comment_id": f"fcc_handan_nader_{sid}",
            "docket_id": "17-108",
            "agency_id": "FCC",
            "submitter_name": "; ".join(filers) if filers else "",
            "submitter_org": "",
            "posted_date": row.get("date_received", ""),
            "comment_text": text or "",
            "attachment_urls": "",  # attachments reference by submission_id
            "raw_metadata": json.dumps(raw_meta, ensure_ascii=False),
        })
        written += 1
        if total % 500_000 == 0:
            logger.info("  submissions: %d processed, %d written", total, written)
    fout.close()
    logger.info("Done. %d submissions written to %s", written, out_path)


if __name__ == "__main__":
    main()
