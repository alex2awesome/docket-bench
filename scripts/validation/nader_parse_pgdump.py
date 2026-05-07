"""Parse Handan-Nader's fcc.pgsql dump into per-table parquet files.

The dump is a standard pg_dump plain-text file with ``COPY public.X (cols) FROM stdin;``
blocks terminated by a ``\\.`` line. We walk the file once, extract rows for the tables
we care about, and write each to ``_nader_fcc_rif/nader_tables/nader_<table>.parquet``.

Outputs are prefixed ``nader_`` to label them unambiguously as Handan-Nader-sourced
ground-truth data (distinct from our own FCC pulls).

Usage:
    python data/bulk_downloads/scripts/nader_parse_pgdump.py \\
        --dump data/bulk_downloads/_nader_fcc_rif/hf_raw/fcc.pgsql \\
        --out  data/bulk_downloads/_nader_fcc_rif/nader_tables
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Tables we care about for validation. Everything else in the dump is skipped.
WANTED_TABLES = {
    "submissions",
    "comments",
    "documents",
    "filers",
    "filers_cited",
    "docs_cited",
    "near_duplicates",
    "exact_duplicates",
    "interest_groups",
    "in_person_exparte",
}


COPY_RE = re.compile(r"^COPY public\.(\w+)\s*\(([^)]+)\)\s+FROM\s+stdin;")


def _unescape(field: str) -> Optional[str]:
    """Reverse the Postgres COPY text-format escapes for a single field."""
    if field == "\\N":
        return None
    # Fast path: no backslashes at all.
    if "\\" not in field:
        return field
    # Escapes we need to handle: \\ \t \n \r \b \f \v
    out = []
    i = 0
    while i < len(field):
        ch = field[i]
        if ch == "\\" and i + 1 < len(field):
            nxt = field[i + 1]
            if nxt == "\\":
                out.append("\\")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "b":
                out.append("\b")
            elif nxt == "f":
                out.append("\f")
            elif nxt == "v":
                out.append("\v")
            else:
                out.append(ch)
                out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_row(line: str) -> List[Optional[str]]:
    # Strip only the trailing newline (not other whitespace — fields may start/end with spaces).
    if line.endswith("\n"):
        line = line[:-1]
    return [_unescape(f) for f in line.split("\t")]


def _iter_copy_blocks(path: Path) -> Iterator[Tuple[str, List[str], Iterator[str]]]:
    """Yield (table_name, columns, row_iterator) for each COPY block in the dump.

    The row_iterator must be fully consumed before the next block is yielded.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = COPY_RE.match(line)
            if not m:
                continue
            table = m.group(1)
            columns = [c.strip() for c in m.group(2).split(",")]

            def row_iter(fh=f):
                for row_line in fh:
                    if row_line == "\\.\n" or row_line.rstrip("\n") == "\\.":
                        return
                    yield row_line

            yield table, columns, row_iter()


def _write_table(table: str, columns: List[str], rows: List[List[Optional[str]]], out_dir: Path) -> None:
    df = pd.DataFrame(rows, columns=columns)
    out_path = out_dir / f"nader_{table}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("wrote %s  rows=%d  cols=%s", out_path.name, len(df), columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only parse these tables (default: all in WANTED_TABLES).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tables whose parquet already exists.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else WANTED_TABLES

    for table, columns, row_iter in _iter_copy_blocks(args.dump):
        if table not in wanted:
            # Must still drain the iterator to advance the file pointer past this block.
            logger.info("skipping table %s (not in --only / WANTED_TABLES)", table)
            for _ in row_iter:
                pass
            continue

        out_path = args.out / f"nader_{table}.parquet"
        if args.skip_existing and out_path.exists():
            logger.info("skipping %s (already exists)", out_path.name)
            for _ in row_iter:
                pass
            continue

        logger.info("parsing %s ...", table)
        rows: List[List[Optional[str]]] = []
        for i, line in enumerate(row_iter):
            rows.append(_parse_row(line))
            if i and i % 1_000_000 == 0:
                logger.info("  %s: %d rows parsed", table, i)
        _write_table(table, columns, rows, args.out)


if __name__ == "__main__":
    main()
