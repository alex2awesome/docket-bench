"""Stage the FULL 24M Nader FCC RIF corpus (express + formal) for matching.

Merges express comments (nader_comments text) with formal filings (PDF text).
For submissions in both, prefers formal filing text. No dedup files written.
"""
from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCKET_ID = "17-108"
AGENCY_ID = "FCC"
MIN_TEXT_LEN = 20
OUTPUT_COLS = [
    "Document ID", "Agency ID", "Docket ID",
    "Document Type", "Posted Date", "Title", "canonical_text",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--formal-csv", required=True, type=Path,
                        help="Path to nader_2017_2018/public_submission_all_text.csv.gz")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=2_000_000,
                        help="Rows per chunk when processing express (default: 2M)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "public_submission_all_text.csv.gz"

    # ── 1. Load formal filings ───────────────────────────────────────────
    logger.info("Loading formal filings from %s ...", args.formal_csv)
    formal = pd.read_csv(args.formal_csv, dtype=str)
    logger.info("  %d formal filings loaded", len(formal))

    # Extract submission_id for dedup against express comments.
    if "submission_id" not in formal.columns:
        formal["submission_id"] = formal["Document ID"].apply(
            lambda x: str(x).split("_")[0] if "_" in str(x) else str(x).replace(".pdf", "")
        )
    formal_sub_ids = set(formal["submission_id"].astype(str))
    logger.info("  %d unique submission_ids in formal filings", len(formal_sub_ids))

    # ── 2. Load submissions ──────────────────────────────────────────────
    logger.info("Loading nader_submissions.parquet ...")
    submissions = pd.read_parquet(
        args.tables_dir / "nader_submissions.parquet",
        columns=["submission_id", "comment_id", "date_received", "submission_type"],
    )
    logger.info("  %d submissions", len(submissions))

    # ── 3. Load comments ─────────────────────────────────────────────────
    logger.info("Loading nader_comments.parquet ...")
    comments = pd.read_parquet(
        args.tables_dir / "nader_comments.parquet",
        columns=["comment_id", "comment_text"],
    )
    logger.info("  %d unique comment bodies", len(comments))

    # ── 4. Split submissions: formal vs express ──────────────────────────
    submissions["submission_id"] = submissions["submission_id"].astype(str)
    has_formal = submissions["submission_id"].isin(formal_sub_ids)
    express = submissions[~has_formal].copy()
    logger.info("  %d express submissions (no formal filing)", len(express))
    logger.info("  %d submissions overlap with formal filings (PDF text preferred)",
                has_formal.sum())
    del submissions

    # Drop express with no comment_id
    express = express.dropna(subset=["comment_id"])
    logger.info("  %d express submissions have a comment_id", len(express))

    # ── 5. Join comment text to express ──────────────────────────────────
    logger.info("Joining comment text to express submissions ...")
    express = express.merge(comments, on="comment_id", how="left")
    del comments

    missing = express["comment_text"].isna().sum()
    if missing:
        logger.warning("  %d express submissions have no matching comment (dropping)", missing)
        express = express.dropna(subset=["comment_text"])

    express["comment_text"] = express["comment_text"].astype(str)
    pre = len(express)
    express = express[express["comment_text"].str.len() >= MIN_TEXT_LEN]
    logger.info("  %d -> %d express after min-length=%d filter", pre, len(express), MIN_TEXT_LEN)

    # ── 6. Prepare formal rows ───────────────────────────────────────────
    logger.info("Preparing formal rows ...")
    formal_out = pd.DataFrame({
        "Document ID": formal["Document ID"].astype(str),
        "Agency ID": formal.get("Agency ID", pd.Series(AGENCY_ID, index=formal.index)),
        "Docket ID": formal.get("Docket ID", pd.Series(DOCKET_ID, index=formal.index)),
        "Document Type": formal.get("Document Type", pd.Series("Public Submission", index=formal.index)),
        "Posted Date": formal.get("Posted Date", pd.Series("", index=formal.index)),
        "Title": formal.get("Title", pd.Series("", index=formal.index)),
        "canonical_text": formal["canonical_text"],
    })
    formal_out["canonical_text"] = formal_out["canonical_text"].astype(str)
    pre = len(formal_out)
    formal_out = formal_out[formal_out["canonical_text"].str.len() >= MIN_TEXT_LEN]
    logger.info("  %d -> %d formal after min-length=%d filter", pre, len(formal_out), MIN_TEXT_LEN)
    del formal

    # ── 7. Write combined output to single gzip stream ───────────────────
    n_formal = len(formal_out)
    n_express = len(express)
    logger.info("Writing %d formal + %d express = %d total to %s ...",
                n_formal, n_express, n_formal + n_express, out_path)

    chunk_size = args.chunk_size
    n_written = 0

    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as fh:
        # Formal rows: write with header
        formal_out[OUTPUT_COLS].to_csv(fh, index=False, header=True)
        n_written += n_formal
        logger.info("  wrote %d formal rows (with header)", n_formal)
        del formal_out

        # Express rows: build pipeline-schema chunks and append (no header)
        for start in range(0, n_express, chunk_size):
            end = min(start + chunk_size, n_express)
            chunk = express.iloc[start:end]
            chunk_out = pd.DataFrame({
                "Document ID": chunk["submission_id"].astype(str).values,
                "Agency ID": AGENCY_ID,
                "Docket ID": DOCKET_ID,
                "Document Type": "Public Submission",
                "Posted Date": chunk["date_received"].astype(str).values,
                "Title": chunk["submission_type"].astype(str).values,
                "canonical_text": chunk["comment_text"].values,
            })
            chunk_out.to_csv(fh, index=False, header=False)
            n_written += len(chunk_out)
            logger.info("  wrote express chunk [%d:%d], total: %d", start, end, n_written)
            del chunk, chunk_out

    del express

    logger.info("=" * 60)
    logger.info("FULL CORPUS STAGING COMPLETE")
    logger.info("  output: %s", out_path)
    logger.info("  total rows: %d", n_written)
    logger.info("  (no dedup files written — run MinHash separately)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
