"""Stage Nader's formal-filing corpus with real PDF text for pipeline matching.

Why a separate script from nader_stage.py: Nader's corpus has two disjoint parts.
  * Express comments (~24M) — short inline-text form letters submitted via ECFS
    Express. These live in `nader_submissions.comment_id` → `nader_comments.comment_text`.
    The FCC does NOT cite these in the final order (ground truth confirms: all 307
    cited submissions share a single placeholder comment_id with NULL text).
  * Formal filings (~12,614 unique documents) — PDFs submitted by organizations
    (ISPs, advocacy groups, etc.). This is Nader's "search corpus" that BM25
    actually runs over. ALL cited submissions live here.

Since the pipeline is matching commenter text against agency responses from
FCC-17-166, we only want the formal corpus. This script:

  1. Loads the formal corpus doc_id list from search_extracted/search/search_index.pickle
  2. For each doc_id, reads the PDF from _nader_fcc_rif/hf_raw/attachments/attachments/
  3. Extracts text via pymupdf
  4. Joins with Nader's documents/submissions tables for metadata
  5. Writes public_submission_all_text.csv.gz + dedup files in pipeline layout

Cluster strategy: singletons (each formal filing is its own cluster). The
near_duplicates/exact_duplicates tables *could* be merged in here, but for
a first validation pass we keep it simple — we're optimising for "every
cited doc is findable", not for dedup fidelity.

Usage:
    python data/bulk_downloads/scripts/nader_stage_formal.py \\
        --tables-dir       data/bulk_downloads/_nader_fcc_rif/nader_tables \\
        --search-index     data/bulk_downloads/_nader_fcc_rif/hf_raw/search_extracted/search/search_index.pickle \\
        --attachments-dir  data/bulk_downloads/_nader_fcc_rif/hf_raw/attachments/attachments \\
        --out-dir          data/bulk_downloads/_nader_fcc_rif/nader_fcc_pilot_2017_2018
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DOCKET_ID = "17-108"
AGENCY_ID = "FCC"


def _install_pandas_numeric_shim() -> None:
    """Restore pandas.core.indexes.numeric namespace for old-pickle compat."""
    class _Shim:
        Int64Index = pd.Index
        UInt64Index = pd.Index
        Float64Index = pd.Index
        NumericIndex = pd.Index

    sys.modules["pandas.core.indexes.numeric"] = _Shim  # type: ignore[assignment]


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF via pymupdf. Returns '' on failure."""
    try:
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning("pdf extract failed for %s: %s", path.name, e)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--search-index", required=True, type=Path)
    parser.add_argument("--attachments-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N unique doc_ids (for smoke testing).",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=200,
        help="Drop docs whose extracted text is shorter than this (default: 200).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Load Nader's formal-corpus doc_id list.
    # ────────────────────────────────────────────────────────────────────
    _install_pandas_numeric_shim()
    import pickle

    logger.info("Loading search_index.pickle ...")
    with open(args.search_index, "rb") as f:
        search_idx = pickle.load(f)
    doc_ids = list(search_idx["doc_id"].unique())
    logger.info("  %d unique formal-filing doc_ids", len(doc_ids))

    if args.limit:
        doc_ids = doc_ids[: args.limit]
        logger.info("  LIMIT: processing only first %d", len(doc_ids))

    # ────────────────────────────────────────────────────────────────────
    # Load metadata tables.
    # ────────────────────────────────────────────────────────────────────
    logger.info("Loading submissions + documents metadata ...")
    documents = pd.read_parquet(
        args.tables_dir / "nader_documents.parquet",
        columns=["submission_id", "document_id", "document_name", "file_extension"],
    )
    submissions = pd.read_parquet(
        args.tables_dir / "nader_submissions.parquet",
        columns=["submission_id", "submission_type", "date_received"],
    )
    docs_cited = pd.read_parquet(args.tables_dir / "nader_docs_cited.parquet")
    cited_doc_ids_set = set(
        f"{r.submission_id}_{r.document_id}.pdf"  # matches Nader's naming convention
        for r in docs_cited.itertuples(index=False)
    )
    logger.info("  %d cited document ids in ground truth", len(cited_doc_ids_set))

    # ────────────────────────────────────────────────────────────────────
    # Extract text from each formal-filing PDF.
    # ────────────────────────────────────────────────────────────────────
    logger.info("Extracting PDF text for %d formal filings ...", len(doc_ids))
    records = []
    missing_files = 0
    empty_text = 0
    for i, doc_id in enumerate(doc_ids):
        if i and i % 500 == 0:
            logger.info(
                "  %d / %d  (%d missing files, %d empty text so far)",
                i, len(doc_ids), missing_files, empty_text,
            )
        pdf_path = args.attachments_dir / doc_id
        if not pdf_path.exists():
            missing_files += 1
            continue
        text = _extract_pdf_text(pdf_path)
        if len(text) < args.min_text_len:
            empty_text += 1
            continue
        # Parse doc_id to recover submission_id + document_id hash
        # Format: "{submission_id}_{hash}.pdf"
        stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
        try:
            submission_id, doc_hash = stem.split("_", 1)
        except ValueError:
            submission_id, doc_hash = stem, ""
        records.append({
            "Document ID": doc_id,  # use full filename-with-.pdf as stable ID
            "submission_id": submission_id,
            "document_hash": doc_hash,
            "canonical_text": text,
        })
    logger.info(
        "  done: %d valid / %d total (%d missing, %d empty)",
        len(records), len(doc_ids), missing_files, empty_text,
    )

    if not records:
        logger.error("No records produced. Aborting.")
        return

    # ────────────────────────────────────────────────────────────────────
    # Join metadata.
    # ────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.merge(
        submissions,
        on="submission_id",
        how="left",
    )

    # Build the pipeline's expected public_submission_all_text schema.
    all_text = pd.DataFrame({
        "Document ID": df["Document ID"],
        "Agency ID": AGENCY_ID,
        "Docket ID": DOCKET_ID,
        "Document Type": "Public Submission",
        "Posted Date": df["date_received"].astype(str),
        "Title": df["submission_type"].astype(str),
        "submission_id": df["submission_id"],
        "canonical_text": df["canonical_text"],
    })

    all_text_path = args.out_dir / "public_submission_all_text.csv.gz"
    all_text.to_csv(all_text_path, index=False, compression="gzip")
    logger.info("wrote %s (%d rows)", all_text_path, len(all_text))

    # ────────────────────────────────────────────────────────────────────
    # Dedup files: every formal filing is its own singleton cluster.
    # ────────────────────────────────────────────────────────────────────
    mapper_rows = []
    clusters_for_docket = []
    for cluster_id, doc_id in enumerate(all_text["Document ID"]):
        cluster_uid = f"{DOCKET_ID}::{cluster_id}"
        mapper_rows.append({
            "agency_id": AGENCY_ID,
            "docket_id": DOCKET_ID,
            "document_id": doc_id,
            "cluster_id": cluster_id,
            "cluster_uid": cluster_uid,
        })
        clusters_for_docket.append([doc_id])

    mapper_path = args.out_dir / "public_submission_all_text__dedup_mapper.csv.gz"
    pd.DataFrame(mapper_rows).to_csv(mapper_path, index=False, compression="gzip")
    logger.info("wrote %s (%d rows)", mapper_path, len(mapper_rows))

    clusters_path = args.out_dir / "public_submission_all_text__dedup_clusters.jsonl"
    with open(clusters_path, "w") as f:
        json.dump(
            {"agency": AGENCY_ID, "docket_id": DOCKET_ID, "clusters": clusters_for_docket},
            f,
        )
        f.write("\n")
    logger.info("wrote %s", clusters_path)

    # ────────────────────────────────────────────────────────────────────
    # Ground-truth check: how many cited docs are in the staged pilot?
    # ────────────────────────────────────────────────────────────────────
    staged_doc_ids = set(all_text["Document ID"])
    cited_in_pilot = staged_doc_ids & cited_doc_ids_set
    logger.info("=" * 60)
    logger.info("STAGING SUMMARY")
    logger.info("  staged documents: %d", len(staged_doc_ids))
    logger.info("  cited docs in ground truth: %d", len(cited_doc_ids_set))
    logger.info("  cited docs in staged pilot: %d", len(cited_in_pilot))
    logger.info("  cited coverage: %.1f%%", 100 * len(cited_in_pilot) / max(len(cited_doc_ids_set), 1))
    logger.info("  avg canonical_text length: %d chars", int(all_text["canonical_text"].str.len().mean()))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
