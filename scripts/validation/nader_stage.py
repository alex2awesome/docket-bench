"""Stage Handan-Nader FCC RIF data into our pipeline's expected layout.

Inputs (from _nader_fcc_rif/nader_tables/, produced by nader_parse_pgdump.py):
    nader_submissions.parquet   - 23,951,967 rows (submission_id, comment_id, ...)
    nader_comments.parquet      -  3,800,693 rows (comment_id, comment_text)
    nader_docs_cited.parquet    -      2,060 rows (cite_id, submission_id, document_id)
    nader_filers_cited.parquet  -      1,886 rows (ground-truth citations)

Outputs (written to _nader_fcc_rif/nader_fcc_pilot_2017_2018/):
    public_submission_all_text.csv.gz            - text + metadata for the pilot subset
    public_submission_all_text__dedup_mapper.csv.gz    - document_id -> cluster_uid
    public_submission_all_text__dedup_clusters.jsonl   - cluster listing per docket

The pilot subset is built as:
    (all submission_ids in docs_cited) ∪ (50K random non-cited submissions)

Clusters are formed by grouping submissions on their comment_id — submissions that
share a comment_id have identical inline text (form letters). Attachment-level
dedup via near_duplicates/exact_duplicates is not included here; it can be added
as a follow-up if needed.

Usage:
    python data/bulk_downloads/scripts/nader_stage.py \\
        --tables-dir data/bulk_downloads/_nader_fcc_rif/nader_tables \\
        --out-dir    data/bulk_downloads/_nader_fcc_rif/nader_fcc_pilot_2017_2018 \\
        --non-cited-sample 50000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DOCKET_ID = "17-108"
AGENCY_ID = "FCC"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--non-cited-sample",
        type=int,
        default=50_000,
        help="How many non-cited submissions to include in the pilot (default: 50000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17108,
        help="RNG seed for the non-cited sample (default: 17108).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Stage the FULL 24M corpus instead of the pilot subset.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Load only the columns we need to keep memory in check.
    # ────────────────────────────────────────────────────────────────────
    logger.info("Loading submissions (id, comment_id, date, type) ...")
    submissions = pd.read_parquet(
        args.tables_dir / "nader_submissions.parquet",
        columns=["submission_id", "comment_id", "date_received", "submission_type"],
    )
    logger.info("  %d submissions", len(submissions))

    logger.info("Loading comments (comment_id, comment_text) ...")
    comments = pd.read_parquet(
        args.tables_dir / "nader_comments.parquet",
        columns=["comment_id", "comment_text"],
    )
    logger.info("  %d unique comment bodies", len(comments))

    logger.info("Loading docs_cited (ground truth) ...")
    docs_cited = pd.read_parquet(args.tables_dir / "nader_docs_cited.parquet")
    cited_submission_ids = set(docs_cited["submission_id"].dropna().unique())
    logger.info(
        "  %d citation-to-document rows; %d unique cited submission_ids",
        len(docs_cited), len(cited_submission_ids),
    )

    # ────────────────────────────────────────────────────────────────────
    # Build the pilot submission set.
    # ────────────────────────────────────────────────────────────────────
    if args.full:
        pilot = submissions.copy()
        logger.info("FULL mode: using all %d submissions", len(pilot))
    else:
        cited_mask = submissions["submission_id"].isin(cited_submission_ids)
        cited = submissions[cited_mask]
        non_cited_pool = submissions[~cited_mask]
        n_sample = min(args.non_cited_sample, len(non_cited_pool))
        non_cited = non_cited_pool.sample(n=n_sample, random_state=args.seed)
        pilot = pd.concat([cited, non_cited], ignore_index=True)
        logger.info(
            "PILOT: %d cited + %d non-cited = %d total",
            len(cited), len(non_cited), len(pilot),
        )

    # Drop submissions with no comment_id (can't recover text for them).
    pilot = pilot.dropna(subset=["comment_id"])
    logger.info("  %d submissions have a comment_id", len(pilot))

    # ────────────────────────────────────────────────────────────────────
    # Join text.
    # ────────────────────────────────────────────────────────────────────
    logger.info("Joining comment text ...")
    pilot_with_text = pilot.merge(comments, on="comment_id", how="left")
    missing = pilot_with_text["comment_text"].isna().sum()
    if missing:
        logger.warning("  %d submissions have no matching comment row (dropping)", missing)
        pilot_with_text = pilot_with_text.dropna(subset=["comment_text"])
    logger.info("  %d submissions with text", len(pilot_with_text))

    # Drop very short text (matches extract_responses_from_gov.py's MIN_TEXT_LEN behaviour
    # indirectly; for matching we want at least a bit of content).
    pilot_with_text["comment_text"] = pilot_with_text["comment_text"].astype(str)
    pre = len(pilot_with_text)
    pilot_with_text = pilot_with_text[pilot_with_text["comment_text"].str.len() >= 20]
    logger.info("  %d -> %d after min-length=20 filter", pre, len(pilot_with_text))

    # ────────────────────────────────────────────────────────────────────
    # Write public_submission_all_text.csv.gz
    # ────────────────────────────────────────────────────────────────────
    logger.info("Writing public_submission_all_text.csv.gz ...")
    all_text = pd.DataFrame({
        "Document ID": pilot_with_text["submission_id"].astype(str),
        "Agency ID": AGENCY_ID,
        "Docket ID": DOCKET_ID,
        "Document Type": "Public Submission",
        "Posted Date": pilot_with_text["date_received"].astype(str),
        "Title": pilot_with_text["submission_type"].astype(str),
        "canonical_text": pilot_with_text["comment_text"],
    })
    all_text_path = args.out_dir / "public_submission_all_text.csv.gz"
    all_text.to_csv(all_text_path, index=False, compression="gzip")
    logger.info("  wrote %s (%d rows)", all_text_path, len(all_text))

    # ────────────────────────────────────────────────────────────────────
    # Build clusters by comment_id (form-letter dedup).
    # Cluster representative = document_id corresponds to submission_id.
    # ────────────────────────────────────────────────────────────────────
    logger.info("Building dedup clusters by comment_id ...")
    # Assign cluster_id per docket (we only have one docket here).
    # Group submissions by comment_id; each group is one cluster.
    grouped = pilot_with_text.groupby("comment_id")["submission_id"].apply(list)

    cluster_records = []  # for dedup_mapper
    clusters_for_docket = []  # list of [doc_id1, doc_id2, ...] for jsonl
    for cluster_id, (comment_id, sub_ids) in enumerate(grouped.items()):
        doc_ids = [str(s) for s in sub_ids]
        clusters_for_docket.append(doc_ids)
        cluster_uid = f"{DOCKET_ID}::{cluster_id}"
        for doc_id in doc_ids:
            cluster_records.append({
                "agency_id": AGENCY_ID,
                "docket_id": DOCKET_ID,
                "document_id": doc_id,
                "cluster_id": cluster_id,
                "cluster_uid": cluster_uid,
            })

    mapper_df = pd.DataFrame(cluster_records)
    mapper_path = args.out_dir / "public_submission_all_text__dedup_mapper.csv.gz"
    mapper_df.to_csv(mapper_path, index=False, compression="gzip")
    logger.info("  wrote %s (%d rows, %d clusters)", mapper_path, len(mapper_df), len(clusters_for_docket))

    clusters_jsonl_path = args.out_dir / "public_submission_all_text__dedup_clusters.jsonl"
    with open(clusters_jsonl_path, "w") as f:
        json.dump(
            {"agency": AGENCY_ID, "docket_id": DOCKET_ID, "clusters": clusters_for_docket},
            f,
        )
        f.write("\n")
    logger.info("  wrote %s", clusters_jsonl_path)

    # ────────────────────────────────────────────────────────────────────
    # Stats summary
    # ────────────────────────────────────────────────────────────────────
    cited_in_pilot = pilot_with_text["submission_id"].isin(cited_submission_ids).sum()
    cluster_sizes = [len(c) for c in clusters_for_docket]
    logger.info("=" * 60)
    logger.info("PILOT STAGING SUMMARY")
    logger.info("  submissions: %d", len(pilot_with_text))
    logger.info("  cited submissions in pilot: %d / %d", cited_in_pilot, len(cited_submission_ids))
    logger.info("  clusters: %d", len(clusters_for_docket))
    logger.info("  median cluster size: %d", int(np.median(cluster_sizes)) if cluster_sizes else 0)
    logger.info("  max cluster size: %d", max(cluster_sizes) if cluster_sizes else 0)
    logger.info("  singleton clusters: %d", sum(1 for s in cluster_sizes if s == 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
