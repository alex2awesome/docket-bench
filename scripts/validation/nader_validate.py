"""Compare our comment-matching pipeline output against Handan-Nader ground truth.

Inputs:
  - our matching output: _nader_fcc_rif/nader_2017_2018/public_submission_all_text__comment_comment_labels.csv.gz
  - our pair scores:     _nader_fcc_rif/nader_2017_2018/public_submission_all_text__comment_pair_scores.csv.gz
  - Nader tables:        _nader_fcc_rif/nader_tables/nader_{docs_cited,filers_cited,near_duplicates}.parquet

Outputs:
  - _nader_fcc_rif/nader_validation_metrics.json
  - _nader_fcc_rif/nader_missed_citations.csv
  - _nader_fcc_rif/nader_validation_report.md

Metrics:
  - Strict recall:    of the cited docs, what fraction does our pipeline label matched=yes?
  - Expanded recall:  of (cited ∪ near-dup-of-cited), what fraction are matched=yes?
  - Precision floor:  our total match rate (matched=yes / total staged)
  - Retrieval reach:  of cited docs, what fraction appear in pair_scores at all (top-k candidates)?

Our doc_id format in the pipeline output is:
    nader__2017_2018__17-108__{submission_id}_{hash}.pdf
Nader's cited format is:
    {submission_id}_{hash}.pdf
We strip the pipeline prefix to join.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_submission_id(doc_id: str) -> str:
    """Extract Nader submission_id from a pipeline doc_id.

    Pipeline format: {dir_name}____{docket}__{submission_id}__{claim_index}
    or comment-level: {dir_name}____{docket}__{submission_id}_{hash}.pdf
    Nader ground truth matches on submission_id.
    """
    if not isinstance(doc_id, str):
        return str(doc_id)
    parts = doc_id.split("__")
    # parts: [dir_name, "", docket, submission_id, claim_index?]
    if len(parts) >= 4:
        sub_id = parts[3]
        # Comment-level: "submission_id_hash.pdf" -> take numeric prefix
        if "_" in sub_id:
            candidate = sub_id.split("_")[0]
            if candidate.isdigit():
                return candidate
        # Claims-level: bare submission_id (numeric)
        if sub_id.isdigit():
            return sub_id
        return sub_id
    return doc_id


def _strip_pipeline_prefix(doc_id: str) -> str:
    """Backward-compatible wrapper."""
    return _extract_submission_id(doc_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("_nader_fcc_rif/nader_2017_2018"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("_nader_fcc_rif/nader_tables"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("_nader_fcc_rif"),
    )
    parser.add_argument(
        "--level",
        type=str,
        default="comment",
        help="Level: 'comment' or 'claims' (default: comment)",
    )
    parser.add_argument(
        "--claims-suffix",
        type=str,
        default="",
        help="Claims suffix, e.g. '_v2' (default: none)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline outputs ─────────────────────────────────────────────
    _cs = args.claims_suffix
    labels_path = args.pilot_dir / f"public_submission_all_text__{args.level}{_cs}_comment_labels.csv.gz"
    pairs_path = args.pilot_dir / f"public_submission_all_text__{args.level}{_cs}_pair_scores.csv.gz"
    logger.info("Loading pipeline outputs ...")
    labels = pd.read_csv(labels_path)
    pairs = pd.read_csv(pairs_path)
    logger.info("  comment_labels: %d rows  (columns: %s)", len(labels), list(labels.columns))
    logger.info("  pair_scores:    %d rows  (columns: %s)", len(pairs), list(pairs.columns))

    labels["nader_doc_id"] = labels["doc_id"].apply(_strip_pipeline_prefix)
    pairs["nader_doc_id"] = pairs["doc_id"].apply(_strip_pipeline_prefix)

    # ── Load staged pilot to get "total docs in corpus" ───────────────────
    staged_path = args.pilot_dir / "public_submission_all_text.csv.gz"
    staged = pd.read_csv(staged_path, usecols=["Document ID"])
    staged_doc_ids = set(staged["Document ID"].astype(str))
    logger.info("  staged pilot: %d docs", len(staged_doc_ids))

    # ── Load Nader ground truth ───────────────────────────────────────────
    logger.info("Loading Nader ground truth ...")
    docs_cited = pd.read_parquet(args.tables_dir / "nader_docs_cited.parquet")
    filers_cited = pd.read_parquet(args.tables_dir / "nader_filers_cited.parquet")
    near_dup = pd.read_parquet(args.tables_dir / "nader_near_duplicates.parquet")

    # Match on submission_id (numeric) since pipeline doc_ids extract to submission_id
    cited_doc_ids = set(str(r.submission_id) for r in docs_cited.itertuples(index=False))
    logger.info("  cited docs: %d citation-doc rows -> %d unique doc ids", len(docs_cited), len(cited_doc_ids))

    # Expand: any doc that is a near-dup of a cited doc is also "relevant"
    # near_duplicates.target_document_id is the canonical form; duplicate_document_id is the copy
    nd_target_to_dups = near_dup.groupby("target_document_id")["duplicate_document_id"].apply(set).to_dict()
    nd_dup_to_target = dict(zip(near_dup["duplicate_document_id"], near_dup["target_document_id"]))
    expanded_relevant = set(cited_doc_ids)
    for cited in cited_doc_ids:
        # forward: dupes of cited
        expanded_relevant.update(nd_target_to_dups.get(cited, set()))
        # reverse: if cited is itself a dup, add its target + target's other dupes
        target = nd_dup_to_target.get(cited)
        if target:
            expanded_relevant.add(target)
            expanded_relevant.update(nd_target_to_dups.get(target, set()))
    logger.info(
        "  expanded (cited ∪ near-dups): %d docs (+%d from near-dup expansion)",
        len(expanded_relevant), len(expanded_relevant) - len(cited_doc_ids),
    )

    # ── Intersect with staged pilot (some cited/near-dup docs may have failed PDF extraction) ─
    # Staged doc IDs are "{submission_id}_{hash}.pdf" — extract submission_id
    staged_sub_ids = set()
    for d in staged_doc_ids:
        d = str(d)
        if "_" in d:
            candidate = d.split("_")[0]
            if candidate.isdigit():
                staged_sub_ids.add(candidate)
        else:
            staged_sub_ids.add(d)
    cited_in_pilot = cited_doc_ids & staged_sub_ids
    expanded_in_pilot = expanded_relevant & staged_sub_ids
    logger.info(
        "  cited docs present in staged pilot:    %d / %d (%.1f%%)",
        len(cited_in_pilot), len(cited_doc_ids),
        100 * len(cited_in_pilot) / max(len(cited_doc_ids), 1),
    )
    logger.info(
        "  expanded relevant in staged pilot:     %d / %d",
        len(expanded_in_pilot), len(expanded_relevant),
    )

    # ── Compute metrics ───────────────────────────────────────────────────
    matched_yes_docs = set(labels.loc[labels["matched"] == "yes", "nader_doc_id"])
    retrieved_docs = set(pairs["nader_doc_id"])  # any doc that appeared in top-k for any response

    def _pct(a: int, b: int) -> float:
        return 100.0 * a / b if b else 0.0

    strict_retrieval_reach = len(cited_in_pilot & retrieved_docs)
    strict_recall = len(cited_in_pilot & matched_yes_docs)
    expanded_retrieval_reach = len(expanded_in_pilot & retrieved_docs)
    expanded_recall = len(expanded_in_pilot & matched_yes_docs)
    total_matched = len(matched_yes_docs)
    match_rate_floor = _pct(total_matched, len(staged_doc_ids))

    metrics = {
        "staged_docs_total": len(staged_doc_ids),
        "cited_docs_ground_truth": len(cited_doc_ids),
        "cited_docs_in_pilot": len(cited_in_pilot),
        "expanded_relevant_ground_truth": len(expanded_relevant),
        "expanded_relevant_in_pilot": len(expanded_in_pilot),
        "pipeline_matched_yes_total": total_matched,
        "pipeline_retrieved_total_in_topk": len(retrieved_docs),
        # headline metrics
        "strict_retrieval_reach": {
            "cited_in_topk": strict_retrieval_reach,
            "cited_in_pilot": len(cited_in_pilot),
            "pct": _pct(strict_retrieval_reach, len(cited_in_pilot)),
        },
        "strict_recall_yes": {
            "cited_matched_yes": strict_recall,
            "cited_in_pilot": len(cited_in_pilot),
            "pct": _pct(strict_recall, len(cited_in_pilot)),
        },
        "expanded_retrieval_reach": {
            "relevant_in_topk": expanded_retrieval_reach,
            "relevant_in_pilot": len(expanded_in_pilot),
            "pct": _pct(expanded_retrieval_reach, len(expanded_in_pilot)),
        },
        "expanded_recall_yes": {
            "relevant_matched_yes": expanded_recall,
            "relevant_in_pilot": len(expanded_in_pilot),
            "pct": _pct(expanded_recall, len(expanded_in_pilot)),
        },
        "match_rate_floor_pct": match_rate_floor,
    }

    metrics_path = args.out_dir / "nader_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", metrics_path)

    # ── Missed citations ─────────────────────────────────────────────────
    missed = cited_in_pilot - matched_yes_docs
    missed_rows = []
    # join with per-doc best score from pair_scores, if present
    best_score_per_doc = (
        pairs.groupby("nader_doc_id")["dense_score"].max().to_dict()
        if "dense_score" in pairs.columns
        else {}
    )
    for doc_id in sorted(missed):
        missed_rows.append({
            "nader_doc_id": doc_id,
            "best_dense_score_in_topk": best_score_per_doc.get(doc_id, None),
            "was_in_topk_candidates": doc_id in retrieved_docs,
        })
    missed_df = pd.DataFrame(missed_rows)
    missed_path = args.out_dir / "nader_missed_citations.csv"
    missed_df.to_csv(missed_path, index=False)
    logger.info("wrote %s (%d missed)", missed_path, len(missed_df))

    # ── Markdown report ───────────────────────────────────────────────────
    report = f"""# Nader FCC RIF Validation Report

Generated: 2026-04-05

## Setup
- Staged pilot: **{len(staged_doc_ids):,}** formal filings from Nader's BM25 search corpus (non-express), text extracted from PDFs in `attachments.tar.gz`.
- Rule text: **FCC-17-166A1** (Restoring Internet Freedom order), 539 pages, 2.46M chars.
- Response extraction: **72 unique comment-response passages** extracted from the order via `extract_responses_from_gov.py` (gpt-5-nano, chunked 4589-token windows).
- Matching pipeline: `match_comments_and_create_indexes.py --level comment` with `sentence-transformers/all-mpnet-base-v2` (384-token cap), top-k=10, gpt-5-nano for threshold labeling over 500 sampled pairs.

## Ground truth
- Cited docs (Nader `docs_cited` table): **{len(cited_doc_ids):,}** unique documents.
- Cited docs present in staged pilot: **{len(cited_in_pilot):,}** ({_pct(len(cited_in_pilot), len(cited_doc_ids)):.1f}%).
- Expanded relevant set (cited ∪ near-duplicates-of-cited): **{len(expanded_relevant):,}** docs.
- Expanded relevant in pilot: **{len(expanded_in_pilot):,}**.

## Headline metrics

| Metric | Value |
|---|---|
| **Strict recall (cited matched=yes)** | **{strict_recall} / {len(cited_in_pilot)} = {_pct(strict_recall, len(cited_in_pilot)):.1f}%** |
| Strict retrieval reach (cited in any top-k) | {strict_retrieval_reach} / {len(cited_in_pilot)} = {_pct(strict_retrieval_reach, len(cited_in_pilot)):.1f}% |
| **Expanded recall (relevant matched=yes)** | **{expanded_recall} / {len(expanded_in_pilot)} = {_pct(expanded_recall, len(expanded_in_pilot)):.1f}%** |
| Expanded retrieval reach | {expanded_retrieval_reach} / {len(expanded_in_pilot)} = {_pct(expanded_retrieval_reach, len(expanded_in_pilot)):.1f}% |
| Pipeline yes-labeled docs (total) | {total_matched:,} |
| Match rate floor (yes / staged_total) | {match_rate_floor:.1f}% |

## Interpretation

- **Strict recall** is the fraction of Nader-cited documents that our pipeline labeled as matched. This is the most direct comparison to Nader's methodology.
- **Retrieval reach** is the ceiling on recall: it's the fraction of cited documents that appeared anywhere in our top-10 retrieval candidates across all 72 responses. Recall cannot exceed reach. If reach is high but recall is low, the retrieval is finding the right candidates but the threshold is rejecting them.
- **Expanded recall** credits our pipeline for matching docs that are near-duplicates of cited docs (Nader's `near_duplicates` table). This adjusts for the fact that cited docs are a sample of a larger set of equally-relevant docs.
- **Match rate floor** is the upper bound on false-positive rate in the pilot: every matched doc that is not in the relevant set is either a false positive or a response that actually does apply to that doc but was never cited in the footnotes.

## Known caveats

1. Phase 3 used gpt-5-nano for response extraction (which hangs with gpt-5-mini on some chunks). Nader's paper used expert-curated query passages; our 72 extracted passages may not perfectly overlap with Nader's ~79.
2. Phase 5 used `sentence-transformers/all-mpnet-base-v2` as primary embedder (max 384 tokens) instead of the production `nvidia/llama-embed-nemotron-8b` (8B params, 4096-token context). The mpnet model truncates long comments at 384 tokens, which likely hurts recall on long formal filings.
3. Singleton clusters: we did NOT run our own MinHash dedup; each formal filing is its own cluster. Nader's near-duplicate table suggests ~300K near-dup edges in the broader corpus but most are within the express-comments pool that we excluded.
4. Latent pipeline bug fixed during this run: `retrieve_for_response` was reading dense scores from retriv's dict return incorrectly (all scores were 0.0). Fixed in this validation run.
5. Phase 5 ran on 8,452 pilot docs (out of 12,614 formal filings); 4,161 were dropped because PDF text extraction returned <200 chars.

See `nader_validation_metrics.json` for the full numbers and `nader_missed_citations.csv` for per-doc failure analysis.
"""
    report_path = args.out_dir / "nader_validation_report.md"
    report_path.write_text(report)
    logger.info("wrote %s", report_path)

    # ── Console summary ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("NADER VALIDATION SUMMARY")
    logger.info("  cited in pilot:        %d / %d", len(cited_in_pilot), len(cited_doc_ids))
    logger.info("  strict retrieval reach: %d (%.1f%%)", strict_retrieval_reach, _pct(strict_retrieval_reach, len(cited_in_pilot)))
    logger.info("  STRICT RECALL (yes):    %d (%.1f%%)", strict_recall, _pct(strict_recall, len(cited_in_pilot)))
    logger.info("  expanded recall (yes):  %d / %d (%.1f%%)", expanded_recall, len(expanded_in_pilot), _pct(expanded_recall, len(expanded_in_pilot)))
    logger.info("  match rate floor:       %.1f%%", match_rate_floor)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
