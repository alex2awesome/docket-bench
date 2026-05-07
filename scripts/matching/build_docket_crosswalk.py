"""Build a docket crosswalk linking orphan rule dockets to comment dockets.

For each final rule filed under a docket with no comments, attempts to find
the proposed rule/notice docket where comments were actually filed.

Linking methods (in priority order):
  1. Federal Register Number — exact match between rule and proposed rule
  2. RIN (Regulation Identifier Number) — shared RIN across dockets
  3. "Comment on Document ID" — explicit back-reference
  4. Title similarity — cosine similarity of rule titles

Outputs:
  - docket_crosswalk.csv: mapping from orphan rule dockets to comment dockets
  - docket_crosswalk_stats.json: summary statistics

Usage:
    python build_docket_crosswalk.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BULK_DIR = SCRIPTS_DIR.parent


def load_all_docs(doc_types=("rule", "proposed_rule", "notice")):
    """Load metadata from all _all_text CSV files."""
    cols = [
        "Document ID", "Docket ID", "Agency ID", "Title",
        "Federal Register Number", "Related RIN(s)",
        "Comment on Document ID", "Posted Date",
    ]
    records = []
    for dt in doc_types:
        for f in sorted(BULK_DIR.rglob(f"{dt}_all_text.csv*")):
            if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
                try:
                    df = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
                    df["doc_type"] = dt
                    df["directory"] = f.parent.name
                    df["agency_dir"] = f.parent.parent.name
                    records.append(df)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", f, e)
    if not records:
        raise RuntimeError("No documents found")
    all_docs = pd.concat(records, ignore_index=True)
    logger.info("Loaded %d documents (%s)", len(all_docs),
                ", ".join(f"{dt}={len(all_docs[all_docs['doc_type']==dt])}" for dt in doc_types))
    return all_docs


def load_comment_dockets():
    """Get the set of dockets that have public comments."""
    dockets = set()
    for f in sorted(BULK_DIR.rglob("public_submission_all_text.csv*")):
        if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
            try:
                df = pd.read_csv(f, usecols=["Docket ID"])
                dockets.update(df["Docket ID"].dropna().astype(str).unique())
            except:
                pass
    logger.info("Found %d dockets with comments", len(dockets))
    return dockets


def find_orphan_rule_dockets(all_docs, comment_dockets):
    """Find rule dockets that have no comments."""
    rule_docs = all_docs[all_docs["doc_type"] == "rule"]
    rule_dockets = set(rule_docs["Docket ID"].dropna().astype(str).unique())
    orphans = rule_dockets - comment_dockets
    logger.info("Rule dockets: %d total, %d orphans (no comments)", len(rule_dockets), len(orphans))
    return orphans


def link_by_fr_number(all_docs, orphan_dockets, comment_dockets):
    """Link orphan rules to comment dockets via shared Federal Register Number."""
    links = []

    # Build FR Number -> dockets mapping
    frn_to_dockets = defaultdict(list)
    for _, row in all_docs.iterrows():
        frn = str(row.get("Federal Register Number", ""))
        dk = str(row.get("Docket ID", ""))
        if frn and frn != "nan" and dk:
            frn_to_dockets[frn].append({
                "docket": dk,
                "doc_type": row.get("doc_type", ""),
                "doc_id": str(row.get("Document ID", "")),
            })

    for frn, entries in frn_to_dockets.items():
        orphan_entries = [e for e in entries if e["docket"] in orphan_dockets]
        comment_entries = [e for e in entries if e["docket"] in comment_dockets]

        if orphan_entries and comment_entries:
            for oe in orphan_entries:
                for ce in comment_entries:
                    links.append({
                        "orphan_docket": oe["docket"],
                        "comment_docket": ce["docket"],
                        "method": "fr_number",
                        "evidence": frn,
                        "orphan_doc_id": oe["doc_id"],
                        "comment_doc_type": ce["doc_type"],
                    })

    logger.info("FR Number linking: %d links found", len(links))
    return links


def link_by_rin(all_docs, orphan_dockets, comment_dockets):
    """Link orphan rules to comment dockets via shared RIN."""
    links = []

    # Build RIN -> dockets mapping
    rin_to_dockets = defaultdict(set)
    for _, row in all_docs.iterrows():
        rins = str(row.get("Related RIN(s)", ""))
        dk = str(row.get("Docket ID", ""))
        if rins and rins != "nan" and dk:
            for rin in re.split(r"[,;\s]+", rins):
                rin = rin.strip()
                if rin:
                    rin_to_dockets[rin].add(dk)

    for rin, dockets in rin_to_dockets.items():
        orphan_in = dockets & orphan_dockets
        comment_in = dockets & comment_dockets
        if orphan_in and comment_in:
            for od in orphan_in:
                for cd in comment_in:
                    links.append({
                        "orphan_docket": od,
                        "comment_docket": cd,
                        "method": "rin",
                        "evidence": rin,
                        "orphan_doc_id": "",
                        "comment_doc_type": "",
                    })

    logger.info("RIN linking: %d links found", len(links))
    return links


def link_by_comment_on_doc(all_docs, orphan_dockets, comment_dockets):
    """Link via 'Comment on Document ID' field."""
    links = []

    # Build Document ID -> Docket ID mapping
    doc_to_docket = {}
    for _, row in all_docs.iterrows():
        doc_id = str(row.get("Document ID", ""))
        dk = str(row.get("Docket ID", ""))
        if doc_id and doc_id != "nan":
            doc_to_docket[doc_id] = dk

    # Check orphan rules for "Comment on Document ID"
    orphan_rules = all_docs[
        (all_docs["doc_type"] == "rule") &
        (all_docs["Docket ID"].astype(str).isin(orphan_dockets))
    ]

    for _, row in orphan_rules.iterrows():
        comment_on = str(row.get("Comment on Document ID", ""))
        if comment_on and comment_on != "nan":
            target_docket = doc_to_docket.get(comment_on, "")
            if target_docket in comment_dockets:
                links.append({
                    "orphan_docket": str(row["Docket ID"]),
                    "comment_docket": target_docket,
                    "method": "comment_on_doc",
                    "evidence": comment_on,
                    "orphan_doc_id": str(row["Document ID"]),
                    "comment_doc_type": "",
                })

    logger.info("Comment-on-Document linking: %d links found", len(links))
    return links


def link_by_title_similarity(all_docs, orphan_dockets, comment_dockets, threshold=0.6):
    """Link orphan rules to proposed rules in comment dockets by title similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    links = []

    # Get orphan rule titles
    orphan_rules = all_docs[
        (all_docs["doc_type"] == "rule") &
        (all_docs["Docket ID"].astype(str).isin(orphan_dockets)) &
        (all_docs["Title"].notna()) &
        (all_docs["Title"].str.len() > 10)
    ].drop_duplicates(subset=["Docket ID"]).copy()

    # Get proposed rule / notice titles from comment dockets
    comment_docs = all_docs[
        (all_docs["doc_type"].isin(["proposed_rule", "notice"])) &
        (all_docs["Docket ID"].astype(str).isin(comment_dockets)) &
        (all_docs["Title"].notna()) &
        (all_docs["Title"].str.len() > 10)
    ].drop_duplicates(subset=["Docket ID"]).copy()

    if orphan_rules.empty or comment_docs.empty:
        logger.info("Title similarity: no candidates")
        return links

    logger.info("Title similarity: comparing %d orphan rules against %d comment docs",
                len(orphan_rules), len(comment_docs))

    # Same agency filter — only compare within same agency
    for agency in orphan_rules["Agency ID"].dropna().unique():
        agency_orphans = orphan_rules[orphan_rules["Agency ID"] == agency]
        agency_comments = comment_docs[comment_docs["Agency ID"] == agency]

        if agency_orphans.empty or agency_comments.empty:
            continue

        all_titles = pd.concat([
            agency_orphans[["Title"]],
            agency_comments[["Title"]],
        ])

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )
            tfidf = vectorizer.fit_transform(all_titles["Title"].fillna(""))

            n_orphans = len(agency_orphans)
            orphan_vecs = tfidf[:n_orphans]
            comment_vecs = tfidf[n_orphans:]

            sims = cosine_similarity(orphan_vecs, comment_vecs)

            for i in range(n_orphans):
                best_j = np.argmax(sims[i])
                best_score = sims[i, best_j]
                if best_score >= threshold:
                    links.append({
                        "orphan_docket": str(agency_orphans.iloc[i]["Docket ID"]),
                        "comment_docket": str(agency_comments.iloc[best_j]["Docket ID"]),
                        "method": "title_similarity",
                        "evidence": f"score={best_score:.3f}",
                        "orphan_doc_id": str(agency_orphans.iloc[i]["Document ID"]),
                        "orphan_title": str(agency_orphans.iloc[i]["Title"])[:120],
                        "comment_title": str(agency_comments.iloc[best_j]["Title"])[:120],
                        "comment_doc_type": str(agency_comments.iloc[best_j]["doc_type"]),
                    })
        except Exception as e:
            logger.warning("Title similarity failed for %s: %s", agency, e)

    logger.info("Title similarity linking: %d links found (threshold=%.2f)", len(links), threshold)
    return links


def main():
    logger.info("Loading all documents...")
    all_docs = load_all_docs()
    comment_dockets = load_comment_dockets()
    orphan_dockets = find_orphan_rule_dockets(all_docs, comment_dockets)

    # Run all linking methods
    all_links = []

    logger.info("\n=== Method 1: Federal Register Number ===")
    all_links.extend(link_by_fr_number(all_docs, orphan_dockets, comment_dockets))

    logger.info("\n=== Method 2: RIN ===")
    all_links.extend(link_by_rin(all_docs, orphan_dockets, comment_dockets))

    logger.info("\n=== Method 3: Comment on Document ID ===")
    all_links.extend(link_by_comment_on_doc(all_docs, orphan_dockets, comment_dockets))

    logger.info("\n=== Method 4: Title Similarity ===")
    all_links.extend(link_by_title_similarity(all_docs, orphan_dockets, comment_dockets))

    # Deduplicate — keep best method per orphan-comment pair
    method_priority = {"fr_number": 0, "rin": 1, "comment_on_doc": 2, "title_similarity": 3}
    links_df = pd.DataFrame(all_links)

    if links_df.empty:
        logger.warning("No links found!")
        return

    links_df["method_rank"] = links_df["method"].map(method_priority)
    links_df = links_df.sort_values("method_rank").drop_duplicates(
        subset=["orphan_docket", "comment_docket"], keep="first"
    ).drop(columns=["method_rank"])

    # Stats
    linked_orphans = links_df["orphan_docket"].nunique()
    total_orphans = len(orphan_dockets)
    by_method = links_df.groupby("method")["orphan_docket"].nunique()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Total orphan rule dockets: %d", total_orphans)
    logger.info("Linked orphan dockets: %d (%.1f%%)", linked_orphans, 100 * linked_orphans / total_orphans)
    logger.info("Unlinked: %d", total_orphans - linked_orphans)
    logger.info("\nBy method (unique orphan dockets linked):")
    for method, count in by_method.items():
        logger.info("  %s: %d", method, count)

    # Save
    out_path = SCRIPTS_DIR / "data" / "docket_crosswalk.csv"
    links_df.to_csv(out_path, index=False)
    logger.info("\nSaved %d links to %s", len(links_df), out_path)

    stats = {
        "total_orphan_dockets": total_orphans,
        "linked_orphan_dockets": linked_orphans,
        "unlinked": total_orphans - linked_orphans,
        "total_links": len(links_df),
        "by_method": by_method.to_dict(),
    }
    stats_path = SCRIPTS_DIR / "data" / "docket_crosswalk_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved stats to %s", stats_path)

    # Print sample links for manual validation
    logger.info("\n=== SAMPLE LINKS FOR VALIDATION ===")
    for method in ["fr_number", "rin", "comment_on_doc", "title_similarity"]:
        sample = links_df[links_df["method"] == method].head(5)
        if not sample.empty:
            logger.info("\n--- %s ---", method)
            for _, row in sample.iterrows():
                logger.info("  %s -> %s  [evidence: %s]",
                            row["orphan_docket"], row["comment_docket"], row["evidence"])
                if "orphan_title" in row and pd.notna(row.get("orphan_title")):
                    logger.info("    orphan: %s", row["orphan_title"])
                    logger.info("    comment: %s", row.get("comment_title", ""))


if __name__ == "__main__":
    main()
