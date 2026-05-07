"""Remap cluster_uids in claims files to match the new minhash mapper.

For each directory:
1. Load new mapper (document_id -> new_cluster_uid)
2. Load claims file
3. For each claims row, look up document_id in mapper -> get correct cluster_uid
4. Replace cluster_uid column
5. Deduplicate (if two old rows map to same new cluster_uid, keep the one
   whose document_id is the new representative = longest text)
6. Write back
"""
import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

total_remapped = 0
total_kept = 0
total_dropped_dups = 0
total_orphaned = 0
dirs_processed = 0

for mapper_gz in sorted(glob.glob("*/*/public_submission_all_text__dedup_mapper.csv.gz")):
    dir_path = "/".join(mapper_gz.split("/")[:2])

    # Find claims file
    claims_path = None
    for ext in [".csv.gz", ".csv"]:
        cp = f"{dir_path}/public_submission_all_text__claims{ext}"
        if os.path.exists(cp):
            claims_path = cp
            break
    if claims_path is None:
        continue

    # Find all_text for representative selection
    all_text_path = None
    for ext in [".csv.gz", ".csv"]:
        at = f"{dir_path}/public_submission_all_text{ext}"
        if os.path.exists(at):
            all_text_path = at
            break
    if all_text_path is None:
        continue

    mapper = pd.read_csv(mapper_gz, low_memory=False)
    mapper["document_id"] = mapper["document_id"].astype(str)
    mapper["cluster_uid"] = mapper["cluster_uid"].astype(str)

    claims = pd.read_csv(claims_path, low_memory=False)
    claims["cluster_uid"] = claims["cluster_uid"].astype(str)
    claims["document_id"] = claims["document_id"].astype(str)

    original_len = len(claims)

    # Build doc_id -> new_cluster_uid lookup
    doc_to_new_uid = dict(zip(mapper["document_id"], mapper["cluster_uid"]))

    # Remap: look up each claim's document_id in the new mapper
    claims["new_cluster_uid"] = claims["document_id"].map(doc_to_new_uid)

    # Claims whose doc_id is not in the new mapper (orphaned singletons from old runs)
    orphaned = claims["new_cluster_uid"].isna()
    n_orphaned = orphaned.sum()

    # For orphaned rows: if doc_id itself looks like a singleton UID and is in the mapper
    # as a document_id under some cluster, it's already handled. Otherwise drop it.
    # Actually keep orphaned rows as-is — they might be singletons that the claims
    # extraction code handles separately (docs not in mapper treated as singletons
    # with cluster_uid = document_id)
    # Don't remap these, keep their original cluster_uid
    claims.loc[~orphaned, "cluster_uid"] = claims.loc[~orphaned, "new_cluster_uid"]
    claims.drop(columns=["new_cluster_uid"], inplace=True)

    n_remapped = (~orphaned).sum()

    # Load all_text for representative selection (to resolve duplicates)
    all_text = pd.read_csv(all_text_path, usecols=["Document ID", "canonical_text"], low_memory=False)
    all_text["Document ID"] = all_text["Document ID"].astype(str)
    all_text["text_len"] = all_text["canonical_text"].fillna("").str.len()
    doc_text_len = dict(zip(all_text["Document ID"], all_text["text_len"]))

    # Deduplicate: if multiple rows have the same cluster_uid,
    # keep the one whose document_id has the longest text (= most likely representative)
    claims["_text_len"] = claims["document_id"].map(doc_text_len).fillna(0)
    before_dedup = len(claims)
    claims = claims.sort_values("_text_len", ascending=False).drop_duplicates(
        subset=["cluster_uid"], keep="first"
    )
    claims.drop(columns=["_text_len"], inplace=True)
    n_dropped = before_dedup - len(claims)

    # Write back (always write as .csv.gz for consistency)
    out_path = f"{dir_path}/public_submission_all_text__claims.csv.gz"
    claims.to_csv(out_path, index=False, compression="gzip")

    # Remove old .csv if we wrote .csv.gz
    old_csv = f"{dir_path}/public_submission_all_text__claims.csv"
    if os.path.exists(old_csv) and out_path.endswith(".gz"):
        os.remove(old_csv)

    total_remapped += n_remapped
    total_kept += len(claims)
    total_dropped_dups += n_dropped
    total_orphaned += n_orphaned
    dirs_processed += 1

    if dirs_processed % 50 == 0:
        logging.info(f"Processed {dirs_processed} dirs, {total_remapped:,} remapped, {total_dropped_dups:,} dups dropped")

print(f"\n{'='*60}")
print(f"REMAP COMPLETE across {dirs_processed} directories")
print(f"  Total claims remapped: {total_remapped:,}")
print(f"  Total claims kept: {total_kept:,}")
print(f"  Duplicates dropped: {total_dropped_dups:,}")
print(f"  Orphaned (kept as-is): {total_orphaned:,}")
