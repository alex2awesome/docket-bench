"""Bridge old singleton cluster_uids in claims files to new mapper cluster_uids.

The old minhash had a bug where comments <80 tokens were dropped from the mapper.
The claims code treated these as singletons using Document ID as cluster_uid.
The new minhash correctly assigns them to clusters with docket::number UIDs.

This script bridges old claims to new UIDs so we don't re-extract them.
"""

import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

total_bridged = 0
total_genuinely_new = 0
total_already_ok = 0
dirs_updated = 0

for mapper_gz in sorted(glob.glob("*/*/public_submission_all_text__dedup_mapper.csv.gz")):
    dir_path = "/".join(mapper_gz.split("/")[:2])

    # Find claims file
    claims_path = f"{dir_path}/public_submission_all_text__claims.csv.gz"
    if not os.path.exists(claims_path):
        claims_path = f"{dir_path}/public_submission_all_text__claims.csv"
    if not os.path.exists(claims_path):
        continue

    # Load mapper and claims
    mapper = pd.read_csv(mapper_gz, low_memory=False)
    mapper["document_id"] = mapper["document_id"].astype(str)
    mapper["cluster_uid"] = mapper["cluster_uid"].astype(str)

    claims = pd.read_csv(claims_path, low_memory=False)
    claims["cluster_uid"] = claims["cluster_uid"].astype(str)

    mapper_uids = set(mapper["cluster_uid"].unique())
    claims_uids = set(claims["cluster_uid"].unique())

    already_ok = mapper_uids & claims_uids
    new_uids = mapper_uids - claims_uids

    if not new_uids:
        total_already_ok += len(already_ok)
        continue

    # Build document_id -> old claims row lookup (old singletons used doc ID as UID)
    orphaned_uids = claims_uids - mapper_uids
    old_singleton_claims = claims[claims["cluster_uid"].isin(orphaned_uids)]
    doc_id_to_old_claims = {}
    for _, row in old_singleton_claims.iterrows():
        doc_id_to_old_claims[row["cluster_uid"]] = row

    # For each new cluster_uid, check if any member doc was an old singleton
    new_mapper = mapper[mapper["cluster_uid"].isin(new_uids)]

    bridged_rows = []
    genuinely_new = 0

    for uid, group in new_mapper.groupby("cluster_uid"):
        member_doc_ids = group["document_id"].tolist()
        found = False
        for doc_id in member_doc_ids:
            if doc_id in doc_id_to_old_claims:
                old_row = doc_id_to_old_claims[doc_id].copy()
                old_row["cluster_uid"] = uid
                bridged_rows.append(old_row)
                found = True
                break
        if not found:
            genuinely_new += 1

    if bridged_rows:
        bridged_df = pd.DataFrame(bridged_rows)
        updated_claims = pd.concat([claims, bridged_df], ignore_index=True)
        out_path = f"{dir_path}/public_submission_all_text__claims.csv.gz"
        updated_claims.to_csv(out_path, index=False, compression="gzip")
        dirs_updated += 1
        total_bridged += len(bridged_rows)
        logging.info(
            "%s: bridged %d, genuinely new %d", dir_path, len(bridged_rows), genuinely_new
        )

    total_genuinely_new += genuinely_new
    total_already_ok += len(already_ok)

print(f"\n=== Bridge Results ===")
print(f"Directories updated: {dirs_updated}")
print(f"Claims bridged (reused): {total_bridged:,}")
print(f"Genuinely new clusters (need LLM): {total_genuinely_new:,}")
print(f"Already matched: {total_already_ok:,}")
