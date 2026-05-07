"""Remap cluster_uids in claims files — V2, NO deduplication.

Keep all claims rows, just fix the cluster_uid to match the new mapper.
Store the original cluster_uid in a separate column for traceability.
load_claims_data will naturally pick the best representative per cluster_uid.
"""
import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

total_remapped = 0
total_unchanged = 0
total_orphaned = 0
dirs_processed = 0

for mapper_gz in sorted(glob.glob("*/*/public_submission_all_text__dedup_mapper.csv.gz")):
    dir_path = "/".join(mapper_gz.split("/")[:2])

    # Find claims file (v1 remap wrote .csv.gz)
    claims_path = None
    for ext in [".csv.gz", ".csv"]:
        cp = f"{dir_path}/public_submission_all_text__claims{ext}"
        if os.path.exists(cp):
            claims_path = cp
            break
    if claims_path is None:
        continue

    mapper = pd.read_csv(mapper_gz, low_memory=False)
    mapper["document_id"] = mapper["document_id"].astype(str)
    mapper["cluster_uid"] = mapper["cluster_uid"].astype(str)

    claims = pd.read_csv(claims_path, low_memory=False)
    claims["cluster_uid"] = claims["cluster_uid"].astype(str)
    claims["document_id"] = claims["document_id"].astype(str)

    # Save original UID for traceability
    claims["original_cluster_uid"] = claims["cluster_uid"]

    # Build doc_id -> new_cluster_uid lookup
    doc_to_new_uid = dict(zip(mapper["document_id"], mapper["cluster_uid"]))

    # Remap
    new_uids = claims["document_id"].map(doc_to_new_uid)
    has_mapping = new_uids.notna()

    n_remapped = (has_mapping & (claims["cluster_uid"] != new_uids)).sum()
    n_unchanged = (has_mapping & (claims["cluster_uid"] == new_uids)).sum()
    n_orphaned = (~has_mapping).sum()

    claims.loc[has_mapping, "cluster_uid"] = new_uids[has_mapping]

    # Write back as .csv.gz
    out_path = f"{dir_path}/public_submission_all_text__claims.csv.gz"
    claims.to_csv(out_path, index=False, compression="gzip")

    # Remove old .csv if exists
    old_csv = f"{dir_path}/public_submission_all_text__claims.csv"
    if os.path.exists(old_csv):
        os.remove(old_csv)

    total_remapped += n_remapped
    total_unchanged += n_unchanged
    total_orphaned += n_orphaned
    dirs_processed += 1

    if dirs_processed % 50 == 0:
        logging.info(
            f"Processed {dirs_processed} dirs: {total_remapped:,} remapped, "
            f"{total_unchanged:,} unchanged, {total_orphaned:,} orphaned"
        )

print(f"\n{'='*60}")
print(f"REMAP V2 COMPLETE across {dirs_processed} directories")
print(f"  Remapped (UID changed): {total_remapped:,}")
print(f"  Unchanged (UID already correct): {total_unchanged:,}")
print(f"  Orphaned (doc not in mapper, kept as-is): {total_orphaned:,}")
print(f"  Total rows preserved: {total_remapped + total_unchanged + total_orphaned:,}")
print(f"  ZERO rows dropped.")
