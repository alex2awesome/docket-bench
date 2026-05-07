"""Build a unified silver label cache from all label collection rounds.

Merges all silver/training label files into a single lookup keyed by
(doc_id, response_text[:200]) → llm_label. Saves as a fast-loading parquet file.

Usage:
    python build_silver_label_cache.py
"""

import glob
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR / "data"
TRAINING_DIR = SCRIPTS_DIR / "training_data"


def load_all_labels() -> pd.DataFrame:
    """Load and merge all silver label sources."""
    sources = []

    # 1. Original training data
    for path in [
        TRAINING_DIR / "llm_labeled_pairs.csv.gz",
        TRAINING_DIR / "llm_labeled_pairs.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df["_label_source"] = path.name
            sources.append(df)
            logger.info("Loaded %d from %s", len(df), path.name)

    # 2. Per-agency silver labels
    path = DATA_DIR / "per_agency_silver_labels.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
        df["_label_source"] = "per_agency_silver"
        sources.append(df)
        logger.info("Loaded %d from %s", len(df), path.name)

    # 3. Crosswalk silver labels
    path = DATA_DIR / "crosswalk_silver_labels.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
        df["_label_source"] = "crosswalk_silver"
        sources.append(df)
        logger.info("Loaded %d from %s", len(df), path.name)

    # 4. Positive-focused labels
    path = DATA_DIR / "positive_focused_silver_labels.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
        df["_label_source"] = "positive_focused"
        sources.append(df)
        logger.info("Loaded %d from %s", len(df), path.name)

    # 5. Imbalanced agency 5K labels
    path = DATA_DIR / "imbalanced_agency_5k_labels.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
        df["_label_source"] = "imbalanced_5k"
        sources.append(df)
        logger.info("Loaded %d from %s", len(df), path.name)

    # 6. Underperforming agency 2K labels
    path = DATA_DIR / "underperforming_agency_2k_labels.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
        df["_label_source"] = "underperforming_2k"
        sources.append(df)
        logger.info("Loaded %d from %s", len(df), path.name)

    # 7. Any additional per-agency label files (ed_4k, cbp_noaa_2k, etc.)
    for extra in sorted(DATA_DIR.glob("*_labels.csv.gz")):
        if extra.name in {p.name for p in [
            DATA_DIR / "per_agency_silver_labels.csv.gz",
            DATA_DIR / "crosswalk_silver_labels.csv.gz",
            DATA_DIR / "positive_focused_silver_labels.csv.gz",
            DATA_DIR / "imbalanced_agency_5k_labels.csv.gz",
            DATA_DIR / "underperforming_agency_2k_labels.csv.gz",
            DATA_DIR / "silver_label_cache.csv.gz",
        ]}:
            continue  # already loaded above or is the cache itself
        try:
            df = pd.read_csv(extra)
            if "llm_label" in df.columns:
                df["_label_source"] = extra.stem
                sources.append(df)
                logger.info("Loaded %d from %s", len(df), extra.name)
        except Exception as e:
            logger.warning("Could not load %s: %s", extra.name, e)

    if not sources:
        raise FileNotFoundError("No silver label files found")

    combined = pd.concat(sources, ignore_index=True)
    logger.info("Combined: %d total rows", len(combined))

    # Normalize agency column — some sources have agency_id (uppercase),
    # others have _agency (lowercase). Reconcile into _agency.
    if "_agency" not in combined.columns:
        combined["_agency"] = None
    if "agency_id" in combined.columns:
        fill_mask = combined["_agency"].isna() & combined["agency_id"].notna()
        combined.loc[fill_mask, "_agency"] = combined.loc[fill_mask, "agency_id"].str.lower()
    # Also try extracting from source_dir (e.g., "epa_2016_2017" → "epa")
    if "source_dir" in combined.columns:
        fill_mask = combined["_agency"].isna() & combined["source_dir"].notna()
        combined.loc[fill_mask, "_agency"] = combined.loc[fill_mask, "source_dir"].str.split("_").str[0]
    if "_source_dir" in combined.columns:
        fill_mask = combined["_agency"].isna() & combined["_source_dir"].notna()
        combined.loc[fill_mask, "_agency"] = combined.loc[fill_mask, "_source_dir"].str.split("_").str[0]

    logger.info("Agency coverage: %d/%d rows have _agency", combined["_agency"].notna().sum(), len(combined))
    return combined


def build_cache(combined: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and build the cache keyed by (doc_id, response_text_prefix)."""
    # Keep only valid labels
    combined = combined[combined["llm_label"].isin(["yes", "no"])].copy()
    logger.info("Valid labels: %d", len(combined))

    # Build dedup key
    combined["_cache_key"] = (
        combined["doc_id"].astype(str)
        + "||"
        + combined["response_text"].fillna("").astype(str).str[:200]
    )

    # Deduplicate — prefer per_agency_silver > crosswalk > positive_focused > original
    source_priority = {
        "per_agency_silver": 0,
        "crosswalk_silver": 1,
        "positive_focused": 2,
        "imbalanced_5k": 3,
        "llm_labeled_pairs.csv.gz": 4,
        "llm_labeled_pairs.csv": 5,
    }
    combined["_priority"] = combined["_label_source"].map(source_priority).fillna(99)
    combined = combined.sort_values("_priority").drop_duplicates(
        subset=["_cache_key"], keep="first"
    )
    logger.info("After dedup: %d unique pairs", len(combined))

    # Select output columns
    out_cols = ["doc_id", "response_text", "llm_label", "_cache_key", "_label_source"]
    # Add optional columns if present
    for col in ["docket_id", "candidate_text", "agency_id", "_agency",
                "dense_score", "claim_ce_score", "comment_ce_score",
                "combined_ce_score", "crosswalk_docket"]:
        if col in combined.columns:
            out_cols.append(col)

    return combined[out_cols].reset_index(drop=True)


def main():
    combined = load_all_labels()
    cache = build_cache(combined)

    # Save as parquet for fast loading
    out_path = DATA_DIR / "silver_label_cache.parquet"
    cache.to_parquet(out_path, index=False)
    logger.info("Saved cache (%d pairs) to %s", len(cache), out_path)

    # Also save as gzipped CSV for compatibility
    csv_path = DATA_DIR / "silver_label_cache.csv.gz"
    cache.to_csv(csv_path, index=False, compression="gzip")
    logger.info("Saved CSV cache to %s", csv_path)

    # Summary stats
    logger.info("\n=== Cache Summary ===")
    logger.info("Total unique labeled pairs: %d", len(cache))
    logger.info("Label distribution: %s", cache["llm_label"].value_counts().to_dict())
    logger.info("By source: %s", cache["_label_source"].value_counts().to_dict())

    if "_agency" in cache.columns:
        logger.info("\nPer-agency coverage:")
        for agency in sorted(cache["_agency"].dropna().unique()):
            sub = cache[cache["_agency"] == agency]
            yes = (sub["llm_label"] == "yes").sum()
            logger.info("  %s: %d pairs (%d yes, %d no)", agency, len(sub), yes, len(sub) - yes)


if __name__ == "__main__":
    main()
