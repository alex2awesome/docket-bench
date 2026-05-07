"""Train per-agency COMMENT-level cross-encoders.

Same as train_per_agency_ce.py but trains on (response_text, comment_text)
pairs instead of (response_text, candidate_text/claim).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_per_agency_comment_ce.py \
        --training-data data/bulk_downloads/scripts/data/per_agency_silver_labels.csv.gz \
        --output-base data/bulk_downloads/scripts/cross_encoder_models/per_agency_comment \
        --min-samples 200
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max length — 8192 for full comments.")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-docket", action="store_true")
    parser.add_argument("--agency", nargs="*", default=None)
    args = parser.parse_args()

    logger.info("Loading training data from %s", args.training_data)
    df = pd.read_csv(args.training_data)
    df = df[df["llm_label"].isin(["yes", "no"])].copy()

    # Must have comment_text
    if "comment_text" not in df.columns:
        logger.error("No comment_text column in training data")
        return

    # Filter out rows with empty/nan comment_text
    df = df[df["comment_text"].notna() & (df["comment_text"].str.len() > 10)]
    logger.info("Loaded %d valid labels with comment_text", len(df))

    # Swap candidate_text -> comment_text for training
    # The train_cross_encoder script uses candidate_text as sentence2
    df["candidate_text"] = df["comment_text"]

    # Also load existing silver labels
    existing_labels = SCRIPTS_DIR / "training_data" / "llm_labeled_pairs.csv.gz"
    existing = pd.DataFrame()
    if existing_labels.exists():
        existing = pd.read_csv(existing_labels)
        existing = existing[existing["llm_label"].isin(["yes", "no"])].copy()
        if "comment_text" in existing.columns:
            existing = existing[existing["comment_text"].notna() & (existing["comment_text"].str.len() > 10)]
            existing["candidate_text"] = existing["comment_text"]
        else:
            existing = pd.DataFrame()  # can't use without comment_text
        if "agency_id" in existing.columns and "_agency" not in existing.columns:
            existing["_agency"] = existing["agency_id"].str.lower()
        logger.info("Loaded %d existing labels with comment_text", len(existing))

    agency_col = "_agency"
    if agency_col not in df.columns:
        if "agency_id" in df.columns:
            df["_agency"] = df["agency_id"].str.lower()

    agencies = sorted(df[agency_col].dropna().unique())
    if args.agency:
        agencies = [a for a in agencies if a in args.agency]

    logger.info("Training comment CEs for %d agencies", len(agencies))

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    trained = 0
    skipped = 0
    for agency in agencies:
        agency_dir = output_base / agency
        final_dir = agency_dir / "final"
        processing_flag = agency_dir / ".processing"

        if final_dir.exists() and (final_dir / "model.safetensors").exists():
            logger.info("Skipping %s — already trained", agency)
            skipped += 1
            continue

        try:
            os.makedirs(agency_dir, exist_ok=True)
            fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}".encode())
            os.close(fd)
        except FileExistsError:
            logger.info("Skipping %s — another process is training it", agency)
            skipped += 1
            continue

        try:
            agency_data = df[df[agency_col] == agency].copy()

            if not existing.empty and "_agency" in existing.columns:
                agency_existing = existing[existing["_agency"] == agency]
                if not agency_existing.empty:
                    common_cols = ["response_text", "candidate_text", "llm_label"]
                    if "docket_id" in agency_data.columns and "docket_id" in agency_existing.columns:
                        common_cols.append("docket_id")
                    agency_data = pd.concat([
                        agency_data[common_cols],
                        agency_existing[common_cols],
                    ], ignore_index=True).drop_duplicates(
                        subset=["response_text", "candidate_text"], keep="first"
                    )
                    logger.info("  %s: merged to %d total labels", agency, len(agency_data))

            if len(agency_data) < args.min_samples:
                logger.info("Skipping %s — only %d labels (need %d)", agency, len(agency_data), args.min_samples)
                skipped += 1
                continue

            n_pos = (agency_data["llm_label"] == "yes").sum()
            n_neg = (agency_data["llm_label"] == "no").sum()
            logger.info("Training comment CE %s: %d labels (%d pos, %d neg)", agency, len(agency_data), n_pos, n_neg)

            agency_train_path = agency_dir / "training_data.csv"
            agency_data.to_csv(agency_train_path, index=False)

            import argparse as _ap
            train_args = _ap.Namespace(
                training_data=str(agency_train_path),
                qa_data=None,
                output_dir=str(agency_dir),
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                val_fraction=0.15,
                test_fraction=0.10,
                seed=args.seed,
                eval_steps=args.eval_steps,
                split_by_docket=args.split_by_docket,
                wandb=False,
            )

            sys.path.insert(0, str(SCRIPTS_DIR))
            from train_cross_encoder import train_cross_encoder
            train_cross_encoder(train_args)

            trained += 1
            logger.info("Finished training comment CE %s", agency)

        except Exception as e:
            logger.error("Failed training %s: %s", agency, e, exc_info=True)
        finally:
            processing_flag.unlink(missing_ok=True)

    logger.info("Done: %d trained, %d skipped", trained, skipped)


if __name__ == "__main__":
    main()
