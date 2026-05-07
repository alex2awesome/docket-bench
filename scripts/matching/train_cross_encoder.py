"""Train a ModernBERT cross-encoder for (response, claim) matching.

Uses sentence-transformers CrossEncoder API with BinaryCrossEntropyLoss.
Trains on centrally collected LLM-labeled pairs from Phase 1 of the
cross-encoder pipeline.

Usage:
    python train_cross_encoder.py \
        --training-data training_data/llm_labeled_pairs.csv \
        --output-dir cross_encoder_models/modernbert-cross-encoder \
        --model-name answerdotai/ModernBERT-base \
        --epochs 3 \
        --batch-size 16 \
        --learning-rate 2e-5 \
        --max-length 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    data_path: Path,
    qa_path: Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
    split_by_docket: bool = False,
):
    """Load LLM-labeled pairs and split into train/val/test.

    If *qa_path* is provided, removes pairs where the primary and QA
    model disagreed (noisy labels).

    If *split_by_docket* is True, splits by docket_id so no docket
    appears in multiple splits (prevents data leakage).

    Returns (train_df, val_df, test_df) — each with ``response_text``,
    ``candidate_text``, and ``label`` (float 0/1) columns.
    """
    df = pd.read_csv(data_path)
    df = df.loc[df["llm_label"].isin(["yes", "no"])].copy()
    df["label"] = (df["llm_label"] == "yes").astype(float)

    logger.info(
        "Loaded %d labeled pairs: %d positive (%.1f%%), %d negative",
        len(df),
        int(df["label"].sum()),
        df["label"].mean() * 100,
        int((1 - df["label"]).sum()),
    )

    # Optionally filter out QA-disagreed pairs
    if qa_path and Path(qa_path).exists():
        qa_df = pd.read_csv(qa_path)
        disagreed = qa_df.loc[qa_df["primary_label"] != qa_df["qa_label"]]
        if len(disagreed) > 0:
            disagree_keys = set(
                zip(disagreed["doc_id"], disagreed["response_text"])
            )
            before = len(df)
            df = df.loc[
                ~df.apply(
                    lambda r: (r["doc_id"], r["response_text"]) in disagree_keys,
                    axis=1,
                )
            ]
            logger.info(
                "Filtered %d QA-disagreed pairs, %d remaining",
                before - len(df),
                len(df),
            )

    if split_by_docket and "docket_id" in df.columns:
        # Split by docket: no docket appears in multiple splits
        rng = np.random.RandomState(seed)
        dockets = df["docket_id"].fillna("unknown").unique()
        rng.shuffle(dockets)
        n_test = max(1, int(len(dockets) * test_fraction))
        n_val = max(1, int(len(dockets) * val_fraction))
        test_dockets = set(dockets[:n_test])
        val_dockets = set(dockets[n_test:n_test + n_val])
        train_dockets = set(dockets[n_test + n_val:])

        test = df[df["docket_id"].isin(test_dockets)]
        val = df[df["docket_id"].isin(val_dockets)]
        train = df[df["docket_id"].isin(train_dockets)]

        logger.info(
            "Docket-based split: %d dockets total -> %d train, %d val, %d test",
            len(dockets), len(train_dockets), len(val_dockets), len(test_dockets),
        )
    else:
        # Stratified split: train / val / test
        train_val, test = train_test_split(
            df,
            test_size=test_fraction,
            random_state=seed,
            stratify=df["label"],
        )
        val_frac_adjusted = val_fraction / (1 - test_fraction)
        train, val = train_test_split(
            train_val,
            test_size=val_frac_adjusted,
            random_state=seed,
            stratify=train_val["label"],
        )

    logger.info(
        "Split: train=%d (%.1f%% pos), val=%d (%.1f%% pos), test=%d (%.1f%% pos)",
        len(train),
        train["label"].mean() * 100,
        len(val),
        val["label"].mean() * 100,
        len(test),
        test["label"].mean() * 100,
    )

    return train, val, test


def compute_pos_weight(labels: pd.Series) -> float:
    """Return ``n_neg / n_pos`` for BinaryCrossEntropyLoss."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cross_encoder(args):  # noqa: C901
    """Main training entry point."""
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator,
    )
    from sentence_transformers.cross_encoder.training_args import (
        CrossEncoderTrainingArguments,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & split data ──
    train_df, val_df, test_df = load_and_prepare_data(
        Path(args.training_data),
        qa_path=args.qa_data,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_by_docket=args.split_by_docket,
    )

    # Save splits for reproducibility
    train_df.to_csv(output_dir / "train_split.csv.gz", index=False)
    val_df.to_csv(output_dir / "val_split.csv.gz", index=False)
    test_df.to_csv(output_dir / "test_split.csv.gz", index=False)

    # ── Initialize model ──
    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
    )

    # ── Prepare HF Datasets ──
    def to_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "sentence1": df["response_text"].tolist(),
            "sentence2": df["candidate_text"].tolist(),
            "label": df["label"].astype(float).tolist(),
        })

    train_dataset = to_dataset(train_df)
    val_dataset = to_dataset(val_df)

    # ── Loss ──
    pos_weight = compute_pos_weight(train_df["label"])
    logger.info("Class imbalance pos_weight: %.2f", pos_weight)

    loss = BinaryCrossEntropyLoss(
        model=model,
        pos_weight=torch.tensor(pos_weight),
    )

    # ── Evaluator ──
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=list(
            zip(
                val_df["response_text"].tolist(),
                val_df["candidate_text"].tolist(),
            )
        ),
        labels=val_df["label"].astype(int).tolist(),
        name="val",
        show_progress_bar=True,
    )

    # ── Training arguments ──
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="val_f1",
        logging_steps=50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=4,
        report_to="wandb" if args.wandb else "none",
        run_name=f"cross-encoder-{args.model_name.split('/')[-1]}",
    )

    # ── Train ──
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # ── Save final model ──
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved final model to %s", final_dir)

    # ── Find optimal threshold on validation set ──
    val_pairs = list(
        zip(
            val_df["response_text"].tolist(),
            val_df["candidate_text"].tolist(),
        )
    )
    val_scores = model.predict(val_pairs, show_progress_bar=True)
    val_labels = val_df["label"].astype(int).values

    best_f1, best_threshold = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (val_scores >= t).astype(int)
        f = f1_score(val_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_threshold = f, t

    logger.info("Best validation threshold: %.3f (F1=%.4f)", best_threshold, best_f1)

    # ── Evaluate on test set ──
    test_pairs = list(
        zip(
            test_df["response_text"].tolist(),
            test_df["candidate_text"].tolist(),
        )
    )
    test_scores = model.predict(test_pairs, show_progress_bar=True)
    test_labels = test_df["label"].astype(int).values
    test_preds = (test_scores >= best_threshold).astype(int)

    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_scores) if len(set(test_labels)) > 1 else 0.0
    test_report = classification_report(
        test_labels, test_preds, target_names=["no", "yes"], zero_division=0,
    )

    logger.info(
        "Test results (threshold=%.3f):\n  F1: %.4f, Precision: %.4f, Recall: %.4f, AUC: %.4f",
        best_threshold,
        test_f1,
        test_precision,
        test_recall,
        test_auc,
    )
    logger.info("\n%s", test_report)

    # ── Save training log & threshold ──
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "training_data": str(args.training_data),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "pos_weight": round(pos_weight, 4),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "best_val_threshold": round(best_threshold, 4),
        "best_val_f1": round(best_f1, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_auc": round(test_auc, 4),
        "test_report": test_report,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_entry, f, indent=2)

    # Save optimal threshold alongside the model for inference
    with open(final_dir / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": round(best_threshold, 4), "val_f1": round(best_f1, 4)}, f)

    logger.info("Training complete. Model at %s", final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT cross-encoder for response–claim matching.",
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to llm_labeled_pairs.csv (from --collect-training-data).",
    )
    parser.add_argument(
        "--qa-data",
        default=None,
        help="Path to QA agreement CSV. Pairs where models disagree are removed.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for model checkpoints and logs.",
    )
    parser.add_argument(
        "--model-name",
        default="answerdotai/ModernBERT-base",
        help="HuggingFace model to fine-tune (default: answerdotai/ModernBERT-base).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max sequence length (ModernBERT supports up to 8192).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=200,
        help="Evaluate and save every N steps (default: 200).",
    )
    parser.add_argument("--split-by-docket", action="store_true",
        help="Split train/val/test by docket instead of random stratified split.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging.",
    )
    args = parser.parse_args()

    train_cross_encoder(args)
