"""Utilities for loading and running the cross-encoder reranker.

Provides a cached model loader and a batch scoring function for use
in the main matching pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_cross_encoder_cache: dict = {}


def load_cross_encoder(model_path: str, max_length: int = 4096):
    """Load a CrossEncoder model, caching it for reuse across directories."""
    if model_path in _cross_encoder_cache:
        return _cross_encoder_cache[model_path]

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_path, max_length=max_length)
    _cross_encoder_cache[model_path] = model
    logger.info("Loaded cross-encoder from %s (max_length=%d)", model_path, max_length)
    return model


def load_optimal_threshold(model_path: str) -> float:
    """Load the optimal threshold saved during training."""
    threshold_path = Path(model_path) / "optimal_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
        threshold = data["threshold"]
        logger.info(
            "Loaded optimal threshold %.3f from %s", threshold, threshold_path
        )
        return threshold
    logger.warning("No optimal_threshold.json at %s, using 0.5", model_path)
    return 0.5


_truncation_tokenizer = None


def _get_truncation_tokenizer():
    """Lazy-load the ModernBERT tokenizer for pre-truncation."""
    global _truncation_tokenizer
    if _truncation_tokenizer is None:
        from transformers import AutoTokenizer
        _truncation_tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
    return _truncation_tokenizer


def _truncate_to_tokens(text: str, max_tokens: int = 10000) -> str:
    """Truncate text to exactly max_tokens using the ModernBERT tokenizer."""
    tok = _get_truncation_tokenizer()
    ids = tok.encode(str(text), add_special_tokens=False, truncation=False)
    if len(ids) <= max_tokens:
        return str(text)
    truncated_ids = ids[:max_tokens]
    return tok.decode(truncated_ids, skip_special_tokens=True)


def rerank_pairs(
    model,
    pairs_df: pd.DataFrame,
    response_col: str = "response_text",
    candidate_col: str = "candidate_text",
    batch_size: int = 64,
    max_tokens: int = 10000,
    min_batch_size: int = 4,
) -> np.ndarray:
    """Score (response, candidate) pairs with the cross-encoder.

    Pre-truncates texts to max_tokens using the ModernBERT tokenizer
    to prevent OOM from very long inputs. On OOM, retries with halved
    batch size down to min_batch_size.

    Returns an array of sigmoid scores in [0, 1], one per row.
    """
    import torch

    responses = pairs_df[response_col].tolist()
    candidates = pairs_df[candidate_col].tolist()

    # Pre-truncate to exact token count
    pairs = [
        (_truncate_to_tokens(r, max_tokens), _truncate_to_tokens(c, max_tokens))
        for r, c in zip(responses, candidates)
    ]

    current_batch_size = batch_size
    while current_batch_size >= min_batch_size:
        try:
            torch.cuda.empty_cache()
            scores = model.predict(
                pairs,
                batch_size=current_batch_size,
                show_progress_bar=True,
            )
            return np.array(scores)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                new_batch_size = max(current_batch_size // 2, min_batch_size)
                if new_batch_size == current_batch_size:
                    # Already at min, re-raise
                    raise
                logger.warning(
                    "OOM with batch_size=%d, retrying with batch_size=%d",
                    current_batch_size, new_batch_size,
                )
                current_batch_size = new_batch_size
            else:
                raise

    # Fallback: shouldn't reach here, but try min batch size once more
    torch.cuda.empty_cache()
    scores = model.predict(
        pairs,
        batch_size=min_batch_size,
        show_progress_bar=True,
    )
    return np.array(scores)
