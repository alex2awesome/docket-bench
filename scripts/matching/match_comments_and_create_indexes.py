"""Retrieval-Based Comment–Response Matching Pipeline.

For each agency/year directory, builds three retrieval indexes over
comment clusters (at either claim or comment level):
  1. BM25 (sparse) — for distribution to users without GPUs
  2. nvidia/llama-embed-nemotron-8b (dense) — primary, used for matching
  3. all-mpnet-base-v2 (dense) — for distribution

Retrieves top-k candidates per government response using the primary
dense model, then uses LLM sampling to find an optimal dense-score
threshold for final labeling.

Outputs per agency/year directory:
  - public_submission_all_text__{level}_response_matches.csv
  - public_submission_all_text__{level}_comment_labels.csv
  - .retriv_indexes/ (BM25 + dense indexes for distribution)

Global output:
  - pipeline_log.csv  (one row per directory, tracks strategy/metrics)

Usage:
    python data/bulk_downloads/scripts/match_comments_and_create_indexes.py --level claims --prompt-backend openai --llm-model gpt-5-mini --top-k 10
    python data/bulk_downloads/scripts/match_comments_and_create_indexes.py --level claims --llm-model gpt-5 gpt-5-mini --collect-training-data --training-samples-per-dir 200
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import atexit
import json
import logging
import os
import random
import re
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# HF_MODULES_CACHE must point to a user-writable directory before any
# transformers import. The shared HF cache at os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
# is read-only for most users, but transformers writes dynamic module files
# under modules/transformers_modules/ when loading trust_remote_code models
# (ModernBERT, nvidia/llama-embed-nemotron-8b, etc.). Leaving this unset
# causes "Permission denied" errors on ~15% of dirs in rescore runs.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
os.environ.setdefault("HF_MODULES_CACHE", str(_script_dir / "data" / ".hf_modules"))
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)

import filelock
import numpy as np
import pandas as pd

# Monkey-patch json.JSONEncoder.default so autofaiss can serialize numpy scalars.
_orig_json_default = json.JSONEncoder.default

def _numpy_json_default(self, obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _orig_json_default(self, obj)

json.JSONEncoder.default = _numpy_json_default  # type: ignore[assignment]
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Ensure notebooks/ is on sys.path so we can import prompt_utils
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent.parent  # regulations-demo/
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")
if not os.environ.get("OPENAI_API_KEY"):
    key_path = _PROJECT_ROOT.parent / ".openai-key.txt"
    if key_path.exists():
        os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()

import prompt_utils  # noqa: E402

# ---------------------------------------------------------------------------
# robust_json_load copied from utils.py to avoid IPython dependency
# ---------------------------------------------------------------------------


def robust_json_load(value):
    """Best-effort parse for JSON-ish strings."""
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return ast.literal_eval(text)
        except (ValueError, SyntaxWarning, SyntaxError):
            try:
                escaped = text.replace("\\", "\\\\")
                return json.loads(escaped)
            except Exception:
                return []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BULK_DIR = _SCRIPTS_DIR.parent  # data/bulk_downloads/
LOG_FILE = _SCRIPTS_DIR / "pipeline_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for noisy_logger in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lock file management (following comments_extract_claims.py pattern)
# ---------------------------------------------------------------------------
_active_processing_files: Set[Path] = set()


def _cleanup_processing_files() -> None:
    for p in list(_active_processing_files):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    _active_processing_files.clear()


def _signal_handler(signum, frame) -> None:
    _cleanup_processing_files()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


atexit.register(_cleanup_processing_files)


# ---------------------------------------------------------------------------
# 1. Load response data
# ---------------------------------------------------------------------------


def _load_gpt5_responses() -> pd.DataFrame:
    """Load the GPT-5 response cache."""
    candidates = [
        _NOTEBOOKS_DIR / "2026-02-10__comment-response-cache.csv.gz",
        _PROJECT_ROOT / "2026-02-10__comment-response-cache.csv.gz",
        Path("2026-02-10__comment-response-cache.csv.gz"),
    ]
    response_cache_path = None
    for c in candidates:
        if c.exists():
            response_cache_path = c
            break
    if response_cache_path is None:
        logger.warning("GPT-5 response cache not found")
        return pd.DataFrame()

    logger.info("Loading GPT-5 response cache from %s", response_cache_path)
    orig = pd.read_csv(response_cache_path, index_col=0).assign(
        parsed_response=lambda df: df["summarized_response"].apply(robust_json_load)
    ).drop(columns="summarized_response")

    proc = (
        orig.assign(
            parsed_response=lambda df: df["parsed_response"].apply(
                lambda x: x if isinstance(x, list) else [x]
            )
        )
        .loc[lambda df: df["parsed_response"].str.len() > 0]
        .explode("parsed_response")
        .reset_index(drop=True)
        .assign(
            parsed_response=lambda df: df["parsed_response"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )
        )
        .pipe(
            lambda df: pd.concat(
                [
                    df[["Agency ID", "Docket ID"]],
                    pd.DataFrame(df["parsed_response"].tolist()),
                ],
                axis=1,
            )
        )
        .drop(
            columns=["error", "detail", "commenter_identifiers_Text"],
            errors="ignore",
        )
    )
    logger.info("Loaded %d GPT-5 response rows", len(proc))
    return proc


def _load_llama_responses() -> pd.DataFrame:
    """Load the Llama-extracted responses from comment_responses.jsonl.

    Also picks up Nader-specific validation responses from
    _nader_fcc_rif/nader_comment_responses_test.jsonl if present (these are
    extracted from FCC-17-166 for the Handan-Nader ground-truth validation).
    """
    candidates = [
        _SCRIPTS_DIR / "data" / "comment_responses.jsonl",
        _SCRIPTS_DIR / "data" / "comment_responses_V2.jsonl",
        BULK_DIR / "_nader_fcc_rif" / "nader_comment_responses_test.jsonl",
    ]
    rows = []
    for jsonl_path in candidates:
        if not jsonl_path.exists():
            continue
        logger.info("Loading Llama responses from %s", jsonl_path)
        with open(jsonl_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                responses = doc.get("responses", [])
                if not responses:
                    continue
                agency_id = doc.get("agency_id", "")
                docket_id = doc.get("docket_id", "")
                for resp in responses:
                    if isinstance(resp, dict):
                        resp["Agency ID"] = agency_id
                        resp["Docket ID"] = docket_id
                        rows.append(resp)
        # Continue loading all available files (V1 + V2)

    if not rows:
        logger.warning("No Llama responses found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalize column names to match GPT-5 format
    df = df.drop(
        columns=["error", "detail", "commenter_identifiers_Text"],
        errors="ignore",
    )
    logger.info("Loaded %d Llama response rows", len(df))
    return df


def load_response_df() -> pd.DataFrame:
    """Load responses from both GPT-5 cache and Llama JSONL, deduplicated."""
    gpt5 = _load_gpt5_responses()
    llama = _load_llama_responses()

    if gpt5.empty and llama.empty:
        raise FileNotFoundError("No response data found from either GPT-5 or Llama sources")

    # Ensure both have the same key columns
    common_cols = ["Agency ID", "Docket ID", "content_of_comment",
                   "summarized_content_of_comment", "response_to_comment"]

    frames = []
    for df, source in [(gpt5, "gpt5"), (llama, "llama")]:
        if df.empty:
            continue
        df = df.copy()
        df["_source"] = source
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: prefer GPT-5 when both have the same (Docket ID, content_of_comment)
    # Use first ~200 chars of content_of_comment as dedup key
    combined["_dedup_key"] = (
        combined["Docket ID"].astype(str) + "||" +
        combined.get("content_of_comment", pd.Series("", index=combined.index))
        .astype(str).str[:200]
    )
    before = len(combined)
    combined = combined.sort_values("_source").drop_duplicates(
        subset=["_dedup_key"], keep="first"  # "first" = gpt5 comes before llama alphabetically
    )
    combined = combined.drop(columns=["_dedup_key", "_source"], errors="ignore")

    logger.info(
        "Combined responses: %d total (%d after dedup, removed %d duplicates)",
        before, len(combined), before - len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# 2. Data loading for claims / comment levels
# ---------------------------------------------------------------------------


def _parse_agency_year(dir_name: str) -> tuple[str, str]:
    """Extract agency and year range from directory name like 'epa_2020_2021'."""
    m = re.match(r"^([a-z]+)_(\d{4}_\d{4})$", dir_name)
    if m:
        return m.group(1), m.group(2)
    return dir_name, ""


def load_claims_data(agency_year_dir: Path, claims_suffix: str = "") -> Optional[pd.DataFrame]:
    """Load claims.csv for a directory. Returns exploded claim rows or None.

    `claims_suffix` lets us select a re-extracted variant (e.g. "_v2"), so
    different claim runs can coexist alongside each other in the same dir.
    """
    claims_path = agency_year_dir / f"public_submission_all_text__claims{claims_suffix}.csv.gz"
    if not claims_path.exists():
        claims_path = agency_year_dir / f"public_submission_all_text__claims{claims_suffix}.csv"
    if not claims_path.exists():
        logger.debug("No claims file at %s", claims_path)
        return None

    agency, year_range = _parse_agency_year(agency_year_dir.name)

    claims_df = pd.read_csv(claims_path, low_memory=False)
    rows = []
    for _, row in claims_df.iterrows():
        cluster_uid = str(row.get("cluster_uid", ""))
        if not cluster_uid:
            continue

        # Try claims_parsed_json first, then claims_fix_parsed_json
        claims_list = None
        if row.get("parse_ok") is True or str(row.get("parse_ok", "")).lower() == "true":
            claims_list = robust_json_load(row.get("claims_parsed_json", ""))
        if not claims_list:
            if row.get("fix_parse_ok") is True or str(row.get("fix_parse_ok", "")).lower() == "true":
                claims_list = robust_json_load(row.get("claims_fix_parsed_json", ""))
        if not claims_list:
            continue

        docket_id = str(row.get("docket_id", ""))
        document_id = str(row.get("document_id", ""))

        for i, claim_text in enumerate(claims_list):
            if not isinstance(claim_text, str) or not claim_text.strip():
                continue
            claim_id = f"{agency}__{year_range}__{docket_id}__{document_id}__{i}"
            rows.append({
                "id": claim_id,
                "text": claim_text.strip(),
                "cluster_uid": cluster_uid,
                "docket_id": docket_id,
                "document_id": document_id,
            })

    if not rows:
        logger.warning("No valid claims found in %s", claims_path)
        return None

    return pd.DataFrame(rows)


def load_comment_data(agency_year_dir: Path) -> Optional[pd.DataFrame]:
    """Load comment text data for a directory, one row per cluster."""
    mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv.gz"
    if not mapper_path.exists():
        mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv"
    all_text_path = agency_year_dir / "public_submission_all_text.csv.gz"
    if not all_text_path.exists():
        all_text_path = agency_year_dir / "public_submission_all_text.csv"

    if not mapper_path.exists():
        logger.warning("No dedup mapper at %s", mapper_path)
        return None
    if not all_text_path.exists():
        logger.warning("No all_text file at %s", all_text_path)
        return None

    agency, year_range = _parse_agency_year(agency_year_dir.name)

    mapper = pd.read_csv(mapper_path, low_memory=False)
    all_text = pd.read_csv(
        all_text_path,
        usecols=["Document ID", "Docket ID", "canonical_text"],
        low_memory=False,
    )

    mapper["document_id"] = mapper["document_id"].astype(str)
    all_text["Document ID"] = all_text["Document ID"].astype(str)

    merged = mapper.merge(
        all_text,
        left_on="document_id",
        right_on="Document ID",
        how="inner",
    )
    if merged.empty:
        logger.warning("Empty merge for %s", agency_year_dir)
        return None

    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    # Pick the longest text per cluster as representative
    reps = (
        merged.sort_values("text_len", ascending=False)
        .groupby("cluster_uid", as_index=False)
        .head(1)
    )

    rows = []
    for _, row in reps.iterrows():
        cluster_uid = str(row.get("cluster_uid", ""))
        text = str(row.get("canonical_text", "")).strip()
        if not cluster_uid or not text:
            continue
        docket_id = str(row.get("docket_id", ""))
        document_id = str(row.get("document_id", ""))
        comment_id = f"{agency}__{year_range}__{docket_id}__{document_id}"
        rows.append({
            "id": comment_id,
            "text": text,
            "cluster_uid": cluster_uid,
            "docket_id": docket_id,
            "document_id": document_id,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_collection(agency_year_dir: Path, level: str, claims_suffix: str = "") -> Optional[pd.DataFrame]:
    """Build retriv collection for a directory. Returns DataFrame with id, text, cluster_uid, docket_id."""
    if level == "claims":
        return load_claims_data(agency_year_dir, claims_suffix=claims_suffix)
    else:
        return load_comment_data(agency_year_dir)


# ---------------------------------------------------------------------------
# 3. Index management
# ---------------------------------------------------------------------------


def _index_exists(base_path: Path, index_name: str, kind: str) -> bool:
    """Check if a retriv index exists on disk."""
    idx_dir = base_path / "collections" / index_name
    if kind == "sparse":
        return (idx_dir / "sr_state.npz").exists()
    else:
        return (idx_dir / "dr_state.npz").exists()


DISTRIBUTION_DENSE_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
]


# Sane max_length caps per model to avoid padding to huge context windows
# (e.g. Llama 3.1's max_position_embeddings=131072 would pad every input
# to 128K tokens, making even 21 short texts take forever).
_MODEL_MAX_LENGTH = {
    "nvidia/llama-embed-nemotron-8b": 4096,  # NVIDIA recommends 4096 in examples
    # mpnet-base has max_position_embeddings=512; use 384 for safety margin.
    "sentence-transformers/all-mpnet-base-v2": 384,
}


# Cache encoders across directories so the 8B model is loaded once and reused.
_encoder_cache: dict[str, object] = {}


def _build_or_load_one_dense(
    index_base: Path,
    index_name: str,
    model_name: str,
    collection: List[dict],
    batch_size: int,
    overwrite: bool,
):
    """Build or load a single dense index. Returns DenseRetriever."""
    from retriv import DenseRetriever

    # Use a suffix derived from the model name for unique index dirs.
    safe_suffix = model_name.replace("/", "_")
    full_name = f"{index_name}_{safe_suffix}"
    max_len = _MODEL_MAX_LENGTH.get(model_name, None)

    if not overwrite and _index_exists(index_base, full_name, "dense"):
        logger.info("Loading existing dense index: %s (model=%s)", full_name, model_name)
        cached = _encoder_cache.get(model_name)
        if cached is not None:
            # Load index without re-loading the encoder, then assign the cached one.
            logger.info("Reusing cached encoder for %s", model_name)
            dr = DenseRetriever.load(full_name, encoder=cached)
        else:
            dr = DenseRetriever.load(full_name)
            # Override max_length on loaded encoder if needed (saved state may
            # have the uncapped value from a previous build).
            if max_len and dr.encoder is not None and dr.encoder.max_length != max_len:
                logger.info("Overriding encoder max_length %s → %s", dr.encoder.max_length, max_len)
                dr.encoder.max_length = max_len
                dr.encoder.tokenizer_kwargs["max_length"] = max_len
            # Ensure encoder is on GPU after loading (retriv may deserialize to CPU).
            # NOTE: Encoder.model is the model NAME (str); the actual AutoModel
            # is stored as Encoder.encoder.
            if dr.encoder is not None and hasattr(dr.encoder, "encoder"):
                import torch
                if torch.cuda.is_available():
                    try:
                        device = next(dr.encoder.encoder.parameters()).device
                        if str(device) == "cpu":
                            target_device = os.environ.get("ENCODER_DEVICE", "cuda")
                            logger.info("Moving encoder to %s (was on %s)", target_device, device)
                            dr.encoder.encoder.to(target_device)
                            dr.encoder.device = target_device
                    except StopIteration:
                        pass
            # Cache the encoder for reuse
            if dr.encoder is not None:
                _encoder_cache[model_name] = dr.encoder
                logger.info("Cached encoder for %s", model_name)
        # Check if the loaded index is stale (fewer docs than collection).
        # Use retriv's incremental .add() to avoid full rebuild.
        if dr.doc_count < len(collection):
            existing_ids = set(dr.id_mapping.values())
            new_docs = [d for d in collection if d["id"] not in existing_ids]
            logger.info(
                "Index %s has %d docs but collection has %d (%d new). "
                "Adding new documents incrementally.",
                full_name, dr.doc_count, len(collection), len(new_docs),
            )
            dr.add(new_docs, batch_size=batch_size, show_progress=True)
    else:
        logger.info(
            "Building dense index: %s (%d docs, model=%s, max_length=%s)",
            full_name,
            len(collection),
            model_name,
            max_len,
        )
        # Reuse cached encoder if available — pass it directly to avoid
        # redundant model loading in the DenseRetriever constructor.
        cached = _encoder_cache.get(model_name)
        if cached is not None:
            logger.info("Building index with cached encoder for %s", model_name)
            dr = DenseRetriever(
                index_name=full_name,
                model=model_name,
                normalize=True,
                use_ann=True,
                max_length=max_len,
                encoder=cached,
            )
        else:
            dr = DenseRetriever(
                index_name=full_name,
                model=model_name,
                normalize=True,
                use_ann=True,
                max_length=max_len,
            )
            if dr.encoder is not None:
                _encoder_cache[model_name] = dr.encoder
                logger.info("Cached encoder for %s", model_name)
        dr.index(collection, batch_size=batch_size, show_progress=True)

    # Fail fast if the ANN index wasn't built (e.g. autofaiss memory error).
    if dr.use_ann and (dr.ann_searcher is None or dr.ann_searcher.faiss_index is None):
        raise RuntimeError(
            f"Dense ANN index build failed for {full_name}. "
            "Delete the .retriv_indexes directory and retry, or check autofaiss logs."
        )
    return dr


def build_or_load_indexes(
    agency_year_dir: Path,
    level: str,
    collection: List[dict],
    primary_embedding_model: str,
    batch_size: int,
    overwrite: bool,
    skip_distribution: bool = False,
    claims_suffix: str = "",
):
    """Build or load BM25 + dense indexes for a directory.

    Builds three indexes:
      1. BM25 (sparse) — for distribution to users without GPUs
      2. Primary dense (primary_embedding_model) — used for matching/search
      3. Distribution dense (all-mpnet-base-v2) — for distribution

    `claims_suffix` (e.g. "_v2") segregates indices for re-extracted claim
    variants. Indices land in `.retriv_indexes{claims_suffix}/` so v1 and
    v2 indices coexist without overwriting each other.

    Returns the primary DenseRetriever (used for search).
    """
    import retriv
    from retriv import SparseRetriever

    index_base = agency_year_dir / f".retriv_indexes{claims_suffix}"
    index_base.mkdir(parents=True, exist_ok=True)
    retriv.set_base_path(str(index_base))

    dir_name = agency_year_dir.name  # e.g. "msha_2017_2018"
    index_name = f"{dir_name}_{level}{claims_suffix}"

    # 1. Sparse (BM25) — for distribution (skip in training-data mode)
    if skip_distribution:
        logger.info("Skipping BM25 index (training data mode)")
    elif not overwrite and _index_exists(index_base, f"{index_name}_bm25", "sparse"):
        logger.info("Loading existing BM25 index: %s", index_name)
        sr = SparseRetriever.load(f"{index_name}_bm25")
        # Check if the loaded BM25 index is stale (fewer docs than collection).
        if sr.doc_count < len(collection):
            existing_ids = set(sr.id_mapping.values())
            new_docs = [d for d in collection if d["id"] not in existing_ids]
            logger.info(
                "BM25 index %s_bm25 has %d docs but collection has %d (%d new). "
                "Adding new documents incrementally.",
                index_name, sr.doc_count, len(collection), len(new_docs),
            )
            sr.add(new_docs, show_progress=True)
    else:
        logger.info("Building BM25 index: %s (%d docs)", index_name, len(collection))
        sr = SparseRetriever(
            index_name=f"{index_name}_bm25",
            model="bm25",
            min_df=1,
        )
        sr.index(collection, show_progress=True)

    # 2. Primary dense index — used for matching/search
    primary_dr = _build_or_load_one_dense(
        index_base, index_name, primary_embedding_model,
        collection, batch_size, overwrite,
    )

    # 3. Distribution dense indexes — built but not used for search
    #    Free each distribution model after building to reclaim GPU memory
    #    so the primary 8B model can be used for query encoding.
    #    Skipped entirely when collecting training data (only the primary
    #    model is needed for retrieval).
    if skip_distribution:
        logger.info("Skipping distribution dense indexes (training data mode)")
        return primary_dr

    for dist_model in DISTRIBUTION_DENSE_MODELS:
        if dist_model == primary_embedding_model:
            continue  # already built above
        try:
            dist_dr = _build_or_load_one_dense(
                index_base, index_name, dist_model,
                collection, batch_size, overwrite,
            )
            # Explicitly free the distribution encoder to reclaim GPU memory.
            if dist_dr.encoder is not None:
                _encoder_cache.pop(dist_model, None)
                del dist_dr.encoder
            del dist_dr
        except Exception as e:
            logger.warning(
                "Failed to build distribution index %s for %s: %s",
                dist_model, index_name, e,
            )

    # Clear GPU cache after building distribution indexes so the primary
    # model has full GPU memory available for query encoding.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return primary_dr


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------


def retrieve_for_response(
    dr,
    response_text: str,
    docket_doc_ids: List[str],
    k: int = 10,
    encoded_query=None,
) -> List[dict]:
    """Query the primary dense index for one response, filtering to docket docs.

    Returns list of dicts with doc_id, dense_score.
    If encoded_query is provided, skip re-encoding for the dense retriever.
    """
    if not docket_doc_ids or not response_text.strip():
        return []

    results = []
    # When the docket covers (nearly) the entire index, skip include_id_list
    # to avoid retriv AssertionError on large ID lists with HNSW indices.
    n_index = getattr(dr, 'n_docs', None) or (len(dr.doc_index) if hasattr(dr, 'doc_index') else 0)
    use_id_filter = len(docket_doc_ids) < n_index * 0.95 if n_index > 0 else True
    try:
        search_kwargs = dict(
            query=response_text if encoded_query is None else None,
            encoded_query=encoded_query,
            return_docs=False,
            cutoff=k,
        )
        if use_id_filter:
            search_kwargs["include_id_list"] = docket_doc_ids
        dense_hits = dr.search(**search_kwargs)
        # retriv returns a dict {doc_id: score} when return_docs=False.
        # Older versions may have returned a list of dicts; handle both.
        if isinstance(dense_hits, dict):
            for doc_id, score in dense_hits.items():
                results.append({"doc_id": doc_id, "dense_score": float(score)})
        else:
            for hit in dense_hits:
                if isinstance(hit, dict):
                    results.append({
                        "doc_id": hit["id"],
                        "dense_score": float(hit.get("score", 0.0)),
                    })
                else:
                    # bare id with no score — fall back to 0.0 and warn once
                    results.append({"doc_id": hit, "dense_score": 0.0})
    except Exception as e:
        logger.warning("Dense search error for %d docs: %s", len(docket_doc_ids), e)

    return results


# ---------------------------------------------------------------------------
# 5. LLM accuracy check
# ---------------------------------------------------------------------------

CLAIMS_MATCHING_PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notice & comment process.
I will show you a government response and a specific claim extracted from a public comment.

Tell me whether this government response is addressing this claim:
either directly or as part of a larger group of similar comments.

IMPORTANT: In regulatory proceedings, the agency's response often discusses an issue by
summarizing arguments from BOTH supporters and opponents, then stating its conclusion.
A response "addresses" a claim if:
- The claim argues FOR the position the response discusses (agreement)
- The claim argues AGAINST the position the response discusses (the response may be rejecting it)
- The claim raises the same specific regulatory question or legal issue the response covers

A response does NOT address a claim if the claim is about a completely different regulatory topic,
even if both are in the same policy domain.

Answer with "yes" or "no". Don't say anything else.

<claim>
{claim}
</claim>

<original_comment_excerpt>
{original_comment}
</original_comment_excerpt>

<response>
{response}
</response>

Your response:
"""

COMMENT_MATCHING_PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notice & comment process.
I will show you a public comment and a government response. You will tell me whether the response is responding to this comment:
either directly as an individual comment or as part of a larger group of similar comments.
Be careful: even comments that are not being responded to are likely to be semantically similar, so really read them carefully.
Ignore any "official"-seeming correlates, like letterhead, signatures, citations of evidence in the comment.
Only look directly at the substantive content of the comment and whether the response is addressing it.
Answer with "yes" or "no". Don't say anything else.

Here are some difficult examples to calibrate your judgment:

<example_1>
<comment>The Department should not expand the program until it can see the successes and failures it has with Tier 1 and Tier 2 facilities. We urge DHS to examine the effectiveness of such screening before proceeding to subject the bulk of CFATS-regulated facilities to these additional measures.</comment>
<response>Four commenters suggested that the Department conduct further assessments on the PSP: One commenter suggested that the Department "should not expand the program until it can see the successes and failures it has with Tier 1 and Tier 2 facilities." A second commenter encouraged the Department to "examine the effectiveness of such screening before proceeding." The Department agreed to a phased implementation approach.</response>
<answer>yes</answer>
<reason>The comment's specific arguments (wait for Tier 1/2 results, examine effectiveness) are directly quoted in the response.</reason>
</example_1>

<example_2>
<comment>The International Liquid Terminals Association believes that the Department of Homeland Security has underestimated the overall burden of the Chemical Facility Anti-Terrorism Standards Personnel Surety Program by excluding certain implementation costs from its analysis.</comment>
<response>Two commenters expressed concern that the ICR did not appear to account for the burden associated with part-time or seasonal employees or contractors that qualify as affected individuals.</response>
<answer>no</answer>
<reason>Both comments are about burden underestimation in the same program, but they make different specific arguments. The comment is about excluded implementation costs; the response is about part-time/seasonal employees. Topical similarity is not enough — the response must address the specific argument.</reason>
</example_2>

<example_3>
<comment>We strongly support the proposed listing of the northern long-eared bat as endangered. White-nose syndrome poses an existential threat to this species and endangered status is necessary to prevent extinction.</comment>
<response>Several commenters expressed strong support for listing the northern long-eared bat as endangered, noting that the species faces severe threats from white-nose syndrome and that endangered status is necessary to prevent extinction. We appreciate these comments and agree that the threats warrant endangered status.</response>
<answer>yes</answer>
<reason>Supportive comments count as matches too. The response summarizes and acknowledges the commenter's position.</reason>
</example_3>

<example_4>
<comment>My dog came from our city pound and was a day away from being put down. He is a Lab-Pitt mix who had been severely abused. I cannot throw a ball to him... he thinks I'm going to hit him. He had a miserable existence.</comment>
<response>One commenter disagreed with our proposed change to business hours, stating that it is unclear what USDA means by "reasonable." The commenter considered "reasonable" to be a minimum of 30 hours a week.</response>
<answer>no</answer>
<reason>The comment is a personal story about pet adoption. The response is about business hour definitions. They are in the same docket (animal welfare) but entirely unrelated in substance.</reason>
</example_4>

<example_5>
<comment>Attached are our comments submitted in response to the proposed rule for Modernized Drawback. The attached document pertains to different references for "normal course of business" and consistency throughout 19 CFR 190.</comment>
<response>One commenter recommended that 19 CFR 113.65(a) be amended in order to establish a sunset date of February 23, 2019 for ESP bond obligations due to the move to electronic proof.</response>
<answer>no</answer>
<reason>Both are about the same regulation (Modernized Drawback), but the comment is about "normal course of business" definitions in Part 190, while the response is about ESP bond sunset dates in Part 113. Same rulemaking, different specific provisions.</reason>
</example_5>

Now evaluate this pair:

<comment>
{comment}
</comment>

<response>
{response}
</response>

Your response:
"""


def _normalize_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    raw = str(value).strip().lower()
    # Check for API error JSON — treat as unlabeled
    if raw.startswith("{") and "error" in raw:
        return ""
    # Extract yes/no from verbose responses (e.g. "Yes, this response...")
    if raw in ("yes", "no"):
        return raw
    # Check if first word is yes/no (common with verbose models)
    first_word = raw.split(",")[0].split(".")[0].split()[0] if raw else ""
    if first_word in ("yes", "no"):
        return first_word
    return raw


def sample_pairs_for_llm(all_pairs_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Stratified sample across dense_score distribution."""
    df = all_pairs_df.copy()
    if len(df) <= n:
        return df

    # Stratified sampling: 10 bins across dense_score
    df["_score_bin"] = pd.cut(df["dense_score"], bins=10, labels=False)
    df["_score_bin"] = df["_score_bin"].fillna(0).astype(int)

    sampled_parts = []
    per_bin = max(1, n // 10)
    for _, bin_df in df.groupby("_score_bin"):
        sample_n = min(len(bin_df), per_bin)
        sampled_parts.append(bin_df.sample(n=sample_n, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # If we have fewer than n, sample more from the full set
    if len(sampled) < n:
        remaining = df.loc[~df.index.isin(sampled.index)]
        extra = min(n - len(sampled), len(remaining))
        if extra > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra, random_state=42)],
                ignore_index=True,
            )

    # If we have more than n, downsample
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=42).reset_index(drop=True)

    sampled = sampled.drop(columns=["_score_bin"], errors="ignore")
    return sampled


def build_matching_prompt(
    level: str,
    response_text: str,
    candidate_text: str,
    original_comment: Optional[str] = None,
) -> str:
    if level == "claims":
        return CLAIMS_MATCHING_PROMPT.format(
            claim=candidate_text[:20000],
            original_comment=(original_comment or "")[:20000],
            response=response_text[:20000],
        )
    else:
        return COMMENT_MATCHING_PROMPT.format(
            comment=candidate_text[:20000],
            response=response_text[:20000],
        )


async def run_llm_check(
    sampled_df: pd.DataFrame,
    level: str,
    backend: str,
    model: str,
    collection_df: pd.DataFrame,
    original_texts: Optional[dict] = None,
) -> pd.DataFrame:
    """Label sampled pairs via LLM. Returns DataFrame with llm_label column."""
    prompts = []
    for _, row in sampled_df.iterrows():
        response_text = str(row.get("response_text", ""))
        candidate_text = str(row.get("candidate_text", ""))
        original_comment = None
        if level == "claims" and original_texts is not None:
            cluster_uid = str(row.get("doc_id", "")).split("::claim_")[0]
            original_comment = original_texts.get(cluster_uid, "")
        prompts.append(
            build_matching_prompt(level, response_text, candidate_text, original_comment)
        )

    logger.info(
        "Querying %d LLM prompts (model=%s, backend=%s)",
        len(prompts),
        model,
        backend,
    )
    raw_labels = await prompt_utils.process_batch(
        prompts=prompts,
        model=model,
        backend=backend,
    )
    sampled_df = sampled_df.copy()
    normalized = [_normalize_label(v) for v in raw_labels]
    sampled_df["llm_label"] = normalized

    # In debug mode, print a sample positive and negative prompt+response
    if logger.isEnabledFor(logging.DEBUG) or os.environ.get("DEBUG"):
        yes_idxs = [i for i, l in enumerate(normalized) if l == "yes"]
        no_idxs = [i for i, l in enumerate(normalized) if l == "no"]
        for tag, idxs in [("POSITIVE (yes)", yes_idxs), ("NEGATIVE (no)", no_idxs)]:
            if idxs:
                i = idxs[0]
                logger.info(
                    "Sample %s prompt:\n%s\nLLM response: %s",
                    tag, prompts[i][:2000], raw_labels[i],
                )

    # Log label distribution and sample non-yes/no responses for debugging
    from collections import Counter
    label_counts = Counter(normalized)
    logger.info("LLM label distribution: %s", dict(label_counts))
    bad_labels = [v for v in raw_labels if _normalize_label(v) not in ("yes", "no")]
    if bad_labels:
        logger.warning(
            "Sample non-yes/no LLM responses (%d total): %s",
            len(bad_labels),
            [str(v)[:200] for v in bad_labels[:5]],
        )

    return sampled_df


async def run_multi_model_llm_check(
    sampled_df: pd.DataFrame,
    level: str,
    backend: str,
    models: List[str],
    collection_df: pd.DataFrame,
    original_texts: Optional[dict] = None,
) -> pd.DataFrame:
    """Label pairs using multiple LLM models with uniform random assignment.

    Each pair is randomly assigned to one of the models. Results are merged
    back with an ``llm_model`` column recording which model labeled each pair.
    """
    if len(models) == 1:
        result = await run_llm_check(
            sampled_df, level, backend, models[0], collection_df, original_texts,
        )
        result["llm_model"] = models[0]
        return result

    # Randomly assign each row to a model (uniform)
    assignments = [random.choice(models) for _ in range(len(sampled_df))]
    sampled_df = sampled_df.copy()
    sampled_df["_assigned_model"] = assignments

    logger.info(
        "Multi-model split: %s",
        {m: assignments.count(m) for m in models},
    )

    parts = []
    for model in models:
        subset = sampled_df.loc[sampled_df["_assigned_model"] == model].copy()
        if subset.empty:
            continue
        labeled = await run_llm_check(
            subset.drop(columns=["_assigned_model"]),
            level, backend, model, collection_df, original_texts,
        )
        labeled["llm_model"] = model
        parts.append(labeled)

    result = pd.concat(parts, ignore_index=True)
    return result


def find_optimal_threshold(labeled_df: pd.DataFrame, score_col: str) -> dict:
    """Grid-search for best F1 threshold on a score column."""
    df = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    y_true = (df["llm_label"] == "yes").astype(int)
    logger.info(
        "Searching threshold for %s using %d labeled pairs", score_col, len(df)
    )

    if len(df) == 0 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {
            "f1": 0.0,
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "report": "No positive or negative samples",
        }

    thresholds = np.arange(
        max(0.0, df[score_col].min()),
        min(1.0, df[score_col].max()) + 0.01,
        0.01,
    )
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (df[score_col] >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds = (df[score_col] >= best_t).astype(int)
    report = classification_report(
        y_true, preds, target_names=["no", "yes"], zero_division=0
    )
    logger.info(
        "Threshold search complete for %s: best F1 %.3f @ %.3f",
        score_col,
        best_f1,
        best_t,
    )
    return {
        "f1": best_f1,
        "threshold": best_t,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "report": report,
    }


# ---------------------------------------------------------------------------
# 6. Training data collection & QA
# ---------------------------------------------------------------------------


def save_training_pairs(
    labeled_df: pd.DataFrame,
    agency_year_dir: Path,
    level: str,
    training_data_dir: Path,
    llm_model: str,
):
    """Append LLM-labeled pairs to a central CSV for cross-encoder training.

    Thread-safe via file lock so multiple pipeline instances can write
    concurrently.
    """
    training_data_dir.mkdir(parents=True, exist_ok=True)
    out_path = training_data_dir / "llm_labeled_pairs.csv.gz"
    lock_path = training_data_dir / "llm_labeled_pairs.csv.gz.lock"

    valid = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    if valid.empty:
        return 0

    valid["source_dir"] = str(agency_year_dir)
    valid["level"] = level
    # Prefer per-row llm_model from multi-model labeling; fall back to arg
    if "llm_model" not in valid.columns:
        valid["llm_model"] = llm_model
    valid["timestamp"] = datetime.now().isoformat()

    cols = [
        "response_text", "candidate_text", "dense_score",
        "llm_label", "agency_id", "docket_id", "doc_id",
        "source_dir", "level", "llm_model", "timestamp",
    ]
    out_df = valid[[c for c in cols if c in valid.columns]]

    with filelock.FileLock(lock_path):
        write_header = not out_path.exists()
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

    logger.info(
        "Saved %d training pairs from %s to %s",
        len(out_df), agency_year_dir.name, out_path,
    )
    return len(out_df)


async def run_qa_check(
    labeled_df: pd.DataFrame,
    level: str,
    backend: str,
    primary_model: str,
    qa_model: str,
    qa_fraction: float,
    collection_df: pd.DataFrame,
    original_texts: Optional[dict],
    training_data_dir: Path,
):
    """Re-label a subset of pairs with a stronger model for agreement analysis."""
    valid = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    if valid.empty or qa_fraction <= 0:
        return

    n_qa = max(1, int(len(valid) * qa_fraction))
    qa_sample = valid.sample(n=min(n_qa, len(valid)), random_state=42).copy()

    logger.info(
        "Running QA check: re-labeling %d pairs with %s",
        len(qa_sample), qa_model,
    )

    # Re-label with the QA model
    qa_labeled = await run_llm_check(
        qa_sample.drop(columns=["llm_label"]),
        level, backend, qa_model, collection_df, original_texts,
    )

    qa_valid = qa_labeled.loc[qa_labeled["llm_label"].isin(["yes", "no"])].copy()
    if qa_valid.empty:
        logger.warning("QA model returned no valid labels")
        return

    # Compare: primary label (from qa_sample) vs QA label (from qa_labeled)
    qa_sample = qa_sample.rename(columns={"llm_label": "primary_label"})
    merged = qa_sample[["doc_id", "response_text", "primary_label"]].merge(
        qa_valid[["doc_id", "response_text", "llm_label"]].rename(
            columns={"llm_label": "qa_label"}
        ),
        on=["doc_id", "response_text"],
        how="inner",
    )

    if merged.empty:
        logger.warning("QA merge produced 0 rows")
        return

    agreement = (merged["primary_label"] == merged["qa_label"]).mean()
    logger.info(
        "QA agreement (%s vs %s): %.1f%% on %d pairs",
        primary_model, qa_model, agreement * 100, len(merged),
    )

    # Save QA results
    training_data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = training_data_dir / "llm_labeled_pairs_gpt5_qa.csv.gz"
    lock_path = training_data_dir / "llm_labeled_pairs_gpt5_qa.csv.gz.lock"

    merged["primary_model"] = primary_model
    merged["qa_model"] = qa_model
    merged["timestamp"] = datetime.now().isoformat()

    with filelock.FileLock(lock_path):
        write_header = not qa_path.exists()
        merged.to_csv(qa_path, mode="a", header=write_header, index=False)

    # Append agreement stats
    report_path = training_data_dir / "agreement_report.json"
    report_entry = {
        "timestamp": datetime.now().isoformat(),
        "primary_model": primary_model,
        "qa_model": qa_model,
        "n_pairs": len(merged),
        "agreement": round(agreement, 4),
        "primary_yes_rate": round((merged["primary_label"] == "yes").mean(), 4),
        "qa_yes_rate": round((merged["qa_label"] == "yes").mean(), 4),
    }
    with open(report_path, "a") as f:
        f.write(json.dumps(report_entry) + "\n")

    logger.info("QA results saved to %s", qa_path)


# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------

_LOG_COLUMNS = [
    "timestamp",
    "directory",
    "level",
    "total_docs",
    "total_pairs",
    "sampled_pairs",
    "labeled_pairs",
    "label_dist_yes",
    "label_dist_no",
    "dense_f1",
    "dense_threshold",
    "dense_precision",
    "dense_recall",
    "match_rate",
    "status",
    "error",
]


def _init_log():
    if not LOG_FILE.exists():
        pd.DataFrame(columns=_LOG_COLUMNS).to_csv(LOG_FILE, index=False)


def log_result(entry: dict):
    entry["timestamp"] = datetime.now().isoformat()
    if "label_dist" in entry:
        dist = entry.pop("label_dist")
        entry["label_dist_yes"] = dist.get("yes", 0)
        entry["label_dist_no"] = dist.get("no", 0)
    row = pd.DataFrame([entry]).reindex(columns=_LOG_COLUMNS)
    row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    logger.info(json.dumps(entry, indent=2, default=str))


# ---------------------------------------------------------------------------
# 7. Output helpers
# ---------------------------------------------------------------------------


def _output_paths(agency_year_dir: Path, level: str, cross_docket: bool = False, claims_suffix: str = ""):
    cd_suffix = "_cross_docket" if cross_docket else ""
    resp_path = agency_year_dir / f"public_submission_all_text__{level}{claims_suffix}{cd_suffix}_response_matches.csv.gz"
    comment_path = agency_year_dir / f"public_submission_all_text__{level}{claims_suffix}{cd_suffix}_comment_labels.csv.gz"
    return resp_path, comment_path


def _outputs_exist(agency_year_dir: Path, level: str, cross_docket: bool = False, claims_suffix: str = "") -> bool:
    resp_path, comment_path = _output_paths(agency_year_dir, level, cross_docket, claims_suffix=claims_suffix)
    return resp_path.exists() and comment_path.exists()


def compute_cluster_stats(all_pairs_df: pd.DataFrame) -> dict:
    """Compute cluster-level match stats from pair-level results."""
    df = all_pairs_df.copy()
    df["_cluster_uid"] = df["doc_id"].str.rsplit("__", n=1).str[0]
    n_clusters = df["_cluster_uid"].nunique()
    matched_clusters = df.loc[df["final_label"] == "yes", "_cluster_uid"].nunique()
    n_responses = df["response_text"].nunique()
    matched_responses = df.loc[
        df["final_label"] == "yes", "response_text"
    ].nunique()
    return {
        "n_clusters": n_clusters,
        "matched_clusters": matched_clusters,
        "cluster_match_rate": round(matched_clusters / n_clusters, 4) if n_clusters > 0 else 0,
        "n_responses": n_responses,
        "matched_responses": matched_responses,
        "response_match_rate": round(matched_responses / n_responses, 4) if n_responses > 0 else 0,
    }


def log_match_summary(name: str, all_pairs_df: pd.DataFrame, log_entry: dict, tag: str = ""):
    """Log a standardized match summary at cluster + response level."""
    stats = compute_cluster_stats(all_pairs_df)
    log_entry.update(stats)
    n_pairs = len(all_pairs_df)
    n_matched_pairs = int((all_pairs_df["final_label"] == "yes").sum())
    prefix = f"MATCH SUMMARY{' (' + tag + ')' if tag else ''}"
    logger.info(
        "%s %s: %d/%d clusters matched (%.1f%%), "
        "%d/%d responses matched (%.1f%%), "
        "%d/%d pairs",
        prefix, name,
        stats["matched_clusters"], stats["n_clusters"], stats["cluster_match_rate"] * 100,
        stats["matched_responses"], stats["n_responses"], stats["response_match_rate"] * 100,
        n_matched_pairs, n_pairs,
    )


def save_outputs(all_pairs_df: pd.DataFrame, agency_year_dir: Path, level: str, cross_docket: bool = False, claims_suffix: str = ""):
    """Save response_matches.csv, comment_labels.csv, cluster_labels.csv, and pair_scores.csv."""
    resp_path, comment_path = _output_paths(agency_year_dir, level, cross_docket, claims_suffix=claims_suffix)

    # --- pair_scores.csv.gz --- (detailed per-pair scores for analysis)
    cd_suffix = "_cross_docket" if cross_docket else ""
    suffix = f"{claims_suffix}{cd_suffix}"
    pairs_path = agency_year_dir / f"public_submission_all_text__{level}{suffix}_pair_scores.csv.gz"
    score_cols = ["doc_id", "docket_id", "response_text", "dense_score", "final_label"]
    # Add optional score columns if present
    for col in ["claim_ce_score", "comment_ce_score", "combined_ce_score",
                 "cross_encoder_score", "comment_text", "candidate_text",
                 "crosswalk_docket"]:
        if col in all_pairs_df.columns:
            score_cols.append(col)
    # Build response_key for joining
    pairs_out = all_pairs_df[score_cols].copy()
    # agency_id may be missing when reusing existing pair_scores in incremental mode
    if "agency_id" in all_pairs_df.columns:
        agency_series = all_pairs_df["agency_id"].astype(str)
    elif "response_key" in all_pairs_df.columns:
        # Parse agency from existing response_key if available
        agency_series = all_pairs_df["response_key"].fillna("").astype(str).str.split("|").str[0]
    else:
        agency_series = pd.Series([""] * len(all_pairs_df), index=all_pairs_df.index)
    pairs_out["response_key"] = (
        agency_series
        + "|"
        + all_pairs_df["docket_id"].astype(str)
        + "|"
        + all_pairs_df["response_text"].fillna("").astype(str).str[:200]
    )
    # Extract cluster_uid from doc_id (e.g., "ams__2016_2017__docket__doc__0" -> cluster uid)
    pairs_out["claim_index"] = pairs_out["doc_id"].str.rsplit("__", n=1).str[-1]
    pairs_out.to_csv(pairs_path, index=False, compression="gzip")
    logger.info("Saved %d pair scores to %s", len(pairs_out), pairs_path)

    # Build response key
    df = all_pairs_df.copy()
    # agency_id may be missing when reusing existing pair_scores in incremental mode
    if "agency_id" not in df.columns:
        if "response_key" in df.columns:
            df["agency_id"] = df["response_key"].fillna("").astype(str).str.split("|").str[0]
        else:
            df["agency_id"] = ""
    df["response_key"] = (
        df["agency_id"].astype(str)
        + "|"
        + df["docket_id"].astype(str)
        + "|"
        + df["response_text"].fillna("").astype(str).str[:200]
    )

    # --- response_matches.csv ---
    matches = df.loc[df["final_label"] == "yes"]
    if not matches.empty:
        resp_agg = (
            matches.groupby("response_key")
            .agg(
                agency_id=("agency_id", "first"),
                docket_id=("docket_id", "first"),
                response_content=("response_text", "first"),
                matched_doc_ids=(
                    "doc_id",
                    lambda x: ";".join(sorted(set(x.astype(str)))),
                ),
                match_count=("doc_id", "nunique"),
            )
            .reset_index()
        )
    else:
        resp_agg = pd.DataFrame(
            columns=[
                "response_key",
                "agency_id",
                "docket_id",
                "response_content",
                "matched_doc_ids",
                "match_count",
            ]
        )
    resp_agg.to_csv(resp_path, index=False)
    logger.info("Saved %d response rows to %s", len(resp_agg), resp_path)

    # --- comment_labels.csv ---
    comment_agg = (
        df.groupby("doc_id")
        .agg(
            cluster_uid=("doc_id", "first"),
            docket_id=("docket_id", "first"),
            matched=(
                "final_label",
                lambda x: "yes" if (x == "yes").any() else "no",
            ),
            matched_response_keys=(
                "response_key",
                lambda x: ";".join(
                    sorted(
                        set(
                            str(v) for v in x.loc[
                                df.loc[x.index, "final_label"] == "yes"
                            ]
                            if pd.notna(v)
                        )
                    )
                )
                if (df.loc[x.index, "final_label"] == "yes").any()
                else "",
            ),
            num_responses_matched=(
                "final_label",
                lambda x: (x == "yes").sum(),
            ),
            best_dense_score=("dense_score", "max"),
            **({
                "best_cross_encoder_score": ("cross_encoder_score", "max"),
            } if "cross_encoder_score" in df.columns else {}),
        )
        .reset_index()
    )
    comment_agg.to_csv(comment_path, index=False)
    logger.info("Saved %d comment rows to %s", len(comment_agg), comment_path)

    # --- cluster_labels.csv --- (aggregate claims to comment/cluster level)
    # Extract cluster_uid from doc_id by stripping the trailing __N claim index
    df["_cluster_uid"] = df["doc_id"].str.rsplit("__", n=1).str[0]

    # Count unique matched response keys per cluster
    def _unique_matched_responses(group):
        matched_mask = df.loc[group.index, "final_label"] == "yes"
        if not matched_mask.any():
            return ""
        return ";".join(sorted(set(str(v) for v in group.loc[matched_mask] if pd.notna(v))))

    cluster_agg = (
        df.groupby("_cluster_uid")
        .agg(
            docket_id=("docket_id", "first"),
            n_claims=("doc_id", "nunique"),
            matched=(
                "final_label",
                lambda x: "yes" if (x == "yes").any() else "no",
            ),
            n_claims_matched=(
                "final_label",
                lambda x: (x == "yes").sum(),
            ),
            n_distinct_responses_matched=(
                "response_key",
                lambda x: len(set(
                    x.loc[df.loc[x.index, "final_label"] == "yes"]
                )) if (df.loc[x.index, "final_label"] == "yes").any() else 0,
            ),
            matched_response_keys=(
                "response_key",
                _unique_matched_responses,
            ),
            best_score=("dense_score", "max"),
            **({
                "best_combined_ce_score": ("combined_ce_score", "max"),
            } if "combined_ce_score" in df.columns else {}),
        )
        .reset_index()
        .rename(columns={"_cluster_uid": "cluster_uid"})
    )

    cluster_path = agency_year_dir / f"public_submission_all_text__{level}{suffix}_cluster_labels.csv.gz"
    cluster_agg.to_csv(cluster_path, index=False, compression="gzip")
    n_matched_clusters = (cluster_agg["matched"] == "yes").sum()
    logger.info(
        "Saved %d cluster rows (%d matched, %.1f%%) to %s",
        len(cluster_agg), n_matched_clusters,
        100 * n_matched_clusters / len(cluster_agg) if len(cluster_agg) > 0 else 0,
        cluster_path,
    )


# ---------------------------------------------------------------------------
# 8. Load original texts for claims-level prompts
# ---------------------------------------------------------------------------


def load_original_texts(agency_year_dir: Path) -> dict:
    """Load original comment texts keyed by cluster_uid.

    Used to provide context in claims-level LLM prompts.
    """
    mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv.gz"
    if not mapper_path.exists():
        mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv"
    all_text_path = agency_year_dir / "public_submission_all_text.csv.gz"
    if not all_text_path.exists():
        all_text_path = agency_year_dir / "public_submission_all_text.csv"

    if not mapper_path.exists() or not all_text_path.exists():
        return {}

    mapper = pd.read_csv(mapper_path, low_memory=False)
    all_text = pd.read_csv(
        all_text_path,
        usecols=["Document ID", "canonical_text"],
        low_memory=False,
    )
    mapper["document_id"] = mapper["document_id"].astype(str)
    all_text["Document ID"] = all_text["Document ID"].astype(str)

    merged = mapper.merge(
        all_text,
        left_on="document_id",
        right_on="Document ID",
        how="inner",
    )
    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    reps = (
        merged.sort_values("text_len", ascending=False)
        .groupby("cluster_uid", as_index=False)
        .head(1)
    )
    return dict(zip(reps["cluster_uid"].astype(str), reps["canonical_text"]))


# ---------------------------------------------------------------------------
# 9. Main pipeline per directory
# ---------------------------------------------------------------------------


def _all_indexes_exist(agency_year_dir: Path, level: str, primary_model: str, claims_suffix: str = "") -> bool:
    """Check if all three indexes (BM25 + primary dense + distribution dense) exist."""
    index_base = agency_year_dir / f".retriv_indexes{claims_suffix}"
    dir_name = agency_year_dir.name
    index_name = f"{dir_name}_{level}{claims_suffix}"
    # BM25
    if not _index_exists(index_base, f"{index_name}_bm25", "sparse"):
        return False
    # Primary dense
    safe_primary = primary_model.replace("/", "_")
    if not _index_exists(index_base, f"{index_name}_{safe_primary}", "dense"):
        return False
    # Distribution dense
    for dist_model in DISTRIBUTION_DENSE_MODELS:
        if dist_model == primary_model:
            continue
        safe_dist = dist_model.replace("/", "_")
        if not _index_exists(index_base, f"{index_name}_{safe_dist}", "dense"):
            return False
    return True


async def process_directory(
    agency_year_dir: Path,
    response_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Process one agency/year directory with lock file protection."""
    level = args.level
    log_entry = {"directory": str(agency_year_dir), "level": level}

    # Check if work is already done
    overwrite_index = getattr(args, "overwrite_index", False)
    overwrite_matches = getattr(args, "overwrite_matches", False)

    if not overwrite_matches:
        if args.index_only:
            # Don't skip when indexes exist — build_or_load_indexes will check
            # if they need incremental updates via .add().
            pass
        elif getattr(args, "incremental", False):
            # Incremental mode: don't skip — we'll check for new pairs inside
            # _process_directory_inner and only score novel ones.
            if _outputs_exist(agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or ""):
                logger.info("Incremental mode: re-entering %s to check for new pairs.", agency_year_dir.name)
        elif _outputs_exist(agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or ""):
            # When collecting training data, don't skip — we need to re-run LLM labeling
            if not getattr(args, "collect_training_data", False):
                logger.info("Outputs already exist for %s, skipping.", agency_year_dir.name)
                log_entry["status"] = "skipped_existing_outputs"
                log_result(log_entry)
                return
            else:
                logger.info("Outputs exist for %s but --collect-training-data is set, continuing.", agency_year_dir.name)

    # Acquire lock file
    processing_flag = agency_year_dir / ".match-processing"
    try:
        fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        logger.info("Skipping %s (another process is working on it)", agency_year_dir.name)
        log_entry["status"] = "skipped_locked"
        log_result(log_entry)
        return
    _active_processing_files.add(processing_flag)

    try:
        await _process_directory_inner(agency_year_dir, response_df, args, log_entry)
    except Exception as e:
        logger.error("Failed on %s: %s", agency_year_dir, e, exc_info=True)
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        log_result(log_entry)
    finally:
        processing_flag.unlink(missing_ok=True)
        _active_processing_files.discard(processing_flag)


async def _process_directory_inner(
    agency_year_dir: Path,
    response_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    log_entry: dict,
) -> None:
    level = args.level
    overwrite_index = getattr(args, "overwrite_index", False)

    # ── Step 1: Load collection data ──
    collection_df = build_collection(agency_year_dir, level, claims_suffix=getattr(args, "claims_suffix", "") or "")
    if collection_df is None or collection_df.empty:
        logger.warning("No data for %s at level=%s, skipping.", agency_year_dir.name, level)
        log_entry["status"] = "skipped_no_data"
        log_result(log_entry)
        return

    log_entry["total_docs"] = len(collection_df)
    logger.info(
        "Loaded %d %s-level docs for %s",
        len(collection_df),
        level,
        agency_year_dir.name,
    )

    # Build doc_id → docket_id mapping (skip NaN/empty docket IDs)
    doc_to_docket = {}
    for doc_id, docket_id in zip(
        collection_df["id"].astype(str), collection_df["docket_id"].astype(str)
    ):
        if docket_id and docket_id != "nan":
            doc_to_docket[doc_id] = docket_id
    # Build docket → list of doc IDs
    docket_to_docs: dict[str, list[str]] = {}
    for doc_id, docket_id in doc_to_docket.items():
        docket_to_docs.setdefault(docket_id, []).append(doc_id)

    # Build collection for retriv (list of {"id": ..., "text": ...})
    collection_list = collection_df[["id", "text"]].to_dict("records")

    # ── Step 2: Build/load indexes ──
    _claims_suffix = getattr(args, "claims_suffix", "") or ""
    dr = build_or_load_indexes(
        agency_year_dir,
        level,
        collection_list,
        args.primary_embedding_model,
        args.batch_size,
        overwrite_index,
        skip_distribution=getattr(args, "collect_training_data", False) or getattr(args, "skip_distribution", False),
        claims_suffix=_claims_suffix,
    )

    # Load BM25 retriever for fusion if requested
    args._bm25_retriever = None
    if getattr(args, "fusion_bm25", False):
        _fusion_index_base = agency_year_dir / f".retriv_indexes{_claims_suffix}"
        _fusion_index_name = f"{agency_year_dir.name}_{level}{_claims_suffix}"
        bm25_index_name = f"{_fusion_index_name}_bm25"
        if _index_exists(_fusion_index_base, bm25_index_name, "sparse"):
            try:
                from retriv import SparseRetriever
                args._bm25_retriever = SparseRetriever.load(bm25_index_name)
                logger.info("BM25 fusion enabled: loaded %s (%d docs)",
                            bm25_index_name, args._bm25_retriever.doc_count)
            except Exception as e:
                logger.warning("BM25 fusion: failed to load %s: %s", bm25_index_name, e)
        else:
            logger.warning("BM25 fusion: index %s not found, running without BM25", bm25_index_name)

    # If --index-only, stop after building indexes.
    if args.index_only:
        logger.info("Index-only mode: finished building indexes for %s", agency_year_dir.name)
        log_entry["status"] = "indexed"
        log_result(log_entry)
        return

    # ── Step 3: Retrieve candidates ──
    # Filter response_df to dockets present in this directory
    dir_dockets = set(docket_to_docs.keys())

    # Expand dir_dockets via crosswalk: if a rule_docket maps to a
    # proposal_docket in dir_dockets, also include that rule_docket so its
    # responses get pulled into resp_subset.
    crosswalk_map = getattr(args, "_crosswalk_map", {})
    if crosswalk_map:
        reverse_crosswalk = {v: k for k, v in crosswalk_map.items()}
        crosswalk_additions = set()
        for proposal_docket in dir_dockets:
            rule_docket = reverse_crosswalk.get(proposal_docket)
            if rule_docket and rule_docket not in dir_dockets:
                crosswalk_additions.add(rule_docket)
        if crosswalk_additions:
            logger.info(
                "Crosswalk expanded dir_dockets by %d rule dockets (was %d, now %d)",
                len(crosswalk_additions), len(dir_dockets),
                len(dir_dockets) + len(crosswalk_additions),
            )
            dir_dockets = dir_dockets | crosswalk_additions

    cross_docket = getattr(args, "cross_docket", False)
    if cross_docket:
        # Cross-docket mode: use ALL responses from the same agency
        dir_agency = agency_year_dir.parent.name  # e.g. "blm", "epa"
        resp_subset = response_df.loc[
            response_df["Agency ID"].astype(str).str.lower() == dir_agency.lower()
        ].copy()
        if resp_subset.empty:
            # Fallback: try matching agency prefix from Docket ID
            resp_subset = response_df.loc[
                response_df["Docket ID"].astype(str).str.upper().str.startswith(dir_agency.upper())
            ].copy()
        logger.info("Cross-docket mode: %d responses from agency %s (vs %d same-docket)",
                     len(resp_subset), dir_agency,
                     len(response_df.loc[response_df["Docket ID"].astype(str).isin(dir_dockets)]))
    else:
        resp_subset = response_df.loc[
            response_df["Docket ID"].astype(str).isin(dir_dockets)
        ].copy()

    if resp_subset.empty:
        logger.warning("No responses for dockets in %s", agency_year_dir.name)
        log_entry["status"] = "skipped_no_responses"
        log_result(log_entry)
        return

    # Determine effective sample size for later pair sampling.
    effective_sample_size = args.llm_sample_size
    if getattr(args, "collect_training_data", False) and getattr(args, "training_samples_per_dir", None):
        effective_sample_size = args.training_samples_per_dir

    # Build response text
    resp_subset["response_text"] = (
        resp_subset.fillna("")
        .apply(
            lambda r: (
                str(r.get("content_of_comment", ""))
                + " "
                + str(r.get("summarized_content_of_comment", ""))
            ),
            axis=1,
        )
        .str.strip()
    )

    # Build text lookup from collection
    doc_id_to_text = dict(zip(collection_df["id"].astype(str), collection_df["text"]))

    # Batch-encode unique response texts for the dense retriever upfront
    # to avoid re-encoding the 8B model per query (massive speedup).
    response_texts = resp_subset["response_text"].tolist()
    unique_texts = list(dict.fromkeys(response_texts))  # preserves order, deduplicates
    logger.info(
        "Batch-encoding %d unique response queries (%d total) for dense retrieval (%s)...",
        len(unique_texts),
        len(response_texts),
        agency_year_dir.name,
    )
    query_bs = args.query_batch_size or args.batch_size
    # Defensive: ensure max_length is capped before query encoding.
    enc_max = getattr(dr.encoder, "max_length", None)
    model_cap = _MODEL_MAX_LENGTH.get(args.primary_embedding_model)
    if model_cap and enc_max and enc_max != model_cap:
        logger.warning(
            "Encoder max_length is %s, overriding to %s before query encoding",
            enc_max, model_cap,
        )
        dr.encoder.max_length = model_cap
        dr.encoder.tokenizer_kwargs["max_length"] = model_cap
    logger.info(
        "Encoder max_length=%s, query_batch_size=%d",
        getattr(dr.encoder, "max_length", "?"), query_bs,
    )
    if args.debug and unique_texts:
        sample = unique_texts[0]
        logger.info(
            "Sample query (%d chars): %s",
            len(sample), sample[:500],
        )
    # Log device info to verify GPU usage.
    # NOTE: Encoder.model is the model NAME (str); the actual AutoModel is
    # stored as Encoder.encoder.
    if hasattr(dr.encoder, "encoder"):
        enc_model = dr.encoder.encoder  # the actual torch model
        try:
            dev = next(enc_model.parameters()).device
            logger.info("Encoder device: %s", dev)
            if str(dev) == "cpu":
                import torch
                if torch.cuda.is_available():
                    target_device = os.environ.get("ENCODER_DEVICE", "cuda")
                    logger.info("Moving encoder to %s for query encoding", target_device)
                    enc_model.to(target_device)
                    dr.encoder.device = target_device
        except StopIteration:
            logger.info("Encoder device: unknown (no parameters)")
    unique_embeddings = dr.encoder(unique_texts, batch_size=query_bs, show_progress=True)
    text_to_embedding = {t: unique_embeddings[i] for i, t in enumerate(unique_texts)}

    all_pairs = []
    skipped_no_docs = 0
    skipped_no_results = 0
    for _, resp_row in tqdm(
        resp_subset.iterrows(),
        total=len(resp_subset),
        desc=f"Retrieving ({agency_year_dir.name})",
    ):
        docket_id = str(resp_row["Docket ID"])
        docket_doc_ids = docket_to_docs.get(docket_id, [])
        # Crosswalk fallback: if no comments in this docket, check linked proposal docket
        crosswalk_docket = None
        if not docket_doc_ids and hasattr(args, "_crosswalk_map") and args._crosswalk_map:
            linked_docket = args._crosswalk_map.get(docket_id)
            if linked_docket:
                docket_doc_ids = docket_to_docs.get(linked_docket, [])
                if docket_doc_ids:
                    crosswalk_docket = linked_docket
        if not docket_doc_ids:
            skipped_no_docs += 1
            continue

        response_text = str(resp_row["response_text"])
        results = retrieve_for_response(
            dr, response_text, docket_doc_ids,
            k=args.top_k,
            encoded_query=text_to_embedding[response_text],
        )

        # BM25 fusion: retrieve from BM25 and merge via RRF
        if getattr(args, "fusion_bm25", False) and hasattr(args, "_bm25_retriever") and args._bm25_retriever is not None:
            try:
                bm25_hits = args._bm25_retriever.search(
                    query=response_text, cutoff=args.top_k * 3,
                )
                docket_set = set(docket_doc_ids)
                bm25_results = []
                if isinstance(bm25_hits, list):
                    for h in bm25_hits:
                        if isinstance(h, dict) and h.get("id") in docket_set:
                            bm25_results.append({"doc_id": h["id"], "dense_score": 0.0})
                elif isinstance(bm25_hits, dict):
                    for did, score in bm25_hits.items():
                        if did in docket_set:
                            bm25_results.append({"doc_id": did, "dense_score": 0.0})
                # Merge: keep dense scores from 8B, add BM25-only docs with score 0
                dense_ids = {r["doc_id"] for r in results}
                for br in bm25_results[:args.top_k]:
                    if br["doc_id"] not in dense_ids:
                        results.append(br)
            except Exception as e:
                logger.debug("BM25 fusion error: %s", e)

        if not results:
            skipped_no_results += 1

        for res in results:
            pair = {
                "agency_id": str(resp_row.get("Agency ID", "")),
                "docket_id": docket_id,
                "response_text": response_text,
                "doc_id": res["doc_id"],
                "candidate_text": doc_id_to_text.get(res["doc_id"], ""),
                "dense_score": res["dense_score"],
            }
            if crosswalk_docket:
                pair["crosswalk_docket"] = crosswalk_docket
            all_pairs.append(pair)

    if not all_pairs:
        logger.warning(
            "No retrieval pairs for %s (skipped_no_docs=%d, skipped_no_results=%d, total_responses=%d)",
            agency_year_dir.name, skipped_no_docs, skipped_no_results, len(resp_subset),
        )
        log_entry["status"] = "skipped_no_pairs"
        log_result(log_entry)
        return

    all_pairs_df = pd.DataFrame(all_pairs)
    log_entry["total_pairs"] = len(all_pairs_df)
    logger.info("Retrieved %d candidate pairs for %s", len(all_pairs_df), agency_year_dir.name)

    # ── Step 3.3: Incremental scoring — load existing pair scores and skip
    # already-scored pairs so reruns with higher top-k or crosswalk changes
    # only score the new pairs. ──
    _cd_suffix = "_cross_docket" if getattr(args, "cross_docket", False) else ""
    _cs_suffix = getattr(args, "claims_suffix", "") or ""
    existing_pairs_path = agency_year_dir / f"public_submission_all_text__{level}{_cs_suffix}{_cd_suffix}_pair_scores.csv.gz"
    existing_pairs_df = None
    if existing_pairs_path.exists() and not getattr(args, "overwrite_matches", False):
        try:
            existing_pairs_df = pd.read_csv(existing_pairs_path)
            # Build a set of (doc_id, response_text[:200]) keys for fast lookup
            existing_keys = set(
                zip(
                    existing_pairs_df["doc_id"].astype(str),
                    existing_pairs_df["response_text"].fillna("").astype(str).str[:200],
                )
            )
            new_keys = set(
                zip(
                    all_pairs_df["doc_id"].astype(str),
                    all_pairs_df["response_text"].fillna("").astype(str).str[:200],
                )
            )
            novel_keys = new_keys - existing_keys
            if not novel_keys:
                logger.info(
                    "Incremental scoring: all %d retrieved pairs already scored, "
                    "reusing existing scores",
                    len(all_pairs_df),
                )
                all_pairs_df = existing_pairs_df.reset_index(drop=True)
                # Determine score_col from existing data
                score_col = "dense_score"
                for col in ["combined_ce_score", "claim_ce_score", "cross_encoder_score"]:
                    if col in all_pairs_df.columns and all_pairs_df[col].notna().any():
                        score_col = col
                        break
                # Jump to final label + save
                if getattr(args, "cross_encoder_threshold", None) is not None and score_col != "dense_score":
                    ce_threshold = args.cross_encoder_threshold
                    all_pairs_df["final_label"] = np.where(
                        all_pairs_df[score_col] >= ce_threshold, "yes", "no"
                    )
                    log_entry["status"] = "completed_cross_encoder"
                    log_entry["score_col"] = score_col
                    log_match_summary(agency_year_dir.name, all_pairs_df, log_entry, tag="cached")
                    log_result(log_entry)
                    save_outputs(all_pairs_df, agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or "")
                    return
            else:
                novel_mask = pd.Series(
                    list(zip(
                        all_pairs_df["doc_id"].astype(str),
                        all_pairs_df["response_text"].fillna("").astype(str).str[:200],
                    ))
                ).isin(novel_keys)
                new_pairs_df = all_pairs_df.loc[novel_mask.values].copy()
                logger.info(
                    "Incremental scoring: %d existing pairs loaded, %d new pairs to score "
                    "(skipping %d already scored)",
                    len(existing_pairs_df), len(new_pairs_df),
                    len(all_pairs_df) - len(new_pairs_df),
                )
                all_pairs_df = new_pairs_df
        except Exception as e:
            logger.warning("Could not load existing pair scores from %s: %s", existing_pairs_path, e)
            existing_pairs_df = None

    if all_pairs_df.empty:
        logger.warning("No new pairs to score for %s", agency_year_dir.name)
        log_entry["status"] = "skipped_no_new_pairs"
        log_result(log_entry)
        return

    # Load original texts early — needed for comment CE and claims-level prompts
    original_texts = None
    if level == "claims":
        original_texts = load_original_texts(agency_year_dir)

    # ── Step 3.4: Apply silver label cache — use existing LLM labels where available ──
    silver_cache_path = _SCRIPTS_DIR / "data" / "silver_label_cache.parquet"
    if silver_cache_path.exists() and not all_pairs_df.empty:
        try:
            cache = pd.read_parquet(silver_cache_path, columns=["_cache_key", "llm_label"])
            cache_lookup = dict(zip(cache["_cache_key"], cache["llm_label"]))
            pair_keys = (
                all_pairs_df["doc_id"].astype(str)
                + "||"
                + all_pairs_df["response_text"].fillna("").astype(str).str[:200]
            )
            cached_labels = pair_keys.map(cache_lookup)
            n_cached = cached_labels.notna().sum()
            if n_cached > 0:
                all_pairs_df["silver_label"] = cached_labels
                logger.info(
                    "Silver label cache: %d/%d pairs have existing LLM labels",
                    n_cached, len(all_pairs_df),
                )
            del cache, cache_lookup
        except Exception as e:
            logger.warning("Could not load silver label cache: %s", e)

    # ── Step 3.4b: Check agency scoring config for LLM-only agencies ──
    scoring_config_path = _SCRIPTS_DIR / "data" / "agency_scoring_config.json"
    dir_agency = agency_year_dir.parent.name.lower()
    agency_scoring = None
    if scoring_config_path.exists():
        try:
            import json as _json
            with open(scoring_config_path) as _f:
                _config = _json.load(_f)
            agency_scoring = _config.get(dir_agency, _config.get("default", {}))
        except Exception:
            pass

    if agency_scoring and agency_scoring.get("method") == "llm":
        # LLM-only agency: score all uncached pairs via LLM
        llm_model = agency_scoring.get("model", "gpt-5-mini")
        # Ensure has_silver has the same index as all_pairs_df
        if "silver_label" in all_pairs_df.columns:
            has_silver = all_pairs_df["silver_label"].notna()
        else:
            has_silver = pd.Series(False, index=all_pairs_df.index)
        uncached = all_pairs_df[~has_silver]
        if len(uncached) > 0:
            logger.info(
                "LLM-only agency %s: scoring %d uncached pairs with %s",
                dir_agency, len(uncached), llm_model,
            )
            llm_labels = await run_llm_check(
                uncached, level, args.prompt_backend,
                llm_model, original_texts,
            )
            all_pairs_df.loc[~has_silver, "silver_label"] = llm_labels["llm_label"].values
        # Use silver labels as final labels
        all_pairs_df["final_label"] = all_pairs_df["silver_label"].fillna("no")
        match_rate = (all_pairs_df["final_label"] == "yes").mean()
        log_entry["status"] = "completed_llm_only"
        log_match_summary(agency_year_dir.name, all_pairs_df, log_entry, tag="LLM-only")
        log_result(log_entry)
        save_outputs(all_pairs_df, agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or "")
        return

    # ── Step 3.5: Cross-encoder reranking (if enabled) ──
    score_col = "dense_score"

    # Check for per-agency CE model
    per_agency_ce_dir = _SCRIPTS_DIR / "cross_encoder_models" / "per_agency" / dir_agency / "final"
    if per_agency_ce_dir.exists() and (per_agency_ce_dir / "model.safetensors").exists():
        # Override the global CE with per-agency CE
        if getattr(args, "cross_encoder_model", None):
            logger.info(
                "Using per-agency claim CE for %s: %s (overriding global)",
                dir_agency, per_agency_ce_dir,
            )
            args = argparse.Namespace(**vars(args))  # shallow copy
            args.cross_encoder_model = str(per_agency_ce_dir)
            # Load per-agency threshold
            threshold_file = per_agency_ce_dir / "optimal_threshold.json"
            if threshold_file.exists():
                import json as _json
                with open(threshold_file) as _f:
                    _thresh_data = _json.load(_f)
                args.cross_encoder_threshold = _thresh_data.get("threshold", 0.5)
                logger.info("  Per-agency threshold: %.3f (val_f1=%.3f)",
                            args.cross_encoder_threshold, _thresh_data.get("val_f1", 0))

    # Check for per-agency comment CE model
    per_agency_comment_ce_dir = _SCRIPTS_DIR / "cross_encoder_models" / "per_agency_comment" / dir_agency / "final"
    if per_agency_comment_ce_dir.exists() and (per_agency_comment_ce_dir / "model.safetensors").exists():
        if getattr(args, "comment_cross_encoder_model", None):
            logger.info(
                "Using per-agency comment CE for %s: %s (overriding global)",
                dir_agency, per_agency_comment_ce_dir,
            )
            args = argparse.Namespace(**vars(args))  # shallow copy if not already
            args.comment_cross_encoder_model = str(per_agency_comment_ce_dir)

    # Claim-level cross-encoder
    if getattr(args, "cross_encoder_model", None):
        from cross_encoder_utils import load_cross_encoder, rerank_pairs, load_optimal_threshold

        ce_model = load_cross_encoder(
            args.cross_encoder_model,
            max_length=args.cross_encoder_max_length,
        )
        logger.info(
            "Reranking %d pairs with claim cross-encoder (%s)",
            len(all_pairs_df), args.cross_encoder_model,
        )
        ce_scores = rerank_pairs(
            ce_model, all_pairs_df,
            batch_size=args.cross_encoder_batch_size,
        )
        all_pairs_df["claim_ce_score"] = ce_scores
        # Backward compat
        all_pairs_df["cross_encoder_score"] = ce_scores
        score_col = "claim_ce_score"

    # Comment-level cross-encoder (scores full comment text)
    comment_ce_model_path = getattr(args, "comment_cross_encoder_model", None)
    if comment_ce_model_path and original_texts:
        from cross_encoder_utils import load_cross_encoder, load_optimal_threshold

        comment_ce_model = load_cross_encoder(
            comment_ce_model_path,
            max_length=getattr(args, "comment_cross_encoder_max_length", 8192),
        )

        # Build (response, full_comment) pairs using original_texts lookup
        claim_id_to_cluster = dict(
            zip(collection_df["id"].astype(str), collection_df["cluster_uid"].astype(str))
        )
        comment_texts = []
        for _, row in all_pairs_df.iterrows():
            cluster_uid = claim_id_to_cluster.get(str(row["doc_id"]), "")
            comment_text = original_texts.get(str(cluster_uid), "")
            comment_texts.append(comment_text)
        all_pairs_df["comment_text"] = comment_texts

        # Score only rows with comment text
        has_comment = all_pairs_df["comment_text"].str.len() > 0
        if has_comment.any():
            comment_pairs_df = all_pairs_df.loc[has_comment].copy()
            comment_pairs_df["candidate_text_backup"] = comment_pairs_df["candidate_text"]
            comment_pairs_df["candidate_text"] = comment_pairs_df["comment_text"]

            logger.info(
                "Scoring %d pairs with comment cross-encoder (%s)",
                len(comment_pairs_df), comment_ce_model_path,
            )
            from cross_encoder_utils import rerank_pairs
            comment_ce_scores = rerank_pairs(
                comment_ce_model, comment_pairs_df,
                batch_size=args.cross_encoder_batch_size,
            )
            all_pairs_df.loc[has_comment, "comment_ce_score"] = comment_ce_scores
        all_pairs_df["comment_ce_score"] = all_pairs_df.get("comment_ce_score", pd.Series(0.5, index=all_pairs_df.index)).fillna(0.5)

        # Combine scores: use per-agency weights from scoring config if available
        w_comment = 0.3
        w_claim = 0.6
        if agency_scoring:
            w_claim = agency_scoring.get("w_claim", w_claim)
            w_comment = agency_scoring.get("w_comment", w_comment)
        claim_scores = all_pairs_df.get("claim_ce_score", pd.Series(0.5, index=all_pairs_df.index)).fillna(0.5)
        all_pairs_df["combined_ce_score"] = (
            w_comment * all_pairs_df["comment_ce_score"]
            + w_claim * claim_scores
        )
        score_col = "combined_ce_score"
        logger.info("Combined CE scores: w_comment=%.1f, w_claim=%.1f (agency=%s)", w_comment, w_claim, dir_agency)

    # ── Step 3.6: Merge new scores with existing scores ──
    if existing_pairs_df is not None and not existing_pairs_df.empty:
        logger.info(
            "Merging %d newly scored pairs with %d existing pairs",
            len(all_pairs_df), len(existing_pairs_df),
        )
        all_pairs_df = pd.concat([existing_pairs_df, all_pairs_df], ignore_index=True)
        # Deduplicate — keep the newly scored version if overlap
        dedup_key = (
            all_pairs_df["doc_id"].astype(str)
            + "||"
            + all_pairs_df["response_text"].fillna("").astype(str).str[:200]
        )
        all_pairs_df = all_pairs_df.loc[~dedup_key.duplicated(keep="last")].reset_index(drop=True)
        logger.info("After merge+dedup: %d total pairs", len(all_pairs_df))
        # Re-determine score_col from merged data
        for col in ["combined_ce_score", "claim_ce_score", "cross_encoder_score"]:
            if col in all_pairs_df.columns and all_pairs_df[col].notna().any():
                score_col = col
                break

    # Fast-path: if a fixed threshold is given, skip LLM labeling entirely
    if getattr(args, "cross_encoder_threshold", None) is not None and score_col != "dense_score":
        # Use per-agency threshold from config if available, otherwise args
        ce_threshold = args.cross_encoder_threshold
        if agency_scoring and "threshold" in agency_scoring:
            ce_threshold = agency_scoring["threshold"]
        logger.info(
            "Applying threshold %.3f on %s (agency=%s)",
            ce_threshold, score_col, dir_agency,
        )
        all_pairs_df["final_label"] = np.where(
            all_pairs_df[score_col] >= ce_threshold, "yes", "no"
        )
        match_rate = (all_pairs_df["final_label"] == "yes").mean()
        n_matched = int((all_pairs_df["final_label"] == "yes").sum())
        n_total_pairs = len(all_pairs_df)
        log_entry["cross_encoder_threshold"] = ce_threshold
        log_entry["score_col"] = score_col
        log_entry["status"] = "completed_cross_encoder"
        log_match_summary(agency_year_dir.name, all_pairs_df, log_entry, tag="CE")
        log_result(log_entry)
        save_outputs(all_pairs_df, agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or "")
        return

    # ── Step 4: LLM accuracy check ──
    # When collecting training data, optionally use a smaller per-dir budget
    effective_sample_size = args.llm_sample_size
    if getattr(args, "collect_training_data", False) and getattr(args, "training_samples_per_dir", None):
        effective_sample_size = args.training_samples_per_dir
        logger.info(
            "Training data mode: sampling %d pairs (--training-samples-per-dir) instead of %d",
            effective_sample_size, args.llm_sample_size,
        )

    sampled = sample_pairs_for_llm(all_pairs_df, n=effective_sample_size)
    log_entry["sampled_pairs"] = len(sampled)

    # ── Comment-level training: pivot from claims to full comments ──
    comment_level_training = (
        level == "claims"
        and getattr(args, "comment_level_training", False)
        and getattr(args, "collect_training_data", False)
        and original_texts
    )

    if comment_level_training:
        # Pivot: replace claim text with full comment text, dedup by (response, comment)
        logger.info("Comment-level training: pivoting %d claim pairs to comment pairs", len(sampled))

        # Build mapping from claim doc_id -> cluster_uid using collection_df
        claim_id_to_cluster = dict(
            zip(collection_df["id"].astype(str), collection_df["cluster_uid"].astype(str))
        )

        sampled = sampled.copy()
        sampled["comment_uid"] = sampled["doc_id"].map(claim_id_to_cluster)
        sampled["full_comment_text"] = sampled["comment_uid"].map(
            lambda uid: original_texts.get(str(uid), "") if pd.notna(uid) else ""
        )

        # Drop pairs where we couldn't find the full comment
        before = len(sampled)
        sampled = sampled[sampled["full_comment_text"].str.len() > 0]
        if len(sampled) < before:
            logger.info("Dropped %d pairs with missing comment text", before - len(sampled))

        # Deduplicate: keep one pair per (response_text, comment_uid), take highest dense_score
        sampled = (
            sampled.sort_values("dense_score", ascending=False)
            .drop_duplicates(subset=["response_text", "comment_uid"], keep="first")
            .reset_index(drop=True)
        )
        logger.info("After dedup: %d unique (response, comment) pairs", len(sampled))

        # Replace candidate_text with full comment for LLM labeling
        sampled["candidate_text"] = sampled["full_comment_text"]

        # Use comment-level prompt for LLM labeling
        llm_level = "comment"
    else:
        llm_level = level

    # args.llm_model is now a list of models; use multi-model labeling
    labeled = await run_multi_model_llm_check(
        sampled,
        llm_level,
        args.prompt_backend,
        args.llm_model,  # list of model names
        collection_df,
        original_texts if not comment_level_training else None,
    )
    labeled_valid = labeled.loc[labeled["llm_label"].isin(["yes", "no"])]
    log_entry["labeled_pairs"] = len(labeled_valid)
    log_entry["label_dist"] = labeled_valid["llm_label"].value_counts().to_dict()
    if "llm_model" in labeled_valid.columns:
        log_entry["model_dist"] = labeled_valid["llm_model"].value_counts().to_dict()

    # ── Step 4b: Save training data & QA (if enabled) ──
    if getattr(args, "collect_training_data", False):
        training_data_dir = Path(
            args.training_data_dir or (_SCRIPTS_DIR / "training_data")
        )
        save_level = "comment" if comment_level_training else level
        # llm_model column is already set per-row by run_multi_model_llm_check
        save_training_pairs(
            labeled, agency_year_dir, save_level, training_data_dir,
            llm_model=", ".join(args.llm_model),
        )
        if getattr(args, "qa_model", None):
            await run_qa_check(
                labeled, level, args.prompt_backend,
                args.llm_model[0], args.qa_model, args.qa_sample_fraction,
                collection_df, original_texts, training_data_dir,
            )

    # ── Step 5: Find optimal threshold ──
    # Always compute dense_score threshold for comparison
    dense_eval = find_optimal_threshold(labeled, "dense_score")
    log_entry["dense_f1"] = dense_eval["f1"]
    log_entry["dense_threshold"] = dense_eval["threshold"]
    log_entry["dense_precision"] = dense_eval["precision"]
    log_entry["dense_recall"] = dense_eval["recall"]

    logger.info(
        "Dense F1: %.3f @ %.3f\n%s",
        dense_eval["f1"],
        dense_eval["threshold"],
        dense_eval["report"],
    )

    # If cross-encoder is active, also compute threshold on cross-encoder scores
    if score_col == "cross_encoder_score":
        ce_eval = find_optimal_threshold(labeled, "cross_encoder_score")
        log_entry["ce_f1"] = ce_eval["f1"]
        log_entry["ce_threshold"] = ce_eval["threshold"]
        log_entry["ce_precision"] = ce_eval["precision"]
        log_entry["ce_recall"] = ce_eval["recall"]
        logger.info(
            "Cross-encoder F1: %.3f @ %.3f\n%s",
            ce_eval["f1"],
            ce_eval["threshold"],
            ce_eval["report"],
        )
        threshold_eval = ce_eval
    else:
        threshold_eval = dense_eval

    # ── Step 6: Apply threshold and save ──
    all_pairs_df["final_label"] = np.where(
        all_pairs_df[score_col] >= threshold_eval["threshold"], "yes", "no"
    )
    match_rate = (all_pairs_df["final_label"] == "yes").mean()
    log_entry["match_rate"] = round(match_rate, 4)
    log_entry["status"] = "completed"
    log_result(log_entry)

    save_outputs(all_pairs_df, agency_year_dir, level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or "")


# ---------------------------------------------------------------------------
# 10. Entry point
# ---------------------------------------------------------------------------


def iter_agency_year_dirs(base_dir: Path):
    """Discover all agency/year directories."""
    dirs = []
    for agency_dir in sorted(base_dir.iterdir()):
        if not agency_dir.is_dir() or agency_dir.name == "scripts":
            continue
        for year_dir in sorted(agency_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            dirs.append(year_dir)
    return dirs


async def run_pipeline(args: argparse.Namespace):
    _init_log()

    # Only load the response cache when we actually need it for matching.
    response_df = None
    if not args.index_only:
        response_df = load_response_df()

    dirs = iter_agency_year_dirs(BULK_DIR)
    if not dirs:
        logger.warning("No agency/year directories found under %s", BULK_DIR)
        return

    if args.dir_filter:
        filter_path = Path(args.dir_filter)
        allowed = {line.strip() for line in filter_path.read_text().splitlines() if line.strip()}
        dirs = [d for d in dirs if str(d.relative_to(BULK_DIR)) in allowed]
        logger.info("--dir-filter: %d directories selected from %s", len(dirs), filter_path)

    if args.debug:
        # Pick one directory with ~1000 claims for quick testing.
        # When not index-only, also require matching responses.
        response_dockets = (
            set(response_df["Docket ID"].astype(str)) if response_df is not None else None
        )
        best_dir, best_count = None, None
        best_has_indexes = False
        target = 1000
        for d in dirs:
            # Skip directories that already have outputs
            if not args.overwrite and _outputs_exist(d, args.level, getattr(args, "cross_docket", False), claims_suffix=getattr(args, "claims_suffix", "") or ""):
                continue
            _cs = getattr(args, "claims_suffix", "") or ""
            claims_path = d / f"public_submission_all_text__claims{_cs}.csv.gz"
            if not claims_path.exists():
                claims_path = d / f"public_submission_all_text__claims{_cs}.csv"
            if not claims_path.exists():
                continue
            collection_df = build_collection(d, args.level, claims_suffix=getattr(args, "claims_suffix", "") or "")
            if collection_df is None or collection_df.empty:
                continue
            # When doing full matching, require responses for this directory.
            # In cross-docket mode, skip this check (we match across dockets).
            if response_dockets is not None and not getattr(args, "cross_docket", False):
                dir_dockets = set(
                    collection_df["docket_id"].astype(str).replace("nan", pd.NA).dropna()
                )
                if not dir_dockets & response_dockets:
                    logger.debug("Debug: %s has no matching responses, skipping", d.name)
                    continue
            n = len(collection_df)
            has_indexes = _all_indexes_exist(d, args.level, args.primary_embedding_model)
            # Prefer dirs with existing indexes to avoid rebuilding
            if best_dir is None:
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            elif has_indexes and not best_has_indexes:
                # Always prefer a dir with indexes over one without
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            elif has_indexes == best_has_indexes and abs(n - target) < abs(best_count - target):
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            if has_indexes and 500 <= n <= 2000:
                break
        if best_dir is None:
            logger.error("Debug mode: no directories with data found.")
            return
        dirs = [best_dir]
        logger.info("Debug mode: selected %s (%d docs)", best_dir.name, best_count)

    # Filter to specific agencies if requested
    if getattr(args, "agency", None):
        agency_set = set(a.lower() for a in args.agency)
        dirs = [d for d in dirs if d.parent.name.lower() in agency_set]
        logger.info("Filtered to %d directories for agencies: %s", len(dirs), sorted(agency_set))

    if args.dir_order == "shuffle":
        random.shuffle(dirs)
    elif args.dir_order == "name":
        dirs.sort(key=lambda p: str(p))

    logger.info(
        "Processing %d directories (level=%s, order=%s, index_only=%s, pid=%d)",
        len(dirs),
        args.level,
        args.dir_order,
        args.index_only,
        os.getpid(),
    )

    for agency_year_dir in tqdm(dirs, desc="Directories"):
        try:
            await process_directory(agency_year_dir, response_df, args)
        except Exception as e:
            logger.error("Failed on %s: %s", agency_year_dir, e, exc_info=True)
            log_result({
                "directory": str(agency_year_dir),
                "level": args.level,
                "status": "error",
                "error": str(e),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieval-based comment matching pipeline."
    )
    parser.add_argument(
        "--level",
        choices=["claims", "comment"],
        required=True,
        help="Match at claim level or comment level.",
    )
    parser.add_argument(
        "--primary-embedding-model",
        default="nvidia/llama-embed-nemotron-8b",
        help="Primary dense model used for matching/search (default: nvidia/llama-embed-nemotron-8b).",
    )
    parser.add_argument(
        "--prompt-backend",
        choices=["openai", "vllm", "vllm-offline"],
        default="openai",
        help="LLM backend for accuracy check (default: openai).",
    )
    parser.add_argument(
        "--llm-model",
        nargs="+",
        default=["gpt-5-mini"],
        help="LLM model(s) for accuracy check (default: gpt-5-mini). "
             "Multiple models are split uniformly at random per pair.",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=1000,
        help="Number of pairs to LLM-label per directory (default: 1000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Candidates per response from each retriever (default: 10).",
    )
    parser.add_argument(
        "--fusion-bm25",
        action="store_true",
        help="Fuse BM25 retrieval with the primary dense retriever. "
             "For each response, retrieves top-k from both 8B dense and BM25, "
             "takes the union (via reciprocal rank fusion), and scores all. "
             "Requires BM25 index to exist (built during indexing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding during indexing (default: 32).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=None,
        help="Batch size for encoding response queries (default: same as --batch-size). "
             "Use a smaller value if response texts are long and cause OOM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild indexes AND overwrite existing match outputs. Equivalent to --overwrite-index --overwrite-matches.",
    )
    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help="Rebuild indexes from scratch (but skip matching if outputs exist).",
    )
    parser.add_argument(
        "--overwrite-matches",
        action="store_true",
        help="Re-run matching even if output files exist (but reuse existing indexes).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Re-enter completed directories to check for new candidate pairs "
             "(e.g., from higher --top-k or crosswalk expansion). Only scores "
             "novel pairs not already in pair_scores.csv.gz, then merges.",
    )
    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Skip building distribution dense indexes (all-mpnet-base-v2). "
             "Useful when these cause CUDA errors.",
    )
    parser.add_argument(
        "--dir-order",
        choices=["shuffle", "name"],
        default="shuffle",
        help="Processing order for directories (default: shuffle).",
    )
    parser.add_argument(
        "--agency",
        nargs="*",
        default=None,
        help="Only process directories for these agencies (e.g., --agency epa fda).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build indexes, skip retrieval and matching.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: pick one directory with ~1000 claims and process only that.",
    )

    # ── Training data collection flags ──
    parser.add_argument(
        "--collect-training-data",
        action="store_true",
        help="Save all LLM-labeled pairs to a central CSV for cross-encoder training.",
    )
    parser.add_argument(
        "--training-samples-per-dir",
        type=int,
        default=None,
        help="When --collect-training-data is set, limit LLM-labeled pairs per directory "
             "to this value (overrides --llm-sample-size) so we spread budget across more "
             "agencies. Default: use --llm-sample-size.",
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help="Directory for centralized training data (default: scripts/training_data/).",
    )
    parser.add_argument(
        "--comment-level-training",
        action="store_true",
        help="When used with --level claims --collect-training-data, retrieves "
             "at claim level (using existing indexes) but pivots to full comment "
             "text for LLM labeling and training data output. Produces "
             "(response, full_comment) pairs labeled at the comment level.",
    )
    parser.add_argument(
        "--qa-model",
        type=str,
        default=None,
        help="Secondary LLM model for quality assurance agreement checks (e.g. gpt-5). "
             "When set, a random subset of pairs is re-labeled with this model.",
    )
    parser.add_argument(
        "--qa-sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled pairs to re-label with --qa-model (default: 0.1).",
    )

    # ── Cross-encoder reranking flags ──
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default=None,
        help="Path to trained cross-encoder model. When set, reranks bi-encoder "
             "candidates and uses cross-encoder scores for thresholding.",
    )
    parser.add_argument(
        "--cross-encoder-threshold",
        type=float,
        default=None,
        help="When set with --cross-encoder-model, skip LLM labeling and apply "
             "this threshold directly (production mode, no LLM cost).",
    )
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=64,
        help="Batch size for cross-encoder scoring (default: 64).",
    )
    parser.add_argument(
        "--cross-encoder-max-length",
        type=int,
        default=4096,
        help="Max sequence length for claim cross-encoder (default: 4096).",
    )
    parser.add_argument(
        "--comment-cross-encoder-model",
        type=str,
        default=None,
        help="Path to trained comment-level cross-encoder model. When set alongside "
             "--cross-encoder-model, uses weighted combination of both scores.",
    )
    parser.add_argument(
        "--comment-cross-encoder-max-length",
        type=int,
        default=8192,
        help="Max sequence length for comment cross-encoder (default: 8192).",
    )
    parser.add_argument(
        "--cross-docket",
        action="store_true",
        help="Match responses against comments across docket boundaries within each directory. "
             "Saves to a separate output file (*_cross_docket_response_matches.csv.gz).",
    )
    parser.add_argument(
        "--claims-suffix",
        type=str,
        default="",
        help="Read __claims{suffix}.csv.gz instead of __claims.csv.gz, and segregate "
             "all derived artifacts (indices, output CSVs) under the same suffix. "
             "Use '_v2' to operate on re-extracted claims without overwriting v1 indices.",
    )
    parser.add_argument(
        "--crosswalk",
        type=str,
        default=None,
        help="Path to rule_proposal_crosswalk CSV. When a response's docket has no comments, "
             "looks up the linked proposal docket and retrieves from there instead.",
    )
    parser.add_argument(
        "--dir-filter",
        type=str,
        default=None,
        help="Path to a text file listing directory names (one per line, e.g. 'epa/epa_2020_2021') "
             "to process. Only these directories will be processed; all others are skipped.",
    )

    args = parser.parse_args()

    # Normalize overwrite flags: --overwrite sets both index and matches
    if args.overwrite:
        args.overwrite_index = True
        args.overwrite_matches = True

    # Load crosswalk if provided
    if args.crosswalk:
        _crosswalk_path = Path(args.crosswalk)
        if _crosswalk_path.exists():
            _xwalk_df = pd.read_csv(_crosswalk_path)
            # Build rule_docket → proposal_docket mapping
            args._crosswalk_map = dict(zip(
                _xwalk_df["rule_docket"].astype(str),
                _xwalk_df["proposal_docket"].astype(str),
            ))
            logger.info("Loaded crosswalk: %d docket mappings from %s", len(args._crosswalk_map), args.crosswalk)
        else:
            logger.warning("Crosswalk file not found: %s", args.crosswalk)
            args._crosswalk_map = {}
    else:
        args._crosswalk_map = {}

    # Register signal handlers for cleanup
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _signal_handler)

    asyncio.run(run_pipeline(args))
