"""Match rules to proposed rules/notices via bi-encoder retrieval + cross-encoder reranking.

Builds two retriv indexes (rules and proposed rules/notices), generates training
data from same-docket pairs, trains a ModernBERT cross-encoder, and produces a
crosswalk linking orphan rule dockets to their comment dockets.

Phases:
  build-indexes      Build retriv dense indexes for rules AND proposed rules/notices
  gen-training       Generate training pairs from same-docket rule/proposed pairs
  train-ce           Train a ModernBERT cross-encoder on generated pairs
  apply              Retrieve + rerank orphan rules → produce crosswalk
  all                Run all phases end-to-end

Usage:
    python match_rules_to_proposals.py all --gpu 5
    python match_rules_to_proposals.py build-indexes --gpu 5
    python match_rules_to_proposals.py gen-training
    python match_rules_to_proposals.py train-ce --gpu 5
    python match_rules_to_proposals.py apply --gpu 5
"""

from __future__ import annotations

# Parse --gpu EARLY before any torch/cuda imports to set CUDA_VISIBLE_DEVICES
import argparse
import os
import sys
from pathlib import Path

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("phase", nargs="?", default="all")
_pre_parser.add_argument("--gpu", type=int, default=0)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")
# Fix ir_datasets home to avoid AFS permission issues
_script_dir = Path(__file__).resolve().parent
_ir_home = str(_script_dir / "data" / ".ir_datasets")
os.makedirs(_ir_home, exist_ok=True)
os.environ["IR_DATASETS_HOME"] = _ir_home
# HF cache: use existing shared cache or local fallback
_hf_shared = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_hf_local = str(_script_dir.parent.parent.parent.parent / ".cache" / "huggingface")
os.environ.setdefault("HF_HOME", _hf_shared if Path(_hf_shared).exists() else _hf_local)
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("HF_MODULES_CACHE", str(_script_dir / "data" / ".hf_modules"))
os.makedirs(os.environ.get("HF_MODULES_CACHE", ""), exist_ok=True)

import json
import logging
import random
from collections import defaultdict

import numpy as np
import pandas as pd

# Monkey-patch json.JSONEncoder so autofaiss can serialize numpy scalars.
_orig_json_default = json.JSONEncoder.default

def _numpy_json_default(self, obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _orig_json_default(self, obj)

json.JSONEncoder.default = _numpy_json_default  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BULK_DIR = SCRIPTS_DIR.parent
DATA_DIR = SCRIPTS_DIR / "data"
INDEX_DIR = DATA_DIR / "rule_matching_indexes"

EMBEDDING_MODEL = "nvidia/llama-embed-nemotron-8b"
MAX_TEXT_TOKENS = 1024
CE_MAX_LENGTH = 2048  # 1024 tokens per doc × 2

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def clean_federal_register_text(text: str) -> str:
    """Strip boilerplate from canonical_text (<<COMMENT>>, FR headers, dates, etc.)."""
    import re
    if not text or text == "nan":
        return ""
    text = str(text)
    # Remove <<COMMENT N>> markers
    text = re.sub(r"<<COMMENT \d+>>\s*", "", text)
    # Remove ISO dates like 2020-07-24T04:00Z
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z?\s*", "", text)
    # Remove standalone true/false
    text = re.sub(r"(?m)^\s*(true|false)\s*$", "", text)
    # Remove standalone numbers (like "0")
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove Federal Register header lines
    text = re.sub(
        r"Federal Register,?\s*Volume\s+\d+\s+Issue\s+\d+\s*\([^)]+\)\s*",
        "", text,
    )
    text = re.sub(r"\[Federal Register Volume \d+, Number \d+ \([^)]+\)\]", "", text)
    # Remove [FR Doc. ...] citations
    text = re.sub(r"\[FR Doc\.[^\]]*\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_structured_text(row) -> str:
    """Extract Title + SUMMARY + ACTION from a document row.

    Returns a short, discriminative text string for embedding/CE input.
    Falls back to title-only if SUMMARY can't be parsed.
    """
    import re as _re
    title = str(row.get("Title", "")).strip()
    raw_text = str(row.get("canonical_text", row.get("_text", "")))
    # Clean boilerplate for parsing
    cleaned = _re.sub(r"<<COMMENT \d+>>\s*", "", raw_text)
    cleaned = _re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z?\s*", "", cleaned)
    cleaned = _re.sub(r"(?m)^\s*(true|false)\s*$", "", cleaned)
    cleaned = _re.sub(r"(?m)^\s*\d+\s*$", "", cleaned)

    # Extract SUMMARY
    m = _re.search(
        r"SUMMARY:\s*(.*?)(?=DATES:|FOR FURTHER|ADDRESSES:|SUPPLEMENTARY)",
        cleaned, _re.DOTALL | _re.IGNORECASE,
    )
    summary = m.group(1).strip() if m else ""

    # OCR garbage check
    if summary:
        words = summary.split()[:30]
        if len(words) > 5 and sum(1 for w in words if len(w) <= 2) / len(words) > 0.4:
            summary = ""

    # Extract ACTION
    a = _re.search(
        r"ACTION:\s*(.*?)(?=SUMMARY:|DATES:|\n\n)",
        cleaned, _re.DOTALL | _re.IGNORECASE,
    )
    action = a.group(1).strip() if a else ""

    # Fallback: if no SUMMARY parsed, grab text from char ~700 onward
    # (where SUMMARY typically starts in FR documents) for up to ~2000 chars
    if not summary and len(cleaned) > 700:
        fallback = cleaned[700:2700].strip()
        # Clean up partial lines at start/end
        if "\n" in fallback:
            fallback = fallback[fallback.index("\n"):].strip()
        summary = fallback

    # Cap summary length to avoid OOM in CE reranking
    if summary and len(summary) > 2000:
        summary = summary[:2000]

    # Build structured text: title first
    parts = [f"Title: {title}"]
    if summary:
        parts.append(f"SUMMARY: {summary}")
    if action:
        parts.append(f"ACTION: {action}")
    return "\n".join(parts)


def truncate_to_tokens(text: str, max_tokens: int = MAX_TEXT_TOKENS) -> str:
    """Truncate text to exactly max_tokens using the embedding model tokenizer."""
    if not text or text == "nan":
        return ""
    tok = _get_tokenizer()
    ids = tok.encode(str(text), add_special_tokens=False, truncation=False)
    if len(ids) <= max_tokens:
        return str(text)
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


def load_all_docs():
    """Load metadata + text from all _all_text CSV files for rules, proposed rules, notices."""
    doc_types = ("rule", "proposed_rule", "notice")
    records = []
    for dt in doc_types:
        for f in sorted(BULK_DIR.rglob(f"{dt}_all_text.csv*")):
            if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
                try:
                    df = pd.read_csv(f, low_memory=False)
                    df["doc_type"] = dt
                    df["source_file"] = str(f)
                    records.append(df)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", f, e)
    if not records:
        raise RuntimeError("No documents found")
    all_docs = pd.concat(records, ignore_index=True)

    # Identify the text column
    text_col = None
    for candidate in ["canonical_text", "text", "Text", "Abstract", "Summary"]:
        if candidate in all_docs.columns:
            text_col = candidate
            break
    if text_col is None:
        logger.warning("No text column found! Columns: %s", list(all_docs.columns))
        all_docs["_text"] = all_docs["Title"].fillna("")
    else:
        all_docs["_text"] = all_docs[text_col].fillna("").apply(clean_federal_register_text)
    logger.info(
        "Loaded %d documents (%s). Text column: %s",
        len(all_docs),
        ", ".join(f"{dt}={len(all_docs[all_docs['doc_type']==dt])}" for dt in doc_types),
        text_col or "Title (fallback)",
    )
    return all_docs


def load_comment_dockets():
    """Get the set of dockets that have public comments."""
    dockets = set()
    for f in sorted(BULK_DIR.rglob("public_submission_all_text.csv*")):
        if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
            try:
                df = pd.read_csv(f, usecols=["Docket ID"])
                dockets.update(df["Docket ID"].dropna().astype(str).unique())
            except Exception:
                pass
    logger.info("Found %d dockets with comments", len(dockets))
    return dockets


def prepare_collection(docs_df: pd.DataFrame, use_structured: bool = True) -> list[dict]:
    """Convert a DataFrame to a retriv collection: list of {"id": ..., "text": ...}."""
    collection = []
    for _, row in docs_df.iterrows():
        doc_id = str(row.get("Document ID", ""))
        if not doc_id or doc_id == "nan":
            continue
        if use_structured:
            text = extract_structured_text(row)
        else:
            text = truncate_to_tokens(str(row.get("_text", "")), MAX_TEXT_TOKENS)
        if not text:
            continue
        collection.append({"id": doc_id, "text": text})
    logger.info("Prepared collection: %d documents", len(collection))
    return collection


# ---------------------------------------------------------------------------
# Phase 1+2: Build per-agency indexes and generate training data
# ---------------------------------------------------------------------------

_encoder_cache = {}


def _load_agency_docs(agency: str):
    """Load docs for a single agency from its CSV files. Much faster than loading all."""
    doc_types = ("rule", "proposed_rule", "notice")
    records = []
    # Agency dirs are like: bulk_downloads/epa/epa_2020_2021/
    agency_lower = agency.lower()
    for agency_parent in BULK_DIR.iterdir():
        if not agency_parent.is_dir():
            continue
        if agency_parent.name.lower() != agency_lower:
            continue
        for dt in doc_types:
            for f in sorted(agency_parent.rglob(f"{dt}_all_text.csv*")):
                if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
                    try:
                        df = pd.read_csv(f, low_memory=False)
                        df["doc_type"] = dt
                        records.append(df)
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", f, e)
    if not records:
        # Try matching by Agency ID column across all dirs
        for dt in doc_types:
            for f in sorted(BULK_DIR.rglob(f"{dt}_all_text.csv*")):
                if f.suffix == ".gz" or (f.suffix == ".csv" and not f.with_suffix(".csv.gz").exists()):
                    try:
                        df = pd.read_csv(f, low_memory=False)
                        if "Agency ID" in df.columns:
                            agency_rows = df[df["Agency ID"] == agency]
                            if not agency_rows.empty:
                                agency_rows = agency_rows.copy()
                                agency_rows["doc_type"] = dt
                                records.append(agency_rows)
                    except Exception:
                        pass
    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)
    # Identify and clean text column
    text_col = None
    for candidate in ["canonical_text", "text", "Text", "Abstract", "Summary"]:
        if candidate in result.columns:
            text_col = candidate
            break
    if text_col:
        result["_text"] = result[text_col].fillna("").apply(clean_federal_register_text)
    else:
        result["_text"] = result.get("Title", pd.Series(dtype=str)).fillna("")
    return result


def _build_agency_indexes(
    agency: str,
    agency_docs: pd.DataFrame | None,
    batch_size: int,
    overwrite: bool,
):
    """Build retriv indexes (rules + proposals) for one agency. GPU-bound, parallelizable.

    If agency_docs is None, loads data lazily for just this agency.
    """
    import filelock
    import retriv
    from retriv import DenseRetriever

    agency_dir = INDEX_DIR / agency
    agency_dir.mkdir(parents=True, exist_ok=True)
    done_marker = agency_dir / ".indexes_summary_done"
    lock_path = agency_dir / ".processing_summary"

    if done_marker.exists() and not overwrite:
        logger.info("[%s] Summary indexes already built, skipping", agency)
        return True

    try:
        lock = filelock.FileLock(str(lock_path), timeout=0)
        lock.acquire()
    except filelock.Timeout:
        logger.info("[%s] Locked by another process, skipping", agency)
        return False

    try:
        # Lazy-load agency docs if not provided
        if agency_docs is None:
            agency_docs = _load_agency_docs(agency)
            if agency_docs.empty:
                logger.info("[%s] No docs found, skipping", agency)
                done_marker.touch()
                return True

        retriv.set_base_path(str(agency_dir))

        rules = agency_docs[agency_docs["doc_type"] == "rule"].drop_duplicates(subset=["Document ID"])
        proposals = agency_docs[
            agency_docs["doc_type"].isin(["proposed_rule", "notice"])
        ].drop_duplicates(subset=["Document ID"])

        if rules.empty or proposals.empty:
            logger.info("[%s] No rules (%d) or proposals (%d), skipping", agency, len(rules), len(proposals))
            done_marker.touch()
            return True

        rules_collection = prepare_collection(rules, use_structured=True)
        if not rules_collection:
            done_marker.touch()
            return True

        # Build rules_summary index
        cached_encoder = _encoder_cache.get(EMBEDDING_MODEL)
        dr_rules = DenseRetriever(
            index_name="rules_summary",
            model=EMBEDDING_MODEL,
            normalize=True,
            use_ann=True,
            max_length=MAX_TEXT_TOKENS,
            **({"encoder": cached_encoder} if cached_encoder else {}),
        )
        dr_rules.index(rules_collection, batch_size=batch_size, show_progress=False)
        if dr_rules.encoder is not None and EMBEDDING_MODEL not in _encoder_cache:
            _encoder_cache[EMBEDDING_MODEL] = dr_rules.encoder
        logger.info("[%s] Rules summary index: %d docs", agency, len(rules_collection))

        # Build proposals_summary index
        proposals_collection = prepare_collection(proposals, use_structured=True)
        if proposals_collection:
            dr_proposals = DenseRetriever(
                index_name="proposals_summary",
                model=EMBEDDING_MODEL,
                normalize=True,
                use_ann=True,
                max_length=MAX_TEXT_TOKENS,
                encoder=dr_rules.encoder,
            )
            dr_proposals.index(proposals_collection, batch_size=batch_size, show_progress=False)
            logger.info("[%s] Proposals summary index: %d docs", agency, len(proposals_collection))

        done_marker.touch()
        return True
    finally:
        lock.release()
        lock_path.unlink(missing_ok=True)


def _get_all_agency_ids():
    """Discover agency IDs from directory names without loading CSVs."""
    agencies = set()
    for d in BULK_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        # Check if any year subdirs exist with rule/proposal files
        for sub in d.iterdir():
            if sub.is_dir() and any(sub.glob("rule_all_text.csv*")):
                agencies.add(d.name.upper())
                break
            if sub.is_dir() and any(sub.glob("proposed_rule_all_text.csv*")):
                agencies.add(d.name.upper())
                break
    return sorted(agencies)


def build_all_indexes(all_docs: pd.DataFrame | None, batch_size: int, overwrite: bool):
    """Build per-agency indexes. Run multiple processes for parallelism.

    If all_docs is None, loads data lazily per agency (no upfront 128K doc load).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if all_docs is not None:
        agencies = sorted(all_docs["Agency ID"].dropna().unique())
    else:
        agencies = _get_all_agency_ids()

    random.shuffle(agencies)
    logger.info("Building indexes for %d agencies", len(agencies))

    done = 0
    for i, agency in enumerate(agencies):
        if all_docs is not None:
            agency_docs = all_docs[all_docs["Agency ID"] == agency]
        else:
            agency_docs = None  # will be lazy-loaded inside _build_agency_indexes
        result = _build_agency_indexes(agency, agency_docs, batch_size, overwrite)
        if result:
            done += 1
        logger.info("[%d/%d] %s — %s", i + 1, len(agencies), agency, "done" if result else "skipped/locked")
    logger.info("Finished: %d/%d agencies indexed", done, len(agencies))


def _noise_title(structured_text: str, rng: random.Random) -> str:
    """Perturb the title in a structured text string to prevent title-matching shortcuts.

    Randomly applies: word deletion, word swap, synonym-ish replacement, or full title drop.
    """
    import re as _re
    m = _re.match(r"(Title: )(.*?)(\n|$)", structured_text, _re.DOTALL)
    if not m:
        return structured_text
    prefix, title, rest_start = m.group(1), m.group(2), m.end()
    rest = structured_text[rest_start:]

    words = title.split()
    if len(words) < 2:
        return structured_text

    op = rng.choice(["drop_words", "swap", "drop_title", "shuffle"])

    if op == "drop_words":
        # Drop 30-60% of words
        keep = max(1, int(len(words) * rng.uniform(0.4, 0.7)))
        kept = rng.sample(words, keep)
        # Maintain original order
        kept_set = set(id(w) for w in kept)
        title = " ".join(w for w in words if id(w) in kept_set)
    elif op == "swap":
        # Swap 2-3 random adjacent pairs
        words = list(words)
        for _ in range(rng.randint(1, min(3, len(words) - 1))):
            i = rng.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        title = " ".join(words)
    elif op == "drop_title":
        # Remove title entirely — force reliance on SUMMARY
        return rest.strip() if rest.strip() else structured_text
    elif op == "shuffle":
        # Fully shuffle words
        words = list(words)
        rng.shuffle(words)
        title = " ".join(words)

    return prefix + title + "\n" + rest


def gen_training_data(all_docs: pd.DataFrame, top_k: int = 20, neg_ratio: int = 1, seed: int = 42):
    """Generate training pairs from all agencies using pre-built summary indexes.

    Positives come only from small, legitimate dockets (not FRDOC, ≤5 docs).
    Uses structured text (Title + SUMMARY + ACTION) for both embedding queries and CE text.
    """
    import retriv
    from retriv import DenseRetriever

    rng = random.Random(seed)
    output_path = DATA_DIR / "rule_matching_training_pairs_v4.csv"

    agencies = sorted(all_docs["Agency ID"].dropna().unique())
    all_rows = []

    for agency in agencies:
        agency_dir = INDEX_DIR / agency
        if not (agency_dir / ".indexes_summary_done").exists():
            continue

        agency_docs = all_docs[all_docs["Agency ID"] == agency]
        rules = agency_docs[agency_docs["doc_type"] == "rule"].drop_duplicates(subset=["Document ID"])
        proposals = agency_docs[
            agency_docs["doc_type"].isin(["proposed_rule", "notice"])
        ].drop_duplicates(subset=["Document ID"])

        if rules.empty or proposals.empty:
            continue

        # Build doc_id → structured text (Title + SUMMARY + ACTION)
        doc_text = {}
        for _, row in agency_docs.iterrows():
            doc_id = str(row.get("Document ID", ""))
            if doc_id and doc_id != "nan":
                doc_text[doc_id] = extract_structured_text(row)

        # Load rules_summary index for retrieval
        retriv.set_base_path(str(agency_dir))
        try:
            dr_rules = DenseRetriever.load("rules_summary")
        except Exception as e:
            logger.warning("[%s] Failed to load rules_summary index: %s", agency, e)
            continue

        # Build docket mappings
        docket_to_rules = defaultdict(list)
        for _, row in rules.iterrows():
            dk, doc_id = str(row.get("Docket ID", "")), str(row.get("Document ID", ""))
            if dk and dk != "nan" and doc_id and doc_id != "nan":
                docket_to_rules[dk].append(doc_id)

        docket_to_proposals = defaultdict(list)
        for _, row in proposals.iterrows():
            dk, doc_id = str(row.get("Docket ID", "")), str(row.get("Document ID", ""))
            if dk and dk != "nan" and doc_id and doc_id != "nan":
                docket_to_proposals[dk].append(doc_id)

        # Positives: same-docket pairs from legitimate dockets only
        # Exclude FRDOC catch-all dockets. Allow dockets up to 20 docs,
        # but cap at 3 pairs per docket to avoid cross-product blowup.
        MAX_DOCKET_SIZE = 20
        MAX_PAIRS_PER_DOCKET = 3
        shared_dockets = set(docket_to_rules.keys()) & set(docket_to_proposals.keys())
        positives = []
        for dk in shared_dockets:
            if "FRDOC" in dk.upper():
                continue
            n_rules = len(docket_to_rules[dk])
            n_props = len(docket_to_proposals[dk])
            if n_rules + n_props > MAX_DOCKET_SIZE:
                continue
            dk_pairs = []
            for rule_id in docket_to_rules[dk]:
                for prop_id in docket_to_proposals[dk]:
                    if doc_text.get(rule_id) and doc_text.get(prop_id):
                        dk_pairs.append({
                            "rule_doc_id": rule_id, "proposal_doc_id": prop_id,
                            "docket_id": dk, "agency": agency, "label": 1, "source": "same_docket",
                        })
            if len(dk_pairs) > MAX_PAIRS_PER_DOCKET:
                rng.shuffle(dk_pairs)
                dk_pairs = dk_pairs[:MAX_PAIRS_PER_DOCKET]
            positives.extend(dk_pairs)
        if not positives:
            continue
        if len(positives) > 15000:
            rng.shuffle(positives)
            positives = positives[:15000]

        # Hard negatives via retrieval: 1 hard neg per positive (sampled from top-k)
        hard_negs = []
        seen_props = set()
        for pair in positives:
            prop_id = pair["proposal_doc_id"]
            if prop_id in seen_props:
                continue
            seen_props.add(prop_id)
            prop_text = doc_text.get(prop_id, "")
            if not prop_text:
                continue
            try:
                hits = dr_rules.search(query=prop_text, return_docs=False, cutoff=top_k)
            except Exception:
                continue
            same_dk_rules = set(docket_to_rules.get(pair["docket_id"], []))
            # Collect valid hard neg candidates, then sample 1
            candidates = [
                hit for hit in hits
                if hit["id"] not in same_dk_rules and doc_text.get(hit["id"])
            ]
            if candidates:
                chosen = rng.choice(candidates)
                hard_negs.append({
                    "rule_doc_id": chosen["id"], "proposal_doc_id": prop_id,
                    "docket_id": pair["docket_id"], "agency": agency, "label": 0, "source": "hard_negative",
                })

        # Easy negatives to fill up to 1:1 ratio
        all_rule_ids = [did for ids in docket_to_rules.values() for did in ids]
        easy_negs = []
        n_easy = max(0, len(positives) - len(hard_negs))
        for pair in positives:
            same_dk = set(docket_to_rules.get(pair["docket_id"], []))
            cands = [r for r in all_rule_ids if r not in same_dk]
            if cands:
                neg_id = rng.choice(cands)
                if doc_text.get(neg_id):
                    easy_negs.append({
                        "rule_doc_id": neg_id, "proposal_doc_id": pair["proposal_doc_id"],
                        "docket_id": pair["docket_id"], "agency": agency, "label": 0, "source": "easy_negative",
                    })
            if len(easy_negs) >= n_easy:
                break

        # Add structured text and collect.
        # Noise titles in ~60% of ALL pairs (both pos and neg) to prevent
        # the CE from relying solely on title matching.
        agency_pairs = positives + hard_negs + easy_negs
        for pair in agency_pairs:
            rt, pt = doc_text.get(pair["rule_doc_id"], ""), doc_text.get(pair["proposal_doc_id"], "")
            if rt and pt:
                if rng.random() < 0.6:
                    rt = _noise_title(rt, rng)
                all_rows.append({**pair, "rule_text": rt, "proposal_text": pt})

        logger.info("[%s] pos=%d, hard_neg=%d, easy_neg=%d", agency, len(positives), len(hard_negs), len(easy_negs))

    if not all_rows:
        logger.warning("No training pairs generated from any agency!")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d training pairs (pos=%d, hard_neg=%d, easy_neg=%d) from %d agencies",
        len(df), int(df["label"].sum()),
        int((df["source"] == "hard_negative").sum()),
        int((df["source"] == "easy_negative").sum()),
        df["agency"].nunique(),
    )


# ---------------------------------------------------------------------------
# Phase 2.5: Silver-label hard negatives with GPT5-mini
# ---------------------------------------------------------------------------


def silver_label_hard_negatives(budget: int = 5000):
    """Use GPT5-mini to check if high-overlap hard negatives are actually matches.

    Flips confirmed matches from label=0 to label=1 in the training data.
    Saves silver labels to a separate file and rewrites training pairs.
    """
    import asyncio
    import re
    from openai import AsyncOpenAI

    # Load OpenAI key
    key_path = SCRIPTS_DIR.parent.parent.parent.parent / ".openai-key.txt"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and key_path.exists():
        api_key = key_path.read_text().strip()
    if not api_key:
        raise RuntimeError(f"No OpenAI API key found (checked OPENAI_API_KEY and {key_path})")

    client = AsyncOpenAI(api_key=api_key)

    training_path = DATA_DIR / "rule_matching_training_pairs.csv"
    silver_path = DATA_DIR / "rule_matching_silver_labels.csv"
    df = pd.read_csv(training_path)
    logger.info("Loaded %d training pairs", len(df))

    # Focus on hard negatives with high word overlap (likely false negatives)
    hard_neg = df[df["source"] == "hard_negative"].copy()

    def word_overlap(a, b):
        wa = set(re.findall(r"\\w+", str(a).lower()[:300]))
        wb = set(re.findall(r"\\w+", str(b).lower()[:300]))
        stop = {"the", "of", "and", "to", "a", "in", "for", "is", "on", "that", "by",
                "this", "with", "from", "or", "an", "be", "as", "at", "are", "was"}
        wa -= stop
        wb -= stop
        if not wa or not wb:
            return 0
        return len(wa & wb) / min(len(wa), len(wb))

    hard_neg["overlap"] = hard_neg.apply(
        lambda r: word_overlap(r["rule_text"], r["proposal_text"]), axis=1
    )

    # Sample from highest overlap first
    candidates = hard_neg.sort_values("overlap", ascending=False).head(budget)
    logger.info(
        "Silver-labeling %d hard negatives (overlap range: %.2f - %.2f)",
        len(candidates), candidates["overlap"].min(), candidates["overlap"].max(),
    )

    PROMPT_TEMPLATE = """Are these two federal regulatory documents about the SAME specific rulemaking action?

Document A (Final Rule):
{rule_text}

Document B (Proposed Rule / Notice):
{proposal_text}

Answer "yes" if they are about the same specific regulatory action (same topic, same agency, same CFR section).
Answer "no" if they are about different regulatory actions, even if they are from the same agency or same general topic area.
Answer ONLY "yes" or "no"."""

    CONCURRENCY = 50  # max concurrent API calls

    async def label_one(sem, idx, row):
        rule_snippet = str(row["rule_text"])[:800]
        prop_snippet = str(row["proposal_text"])[:800]
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(
                            rule_text=rule_snippet, proposal_text=prop_snippet
                        ),
                    }],
                    max_tokens=5,
                    temperature=0,
                )
                answer = resp.choices[0].message.content.strip().lower()
                is_match = answer.startswith("yes")
            except Exception as e:
                logger.warning("API error for idx %s: %s", idx, e)
                is_match = False
                answer = "error"
        return {
            "index": idx,
            "rule_doc_id": row["rule_doc_id"],
            "proposal_doc_id": row["proposal_doc_id"],
            "overlap": row["overlap"],
            "llm_answer": answer,
            "is_match": is_match,
        }

    async def run_all():
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            label_one(sem, idx, row)
            for idx, row in candidates.iterrows()
        ]
        results = []
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1
            if done % 500 == 0:
                flipped_so_far = sum(1 for r in results if r["is_match"])
                logger.info(
                    "  Silver-labeled %d/%d (flipped %d so far, %.1f%%)",
                    done, len(tasks), flipped_so_far, 100 * flipped_so_far / done,
                )
        return results

    results = asyncio.run(run_all())
    flipped = sum(1 for r in results if r["is_match"])

    silver_df = pd.DataFrame(results)
    silver_df.to_csv(silver_path, index=False)
    logger.info(
        "Silver labeling complete: %d/%d flipped to positive (%.1f%%)",
        flipped, len(results), 100 * flipped / len(results) if results else 0,
    )

    # Apply silver labels to training data: flip confirmed matches
    flip_indices = silver_df[silver_df["is_match"]]["index"].values
    before_pos = df["label"].sum()
    df.loc[flip_indices, "label"] = 1
    df.loc[flip_indices, "source"] = "silver_positive"
    after_pos = df["label"].sum()
    logger.info("Flipped %d labels: positives %d -> %d", len(flip_indices), int(before_pos), int(after_pos))

    # Save updated training data
    df.to_csv(training_path, index=False)
    logger.info("Saved updated training pairs to %s", training_path)


# ---------------------------------------------------------------------------
# Phase 3: Train cross-encoder
# ---------------------------------------------------------------------------


def train_ce(gpu: int, epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
    """Train a ModernBERT cross-encoder on rule-proposal pairs."""
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # GPU already set at module level via --gpu flag

    # Use v3 training pairs if available, fall back to v2
    training_path = DATA_DIR / "rule_matching_training_pairs_v4.csv"
    if not training_path.exists():
        training_path = DATA_DIR / "rule_matching_training_pairs.csv"
    output_dir = SCRIPTS_DIR / "cross_encoder_models" / "modernbert-rule-matching-ce-v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    df = pd.read_csv(training_path)
    logger.info("Loaded %d training pairs (%.1f%% positive)", len(df), 100 * df["label"].mean())

    # Split by docket to prevent leakage
    dockets = df["docket_id"].fillna("unknown").unique()
    rng = np.random.RandomState(42)
    rng.shuffle(dockets)
    n_test = max(1, int(len(dockets) * 0.10))
    n_val = max(1, int(len(dockets) * 0.15))
    test_dockets = set(dockets[:n_test])
    val_dockets = set(dockets[n_test:n_test + n_val])

    test_df = df[df["docket_id"].isin(test_dockets)]
    val_df = df[df["docket_id"].isin(val_dockets)]
    train_df = df[~df["docket_id"].isin(test_dockets | val_dockets)]

    logger.info(
        "Split: train=%d (%.1f%% pos), val=%d (%.1f%% pos), test=%d (%.1f%% pos)",
        len(train_df), 100 * train_df["label"].mean(),
        len(val_df), 100 * val_df["label"].mean(),
        len(test_df), 100 * test_df["label"].mean(),
    )

    # Save splits
    train_df.to_csv(output_dir / "train_split.csv.gz", index=False)
    val_df.to_csv(output_dir / "val_split.csv.gz", index=False)
    test_df.to_csv(output_dir / "test_split.csv.gz", index=False)

    # Initialize model
    model = CrossEncoder("answerdotai/ModernBERT-base", num_labels=1, max_length=CE_MAX_LENGTH)

    # Prepare HF datasets
    def to_dataset(d):
        return Dataset.from_dict({
            "sentence1": d["rule_text"].tolist(),
            "sentence2": d["proposal_text"].tolist(),
            "label": d["label"].astype(float).tolist(),
        })

    train_dataset = to_dataset(train_df)
    val_dataset = to_dataset(val_df)

    # Loss with class imbalance weighting
    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info("Class imbalance pos_weight: %.2f", pos_weight)

    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(pos_weight))

    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=list(zip(val_df["rule_text"].tolist(), val_df["proposal_text"].tolist())),
        labels=val_df["label"].astype(int).tolist(),
        name="val",
        show_progress_bar=True,
    )

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="val_f1",
        logging_steps=50,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))

    # Find optimal threshold
    val_pairs = list(zip(val_df["rule_text"].tolist(), val_df["proposal_text"].tolist()))
    val_scores = model.predict(val_pairs, show_progress_bar=True)
    val_labels = val_df["label"].astype(int).values

    best_f1, best_threshold = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (val_scores >= t).astype(int)
        f = f1_score(val_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_threshold = f, t

    logger.info("Best validation threshold: %.3f (F1=%.4f)", best_threshold, best_f1)

    # Test evaluation
    test_pairs = list(zip(test_df["rule_text"].tolist(), test_df["proposal_text"].tolist()))
    test_scores = model.predict(test_pairs, show_progress_bar=True)
    test_labels = test_df["label"].astype(int).values
    test_preds = (test_scores >= best_threshold).astype(int)

    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_scores) if len(set(test_labels)) > 1 else 0.0

    logger.info(
        "Test: F1=%.4f, P=%.4f, R=%.4f, AUC=%.4f (threshold=%.3f)",
        test_f1, test_precision, test_recall, test_auc, best_threshold,
    )
    logger.info("\n%s", classification_report(test_labels, test_preds, target_names=["no", "yes"], zero_division=0))

    # Save threshold and log
    with open(final_dir / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": round(best_threshold, 4), "val_f1": round(best_f1, 4)}, f)

    log_entry = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "pos_weight": round(pos_weight, 4),
        "best_val_threshold": round(best_threshold, 4),
        "best_val_f1": round(best_f1, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_auc": round(test_auc, 4),
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_entry, f, indent=2)

    logger.info("Training complete. Model at %s", final_dir)


# ---------------------------------------------------------------------------
# Phase 4: Apply — retrieve + rerank orphan rules → crosswalk
# ---------------------------------------------------------------------------


def _link_by_fr_number_and_rin(all_docs: pd.DataFrame, orphan_dockets: set, comment_dockets: set):
    """Link orphan rule dockets to proposal dockets via FR Number and RIN.

    Returns a DataFrame of matches with columns: rule_doc_id, rule_docket,
    rule_agency, rule_title, proposal_doc_id, proposal_docket, proposal_title,
    proposal_doc_type, method.
    """
    import re as _re
    from collections import defaultdict

    rules = all_docs[all_docs["doc_type"] == "rule"].drop_duplicates(subset=["Document ID"])
    proposals = all_docs[all_docs["doc_type"].isin(["proposed_rule", "notice"])].drop_duplicates(subset=["Document ID"])

    orphan_rules = rules[rules["Docket ID"].astype(str).isin(orphan_dockets)]

    # --- FR Number linking ---
    frn_to_docs = defaultdict(list)
    for _, row in all_docs.iterrows():
        frn = str(row.get("Federal Register Number", ""))
        dk = str(row.get("Docket ID", ""))
        if frn and frn != "nan" and dk and dk != "nan":
            frn_to_docs[frn].append({
                "doc_id": str(row.get("Document ID", "")),
                "docket": dk,
                "doc_type": row.get("doc_type", ""),
                "title": str(row.get("Title", "")),
                "agency": str(row.get("Agency ID", "")),
            })

    fr_matches = []
    for frn, entries in frn_to_docs.items():
        orphan_entries = [e for e in entries if e["docket"] in orphan_dockets]
        proposal_entries = [e for e in entries if e["doc_type"] in ("proposed_rule", "notice") and e["docket"] not in orphan_dockets]
        if orphan_entries and proposal_entries:
            for oe in orphan_entries:
                for pe in proposal_entries:
                    if oe["docket"] != pe["docket"]:
                        fr_matches.append({
                            "rule_doc_id": oe["doc_id"],
                            "rule_docket": oe["docket"],
                            "rule_agency": oe["agency"],
                            "rule_title": oe["title"][:200],
                            "proposal_doc_id": pe["doc_id"],
                            "proposal_docket": pe["docket"],
                            "proposal_title": pe["title"][:200],
                            "proposal_doc_type": pe["doc_type"],
                            "method": "fr_number",
                            "evidence": frn,
                        })

    # --- RIN linking ---
    rin_to_dockets = defaultdict(set)
    rin_to_docs = defaultdict(list)
    for _, row in all_docs.iterrows():
        rins = str(row.get("Related RIN(s)", ""))
        dk = str(row.get("Docket ID", ""))
        if rins and rins != "nan" and dk and dk != "nan":
            for rin in _re.split(r"[,;\s]+", rins):
                rin = rin.strip()
                if rin:
                    rin_to_dockets[rin].add(dk)
                    rin_to_docs[rin].append({
                        "doc_id": str(row.get("Document ID", "")),
                        "docket": dk,
                        "doc_type": row.get("doc_type", ""),
                        "title": str(row.get("Title", "")),
                        "agency": str(row.get("Agency ID", "")),
                    })

    rin_matches = []
    for rin, entries in rin_to_docs.items():
        orphan_entries = [e for e in entries if e["docket"] in orphan_dockets]
        proposal_entries = [e for e in entries if e["doc_type"] in ("proposed_rule", "notice") and e["docket"] not in orphan_dockets]
        if orphan_entries and proposal_entries:
            for oe in orphan_entries:
                for pe in proposal_entries:
                    if oe["docket"] != pe["docket"]:
                        rin_matches.append({
                            "rule_doc_id": oe["doc_id"],
                            "rule_docket": oe["docket"],
                            "rule_agency": oe["agency"],
                            "rule_title": oe["title"][:200],
                            "proposal_doc_id": pe["doc_id"],
                            "proposal_docket": pe["docket"],
                            "proposal_title": pe["title"][:200],
                            "proposal_doc_type": pe["doc_type"],
                            "method": "rin",
                            "evidence": rin,
                        })

    all_metadata_matches = fr_matches + rin_matches
    if not all_metadata_matches:
        return pd.DataFrame()

    df = pd.DataFrame(all_metadata_matches)
    # Deduplicate: keep FR number match over RIN (higher confidence)
    method_rank = {"fr_number": 0, "rin": 1}
    df["_rank"] = df["method"].map(method_rank)
    df = df.sort_values("_rank").drop_duplicates(subset=["rule_docket"], keep="first").drop(columns=["_rank"])

    logger.info(
        "Metadata linking: %d orphan dockets matched (fr_number=%d, rin=%d)",
        len(df),
        (df["method"] == "fr_number").sum(),
        (df["method"] == "rin").sum(),
    )
    return df


def apply_crosswalk(
    all_docs: pd.DataFrame,
    gpu: int,
    top_k: int = 20,
    batch_size: int = 64,
):
    """Match orphan rules to proposals using:
    1. FR Number / RIN metadata linking (exact, high confidence)
    2. Bi-encoder retrieval + CE reranking (for remaining unmatched)
    """
    import retriv
    from retriv import DenseRetriever
    from cross_encoder_utils import load_cross_encoder, load_optimal_threshold, rerank_pairs

    # Find orphan rule dockets
    comment_dockets = load_comment_dockets()
    rules = all_docs[all_docs["doc_type"] == "rule"].drop_duplicates(subset=["Document ID"])
    rule_dockets = set(rules["Docket ID"].dropna().astype(str).unique())
    orphan_dockets = rule_dockets - comment_dockets
    logger.info("Orphan rule dockets: %d", len(orphan_dockets))

    # ── Step 1: FR Number / RIN linking ──
    logger.info("\n=== Step 1: FR Number / RIN linking ===")
    metadata_matches = _link_by_fr_number_and_rin(all_docs, orphan_dockets, comment_dockets)
    metadata_matched_dockets = set(metadata_matches["rule_docket"]) if not metadata_matches.empty else set()
    logger.info("Metadata matched: %d dockets", len(metadata_matched_dockets))

    # ── Step 2: CE matching for remaining orphans ──
    remaining_orphans = orphan_dockets - metadata_matched_dockets
    logger.info("\n=== Step 2: CE matching for %d remaining orphans ===", len(remaining_orphans))

    # Load CE model
    ce_model_path = str(SCRIPTS_DIR / "cross_encoder_models" / "modernbert-rule-matching-ce-v4" / "final")
    if not Path(ce_model_path).exists():
        ce_model_path = str(SCRIPTS_DIR / "cross_encoder_models" / "modernbert-rule-matching-ce" / "final")
    ce_model = load_cross_encoder(ce_model_path, max_length=CE_MAX_LENGTH)
    threshold = load_optimal_threshold(ce_model_path)
    logger.info("Loaded CE model from %s (threshold=%.3f)", ce_model_path, threshold)

    orphan_rules = rules[rules["Docket ID"].astype(str).isin(remaining_orphans)].copy()
    orphan_rules = orphan_rules.drop_duplicates(subset=["Docket ID"])

    # Build proposal metadata
    proposals = all_docs[all_docs["doc_type"].isin(["proposed_rule", "notice"])].drop_duplicates(subset=["Document ID"])
    prop_meta = {}
    for _, row in proposals.iterrows():
        doc_id = str(row.get("Document ID", ""))
        if doc_id and doc_id != "nan":
            prop_meta[doc_id] = {
                "docket_id": str(row.get("Docket ID", "")),
                "agency_id": str(row.get("Agency ID", "")),
                "title": str(row.get("Title", "")),
                "doc_type": str(row.get("doc_type", "")),
                "structured_text": extract_structured_text(row),
            }

    # Retrieve per agency
    all_candidates = []
    agencies = sorted(orphan_rules["Agency ID"].dropna().unique())

    for agency in agencies:
        agency_dir = INDEX_DIR / agency
        if not (agency_dir / ".indexes_summary_done").exists():
            logger.warning("[%s] No summary index, skipping", agency)
            continue

        retriv.set_base_path(str(agency_dir))
        try:
            dr_proposals = DenseRetriever.load("proposals_summary")
        except Exception as e:
            logger.warning("[%s] Failed to load proposals_summary: %s", agency, e)
            continue

        agency_orphans = orphan_rules[orphan_rules["Agency ID"] == agency]
        logger.info("[%s] Retrieving for %d orphan rules...", agency, len(agency_orphans))

        for _, rule_row in agency_orphans.iterrows():
            rule_id = str(rule_row["Document ID"])
            rule_docket = str(rule_row["Docket ID"])
            rule_structured = extract_structured_text(rule_row)

            try:
                hits = dr_proposals.search(
                    query=rule_structured, return_docs=False, cutoff=top_k,
                )
            except Exception:
                continue

            for hit in hits:
                prop_id = hit["id"]
                meta = prop_meta.get(prop_id, {})
                if meta.get("docket_id") == rule_docket:
                    continue
                all_candidates.append({
                    "rule_doc_id": rule_id,
                    "rule_docket": rule_docket,
                    "rule_agency": agency,
                    "rule_title": str(rule_row.get("Title", ""))[:200],
                    "rule_text": rule_structured,
                    "proposal_doc_id": prop_id,
                    "proposal_docket": meta.get("docket_id", ""),
                    "proposal_title": meta.get("title", "")[:200],
                    "proposal_doc_type": meta.get("doc_type", ""),
                    "proposal_text": meta.get("structured_text", ""),
                    "dense_score": hit["score"],
                })

    # Rerank with CE
    ce_matches = pd.DataFrame()
    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates)
        logger.info("Retrieved %d candidates for %d rules, reranking...",
                     len(candidates_df), candidates_df["rule_doc_id"].nunique())

        ce_scores = rerank_pairs(
            ce_model, candidates_df,
            response_col="rule_text", candidate_col="proposal_text",
            batch_size=batch_size, max_tokens=MAX_TEXT_TOKENS,
        )
        candidates_df["ce_score"] = ce_scores

        ce_matches = candidates_df[candidates_df["ce_score"] >= threshold].copy()
        ce_matches = ce_matches.sort_values("ce_score", ascending=False).drop_duplicates(
            subset=["rule_docket"], keep="first"
        )
        ce_matches["method"] = "ce"
        ce_matches["evidence"] = ce_matches["ce_score"].apply(lambda x: f"ce={x:.3f}")

        # Save full candidates
        candidates_df.to_csv(DATA_DIR / "rule_matching_all_candidates_v4.csv.gz", index=False)

    # ── Combine metadata + CE matches ──
    out_cols = [
        "rule_doc_id", "rule_docket", "rule_agency", "rule_title",
        "proposal_doc_id", "proposal_docket", "proposal_title", "proposal_doc_type",
        "method", "evidence",
    ]

    parts = []
    if not metadata_matches.empty:
        parts.append(metadata_matches[out_cols])
    if not ce_matches.empty:
        parts.append(ce_matches[[c for c in out_cols if c in ce_matches.columns]])

    if not parts:
        logger.warning("No matches found!")
        return

    matches = pd.concat(parts, ignore_index=True)
    # Deduplicate: metadata wins over CE
    method_rank = {"fr_number": 0, "rin": 1, "ce": 2}
    matches["_rank"] = matches["method"].map(method_rank)
    matches = matches.sort_values("_rank").drop_duplicates(subset=["rule_docket"], keep="first").drop(columns=["_rank"])

    matches["proposal_has_comments"] = matches["proposal_docket"].isin(comment_dockets)

    crosswalk_path = DATA_DIR / "rule_proposal_crosswalk_v4.csv"
    matches.to_csv(crosswalk_path, index=False)

    n_with_comments = matches["proposal_has_comments"].sum()
    by_method = matches.groupby("method")["rule_docket"].nunique()
    logger.info(
        "\n=== CROSSWALK RESULTS ===\n"
        "Orphan rule dockets: %d\n"
        "Matched to proposals: %d (%.1f%%)\n"
        "  fr_number: %d\n"
        "  rin: %d\n"
        "  ce: %d\n"
        "  of which proposal has comments: %d\n"
        "Saved to %s",
        len(orphan_dockets), len(matches),
        100 * len(matches) / len(orphan_dockets) if orphan_dockets else 0,
        by_method.get("fr_number", 0), by_method.get("rin", 0), by_method.get("ce", 0),
        n_with_comments, crosswalk_path,
    )

    # Sample for validation
    logger.info("\n=== SAMPLE MATCHES ===")
    for method in ["fr_number", "rin", "ce"]:
        sample = matches[matches["method"] == method].head(5)
        if not sample.empty:
            logger.info("\n--- %s ---", method)
            for _, row in sample.iterrows():
                logger.info(
                    "  %s -> %s [%s, has_comments=%s]",
                    row["rule_docket"], row["proposal_docket"],
                    row["evidence"], row["proposal_has_comments"],
                )
                logger.info("    rule:     %s", row["rule_title"])
                logger.info("    proposal: %s", row["proposal_title"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["build-indexes", "gen-training", "silver-label", "train-ce", "apply", "all"])
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding/CE")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k for retrieval")
    parser.add_argument("--neg-ratio", type=int, default=1, help="Negative:positive ratio for training")
    parser.add_argument("--epochs", type=int, default=3, help="CE training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild indexes even if they exist")
    parser.add_argument("--silver-label-budget", type=int, default=15000, help="Max pairs to silver-label with LLM")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    phases = [args.phase] if args.phase != "all" else ["build-indexes", "gen-training", "silver-label", "train-ce", "apply"]

    all_docs = None  # lazy-loaded only when needed

    for phase in phases:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE: %s", phase)
        logger.info("=" * 60)

        if phase == "build-indexes":
            # No need to load all docs — each agency loads its own lazily
            build_all_indexes(None, batch_size=args.batch_size, overwrite=args.overwrite)
        elif phase == "gen-training":
            if all_docs is None:
                logger.info("Loading all documents for training data generation...")
                all_docs = load_all_docs()
            gen_training_data(all_docs, top_k=args.top_k, neg_ratio=args.neg_ratio)
        elif phase == "silver-label":
            silver_label_hard_negatives(budget=args.silver_label_budget)
        elif phase == "train-ce":
            train_ce(gpu=args.gpu, epochs=args.epochs, learning_rate=args.learning_rate)
        elif phase == "apply":
            if all_docs is None:
                logger.info("Loading all documents for crosswalk application...")
                all_docs = load_all_docs()
            apply_crosswalk(all_docs, gpu=args.gpu, top_k=args.top_k, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
