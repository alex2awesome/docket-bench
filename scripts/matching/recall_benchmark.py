"""Recall benchmark: exhaustively label all (response, cluster) pairs for
selected directories, then measure recall@k for different retriever combos.

Steps:
1. For each benchmark dir, generate ALL (response × claim_doc) pairs
2. Score every pair with GPT-5-mini → ground truth
3. For each retriever (8B, SBERT, BM25) retrieve top-k candidates
4. Measure: of the true positives, how many were retrieved?

Usage:
    python recall_benchmark.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# HF cache fix
os.environ.setdefault("HF_MODULES_CACHE",
    str(Path(__file__).resolve().parent / "data" / ".hf_modules"))
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BULK_DIR = SCRIPTS_DIR.parent
_notebooks_dir = SCRIPTS_DIR.parent.parent.parent / "notebooks"

BENCHMARK_DIRS = [
    ("hhs", "hhs_2017_2018"),       # 19 resp × 49 clust = 931 pairs, $0.26
    ("doj", "doj_2023_2024"),       # 20 × 13 = 260, $0.07
    ("fema", "fema_2021_2022"),     # 8 × 33 = 264, $0.07
    ("blm", "blm_2018_2019"),       # 12 × 181 = 2172, $0.61
    ("tsa", "tsa_2023_2024"),       # 44 × 36 = 1584, $0.44
    ("cdc", "cdc_2021_2022"),       # 38 × 97 = 3686, $1.03
    ("dod", "dod_2020_2021"),       # 37 × 56 = 2072, $0.58
    ("ed", "ed_2026_2027"),         # 17 × 118 = 2006, $0.56
    ("nist", "nist_2023"),          # 103 × 57 = 5871, $1.64
    ("usace", "usace_2023_2024"),   # 7 × 72 = 504, $0.14
]

PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notice & comment process.
I will show you a government response and a specific claim extracted from a public comment.
Tell me whether this government response is addressing this claim:
either directly or as part of a larger group of similar comments.
Answer with "yes" or "no". Don't say anything else.

<claim>
{claim}
</claim>

<response>
{response}
</response>

Your response:
"""


def normalize_label(raw: str) -> str:
    if not raw:
        return "unknown"
    raw = str(raw).strip().lower()
    if raw.startswith("yes"):
        return "yes"
    if raw.startswith("no"):
        return "no"
    return "unknown"


def generate_exhaustive_pairs(dir_path: Path, dir_name: str):
    """Generate ALL (response_text, claim_doc_id, claim_text) pairs."""
    import retriv
    from retriv import DenseRetriever

    resp = pd.read_csv(dir_path / "responses_combined.csv.gz")
    if "response_to_comment" in resp.columns:
        resp["text"] = resp["response_to_comment"]
    else:
        resp["text"] = resp.get("content_of_comment", resp.get("content", ""))

    # Get claim texts from the index docs.jsonl (has the actual text)
    index_base = dir_path / ".retriv_indexes"
    retriv.set_base_path(str(index_base))
    idx_name = "{}_claims_nvidia_llama-embed-nemotron-8b".format(dir_name)
    dr = DenseRetriever.load(idx_name)

    # Read docs.jsonl for claim texts
    docs_path = index_base / "collections" / idx_name / "docs.jsonl"
    claim_docs = {}
    with open(docs_path) as f:
        for line in f:
            doc = json.loads(line)
            claim_docs[doc["id"]] = doc.get("text", "")

    # Build docket mapping
    all_ids = list(claim_docs.keys())
    docket_to_ids = defaultdict(list)
    for doc_id in all_ids:
        parts = doc_id.split("__")
        if len(parts) >= 3:
            docket_to_ids[parts[2]].append(doc_id)

    MAX_CLAIMS_PER_DOCKET = 300  # Cap to keep pair count manageable
    MAX_RESPONSES_PER_DOCKET = 30

    pairs = []
    for _, resp_row in resp.sample(min(MAX_RESPONSES_PER_DOCKET, len(resp)), random_state=42).iterrows():
        resp_text = str(resp_row["text"])
        if not resp_text or resp_text == "nan":
            continue
        docket = resp_row.get("docket_id", "")
        claim_ids = docket_to_ids.get(docket, [])
        if not claim_ids:
            continue
        # Sample claims if too many
        import random
        random.seed(42)
        if len(claim_ids) > MAX_CLAIMS_PER_DOCKET:
            claim_ids = random.sample(claim_ids, MAX_CLAIMS_PER_DOCKET)
        for cid in claim_ids:
            pairs.append({
                "response_text": resp_text[:1500],
                "claim_doc_id": cid,
                "claim_text": claim_docs.get(cid, "")[:1000],
                "docket_id": docket,
            })

    return pairs, dr, docket_to_ids


async def llm_label_pairs(pairs: list[dict], model="gpt-5-mini") -> list[str]:
    """Label all pairs with GPT-5-mini."""
    if not os.environ.get("OPENAI_API_KEY"):
        for p in [Path.home() / ".openai-key.txt"]:
            if p.exists():
                os.environ["OPENAI_API_KEY"] = p.read_text().strip()
                break

    prompts = [PROMPT.format(claim=p["claim_text"], response=p["response_text"]) for p in pairs]

    try:
        sys.path.insert(0, str(_notebooks_dir))
        import prompt_utils
        results = await prompt_utils.process_batch(prompts=prompts, model=model, concurrency=200)
        return [normalize_label(r) for r in results]
    except (ImportError, AttributeError):
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(50)
        async def _call(prompt):
            async with sem:
                r = await client.responses.create(model=model, input=prompt, max_output_tokens=10)
                return r.output_text.strip()
        results = await asyncio.gather(*[_call(p) for p in prompts])
        return [normalize_label(r) for r in results]


def measure_recall(true_positives: set, retrieved_ids: list[str], k_values=[10, 25, 50, 100, 200]):
    """Compute recall@k for different cutoffs."""
    results = {}
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        found = len(true_positives & top_k)
        results[k] = found / len(true_positives) if true_positives else 0
    return results


def run_retrieval(dr, query: str, doc_ids: list[str], k: int = 200, label: str = ""):
    """Run retrieval and return ranked list of doc_ids."""
    try:
        if label == "bm25":
            hits = dr.search(query=query, cutoff=k * 3)
            doc_set = set(doc_ids)
            if isinstance(hits, list):
                return [h["id"] for h in hits if isinstance(h, dict) and h.get("id") in doc_set][:k]
            elif isinstance(hits, dict):
                return [did for did in sorted(hits, key=hits.get, reverse=True) if did in doc_set][:k]
        else:
            hits = dr.search(query=query, include_id_list=doc_ids, return_docs=False, cutoff=k)
            if isinstance(hits, list):
                return [h["id"] for h in hits if isinstance(h, dict)]
            elif isinstance(hits, dict):
                return sorted(hits, key=hits.get, reverse=True)
    except Exception as e:
        logger.debug("Search error: %s", e)
    return []


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k=60) -> list[str]:
    """Fuse multiple ranked lists via RRF."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def main():
    import retriv
    from retriv import DenseRetriever, SparseRetriever

    output_path = SCRIPTS_DIR / "data" / "recall_benchmark_results.csv"
    all_results = []
    _encoder_cache = {}

    k_values = [10, 25, 50, 100, 200]
    combos = {
        "8b": ["8b"],
        "sbert": ["sbert"],
        "bm25": ["bm25"],
        "8b+bm25": ["8b", "bm25"],
        "sbert+bm25": ["sbert", "bm25"],
        "8b+sbert+bm25": ["8b", "sbert", "bm25"],
    }

    for agency, dir_name in BENCHMARK_DIRS:
        dir_path = BULK_DIR / agency / dir_name
        if not (dir_path / "responses_combined.csv.gz").exists():
            logger.warning("Skip %s: no responses", dir_name)
            continue

        logger.info("=== %s ===", dir_name)

        # Step 1: Generate exhaustive pairs
        try:
            pairs, dr_8b, docket_to_ids = generate_exhaustive_pairs(dir_path, dir_name)
        except Exception as e:
            logger.error("Failed to generate pairs for %s: %s", dir_name, e)
            continue
        logger.info("  %d exhaustive pairs", len(pairs))

        if not pairs:
            continue

        # Step 2: LLM label
        labels = asyncio.run(llm_label_pairs(pairs))
        for p, l in zip(pairs, labels):
            p["llm_label"] = l

        n_pos = sum(1 for p in pairs if p["llm_label"] == "yes")
        logger.info("  %d/%d positive (%.1f%%)", n_pos, len(pairs), n_pos / len(pairs) * 100)

        # Save per-dir labels
        pd.DataFrame(pairs).to_csv(
            SCRIPTS_DIR / "data" / "recall_benchmark_{}.csv.gz".format(dir_name),
            index=False, compression="gzip")

        # Step 3: Load all retrievers
        index_base = dir_path / ".retriv_indexes"
        retriv.set_base_path(str(index_base))

        retrievers = {}
        for label, suffix in [("8b", "nvidia_llama-embed-nemotron-8b"),
                               ("sbert", "sentence-transformers_all-mpnet-base-v2")]:
            idx = "{}_claims_{}".format(dir_name, suffix)
            idx_dir = index_base / "collections" / idx
            if not idx_dir.exists():
                continue
            try:
                cached = _encoder_cache.get(label)
                if cached:
                    retrievers[label] = DenseRetriever.load(idx, encoder=cached)
                else:
                    retrievers[label] = DenseRetriever.load(idx)
                    _encoder_cache[label] = retrievers[label].encoder
            except Exception as e:
                logger.warning("  %s load failed: %s", label, e)

        bm25_name = "{}_claims_bm25".format(dir_name)
        if (index_base / "collections" / bm25_name).exists():
            try:
                retrievers["bm25"] = SparseRetriever.load(bm25_name)
            except Exception as e:
                logger.warning("  bm25 load failed: %s", label, e)

        logger.info("  retrievers loaded: %s", list(retrievers.keys()))

        # Step 4: For each response, measure recall
        resp = pd.read_csv(dir_path / "responses_combined.csv.gz")
        if "response_to_comment" in resp.columns:
            resp["text"] = resp["response_to_comment"]
        else:
            resp["text"] = resp.get("content_of_comment", resp.get("content", ""))

        for _, resp_row in resp.iterrows():
            query = str(resp_row["text"])
            if not query or query == "nan":
                continue
            docket = resp_row.get("docket_id", "")
            doc_ids = docket_to_ids.get(docket, [])
            if not doc_ids:
                continue

            # True positives for this response
            tp = set()
            for p in pairs:
                if p["docket_id"] == docket and p["llm_label"] == "yes":
                    if p["response_text"][:100] == query[:100]:
                        tp.add(p["claim_doc_id"])

            if not tp:
                continue

            # Retrieve with each retriever
            per_retriever_ranked = {}
            for label, ret in retrievers.items():
                ranked = run_retrieval(ret, query, doc_ids, k=200, label=label)
                per_retriever_ranked[label] = ranked

            # Measure recall for each combo
            for combo_name, combo_retrievers in combos.items():
                available = [per_retriever_ranked[r] for r in combo_retrievers if r in per_retriever_ranked]
                if not available:
                    continue
                if len(available) == 1:
                    fused = available[0]
                else:
                    fused = reciprocal_rank_fusion(available)

                recalls = measure_recall(tp, fused, k_values)
                for k, recall in recalls.items():
                    all_results.append({
                        "dir": dir_name,
                        "agency": agency,
                        "docket": docket,
                        "combo": combo_name,
                        "k": k,
                        "recall": recall,
                        "n_true_positives": len(tp),
                        "n_docket_docs": len(doc_ids),
                    })

    # Summary
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    print("\n" + "=" * 70)
    print("RECALL BENCHMARK RESULTS")
    print("=" * 70)
    for combo in combos:
        sub = df[df.combo == combo]
        if sub.empty:
            continue
        print("\n  {}:".format(combo))
        for k in k_values:
            ksub = sub[sub.k == k]
            if ksub.empty:
                continue
            print("    recall@{:<4} = {:.1%}  (n={} response-dockets)".format(
                k, ksub.recall.mean(), len(ksub)))


if __name__ == "__main__":
    main()
