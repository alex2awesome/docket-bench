"""Run Axis B claim pair judging for new augmented configs.

Embed claims with MiniLM (cached), top-3 retrieval, batch-judge with gpt-5-mini (k=10).

Usage:
    OPENAI_API_KEY=$(cat ~/.openai-key.txt) \
        python scripts/run_claim_pair_judging.py
"""

import asyncio
import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(os.environ.get("DOCKET_BASE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RUNS = BASE / "data" / "derived" / "benchmark_runs"
DU = BASE / "data" / "bulk_downloads" / "deontic_units"
MATCH2 = DU / "match2_llm_labeled.jsonl"
CACHE_DIR = BASE / "data" / "derived" / "embedding_cache"
BENCH_PATH = DU / "comment_anticipation_benchmark_gpt5mini.jsonl"

MODEL = "gpt-4o-mini"
CONCURRENCY = 50
BATCH_SIZE = 10
CLAIM_RE = re.compile(r'^\s*\[P(\d+)\]\s*(.+?)\s*$', re.MULTILINE)

JUDGE_BATCH = """Classify each pair of public comment claims on a federal regulation. For each, respond with one word: same, related, opposing, or different.

- same: both make the same specific argument AND take the same position
- related: same topic but different specific arguments
- opposing: same point, opposite positions
- different: entirely different provisions or issues

{pairs}

{n} labels, one per line:"""

# The 12 new configs to judge
TARGET_CONFIGS = [
    "llama-3.3-70b_vanilla", "llama-3.3-70b_subgroup",
]


def parse_batch_labels(raw, n):
    labels = [w for w in re.findall(r'[a-z]+', raw.lower())
              if w in {'same', 'related', 'opposing', 'different'}]
    while len(labels) < n:
        labels.append('error')
    return labels[:n]


async def main():
    client = AsyncOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    # Load benchmark dockets
    bench_dks = set()
    for line in open(BENCH_PATH):
        bench_dks.add(json.loads(line)['docket_id'])

    # Load real claims
    logger.info("Loading real claims...")
    claims_by_dk = defaultdict(set)
    for line in open(MATCH2):
        if '"llm_label": 1' not in line:
            continue
        r = json.loads(line)
        if r.get('llm_label') == 1 and r['docket_id'] in bench_dks:
            claims_by_dk[r['docket_id']].add(r['claim_text'])
    claims_by_dk = {dk: list(v) for dk, v in claims_by_dk.items()}
    logger.info("Loaded %d unique real claims across %d dockets",
                sum(len(v) for v in claims_by_dk.values()), len(claims_by_dk))

    # Embed real claims (cached)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "real_claim_embs_minilm_local.pkl"

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if cache_file.exists():
        logger.info("Loading cached claim embeddings...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        claim_texts_global = cache["texts"]
        claim_embs_global = cache["embs"]
        claim_dk_idx = cache["dk_idx"]
    else:
        logger.info("Embedding real claims...")
        text_to_idx = {}
        claim_texts_global = []
        claim_dk_idx = {}
        for dk, texts in claims_by_dk.items():
            indices = []
            for t in texts:
                if t not in text_to_idx:
                    text_to_idx[t] = len(claim_texts_global)
                    claim_texts_global.append(t)
                indices.append(text_to_idx[t])
            claim_dk_idx[dk] = indices

        claim_embs_global = embedder.encode(claim_texts_global, normalize_embeddings=True,
                                             batch_size=256, show_progress_bar=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"texts": claim_texts_global, "embs": claim_embs_global, "dk_idx": claim_dk_idx}, f)
        logger.info("Cached %d claim embeddings", len(claim_texts_global))

    def find_top_k(query_embs, corpus_embs, k=3, min_score=0.3):
        sims = query_embs @ corpus_embs.T
        results = []
        for i in range(len(query_embs)):
            top_idx = np.argsort(sims[i])[-k:][::-1]
            results.append([(int(j), float(sims[i][j])) for j in top_idx if sims[i][j] > min_score])
        return results

    # Process each config
    for config_name in TARGET_CONFIGS:
        config_dir = RUNS / config_name
        gc_path = config_dir / "generated_claims.jsonl"
        pj_path = config_dir / "pair_judgments.jsonl"

        if not gc_path.exists():
            logger.info("SKIP %s: no generated_claims.jsonl", config_name)
            continue

        if pj_path.exists() and sum(1 for _ in open(pj_path)) > 100:
            logger.info("SKIP %s: already judged (%d pairs)", config_name,
                        sum(1 for _ in open(pj_path)))
            continue

        generated = [json.loads(l) for l in open(gc_path)]
        logger.info("=== %s: %d dockets ===", config_name, len(generated))

        # Embed LLM claims
        all_llm_claims = []
        claim_offsets = []
        for rec in generated:
            offset = len(all_llm_claims)
            claims = [c['claim_text'] for c in rec.get('parsed_claims', [])]
            all_llm_claims.extend(claims)
            claim_offsets.append((offset, len(claims)))

        if not all_llm_claims:
            logger.info("  No claims to judge")
            continue

        logger.info("  Embedding %d LLM claims...", len(all_llm_claims))
        llm_embs = embedder.encode(all_llm_claims, normalize_embeddings=True,
                                    batch_size=256, show_progress_bar=False)

        # Build all pairs via top-3 retrieval
        all_calls = []  # (pair_data_list, prompt)
        all_pairs = []  # flat list of all pairs for output

        for rec_idx, rec in enumerate(generated):
            dk = rec['docket_id']
            offset, n_claims = claim_offsets[rec_idx]
            if n_claims == 0 or dk not in claim_dk_idx:
                continue

            dk_indices = np.array(claim_dk_idx[dk])
            dk_embs = claim_embs_global[dk_indices]
            dk_texts = [claim_texts_global[i] for i in dk_indices]
            q_embs = llm_embs[offset:offset + n_claims]
            q_texts = all_llm_claims[offset:offset + n_claims]

            top_k = find_top_k(q_embs, dk_embs, k=3)

            batch_buffer = []
            for qi, neighbors in enumerate(top_k):
                for ci, score in neighbors:
                    pair = {
                        'docket_id': dk,
                        'llm_text': q_texts[qi],
                        'real_text': dk_texts[ci],
                        'cosine': score,
                        'label': 'pending',
                        'model': rec.get('model', ''),
                        'variant': rec.get('variant', ''),
                    }
                    pair_idx = len(all_pairs)
                    all_pairs.append(pair)
                    batch_buffer.append((pair_idx, dk_texts[ci], q_texts[qi]))

                    if len(batch_buffer) == BATCH_SIZE:
                        pairs_text = "\n".join(
                            f"{j+1}. A: \"{rt[:200]}\" | B: \"{lt[:200]}\""
                            for j, (_, rt, lt) in enumerate(batch_buffer))
                        prompt = JUDGE_BATCH.format(pairs=pairs_text, n=len(batch_buffer))
                        all_calls.append(([idx for idx, _, _ in batch_buffer], prompt))
                        batch_buffer = []

            # Flush remaining
            if batch_buffer:
                pairs_text = "\n".join(
                    f"{j+1}. A: \"{rt[:200]}\" | B: \"{lt[:200]}\""
                    for j, (_, rt, lt) in enumerate(batch_buffer))
                prompt = JUDGE_BATCH.format(pairs=pairs_text, n=len(batch_buffer))
                all_calls.append(([idx for idx, _, _ in batch_buffer], prompt))

        logger.info("  %d pairs, %d API calls (batch=%d)", len(all_pairs), len(all_calls), BATCH_SIZE)

        # Fire all calls
        completed = [0]
        t0 = time.time()

        async def judge_one(call_info):
            pair_indices, prompt = call_info
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=200,
                        )
                    raw = resp.choices[0].message.content or ""
                    labels = parse_batch_labels(raw, len(pair_indices))
                except Exception as e:
                    labels = ['error'] * len(pair_indices)

            for idx, label in zip(pair_indices, labels):
                all_pairs[idx]['label'] = label

            completed[0] += 1
            if completed[0] % 500 == 0:
                logger.info("    %d/%d calls (%.0fs)", completed[0], len(all_calls), time.time() - t0)

        await asyncio.gather(*[judge_one(c) for c in all_calls])

        # Save
        with open(pj_path, 'w') as f:
            for p in all_pairs:
                f.write(json.dumps(p) + '\n')

        # Stats
        from collections import Counter
        labels = Counter(p['label'] for p in all_pairs)
        total = len(all_pairs)
        same = labels.get('same', 0)
        related = labels.get('related', 0)
        logger.info("  %s: %d same (%.1f%%), %d related (%.1f%%), %d total, %d error",
                    config_name, same, same/max(total,1)*100,
                    related, related/max(total,1)*100, total, labels.get('error', 0))
        logger.info("  Done in %.0fs", time.time() - t0)

    logger.info("=== ALL PAIR JUDGING DONE ===")


if __name__ == '__main__':
    asyncio.run(main())
