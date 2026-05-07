"""Match framing/structure via OpenRouter (Llama-3.3-70B).

Uses clustered real taxonomy + batched judging (5 pairs per prompt).

Usage:
    OPENROUTER_KEY=$(cat ~/.openrouter-key.txt) \
        python scripts/match_via_openrouter.py --configs full_gpt-5-mini_subgroup,full_llama-3.3-70b_vanilla
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(os.environ.get("DOCKET_BASE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RUNS = BASE / "data" / "derived" / "benchmark_runs"
CLUSTER_DIR = BASE / "data" / "derived" / "clustered_taxonomy"

MODEL = "meta-llama/llama-3.3-70b-instruct"
CONCURRENCY = 50
BATCH_SIZE = 5

JUDGE_FRAME_BATCH = """Classify each pair of argumentative framings. For each, respond with one word: same, related, or different.

- same: same analytical approach, even if worded differently
- related: same broad topic but different specific approaches
- different: completely unrelated types of arguments

{pairs}

{n} labels, one per line:"""

JUDGE_STRUCT_BATCH = """Classify each pair of writing/evidence techniques. For each, respond with one word: same, related, or different.

- same: same technique or evidence form, even if worded differently
- related: both about presentation but different techniques
- different: completely unrelated

{pairs}

{n} labels, one per line:"""


def load_clusters(dim):
    path = CLUSTER_DIR / f"{dim}_clusters.jsonl"
    by_dk = {}
    for line in open(path):
        r = json.loads(line)
        by_dk[r['docket_id']] = list(set(r['item_to_canonical'].values()))
    return by_dk


def parse_batch_labels(raw, n):
    labels = [w for w in re.findall(r'[a-z]+', raw.lower()) if w in {'same', 'related', 'different'}]
    while len(labels) < n:
        labels.append('error')
    return labels[:n]


async def match_config(client, sem, config_name, frame_clusters, struct_clusters):
    extract_path = RUNS / config_name / 'extracted_llama_v2.jsonl'
    match_path = RUNS / config_name / 'match_results_v3.jsonl'

    if match_path.exists() and sum(1 for _ in open(match_path)) > 100:
        logger.info("SKIP %s: already matched", config_name)
        return

    extractions = [json.loads(l) for l in open(extract_path)]
    logger.info("=== MATCHING %s: %d dockets ===", config_name, len(extractions))

    all_results = []
    t0 = time.time()

    # Pre-build all results and collect all API calls
    all_calls = []  # (call_idx, rec_idx, dim, pair_indices, batch_pairs)

    for rec_idx, rec in enumerate(extractions):
        dk = rec['docket_id']
        result = {
            'docket_id': dk,
            'model': rec.get('model', ''),
            'variant': rec.get('variant', ''),
            'stakeholder': rec.get('stakeholder'),
            'claim_pairs': [],
            'framing_pairs': [],
            'structure_pairs': [],
        }

        for dim, prompt_tmpl, clusters_by_dk in [
            ('framing', JUDGE_FRAME_BATCH, frame_clusters),
            ('structure', JUDGE_STRUCT_BATCH, struct_clusters),
        ]:
            llm_items = rec.get(dim, [])
            canonicals = clusters_by_dk.get(dk, [])
            key = f'{dim}_pairs'

            if not llm_items or not canonicals:
                continue

            all_pairs = [(li, rc) for li in llm_items for rc in canonicals]

            for bi in range(0, len(all_pairs), BATCH_SIZE):
                batch = all_pairs[bi:bi + BATCH_SIZE]
                pair_indices = []
                for li, rc in batch:
                    result[key].append({'llm_text': li, 'real_text': rc, 'cosine': 0.0, 'label': 'pending'})
                    pair_indices.append(len(result[key]) - 1)

                pairs_text = "\n".join(f"{j+1}. A: \"{rc}\" | B: \"{li}\"" for j, (li, rc) in enumerate(batch))
                prompt = prompt_tmpl.format(pairs=pairs_text, n=len(batch))
                all_calls.append((rec_idx, dim, pair_indices, prompt))

        all_results.append(result)

    logger.info("  Built %d API calls across %d dockets", len(all_calls), len(extractions))

    # Fire all calls with concurrency
    completed = [0]

    async def judge_one(call_info):
        rec_idx, dim, pair_indices, prompt = call_info
        key = f'{dim}_pairs'
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50, temperature=0)
                raw = resp.choices[0].message.content or ""
                labels = parse_batch_labels(raw, len(pair_indices))
            except Exception as e:
                labels = ['error'] * len(pair_indices)

        for idx, label in zip(pair_indices, labels):
            all_results[rec_idx][key][idx]['label'] = label

        completed[0] += 1
        if completed[0] % 500 == 0:
            elapsed = time.time() - t0
            logger.info("    %d/%d calls (%.0fs)", completed[0], len(all_calls), elapsed)

    await asyncio.gather(*[judge_one(c) for c in all_calls])

    # Save
    with open(match_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    # Stats
    for dim in ['framing', 'structure']:
        key = f'{dim}_pairs'
        total = sum(len(r[key]) for r in all_results)
        same = sum(1 for r in all_results for p in r[key] if p.get('label') == 'same')
        related = sum(1 for r in all_results for p in r[key] if p.get('label') == 'related')
        error = sum(1 for r in all_results for p in r[key] if p.get('label') == 'error')
        logger.info("  %s %s: %d same (%.1f%%), %d related (%.1f%%), %d error / %d total",
                    config_name, dim, same, same/max(total,1)*100,
                    related, related/max(total,1)*100, error, total)

    elapsed = time.time() - t0
    logger.info("  Done %s in %.0fs (%d API calls)", config_name, elapsed, total_calls)


async def main(configs):
    client = AsyncOpenAI(
        api_key=os.environ['OPENROUTER_KEY'],
        base_url="https://openrouter.ai/api/v1"
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    frame_clusters = load_clusters("framing")
    struct_clusters = load_clusters("structure")
    logger.info("Loaded clusters: %d framing, %d structure dockets",
                len(frame_clusters), len(struct_clusters))

    for config_name in configs:
        await match_config(client, sem, config_name, frame_clusters, struct_clusters)

    logger.info("=== ALL DONE ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', required=True, type=str)
    args = parser.parse_args()
    asyncio.run(main(args.configs.split(',')))
