"""Compute pluralism grid results from match results.

Takes match_results_v3.jsonl per config and produces:
  1. Per-config Overton scores (claim/framing/structure recall)
  2. Per-config Steerable scores (persona Match/Pop by org_type)
  3. Aggregate results table for the paper

Usage:
    python scripts/compute_pluralism_results.py
"""

import json
import logging
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(os.environ.get("DOCKET_BASE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RUNS = BASE / "data" / "derived" / "benchmark_runs"
REAL_TAX = BASE / "data" / "derived" / "real_comment_taxonomy_llama_v3_deduped.jsonl"
MATCH2 = BASE / "data" / "bulk_downloads" / "deontic_units" / "match2_llm_labeled.jsonl"
ORG_LOOKUP = BASE / "data" / "policy-feedback-bench" / "benchmark" / "commenter_org_type_lookup_558.csv.gz"
PERSONA_WEIGHTS = BASE / "data" / "policy-feedback-bench" / "benchmark" / "docket_persona_weights.json"

ORG_BUCKETS = {
    'corporation': 'industry', 'trade_association': 'industry',
    'law_firm': 'lawfirm',
    'labor_union': 'labor',
    'ngo_advocacy': 'advocacy',
    'academic': 'expert', 'professional_society': 'expert',
    'government': 'gov',
    'individual': 'individual',
}


def load_org_lookup():
    """Load cluster_uid → org_type mapping."""
    df = pd.read_csv(ORG_LOOKUP)
    lookup = {}
    for _, row in df.iterrows():
        cuid = row['cluster_uid']
        org = ORG_BUCKETS.get(row.get('org_type', ''), None)
        if org:
            lookup[cuid] = org
    logger.info("Org lookup: %d cluster_uids mapped", len(lookup))
    return lookup


def load_real_taxonomy_by_docket():
    """Load real framing/structure grouped by docket."""
    by_dk = defaultdict(lambda: {'framing': set(), 'structure': set(), 'org_types': Counter()})
    for line in open(REAL_TAX):
        r = json.loads(line)
        dk = r['docket_id']
        for f in r.get('framing', []):
            by_dk[dk]['framing'].add(f.lower())
        for s in r.get('structure', []):
            by_dk[dk]['structure'].add(s.lower())
        org = ORG_BUCKETS.get(r.get('org_type', ''), None)
        if org:
            by_dk[dk]['org_types'][org] += 1
    return by_dk


def load_real_taxonomy_by_docket_and_org(org_lookup):
    """Load real framing/structure grouped by (docket, org_type)."""
    by_dk_org = defaultdict(lambda: defaultdict(lambda: {'framing': set(), 'structure': set()}))
    for line in open(REAL_TAX):
        r = json.loads(line)
        dk = r['docket_id']
        org = ORG_BUCKETS.get(r.get('org_type', ''), None)
        if not org:
            continue
        for f in r.get('framing', []):
            by_dk_org[dk][org]['framing'].add(f.lower())
        for s in r.get('structure', []):
            by_dk_org[dk][org]['structure'].add(s.lower())
    return by_dk_org


def compute_overton(match_results):
    """Compute Overton recall: fraction of LLM items that match real items.

    For each dimension (claim/framing/structure), what fraction of LLM-generated
    items were judged as 'same' to at least one real item?
    """
    metrics = {}
    for dim in ['claim', 'framing', 'structure']:
        key = f'{dim}_pairs'
        total_llm_items = set()
        matched_llm_items = set()

        for r in match_results:
            dk = r['docket_id']
            for p in r.get(key, []):
                item_key = (dk, p['llm_text'])
                total_llm_items.add(item_key)
                if p.get('label') == 'same':
                    matched_llm_items.add(item_key)

        n_total = len(total_llm_items)
        n_matched = len(matched_llm_items)

        # Also compute per-docket recall and average
        per_docket = []
        for r in match_results:
            dk_total = set()
            dk_matched = set()
            for p in r.get(key, []):
                dk_total.add(p['llm_text'])
                if p.get('label') == 'same':
                    dk_matched.add(p['llm_text'])
            if dk_total:
                per_docket.append(len(dk_matched) / len(dk_total))

        metrics[dim] = {
            'global_recall': n_matched / max(n_total, 1),
            'mean_per_docket_recall': np.mean(per_docket) if per_docket else 0,
            'n_total': n_total,
            'n_matched': n_matched,
            'n_dockets_with_pairs': len(per_docket),
        }

    return metrics


def compute_steerable(match_results, org_lookup, real_by_dk_org, persona_weights):
    """Compute Steerable Match/Pop per org_type group.

    For persona-conditioned generation, does the persona's output
    disproportionately match its real counterpart's patterns?
    """
    # Count matches by (docket, org_type of matched real item)
    matches_by_org = Counter()
    total_by_org = Counter()

    for r in match_results:
        dk = r['docket_id']
        stakeholder = r.get('stakeholder')  # persona used for generation

        for dim in ['claim', 'framing', 'structure']:
            key = f'{dim}_pairs'
            for p in r.get(key, []):
                if p.get('label') == 'same':
                    # Try to find org_type of the matched real item
                    # For claims, we need cluster_uid → org_type
                    # For framing/structure, use the real taxonomy org_type
                    # For now, count by the persona used
                    if stakeholder:
                        matches_by_org[(stakeholder, dim)] += 1
                total_by_org[(stakeholder or 'vanilla', dim)] += 1

    return matches_by_org, total_by_org


def compute_results_table():
    """Compute full results table across all configs."""
    org_lookup = load_org_lookup()
    real_by_dk = load_real_taxonomy_by_docket()
    real_by_dk_org = load_real_taxonomy_by_docket_and_org(org_lookup)
    pw = json.load(open(PERSONA_WEIGHTS)).get('per_docket', {})

    rows = []

    for d in sorted(RUNS.iterdir()):
        if not d.name.startswith('full_'):
            continue
        match_path = d / 'match_results_v3.jsonl'
        if not match_path.exists():
            continue

        match_results = [json.loads(l) for l in open(match_path)]

        # Check if results are non-empty
        total_pairs = sum(
            len(r.get('claim_pairs', [])) + len(r.get('framing_pairs', [])) + len(r.get('structure_pairs', []))
            for r in match_results
        )
        if total_pairs == 0:
            logger.warning("SKIP %s: 0 pairs", d.name)
            continue

        # Parse config name
        parts = d.name.replace('full_', '').split('_')
        # e.g. "gpt-4o-mini_vanilla" or "gpt-5-mini_rag_persona"

        overton = compute_overton(match_results)
        matches_by_org, total_by_org = compute_steerable(
            match_results, org_lookup, real_by_dk_org, pw)

        row = {
            'config': d.name,
            # Overton: global recall per dimension
            'O_claim': overton['claim']['global_recall'],
            'O_framing': overton['framing']['global_recall'],
            'O_structure': overton['structure']['global_recall'],
            # Per-docket mean recall
            'O_claim_perdocket': overton['claim']['mean_per_docket_recall'],
            'O_framing_perdocket': overton['framing']['mean_per_docket_recall'],
            'O_structure_perdocket': overton['structure']['mean_per_docket_recall'],
            # Counts
            'n_claim_pairs': overton['claim']['n_total'],
            'n_claim_matched': overton['claim']['n_matched'],
            'n_framing_pairs': overton['framing']['n_total'],
            'n_framing_matched': overton['framing']['n_matched'],
            'n_structure_pairs': overton['structure']['n_total'],
            'n_structure_matched': overton['structure']['n_matched'],
            'n_dockets_claims': overton['claim']['n_dockets_with_pairs'],
            'n_dockets_framing': overton['framing']['n_dockets_with_pairs'],
            'n_dockets_structure': overton['structure']['n_dockets_with_pairs'],
        }
        rows.append(row)

        logger.info("%s: O_claim=%.3f O_frame=%.3f O_struct=%.3f (pairs: %d/%d/%d)",
                    d.name,
                    row['O_claim'], row['O_framing'], row['O_structure'],
                    row['n_claim_pairs'], row['n_framing_pairs'], row['n_structure_pairs'])

    if not rows:
        logger.warning("No match results found!")
        return

    df = pd.DataFrame(rows)

    # Print summary table
    print("\n" + "=" * 90)
    print("PLURALISM GRID: Overton Recall (fraction of LLM items matching real)")
    print("=" * 90)
    print(f"{'Config':<45} {'Claims':>8} {'Framing':>8} {'Structure':>8} {'Pairs':>12}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['config']:<45} {row['O_claim']:>8.3f} {row['O_framing']:>8.3f} {row['O_structure']:>8.3f} "
              f"{row['n_claim_pairs']:>4.0f}/{row['n_framing_pairs']:>3.0f}/{row['n_structure_pairs']:>3.0f}")

    # Per-docket mean
    print(f"\n{'Config':<45} {'Claims':>8} {'Framing':>8} {'Structure':>8}  (per-docket mean)")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['config']:<45} {row['O_claim_perdocket']:>8.3f} {row['O_framing_perdocket']:>8.3f} {row['O_structure_perdocket']:>8.3f}")

    # Save
    out_path = BASE / "data" / "derived" / "pluralism_results.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)

    # Also print label distributions per config
    print("\n" + "=" * 90)
    print("LABEL DISTRIBUTIONS")
    print("=" * 90)
    for d_dir in sorted(RUNS.iterdir()):
        if not d_dir.name.startswith('full_'):
            continue
        match_path = d_dir / 'match_results_v3.jsonl'
        if not match_path.exists():
            continue

        match_results = [json.loads(l) for l in open(match_path)]

        for dim in ['claim', 'framing', 'structure']:
            key = f'{dim}_pairs'
            labels = Counter()
            for r in match_results:
                for p in r.get(key, []):
                    labels[p.get('label', 'missing')] += 1
            if labels:
                total = sum(labels.values())
                dist = ', '.join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in labels.most_common())
                print(f"  {d_dir.name} {dim}: {dist}")


def compute_bootstrap_ci(match_results, n_resamples=1000, seed=42):
    """Bootstrap 95% CI for Overton recall."""
    rng = np.random.RandomState(seed)

    results = {}
    for dim in ['claim', 'framing', 'structure']:
        key = f'{dim}_pairs'

        # Per-docket recall values
        per_docket = []
        for r in match_results:
            dk_total = set()
            dk_matched = set()
            for p in r.get(key, []):
                dk_total.add(p['llm_text'])
                if p.get('label') == 'same':
                    dk_matched.add(p['llm_text'])
            if dk_total:
                per_docket.append(len(dk_matched) / len(dk_total))

        if not per_docket:
            results[dim] = {'mean': 0, 'ci_lo': 0, 'ci_hi': 0}
            continue

        per_docket = np.array(per_docket)
        boot_means = []
        for _ in range(n_resamples):
            sample = rng.choice(per_docket, size=len(per_docket), replace=True)
            boot_means.append(sample.mean())

        boot_means = np.array(boot_means)
        results[dim] = {
            'mean': per_docket.mean(),
            'ci_lo': np.percentile(boot_means, 2.5),
            'ci_hi': np.percentile(boot_means, 97.5),
        }

    return results


def inspect_matches(match_path, n_samples=10):
    """Print sample matches for manual inspection."""
    import random
    random.seed(42)

    match_results = [json.loads(l) for l in open(match_path)]

    for dim in ['claim', 'framing', 'structure']:
        key = f'{dim}_pairs'
        same_pairs = []
        for r in match_results:
            for p in r.get(key, []):
                if p.get('label') == 'same':
                    same_pairs.append((r['docket_id'], p))

        print(f"\n{'='*60}")
        print(f"SAME matches — {dim} ({len(same_pairs)} total)")
        print(f"{'='*60}")

        if same_pairs:
            samples = random.sample(same_pairs, min(n_samples, len(same_pairs)))
            for dk, p in samples:
                print(f"  [{dk}] cos={p['cosine']:.3f}")
                print(f"    Real: {p['real_text'][:80]}")
                print(f"    LLM:  {p['llm_text'][:80]}")
                print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', type=str, help='Config name to inspect matches')
    args = parser.parse_args()

    if args.inspect:
        match_path = RUNS / args.inspect / 'match_results_v3.jsonl'
        if match_path.exists():
            inspect_matches(match_path)
        else:
            print(f"No match results at {match_path}")
    else:
        compute_results_table()
