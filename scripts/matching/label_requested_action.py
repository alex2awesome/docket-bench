"""Add requested_action dimension to existing claim taxonomy labels.

Categories:
  - add_exception: commenter requests a carve-out, exemption, or waiver
  - remove_provision: commenter wants a provision deleted entirely
  - strengthen: commenter wants stricter requirements, more enforcement, broader scope
  - weaken: commenter wants looser requirements, fewer restrictions, narrower scope
  - add_requirement: commenter wants a NEW obligation, prohibition, or condition added
  - clarify: commenter wants language clarified, definitions tightened, ambiguity resolved
  - extend_timeline: commenter wants more time for compliance, longer phase-in
  - no_change: commenter supports the provision as-is
  - other: doesn't fit the above

Usage:
    python label_requested_action.py [--max-claims 1000] [--model gpt-4o-mini]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT = """Classify this public comment's requested action into ONE of these categories:

- add_exception: requests a carve-out, exemption, safe harbor, or waiver from a requirement
- remove_provision: wants a provision, section, or requirement deleted entirely
- strengthen: wants stricter standards, more enforcement, broader applicability, or tighter limits
- weaken: wants looser standards, fewer restrictions, higher thresholds, or narrower applicability
- add_requirement: wants a NEW obligation, prohibition, condition, or reporting requirement added that doesn't exist in the proposal
- clarify: wants language clarified, terms defined, ambiguity resolved, or guidance added
- extend_timeline: wants more time for compliance, longer comment periods, phased implementation
- no_change: supports the provision as proposed, no changes requested
- other: doesn't fit any of the above

Comment claim:
{claim_text}

Respond with ONLY the category name (e.g., "add_exception"). Nothing else."""


async def label_claims(claims: list[dict], model: str = "gpt-4o-mini",
                       concurrency: int = 100) -> list[str]:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        for p in [Path.home() / ".openai-key.txt"]:
            if p.exists():
                os.environ["OPENAI_API_KEY"] = p.read_text().strip()
                break

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(claims)

    async def _call(i, claim):
        async with sem:
            prompt = PROMPT.format(claim_text=claim["claim_text"][:1000])
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0,
                )
                results[i] = r.choices[0].message.content.strip().lower()
            except Exception as e:
                logger.debug("claim %d failed: %s", i, e)
                results[i] = "other"

    tasks = [_call(i, c) for i, c in enumerate(claims)]
    # Process in batches
    batch_size = 500
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        await asyncio.gather(*batch)
        logger.info("  labeled %d/%d", min(start + batch_size, len(claims)), len(claims))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-claims", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    input_path = Path("data/bulk_downloads/deontic_units/claim_taxonomy_labels.jsonl")
    output_path = Path("data/bulk_downloads/deontic_units/claim_taxonomy_labels_v2.jsonl")

    claims = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    # Skip error rows without claim_text
    claims = [c for c in claims if c.get("claim_text")]
    logger.info("Loaded %d claims (skipped rows without claim_text)", len(claims))

    # Skip claims already labeled
    if output_path.exists():
        done = {json.loads(l)["claim_id"] for l in output_path.read_text().splitlines() if l.strip()}
        claims = [c for c in claims if c.get("claim_id") not in done]
        logger.info("After skipping done: %d remaining", len(claims))

    if args.max_claims:
        claims = claims[:args.max_claims]

    if not claims:
        logger.info("Nothing to do")
        return

    # Estimate cost
    avg_tokens = 200  # input
    total_input = len(claims) * avg_tokens
    cost = total_input / 1e6 * 0.15 + len(claims) * 10 / 1e6 * 0.60  # gpt-4o-mini pricing
    logger.info("Estimated cost: $%.2f for %d claims", cost, len(claims))

    labels = asyncio.run(label_claims(claims, model=args.model))

    # Write output
    with open(output_path, "a") as f:
        for claim, label in zip(claims, labels):
            claim["requested_action"] = label
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    dist = Counter(labels)
    logger.info("Distribution:")
    for k, v in dist.most_common():
        logger.info("  %s: %d (%.1f%%)", k, v, v / len(labels) * 100)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent.parent)
    main()
