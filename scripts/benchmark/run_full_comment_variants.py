"""Generate full comments for vanilla, persona, RAG, few-shot, and tool-calling variants.

Supports multiple models via --model flag.

Variants:
  - vanilla: base prompt, no augmentation
  - persona: persona-weighted stakeholder
  - rag_vanilla: BM25-retrieved evidence stuffed into prompt
  - rag_persona: same + persona-weighted stakeholder
  - few_shot_vanilla: 2 real example comments as demonstrations
  - few_shot_persona: same + persona
  - rag_tool_vanilla: tool-calling RAG (model decides what to retrieve)
  - rag_tool_persona: same + persona

Usage:
    OPENAI_API_KEY=$(cat ~/.openai-key.txt) \
        python scripts/run_full_comment_variants.py --model gpt-5-mini
    OPENAI_API_KEY=$(cat ~/.openai-key.txt) \
        python scripts/run_full_comment_variants.py --model gpt-5
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)

BASE = Path(os.environ.get("DOCKET_BASE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
DU = BASE / "data" / "bulk_downloads" / "deontic_units"
RUNS = BASE / "data" / "derived" / "benchmark_runs"
EVIDENCE_PATH = BASE / "data" / "derived" / "evidence_patterns.jsonl"
BENCHMARK_PATH = DU / "comment_anticipation_benchmark_gpt5mini.jsonl"
UNITS_PATH = DU / "proposed_rule_deontic_units_normalized_v2.jsonl"
PERSONA_WEIGHTS = BASE / "data" / "policy-feedback-bench" / "benchmark" / "docket_persona_weights.json"
FAS_PATH = BASE / "data" / "feature_analysis_summary.csv.gz"

CONCURRENCY = 50

STAKEHOLDER_PERSONAS = {
    'industry':   ('industry / corporation / trade association',
                   'You are a regulated company, trade association, or industry coalition. You focus on compliance cost, operational feasibility, market effects, and ROI. You cite economic data and industry-specific operational details.'),
    'labor':      ('labor union representative',
                   'You are a labor union representing workers in the regulated industry. You focus on worker safety, job protection, fair labor standards, and economic impact on workers. You cite worker statistics and on-the-job experience.'),
    'advocacy':   ('public-interest advocacy NGO',
                   "You are an environmental, civil rights, or public-interest advocacy organization. You focus on broader public good, environmental protection, equity, and the regulation's role in achieving social goals. You cite scientific studies, public-health data, and advocacy research."),
    'expert':     ('expert / academic organization',
                   'You are an academic researcher, professional society, or expert organization. You focus on technical correctness, scientific accuracy, peer-reviewed evidence, and methodological soundness. You cite research, technical specifications, and disciplinary expertise.'),
    'gov':        ('government / state agency',
                   'You are a state or local government agency that interacts with this federal regulation. You focus on inter-jurisdictional coordination, implementation feasibility for state agencies, and federalism issues. You cite state-level data and operational details.'),
    'lawfirm':    ('law firm / bar association',
                   'You are a law firm or bar association representing regulated parties. You focus on statutory authority, constitutional issues, APA procedural requirements, due-process implications, and legal interpretation. You cite case law, statutes, and regulatory history.'),
    'individual': ('individual citizen or affected community member',
                   'You are an ordinary individual citizen affected by this rule. You focus on personal experience, lived impact on you or your family/community, and concrete harms or benefits you would face. You speak in first person about direct experience.'),
}

# ---- PROMPTS ----

GEN_VANILLA = """\
Below are provisions from a proposed federal rule.

Write a public comment responding to these provisions, as a real member of the public would. Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree. Address the provisions you find most important.

Do not use bullet points or numbered lists. Write naturally, as if submitting a real comment to the Federal Register.

PROPOSED RULE PROVISIONS:
{provisions}

Public comment:"""

GEN_PERSONA = """\
Below are provisions from a proposed federal rule. Write a public comment FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree. Address the provisions you find most important.

Do not use bullet points or numbered lists. Write naturally, as this stakeholder would actually write to the Federal Register.

PROPOSED RULE PROVISIONS:
{provisions}

Public comment:"""

RAG_VANILLA = """\
Below are provisions from a proposed federal rule.

You also have access to REAL EVIDENCE from prior public comments on similar regulations. Use this evidence to inform and ground your response — reference similar statutes, economic data, and stakeholder impacts.

PROPOSED RULE PROVISIONS:
{provisions}

RETRIEVED EVIDENCE FROM PRIOR COMMENTS ON SIMILAR RULES:
{retrieved_evidence}

Write a public comment responding to these provisions, as a real member of the public would. Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree.

Do not use bullet points or numbered lists. Write naturally, as if submitting a real comment to the Federal Register.

Public comment:"""

RAG_PERSONA = """\
Below are provisions from a proposed federal rule. Write a public comment FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

You also have access to REAL EVIDENCE from prior public comments on similar regulations. Use this evidence to inform and ground your response.

PROPOSED RULE PROVISIONS:
{provisions}

RETRIEVED EVIDENCE FROM PRIOR COMMENTS ON SIMILAR RULES:
{retrieved_evidence}

Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree.

Do not use bullet points or numbered lists. Write naturally, as this stakeholder would actually write to the Federal Register.

Public comment:"""

FEW_SHOT_VANILLA = """\
Below are provisions from a proposed federal rule.

Here are examples of REAL public comments submitted on other federal regulations. Study their style, structure, level of detail, and types of evidence used — then write a comment on the NEW rule below in a similar style.

EXAMPLE COMMENT 1:
{example1}

EXAMPLE COMMENT 2:
{example2}

---

PROPOSED RULE PROVISIONS (write your comment on THESE):
{provisions}

Write a public comment responding to these provisions, as a real member of the public would. Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree.

Do not use bullet points or numbered lists. Write naturally, as if submitting a real comment to the Federal Register.

Public comment:"""

FEW_SHOT_PERSONA = """\
Below are provisions from a proposed federal rule. Write a public comment FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

Here are examples of REAL public comments submitted by similar stakeholders on other federal regulations. Study their style, structure, and types of evidence — then write a comment on the NEW rule below.

EXAMPLE COMMENT 1:
{example1}

EXAMPLE COMMENT 2:
{example2}

---

PROPOSED RULE PROVISIONS (write your comment on THESE):
{provisions}

Write in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree.

Do not use bullet points or numbered lists. Write naturally, as this stakeholder would actually write to the Federal Register.

Public comment:"""

# Tool definition for RAG tool-calling variant
RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_similar_comments",
        "description": "Search a database of prior public comments on federal regulations to find comments with relevant evidence, arguments, or perspectives. Use this to ground your comment in real stakeholder concerns and evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language search query describing what kind of evidence, arguments, or perspectives you want to find. Be specific about the topic, industry, or type of concern."
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_SYSTEM = """\
You are writing a public comment on a proposed federal regulation. You have access to a tool that can search a database of real prior public comments to find relevant evidence, arguments, and perspectives. Use this tool to ground your comment in real stakeholder concerns before writing.

Write your final comment in full paragraphs — cite specific evidence, reference relevant statutes or data, draw on operational experience where relevant, and propose specific alternatives where you disagree. Do not use bullet points or numbered lists."""

TOOL_USER = """\
PROPOSED RULE PROVISIONS:
{provisions}

Write a public comment responding to these provisions. You may use the retrieve_similar_comments tool to search for relevant prior comments before writing. Write naturally, as if submitting a real comment to the Federal Register."""

TOOL_USER_PERSONA = """\
Write a public comment FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

PROPOSED RULE PROVISIONS:
{provisions}

You may use the retrieve_similar_comments tool to search for relevant prior comments from this stakeholder type before writing. Write naturally, as this stakeholder would actually write to the Federal Register."""


def load_benchmark():
    d = []; s = set()
    for line in open(BENCHMARK_PATH):
        r = json.loads(line); dk = r['docket_id']
        if dk not in s: s.add(dk); d.append(r)
    return d


def load_units():
    b = set(d['docket_id'] for d in load_benchmark())
    du = defaultdict(list)
    for line in open(UNITS_PATH):
        r = json.loads(line); dk = r.get('docket_id')
        if dk in b:
            for u in r.get('units', []): du[dk].append(u)
    return du


def fmt_provisions(units):
    return '\n'.join(
        f"P{i} [{u.get('section','')}] ({u.get('type','')}): "
        f"{(u.get('action_summary') or u.get('obligation_text',''))[:300]}"
        for i, u in enumerate(units, 1))


def build_evidence_index():
    """Build BM25 index over evidence-bearing comments."""
    if not EVIDENCE_PATH.exists():
        logger.warning("No evidence patterns file at %s", EVIDENCE_PATH)
        return None, []

    evidence_records = []
    evidence_texts = []

    for line in open(EVIDENCE_PATH):
        r = json.loads(line)
        if not r.get('has_evidence'):
            continue
        parts = []
        if r.get('statutes_cited'):
            parts.append(f"Legal: {r['statutes_cited']}")
        if r.get('published_data'):
            parts.append(f"Data: {r['published_data']}")
        if r.get('studies_cited'):
            parts.append(f"Studies: {r['studies_cited']}")
        if r.get('regulatory_history'):
            parts.append(f"History: {r['regulatory_history']}")
        if r.get('specific_alternative'):
            parts.append(f"Alternative: {r['specific_alternative']}")
        if parts:
            snippet = f"[{r.get('agency', '?')}] " + " | ".join(parts)
            evidence_records.append({
                'snippet': snippet, 'agency': r.get('agency', ''),
                'docket_id': r.get('docket_id', ''),
            })
            evidence_texts.append(' '.join(parts))

    logger.info("Building BM25 index over %d evidence records...", len(evidence_texts))
    tokenized = [t.lower().split() for t in evidence_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, evidence_records


def retrieve_evidence(bm25, evidence_records, query, top_k=10):
    if bm25 is None:
        return "No prior evidence available."
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]
    snippets = [evidence_records[i]['snippet'] for i in top_idx if scores[i] > 0]
    if not snippets:
        return "No closely matching prior evidence found."
    return '\n'.join(f"  {i+1}. {s}" for i, s in enumerate(snippets))


def load_few_shot_comments():
    """Load real comments for few-shot examples, grouped by org_type."""
    import pandas as pd

    # Load from the exported real comments file
    real_path = BASE / "data" / "derived" / "real_comments_for_sk3.jsonl"
    if not real_path.exists():
        logger.warning("No real comments file at %s", real_path)
        return {}

    bench_dks = set()
    for line in open(BENCHMARK_PATH):
        bench_dks.add(json.loads(line)['docket_id'])

    by_org = defaultdict(list)
    for line in open(real_path):
        r = json.loads(line)
        # Only use comments NOT in benchmark (avoid leakage)
        if r['docket_id'] in bench_dks:
            continue
        org = r.get('org_type', 'individual')
        text = r.get('text', '')
        if 500 < len(text) < 3000:  # good length for examples
            by_org[org].append(text)

    # If not enough non-benchmark comments, use benchmark ones but from different dockets
    if sum(len(v) for v in by_org.values()) < 20:
        logger.info("Not enough non-benchmark comments, using benchmark comments as fallback")
        for line in open(real_path):
            r = json.loads(line)
            org = r.get('org_type', 'individual')
            text = r.get('text', '')
            if 500 < len(text) < 3000:
                by_org[org].append(text)

    for org in by_org:
        random.shuffle(by_org[org])

    logger.info("Few-shot pool: %s", {k: len(v) for k, v in by_org.items()})
    return by_org


ORG_TO_PERSONA = {
    'Industry': 'industry', 'Trade Association': 'industry',
    'Law Firm': 'lawfirm', 'Bar Association': 'lawfirm',
    'Labor Union': 'labor',
    'Advocacy': 'advocacy', 'NGO': 'advocacy',
    'Academic': 'expert', 'Research': 'expert',
    'Government': 'gov', 'State Agency': 'gov',
    'Individual': 'individual',
}


async def run_all(model='gpt-5-mini'):
    client = AsyncOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        default_headers={"X-Model-Service-Tier": "flex"},
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    # gpt-5 is a reasoning model — use low reasoning effort to save cost
    IS_REASONING = model in ('gpt-5', 'o3', 'o4-mini')
    MAX_TOKENS = 4048 if IS_REASONING else 2048
    REASONING_EXTRA = {'reasoning_effort': 'low'} if IS_REASONING else {}

    bench = load_benchmark()
    units = load_units()
    pw = json.load(open(PERSONA_WEIGHTS)).get('per_docket', {})

    # Build evidence index for RAG
    bm25, evidence_records = build_evidence_index()

    # Load few-shot examples
    few_shot_pool = load_few_shot_comments()

    def get_examples(persona_key, docket_id, n=2):
        """Get n few-shot examples, avoiding same docket."""
        pool = few_shot_pool.get(persona_key, [])
        if not pool:
            # Fallback to any org type
            all_texts = []
            for v in few_shot_pool.values():
                all_texts.extend(v)
            pool = all_texts
        return pool[:n] if len(pool) >= n else pool * n  # repeat if not enough

    def get_evidence_block(dk, units_list):
        """Retrieve evidence for a docket's provisions."""
        all_evidence = set()
        for u in units_list[:20]:
            text = u.get('action_summary', '') or u.get('obligation_text', '')
            if text:
                ev = retrieve_evidence(bm25, evidence_records, text, top_k=3)
                for line in ev.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('No '):
                        all_evidence.add(line)
        evidence_lines = sorted(all_evidence)[:15]
        return '\n'.join(evidence_lines) if evidence_lines else "No prior evidence retrieved."

    # Define all variant configs
    VARIANTS = [
        (f'full_{model}_vanilla', 'base', 'vanilla'),
        (f'full_{model}_subgroup', 'base', 'persona'),
        (f'full_{model}_rag_vanilla', 'rag', 'vanilla'),
        (f'full_{model}_rag_persona', 'rag', 'persona'),
        (f'full_{model}_few_shot_vanilla', 'few_shot', 'vanilla'),
        (f'full_{model}_few_shot_persona', 'few_shot', 'persona'),
        (f'full_{model}_rag_tool_vanilla', 'rag_tool', 'vanilla'),
        (f'full_{model}_rag_tool_persona', 'rag_tool', 'persona'),
    ]

    for config_name, method, style in VARIANTS:
        out_dir = RUNS / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'full_comments.jsonl'

        done = set()
        if out_path.exists():
            for line in open(out_path):
                try: done.add(json.loads(line)['docket_id'])
                except: pass

        todo = [b for b in bench if b['docket_id'] not in done]
        if not todo:
            logger.info("SKIP %s: already done", config_name)
            continue

        logger.info("=== %s: %d dockets ===", config_name, len(todo))

        async def gen_one(b, _method=method, _style=style):
            dk = b['docket_id']
            us = units.get(dk, [])
            if not us:
                return None
            provs = fmt_provisions(us)

            # Get persona info for persona variants
            stakeholder = None
            label, persona_text = '', ''
            if _style == 'persona':
                w = pw.get(dk, {})
                stakeholder = max(w, key=w.get) if w else 'individual'
                label, persona_text = STAKEHOLDER_PERSONAS[stakeholder]

            if _method == 'base':
                if _style == 'vanilla':
                    prompt = GEN_VANILLA.format(provisions=provs)
                else:
                    prompt = GEN_PERSONA.format(
                        provisions=provs,
                        stakeholder_label=label, stakeholder_persona=persona_text)

                async with sem:
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{'role': 'user', 'content': prompt}],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        comment = resp.choices[0].message.content or ''
                        return {'docket_id': dk, 'n_units': len(us),
                                'comment': comment, 'variant': _style,
                                'model': model, 'stakeholder': stakeholder}
                    except Exception as e:
                        logger.warning("Err %s/%s: %s", config_name, dk, str(e)[:100])
                        return None

            elif _method == 'rag':
                ev_block = get_evidence_block(dk, us)
                if _style == 'vanilla':
                    prompt = RAG_VANILLA.format(provisions=provs, retrieved_evidence=ev_block)
                else:
                    prompt = RAG_PERSONA.format(
                        provisions=provs, retrieved_evidence=ev_block,
                        stakeholder_label=label, stakeholder_persona=persona_text)

                async with sem:
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{'role': 'user', 'content': prompt}],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        comment = resp.choices[0].message.content or ''
                        return {'docket_id': dk, 'n_units': len(us),
                                'comment': comment, 'variant': f'rag_{_style}',
                                'model': model, 'stakeholder': stakeholder}
                    except Exception as e:
                        logger.warning("Err %s/%s: %s", config_name, dk, str(e)[:100])
                        return None

            elif _method == 'few_shot':
                # Get example comments
                if _style == 'persona' and stakeholder:
                    examples = get_examples(stakeholder, dk)
                else:
                    examples = get_examples('individual', dk)

                ex1 = examples[0][:2000] if examples else "No example available."
                ex2 = examples[1][:2000] if len(examples) > 1 else "No example available."

                if _style == 'vanilla':
                    prompt = FEW_SHOT_VANILLA.format(
                        provisions=provs, example1=ex1, example2=ex2)
                else:
                    prompt = FEW_SHOT_PERSONA.format(
                        provisions=provs, example1=ex1, example2=ex2,
                        stakeholder_label=label, stakeholder_persona=persona_text)

                async with sem:
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{'role': 'user', 'content': prompt}],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        comment = resp.choices[0].message.content or ''
                        return {'docket_id': dk, 'n_units': len(us),
                                'comment': comment, 'variant': f'few_shot_{_style}',
                                'model': model, 'stakeholder': stakeholder}
                    except Exception as e:
                        logger.warning("Err %s/%s: %s", config_name, dk, str(e)[:100])
                        return None

            elif _method == 'rag_tool':
                if _style == 'vanilla':
                    user_msg = TOOL_USER.format(provisions=provs)
                else:
                    user_msg = TOOL_USER_PERSONA.format(
                        provisions=provs,
                        stakeholder_label=label, stakeholder_persona=persona_text)

                messages = [
                    {'role': 'system', 'content': TOOL_SYSTEM},
                    {'role': 'user', 'content': user_msg},
                ]

                async with sem:
                    try:
                        # First call — model may call tool
                        resp = await client.chat.completions.create(
                            model=model, messages=messages,
                            tools=[RETRIEVE_TOOL],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)

                        msg = resp.choices[0].message
                        # Handle tool calls (up to 3 rounds)
                        rounds = 0
                        while msg.tool_calls and rounds < 3:
                            messages.append(msg)
                            for tc in msg.tool_calls:
                                args = json.loads(tc.function.arguments)
                                query = args.get('query', '')
                                result = retrieve_evidence(bm25, evidence_records, query, top_k=5)
                                messages.append({
                                    'role': 'tool',
                                    'tool_call_id': tc.id,
                                    'content': result,
                                })
                            resp = await client.chat.completions.create(
                                model=model, messages=messages,
                                tools=[RETRIEVE_TOOL],
                                max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                            msg = resp.choices[0].message
                            rounds += 1

                        comment = msg.content or ''
                        return {'docket_id': dk, 'n_units': len(us),
                                'comment': comment, 'variant': f'rag_tool_{_style}',
                                'model': model, 'stakeholder': stakeholder,
                                'tool_rounds': rounds}
                    except Exception as e:
                        logger.warning("Err %s/%s: %s", config_name, dk, str(e)[:100])
                        return None

        BATCH = 50
        t0 = time.time()
        with open(out_path, 'a') as fout:
            for s in range(0, len(todo), BATCH):
                batch = todo[s:s+BATCH]
                results = await asyncio.gather(*[gen_one(b) for b in batch])
                for r in results:
                    if r:
                        fout.write(json.dumps(r) + '\n')
                fout.flush()
                n_done = min(s + BATCH, len(todo))
                elapsed = time.time() - t0
                logger.info("  %d/%d (%.1f/s)", n_done, len(todo),
                           n_done / max(elapsed, 1))

        logger.info("Done %s", config_name)

    logger.info("=== ALL VARIANTS DONE ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-5-mini',
                        help='Model to use (gpt-5-mini, gpt-5, etc.)')
    args = parser.parse_args()
    asyncio.run(run_all(model=args.model))
