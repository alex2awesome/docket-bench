"""Generate claim-list format for RAG, few-shot, and tool-calling variants.

Mirrors run_full_comment_variants.py but generates [P#] tagged claims
instead of full paragraphs — for content matching via the existing Axis B pipeline.

Supports resume: checks existing generated_claims.jsonl and skips done dockets.

Usage:
    OPENAI_API_KEY=$(cat ~/.openai-key.txt) \
        python scripts/run_claim_list_variants.py --model gpt-5-mini
    OPENAI_API_KEY=$(cat ~/.openai-key.txt) \
        python scripts/run_claim_list_variants.py --model gpt-5 --sample 100
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

CONCURRENCY = 50
CLAIM_RE = re.compile(r'^\s*\[P(\d+)\]\s*(.+?)\s*$', re.MULTILINE)

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

# ---- CLAIM-LIST PROMPTS ----

CLAIM_VANILLA = """\
Below are provisions from a proposed federal rule.

What are public comments members of the public are likely to submit on the following provisions? For each comment, express it as a specific claim — a concrete point about a specific provision.

Format each claim on its own line, prefixed with the provision number it addresses:
[P1] claim about provision 1
[P3] claim about provision 3

Generate as many distinct claims as you think real commenters would submit. Be specific — reference particular requirements, thresholds, or actors mentioned in the provisions.

PROPOSED RULE PROVISIONS:
{provisions}

Generate the anticipated public comments:"""

CLAIM_PERSONA = """\
Below are provisions from a proposed federal rule. You will write public-comment claims FROM THE PERSPECTIVE OF A SPECIFIC STAKEHOLDER GROUP.

STAKEHOLDER PERSPECTIVE: {stakeholder_persona}

Generate the public-comment claims this stakeholder would submit. Each claim should be a concrete point about a specific provision.

Format each claim on its own line, prefixed with the provision number it addresses:
[P1] claim about provision 1
[P3] claim about provision 3

Be specific — reference particular requirements, thresholds, or actors mentioned in the provisions.

PROPOSED RULE PROVISIONS:
{provisions}

{stakeholder_label} comments on this proposal:"""

CLAIM_RAG_VANILLA = """\
Below are provisions from a proposed federal rule.

You also have access to REAL EVIDENCE from prior public comments on similar regulations. Use this evidence to ground your claims.

PROPOSED RULE PROVISIONS:
{provisions}

RETRIEVED EVIDENCE FROM PRIOR COMMENTS ON SIMILAR RULES:
{retrieved_evidence}

What are public comments members of the public are likely to submit? For each comment, express it as a specific claim — a concrete point about a specific provision. Ground claims in specific evidence.

Format each claim on its own line:
[P1] claim about provision 1

Generate the anticipated public comments:"""

CLAIM_RAG_PERSONA = """\
Below are provisions from a proposed federal rule. Generate claims FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

PROPOSED RULE PROVISIONS:
{provisions}

RETRIEVED EVIDENCE FROM PRIOR COMMENTS ON SIMILAR RULES:
{retrieved_evidence}

Generate claims this stakeholder would submit, grounded in the retrieved evidence.

Format each claim on its own line:
[P1] claim about provision 1

{stakeholder_label} comments:"""

CLAIM_FEW_SHOT_VANILLA = """\
Below are provisions from a proposed federal rule.

Here are examples of REAL public comment claims on other regulations:
{examples}

Now generate claims for THIS rule:

PROPOSED RULE PROVISIONS:
{provisions}

Format each claim on its own line:
[P1] claim about provision 1

Generate the anticipated public comments:"""

CLAIM_FEW_SHOT_PERSONA = """\
Below are provisions from a proposed federal rule. Generate claims FROM THE PERSPECTIVE OF: {stakeholder_label}.

{stakeholder_persona}

Here are examples of REAL claims from similar stakeholders:
{examples}

PROPOSED RULE PROVISIONS:
{provisions}

Format each claim on its own line:
[P1] claim about provision 1

{stakeholder_label} comments:"""

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_similar_comments",
        "description": "Search a database of prior public comments on federal regulations to find relevant evidence and claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for relevant prior comments."
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_SYSTEM_CLAIMS = """\
You are generating anticipated public comments on a proposed federal regulation. You have access to a tool that searches real prior comments. Use it to ground your claims in real evidence.

Format each claim on its own line, prefixed with the provision number:
[P1] claim about provision 1"""

TOOL_USER_CLAIMS = """\
PROPOSED RULE PROVISIONS:
{provisions}

Generate anticipated public comment claims. Use the retrieve tool first."""

TOOL_USER_CLAIMS_PERSONA = """\
Generate claims FROM THE PERSPECTIVE OF: {stakeholder_label}.
{stakeholder_persona}

PROPOSED RULE PROVISIONS:
{provisions}

Generate claims this stakeholder would submit. Use the retrieve tool first."""


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


def parse_claims(text, n_units):
    return [{'provision_idx': int(m.group(1)), 'claim_text': m.group(2).strip(), 'stakeholder': None}
            for m in CLAIM_RE.finditer(text) if 1 <= int(m.group(1)) <= n_units]


def build_evidence_index():
    if not EVIDENCE_PATH.exists():
        return None, []
    evidence_records = []; evidence_texts = []
    for line in open(EVIDENCE_PATH):
        r = json.loads(line)
        if not r.get('has_evidence'): continue
        parts = []
        for k in ['statutes_cited','published_data','studies_cited','regulatory_history','specific_alternative']:
            if r.get(k): parts.append(f"{k}: {r[k]}")
        if parts:
            evidence_records.append({'snippet': f"[{r.get('agency','?')}] " + " | ".join(parts)})
            evidence_texts.append(' '.join(parts))
    tokenized = [t.lower().split() for t in evidence_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, evidence_records


def retrieve_evidence(bm25, evidence_records, query, top_k=10):
    if bm25 is None: return "No prior evidence available."
    scores = bm25.get_scores(query.lower().split())
    top_idx = scores.argsort()[-top_k:][::-1]
    snippets = [evidence_records[i]['snippet'] for i in top_idx if scores[i] > 0]
    return '\n'.join(f"  {i+1}. {s}" for i, s in enumerate(snippets)) if snippets else "No matching evidence."


def load_few_shot_claims():
    """Load real claims as few-shot examples."""
    examples_by_org = defaultdict(list)
    count = 0
    for line in open(DU / "match2_llm_labeled.jsonl"):
        if count > 50000: break
        if '"llm_label": 1' not in line: continue
        r = json.loads(line)
        if r.get('llm_label') == 1:
            examples_by_org['all'].append(f"[P] {r['claim_text']}")
            count += 1
    random.shuffle(examples_by_org['all'])
    return examples_by_org


async def run_all(model='gpt-5-mini', sample=None):
    client = AsyncOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        default_headers={"X-Model-Service-Tier": "flex"},
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    IS_REASONING = model in ('gpt-5', 'gpt-5-mini', 'o3', 'o4-mini')
    MAX_TOKENS = 4048 if IS_REASONING else 2048
    REASONING_EXTRA = {'reasoning_effort': 'low'} if IS_REASONING else {}

    bench = load_benchmark()
    units = load_units()
    pw = json.load(open(PERSONA_WEIGHTS)).get('per_docket', {})
    bm25, evidence_records = build_evidence_index()
    few_shot_pool = load_few_shot_claims()

    if sample:
        bench = random.sample(bench, min(sample, len(bench)))
        logger.info("Sampled %d dockets", len(bench))

    VARIANTS = [
        (f'{model}_rag_vanilla', 'rag', 'vanilla'),
        (f'{model}_rag_persona', 'rag', 'persona'),
        (f'{model}_few_shot_vanilla', 'few_shot', 'vanilla'),
        (f'{model}_few_shot_persona', 'few_shot', 'persona'),
        (f'{model}_rag_tool_vanilla', 'rag_tool', 'vanilla'),
        (f'{model}_rag_tool_persona', 'rag_tool', 'persona'),
    ]

    for config_name, method, style in VARIANTS:
        out_dir = RUNS / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'generated_claims.jsonl'

        done = set()
        if out_path.exists():
            for line in open(out_path):
                try: done.add(json.loads(line)['docket_id'])
                except: pass

        todo = [b for b in bench if b['docket_id'] not in done]
        if not todo:
            logger.info("SKIP %s: already done (%d)", config_name, len(done))
            continue

        logger.info("=== %s: %d dockets (resume from %d) ===", config_name, len(todo), len(done))

        async def gen_one(b, _method=method, _style=style):
            dk = b['docket_id']
            us = units.get(dk, [])
            if not us: return None
            provs = fmt_provisions(us)
            n_units = len(us)

            stakeholder = None
            label, persona_text = '', ''
            if _style == 'persona':
                w = pw.get(dk, {})
                stakeholder = max(w, key=w.get) if w else 'individual'
                label, persona_text = STAKEHOLDER_PERSONAS[stakeholder]

            try:
                if _method == 'rag':
                    ev = '\n'.join(sorted({
                        line.strip() for u in us[:20]
                        for line in retrieve_evidence(bm25, evidence_records,
                            u.get('action_summary','') or u.get('obligation_text',''), top_k=3).split('\n')
                        if line.strip() and not line.strip().startswith('No ')
                    })[:15])
                    if _style == 'vanilla':
                        prompt = CLAIM_RAG_VANILLA.format(provisions=provs, retrieved_evidence=ev)
                    else:
                        prompt = CLAIM_RAG_PERSONA.format(provisions=provs, retrieved_evidence=ev,
                            stakeholder_label=label, stakeholder_persona=persona_text)

                    async with sem:
                        resp = await client.chat.completions.create(
                            model=model, messages=[{'role':'user','content':prompt}],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        raw = resp.choices[0].message.content or ''

                elif _method == 'few_shot':
                    exs = '\n'.join(few_shot_pool['all'][:10])
                    if _style == 'vanilla':
                        prompt = CLAIM_FEW_SHOT_VANILLA.format(provisions=provs, examples=exs)
                    else:
                        prompt = CLAIM_FEW_SHOT_PERSONA.format(provisions=provs, examples=exs,
                            stakeholder_label=label, stakeholder_persona=persona_text)

                    async with sem:
                        resp = await client.chat.completions.create(
                            model=model, messages=[{'role':'user','content':prompt}],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        raw = resp.choices[0].message.content or ''

                elif _method == 'rag_tool':
                    if _style == 'vanilla':
                        user_msg = TOOL_USER_CLAIMS.format(provisions=provs)
                    else:
                        user_msg = TOOL_USER_CLAIMS_PERSONA.format(provisions=provs,
                            stakeholder_label=label, stakeholder_persona=persona_text)

                    messages = [{'role':'system','content':TOOL_SYSTEM_CLAIMS},
                                {'role':'user','content':user_msg}]
                    async with sem:
                        resp = await client.chat.completions.create(
                            model=model, messages=messages, tools=[RETRIEVE_TOOL],
                            max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                        msg = resp.choices[0].message
                        rounds = 0
                        while msg.tool_calls and rounds < 3:
                            messages.append(msg)
                            for tc in msg.tool_calls:
                                args = json.loads(tc.function.arguments)
                                result = retrieve_evidence(bm25, evidence_records, args.get('query',''), top_k=5)
                                messages.append({'role':'tool','tool_call_id':tc.id,'content':result})
                            resp = await client.chat.completions.create(
                                model=model, messages=messages, tools=[RETRIEVE_TOOL],
                                max_completion_tokens=MAX_TOKENS, temperature=1.0, **REASONING_EXTRA)
                            msg = resp.choices[0].message
                            rounds += 1
                        raw = msg.content or ''

                claims = parse_claims(raw, n_units)
                return {
                    'docket_id': dk, 'n_units': n_units,
                    'parsed_claims': claims, 'variant': f'{_method}_{_style}',
                    'model': model, 'stakeholder': stakeholder,
                    'n_claims': len(claims),
                }
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
                    if r: fout.write(json.dumps(r) + '\n')
                fout.flush()
                n_done = min(s + BATCH, len(todo))
                logger.info("  %d/%d (%.1f/s)", n_done, len(todo), n_done/max(time.time()-t0,1))

        logger.info("Done %s", config_name)

    logger.info("=== ALL CLAIM-LIST VARIANTS DONE ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--sample', type=int, default=None, help='Sample N dockets (for quick testing)')
    args = parser.parse_args()
    asyncio.run(run_all(model=args.model, sample=args.sample))
