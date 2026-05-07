#!/usr/bin/env python3
"""Extract government responses to public comments from regulatory documents.

Parses rule/proposed_rule preambles using an LLM to identify passages where
agencies respond to public comments. Supports both API models (OpenAI/Anthropic)
and local vLLM for open-source models.

Usage:
    # OpenAI API:
    python extract_responses_from_gov.py \
        --base-dir data/bulk_downloads --doc-types rule \
        --backend openai --model gpt-5-mini \
        --output data/comment_responses.jsonl

    # vLLM (local GPU):
    python extract_responses_from_gov.py \
        --base-dir data/bulk_downloads --doc-types rule \
        --backend vllm --model meta-llama/Llama-3.3-70B-Instruct \
        --vllm-base-url http://127.0.0.1:8002/v1 \
        --output data/comment_responses.jsonl

    # vLLM offline (batch):
    python extract_responses_from_gov.py \
        --base-dir data/bulk_downloads --doc-types rule \
        --backend vllm-offline --model meta-llama/Llama-3.3-70B-Instruct \
        --output data/comment_responses.jsonl
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import re
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import filelock
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent
DOC_TYPES = ["rule", "proposed_rule", "notice"]
MIN_TEXT_LEN = 500  # skip very short documents
MAX_INPUT_TOKENS = 6000  # truncate input to this many tokens (~24K chars)
MAX_INPUT_CHARS = MAX_INPUT_TOKENS * 4

# ---------------------------------------------------------------------------
# Lock file management for parallel execution
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
for _sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(_sig, _signal_handler)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a legal analyst specializing in US federal regulatory documents. Your task is to extract structured comment-response pairs from government rulemaking documents.

You will be given a government document (rule, proposed rule, or notice) that may contain passages where the agency responds to public comments received during the notice-and-comment process. These passages are typically found in the preamble section of final rules.

IMPORTANT: The document may NOT contain responses to public comments. In those cases, return an empty list []. Do not fabricate comment-response pairs that are not present in the text. Also note that documents contain boilerplate text (headers, dates, addresses, authority citations, cost analyses, etc.) that should be ignored — only extract actual comment-response pairs.

COMPLETENESS: You MUST extract ALL comment-response pairs in the text, including:
- Comments where the agency AGREED and made changes
- Comments expressing SUPPORT for the rule (even if the agency simply acknowledged them)
- Comments the agency found outside scope or irrelevant
- Comments where the agency deferred action to future rulemaking
- Implicit responses: passages where the agency clearly addresses public input even if the comments themselves are not directly quoted (e.g., "After considering public input, the agency decided to..." or "In response to concerns raised during the comment period...")
Do NOT skip supportive or agreement comments — they are just as important as disagreements. Be exhaustive.

DETAIL: For content_of_comment, quote the FULL passage from the document that describes what the commenter said. Include the commenter's reasoning, specific arguments, proposed alternatives, and any statutory/regulatory references — not just a one-sentence summary. When multiple commenters are discussed in one passage, capture all of their perspectives.

For each comment-response pair you find, extract it into a structured JSON object."""

FEW_SHOT_EXAMPLES = [
    # --- Example 1: Mixed document with regulatory boilerplate AND comment responses ---
    {
        "role": "user",
        "content": """Extract comment-response pairs from this document:

<document>
ENVIRONMENTAL PROTECTION AGENCY
40 CFR Part 60
[EPA-HQ-OAR-1989-0098; FRL-5765-2]
RIN 2060-AH23

Performance Specification 4A—Specifications and Test Procedures for Continuous Emission Monitoring Systems

AGENCY: Environmental Protection Agency (EPA).
ACTION: Final rule.

SUMMARY: This action promulgates Performance Specification 4A (PS-4A) which establishes the minimum specifications and test procedures for carbon monoxide (CO) continuous emission monitoring systems (CEMS). The EPA received comments from two sources during the public comment period. This document addresses those comments and finalizes PS-4A with modifications.

DATES: Effective Date: This rule is effective June 15, 2001.
ADDRESSES: Docket No. A-89-09 contains the supporting information. The docket is located at the EPA's Air and Radiation Docket, Room M-1500, 401 M Street SW., Washington, DC.

FOR FURTHER INFORMATION CONTACT: Mr. William J. Grimley, Emission Measurement Center (MD-19), Office of Air Quality Planning and Standards, EPA, Research Triangle Park, NC 27711; telephone (919) 541-1065.

PS-4A establishes the procedures for evaluating the acceptability of CO CEMS at the time of installation. The specification requires an initial calibration error test, a relative accuracy test, and a 7-day calibration drift test. These requirements ensure that CO CEMS produce data of known and acceptable quality.

Both of the commenters thought that because PS-4A required that a CEMS be free from the effects of any interferences, EPA should include a procedure for demonstrating that interferences are not a problem. EPA has decided not to add a single required procedure for evaluating the CEMS for interferences because of the wide variety of operating principles used by CO CEMS. The Agency believes that the RA test requirement in PS-4A ensures adequate performance without mandating a specific interference check procedure.

One of the commenters thought if PS-4A allows a cylinder gas audit to be substituted for the RA test where source emissions are less than 10 percent of the applicable standard, then the appropriate reference method should be used to demonstrate the applicability of the substitution. EPA agrees and has revised the method accordingly to require use of the reference method (Method 10) to verify emissions.

The other commenter thought that the calibration drift limits in PS-4A should be tightened from 5 percent to 3 percent. EPA disagrees because current CEMS technology does not consistently support tighter drift limits, and maintaining 5 percent ensures broad applicability while still ensuring data quality.

Under Executive Order 12866, this action is not a "significant regulatory action." This rule does not contain any information collection requirements subject to OMB approval under the Paperwork Reduction Act.
</document>

Your response:"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "content_of_comment": "Both of the commenters thought that because PS-4A required that a CEMS be free from the effects of any interferences, EPA should include a procedure for demonstrating that interferences are not a problem.",
    "summarized_content_of_comment": "Request to include a specific interference check procedure in PS-4A.",
    "response_to_comment": "EPA declined to add a single required procedure, citing the need for flexibility due to varying CEMS operating principles and stating that the RA test sufficiently ensures performance without sacrificing accuracy.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "disagree",
    "reference_scope": "group_of_comments",
    "commenter_identifiers_text": "Both of the commenters",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_considered_but_not_made"
  },
  {
    "content_of_comment": "One of the commenters thought if PS-4A allows a cylinder gas audit to be substituted for the RA test where source emissions are less than 10 percent of the applicable standard, then the appropriate reference method should be used to demonstrate the applicability of the substitution.",
    "summarized_content_of_comment": "Require use of a reference method to verify eligibility for substituting a cylinder gas audit for RA when emissions are <10% of the standard.",
    "response_to_comment": "EPA agreed and revised the method to require use of the reference method (Method 10) to verify emissions are below 10% when using the substitution.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "accepted",
    "reference_scope": "single_comment",
    "commenter_identifiers_text": "One of the commenters",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_made"
  },
  {
    "content_of_comment": "The other commenter thought that the calibration drift limits in PS-4A should be tightened from 5 percent to 3 percent.",
    "summarized_content_of_comment": "Request to tighten calibration drift limits from 5% to 3%.",
    "response_to_comment": "EPA disagreed because current CEMS technology does not consistently support tighter drift limits, and 5% ensures broad applicability while maintaining data quality.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "disagree",
    "reference_scope": "single_comment",
    "commenter_identifiers_text": "The other commenter",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_considered_but_not_made"
  }
]"""
    },
    # --- Example 2: CMS proposed rule withdrawal with comment responses ---
    {
        "role": "user",
        "content": """Extract comment-response pairs from this document:

<document>
We received significant comments on the proposed rule regarding its potential impact on states and their budgets, Medicaid providers and Medicaid beneficiary access to needed services. Many commenters stated their belief that the proposed rule did not include adequate analysis of these matters. After consideration of the comments received, CMS has decided to withdraw this proposed rule. We are withdrawing this proposed rule so that we may re-examine the issues raised and fully analyze them, including in light of the new statutory reporting requirements, and to consider alternative approaches to the issues addressed in this proposed rule.

Numerous commenters indicated that CMS, in some instances, lacked statutory authority for its proposals and was creating regulatory provisions that were ambiguous or unclear and subject to excessive Agency discretion. CMS is withdrawing this proposed rule to allow a re-examination of these issues and to explore options that best implement the Medicaid statute.
</document>

Your response:"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "content_of_comment": "We received significant comments on the proposed rule regarding its potential impact on states and their budgets, Medicaid providers and Medicaid beneficiary access to needed services. Many commenters stated their belief that the proposed rule did not include adequate analysis of these matters.",
    "summarized_content_of_comment": "Commenters said CMS had not adequately analyzed the rule's impacts on state budgets, providers, and beneficiary access to services.",
    "response_to_comment": "CMS withdrew the proposed rule to re-examine and fully analyze the issues, including in light of new statutory reporting requirements, and to consider alternative approaches.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "partially_accepted",
    "reference_scope": "group_of_comments",
    "commenter_identifiers_text": "Many commenters",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_made"
  },
  {
    "content_of_comment": "Numerous commenters indicated that CMS, in some instances, lacked statutory authority for its proposals and was creating regulatory provisions that were ambiguous or unclear and subject to excessive Agency discretion.",
    "summarized_content_of_comment": "Commenters argued CMS lacked statutory authority for parts of the proposal and that some provisions were ambiguous and overly discretionary.",
    "response_to_comment": "CMS withdrew the proposed rule to allow re-examination of the issues and exploration of options that best implement the Medicaid statute.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "unclear",
    "reference_scope": "group_of_comments",
    "commenter_identifiers_text": "Numerous commenters",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_made"
  }
]"""
    },
    # --- Example 3: Document with supportive comments, implicit responses, and out-of-scope ---
    {
        "role": "user",
        "content": """Extract comment-response pairs from this document:

<document>
DEPARTMENT OF THE INTERIOR
Fish and Wildlife Service
50 CFR Part 17

Endangered and Threatened Wildlife and Plants; Final Rule To List the Northern Long-Eared Bat as Endangered

AGENCY: Fish and Wildlife Service, Interior.
ACTION: Final rule.

SUMMARY: We, the U.S. Fish and Wildlife Service, are listing the northern long-eared bat as an endangered species under the Endangered Species Act. We received 108 comments during the public comment period.

Discussion of Comments

Several commenters expressed strong support for listing the northern long-eared bat as endangered, noting that the species faces severe threats from white-nose syndrome and that endangered status is necessary to prevent extinction. We appreciate these comments and agree that the threats warrant endangered status rather than the previously proposed threatened status.

Two industry associations argued that the economic impacts of the listing had not been adequately analyzed and that the Service should conduct a cost-benefit analysis before finalizing the rule. The Service notes that, under the ESA, listing determinations must be based solely on the best scientific and commercial data available, without consideration of economic impacts. We therefore decline to conduct such an analysis for the listing determination itself.

One commenter submitted a letter about highway construction in Ohio that was unrelated to this rulemaking. This comment is outside the scope of this action and is not addressed further.

After considering the peer review and public comments received, the Service has determined that revising the proposed listing from threatened to endangered is warranted based on the severity and immediacy of the threats.
</document>

Your response:"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "content_of_comment": "Several commenters expressed strong support for listing the northern long-eared bat as endangered, noting that the species faces severe threats from white-nose syndrome and that endangered status is necessary to prevent extinction.",
    "summarized_content_of_comment": "Multiple commenters supported endangered listing, citing white-nose syndrome as a severe threat requiring stronger protection than threatened status.",
    "response_to_comment": "The Service appreciates these comments and agrees that the threats warrant endangered status rather than the previously proposed threatened status.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "accepted",
    "reference_scope": "group_of_comments",
    "commenter_identifiers_text": "Several commenters",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_made"
  },
  {
    "content_of_comment": "Two industry associations argued that the economic impacts of the listing had not been adequately analyzed and that the Service should conduct a cost-benefit analysis before finalizing the rule.",
    "summarized_content_of_comment": "Industry associations requested a cost-benefit analysis, arguing economic impacts were not adequately assessed.",
    "response_to_comment": "The Service notes that under the ESA, listing determinations must be based solely on the best scientific and commercial data available, without consideration of economic impacts, and therefore declines to conduct such an analysis for the listing determination.",
    "quoted_or_paraphrased": "quoted",
    "type_of_response": "disagree",
    "reference_scope": "group_of_comments",
    "commenter_identifiers_text": "Two industry associations",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_considered_but_not_made"
  },
  {
    "content_of_comment": "One commenter submitted a letter about highway construction in Ohio that was unrelated to this rulemaking.",
    "summarized_content_of_comment": "Commenter submitted an off-topic letter about highway construction.",
    "response_to_comment": "This comment is outside the scope of this action and is not addressed further.",
    "quoted_or_paraphrased": "paraphrased",
    "type_of_response": "not_relevant",
    "reference_scope": "single_comment",
    "commenter_identifiers_text": "One commenter",
    "response_engagement_type": "procedural_response",
    "rule_change_outcome": "no_change_discussed"
  },
  {
    "content_of_comment": "After considering the peer review and public comments received, the Service has determined that revising the proposed listing from threatened to endangered is warranted based on the severity and immediacy of the threats.",
    "summarized_content_of_comment": "Implicit response: the agency revised the listing from threatened to endangered after considering all public comments and peer review.",
    "response_to_comment": "The Service determined that the overall weight of public comments and scientific evidence supported upgrading the listing from threatened to endangered status.",
    "quoted_or_paraphrased": "paraphrased",
    "type_of_response": "accepted",
    "reference_scope": "all_comments",
    "commenter_identifiers_text": "peer review and public comments",
    "response_engagement_type": "substantive_response",
    "rule_change_outcome": "change_made"
  }
]"""
    },
    # --- Example 4: Document with NO comment responses ---
    {
        "role": "user",
        "content": """Extract comment-response pairs from this document:

<document>
DEPARTMENT OF TRANSPORTATION
Federal Aviation Administration
14 CFR Part 39
[Docket No. FAA-2024-1847; Product Identifier 2024-NM-089-AD]
RIN 2120-AA64

Airworthiness Directives; The Boeing Company Airplanes

AGENCY: Federal Aviation Administration (FAA), DOT.
ACTION: Final rule.

SUMMARY: The FAA is adopting a new airworthiness directive (AD) for certain Boeing Model 737-8 and 737-8200 airplanes. This AD was prompted by a report indicating that certain rudder actuator components may not meet design specifications. This AD requires inspection and, if necessary, replacement of the affected components.

DATES: This AD is effective August 15, 2024.

SUPPLEMENTARY INFORMATION:

Discussion
The FAA has examined the underlying safety concern and determined that unsafe conditions exist that require the issuance of this AD. Transport Canada Civil Aviation issued Canadian AD CF-2024-15 to address the unsafe condition on these products. The FAA reviewed the relevant data and determined that the unsafe condition described previously is likely to exist or develop in other airplanes of the same type design.

Costs of Compliance
The FAA estimates that this AD affects 189 airplanes of U.S. registry. The FAA estimates the following costs to comply with this AD: labor costs of approximately 2 work-hours per product at an average rate of $85 per work-hour. Based on these figures, the estimated cost of this AD on U.S. operators is $32,130.

Authority for This Rulemaking
Title 49 of the United States Code specifies the FAA's authority to issue rules on aviation safety.
</document>

Your response:"""
    },
    {
        "role": "assistant",
        "content": "[]"
    },
]

USER_PROMPT_TEMPLATE = """Extract comment-response pairs from this document:

<document>
{input_text}
</document>

Your response:"""

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def _call_openai(prompts: List[str], model: str, **kwargs) -> List[str]:
    """Call OpenAI API (async, batched).

    Individual calls have a hard timeout so a single stuck reasoning call
    cannot stall the entire batch. Stuck calls are surfaced as empty-string
    results, which the pipeline retries via its normal parse-failure path.
    """
    import asyncio
    from openai import AsyncOpenAI

    # Per-request timeout in seconds. gpt-5 reasoning calls legitimately take
    # a few minutes on dense chunks; 300s balances that against head-of-line
    # blocking when one chunk goes pathological.
    per_request_timeout = float(kwargs.get("per_request_timeout", 300))
    client = AsyncOpenAI(timeout=per_request_timeout)
    semaphore = asyncio.Semaphore(kwargs.get("concurrency", 30))
    temp = kwargs.get("temperature", 0.0)

    # gpt-5-*, o1-*, o3-* require max_completion_tokens instead of max_tokens
    # and only accept temperature=1.
    uses_new_api = model.startswith(("gpt-5", "o1", "o3", "o4"))

    async def _call_one(prompt: str) -> str:
        async with semaphore:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(FEW_SHOT_EXAMPLES)
            messages.append({"role": "user", "content": prompt})
            try:
                if uses_new_api:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_completion_tokens=8192,
                        ),
                        timeout=per_request_timeout,
                    )
                else:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temp,
                            max_tokens=8192,
                        ),
                        timeout=per_request_timeout,
                    )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                logging.warning(
                    "OpenAI call exceeded %ds timeout for model=%s; returning empty result (will be retried).",
                    per_request_timeout, model,
                )
                return ""
            except Exception as e:
                logging.warning("OpenAI call failed: %s; returning empty result.", e)
                return ""

    async def _run(prompts):
        return await asyncio.gather(*[_call_one(p) for p in prompts])

    return asyncio.run(_run(prompts))


def _call_anthropic(prompts: List[str], model: str, **kwargs) -> List[str]:
    """Call Anthropic API."""
    from anthropic import Anthropic
    from concurrent.futures import ThreadPoolExecutor

    client = Anthropic()
    temp = kwargs.get("temperature", 0.0)

    def _call_one(prompt: str) -> str:
        messages = list(FEW_SHOT_EXAMPLES) + [{"role": "user", "content": prompt}]
        resp = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=temp,
            max_tokens=8192,
        )
        return resp.content[0].text if resp.content else ""

    with ThreadPoolExecutor(max_workers=min(len(prompts), 8)) as pool:
        return list(pool.map(_call_one, prompts))


def _call_vllm_online(prompts: List[str], model: str, **kwargs) -> List[str]:
    """Call a running vLLM server (OpenAI-compatible)."""
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor

    base_url = kwargs.get("vllm_base_url", "http://127.0.0.1:8002/v1")
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    temp = kwargs.get("temperature", 0.0)

    def _call_one(prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=8192,
        )
        return resp.choices[0].message.content or ""

    with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as pool:
        return list(pool.map(_call_one, prompts))


_VLLM_CACHE = {}  # cache LLM + tokenizer across calls


def _call_vllm_offline(prompts: List[str], model: str, **kwargs) -> List[str]:
    """Use vLLM batch inference (requires GPU). Caches the LLM across calls."""
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from vllm import LLM, SamplingParams

    max_model_len = kwargs.get("max_model_len", 16384)
    tp = kwargs.get("tp", 1)

    cache_key = (model, max_model_len, tp)
    if cache_key not in _VLLM_CACHE:
        llm_kwargs = {"max_model_len": max_model_len}
        if tp > 1:
            llm_kwargs["tensor_parallel_size"] = tp
        llm = LLM(model=model, **llm_kwargs)
        tokenizer = llm.get_tokenizer()
        _VLLM_CACHE[cache_key] = (llm, tokenizer)

    llm, tokenizer = _VLLM_CACHE[cache_key]
    temp = kwargs.get("temperature", 0.0)
    sampling_params = SamplingParams(temperature=temp, max_tokens=8192)

    # Build chat-formatted prompts
    formatted = []
    for prompt in prompts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    outputs = llm.generate(formatted, sampling_params)
    results = []
    for o in outputs:
        results.append(o.outputs[0].text if o.outputs else "")

    return results


BACKENDS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "vllm": _call_vllm_online,
    "vllm-offline": _call_vllm_offline,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def iter_input_files(base_dir: Path, doc_type: str):
    """Iterate over _all_text.csv.gz files."""
    gz = sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv.gz"))
    plain = sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv"))
    seen = set()
    result = []
    for p in gz + plain:
        key = p.with_suffix("").with_suffix("")
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def load_documents(
    base_dir: Path,
    doc_type: str,
    agencies: Optional[List[str]] = None,
    min_text_len: int = MIN_TEXT_LEN,
) -> pd.DataFrame:
    """Load documents with canonical_text from bulk CSVs."""
    frames = []
    for csv_path in iter_input_files(base_dir, doc_type):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["Document ID", "Agency ID", "Docket ID",
                         "Document Type", "Posted Date", "canonical_text"],
                low_memory=False,
            )
        except (ValueError, KeyError):
            continue
        df = df.dropna(subset=["canonical_text"])
        df = df[df["canonical_text"].str.len() >= min_text_len]
        if agencies:
            df = df[df["Agency ID"].isin(set(agencies))]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Document ID"])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_llm_response(text: str) -> Optional[List[Dict]]:
    """Parse LLM output as JSON array, handling common issues.

    Returns None if the output could not be parsed (signals need for retry),
    or a list of dicts (possibly empty) on success.
    """
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Find the LAST complete JSON array in the text (handles trailing commentary)
    # Use a greedy search from the first [ to the last ]
    bracket_depth = 0
    arr_start = None
    arr_end = None
    for i, ch in enumerate(text):
        if ch == "[" and arr_start is None:
            arr_start = i
            bracket_depth = 1
        elif ch == "[" and arr_start is not None:
            bracket_depth += 1
        elif ch == "]" and arr_start is not None:
            bracket_depth -= 1
            if bracket_depth == 0:
                arr_end = i + 1
                break  # found the complete array

    if arr_start is not None and arr_end is not None:
        candidate = text[arr_start:arr_end]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try to fix common JSON issues: trailing commas, unescaped quotes
    # by finding individual objects
    if arr_start is not None:
        # Try fixing trailing comma before ]
        candidate = text[arr_start:arr_end] if arr_end else text[arr_start:]
        candidate = re.sub(r",\s*\]", "]", candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Could not parse — return None to signal retry needed
    return None


# ---------------------------------------------------------------------------
# Tokenizer + chunking
# ---------------------------------------------------------------------------


def _get_tokenizer(backend: str, model: str):
    """Load tokenizer appropriate for the backend."""
    if backend in ("openai",):
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")
    else:
        # HuggingFace tokenizer for vllm / anthropic
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens using the appropriate tokenizer."""
    if hasattr(tokenizer, "encode"):
        # tiktoken or HF tokenizer
        try:
            return len(tokenizer.encode(text, allowed_special="all"))
        except TypeError:
            return len(tokenizer.encode(text))
    return len(text) // 4  # fallback


def _compute_prompt_overhead(tokenizer, backend: str) -> int:
    """Compute token count of system prompt + few-shot examples + template wrapper."""
    parts = [SYSTEM_PROMPT]
    for msg in FEW_SHOT_EXAMPLES:
        parts.append(msg["content"])
    # Add the user prompt template (without the document text)
    parts.append(USER_PROMPT_TEMPLATE.replace("{input_text}", ""))

    if hasattr(tokenizer, "apply_chat_template"):
        # HF tokenizer — use chat template for accurate count
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{input_text}", "")})
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return len(formatted)
        except Exception:
            pass

    return sum(_count_tokens(tokenizer, p) for p in parts) + 100  # +100 for chat formatting overhead


CHUNK_OVERLAP_TOKENS = 500  # overlap between consecutive chunks


def chunk_document(text: str, tokenizer, max_doc_tokens: int) -> List[str]:
    """Split a document into chunks that each fit within max_doc_tokens.

    Tries to split on paragraph boundaries (double newlines) to keep
    coherent sections together. Includes ~CHUNK_OVERLAP_TOKENS of overlap
    between consecutive chunks to avoid losing comment-response pairs that
    span chunk boundaries.
    """
    total_tokens = _count_tokens(tokenizer, text)
    if total_tokens <= max_doc_tokens:
        return [text]

    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", text)
    # Pre-compute token counts for each paragraph
    para_tokens_list = [_count_tokens(tokenizer, p) for p in paragraphs]

    chunks = []
    current_chunk = []
    current_tokens = 0
    # Track paragraph indices for overlap
    chunk_start_indices = []
    current_start_idx = 0

    for idx, (para, para_tokens) in enumerate(zip(paragraphs, para_tokens_list)):
        if para_tokens > max_doc_tokens:
            # Single paragraph too long — hard split by tokens
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                chunk_start_indices.append(current_start_idx)
                current_chunk = []
                current_tokens = 0
                current_start_idx = idx
            # Tokenize and split
            try:
                tokens = tokenizer.encode(para, allowed_special="all")
            except TypeError:
                tokens = tokenizer.encode(para)
            for i in range(0, len(tokens), max_doc_tokens):
                chunk_tokens = tokens[i:i + max_doc_tokens]
                if hasattr(tokenizer, "decode"):
                    chunks.append(tokenizer.decode(chunk_tokens))
                else:
                    ratio = len(para) / max(len(tokens), 1)
                    start_char = int(i * ratio)
                    end_char = int(min((i + max_doc_tokens) * ratio, len(para)))
                    chunks.append(para[start_char:end_char])
                chunk_start_indices.append(idx)
            current_start_idx = idx + 1
        elif current_tokens + para_tokens > max_doc_tokens:
            # Adding this paragraph would exceed the limit
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                chunk_start_indices.append(current_start_idx)

            # Overlap: walk backwards from current chunk to include ~CHUNK_OVERLAP_TOKENS
            overlap_paras = []
            overlap_tokens = 0
            for back_idx in range(len(current_chunk) - 1, -1, -1):
                back_para = current_chunk[back_idx]
                bt = _count_tokens(tokenizer, back_para)
                if overlap_tokens + bt > CHUNK_OVERLAP_TOKENS:
                    break
                overlap_paras.insert(0, back_para)
                overlap_tokens += bt

            current_chunk = overlap_paras + [para]
            current_tokens = overlap_tokens + para_tokens
            current_start_idx = idx - len(overlap_paras)
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Second-pass completeness check (#5)
# ---------------------------------------------------------------------------

SECOND_PASS_TEMPLATE = """You previously extracted the following comment-response pairs from a government document. Review the ORIGINAL document text below and determine if any comment-response pairs were MISSED.

Previously extracted pairs (summarized):
{extracted_summaries}

Original document text:
<document>
{input_text}
</document>

If there are comment-response pairs in the document that were NOT captured above, extract ONLY the missing ones as a JSON array. If nothing was missed, return [].

Your response:"""


def _run_second_pass(
    backend_fn,
    model: str,
    doc_text: str,
    existing_responses: List[Dict],
    **backend_kwargs,
) -> List[Dict]:
    """Run a completeness-check second pass on a document."""
    if not existing_responses:
        return []

    # Build summary of what was already extracted
    summaries = []
    for i, resp in enumerate(existing_responses, 1):
        summary = resp.get("summarized_content_of_comment", "")
        if not summary:
            summary = str(resp.get("content_of_comment", ""))[:150]
        summaries.append(f"{i}. {summary}")
    extracted_summaries = "\n".join(summaries)

    prompt = SECOND_PASS_TEMPLATE.format(
        extracted_summaries=extracted_summaries,
        input_text=doc_text[:MAX_INPUT_CHARS],
    )

    try:
        results = backend_fn([prompt], model, temperature=0.2, **backend_kwargs)
        parsed = parse_llm_response(results[0])
        if parsed:
            return parsed
    except Exception as e:
        logging.warning("Second pass failed: %s", e)
    return []


# ---------------------------------------------------------------------------
# Cross-chunk deduplication (#7)
# ---------------------------------------------------------------------------


def _deduplicate_responses(responses: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """Remove near-duplicate responses extracted from overlapping chunks.

    Uses simple character-level similarity on content_of_comment.
    """
    if len(responses) <= 1:
        return responses

    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _char_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_set = set(a.split())
        b_set = set(b.split())
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        return intersection / min(len(a_set), len(b_set))

    keep = []
    for resp in responses:
        content = _normalize(resp.get("content_of_comment", ""))
        is_dup = False
        for kept in keep:
            kept_content = _normalize(kept.get("content_of_comment", ""))
            if _char_overlap(content, kept_content) >= threshold:
                # Keep the longer one
                if len(content) > len(kept_content):
                    keep.remove(kept)
                    keep.append(resp)
                is_dup = True
                break
        if not is_dup:
            keep.append(resp)

    if len(keep) < len(responses):
        logging.info(
            "Deduplication: %d -> %d responses (removed %d duplicates)",
            len(responses), len(keep), len(responses) - len(keep),
        )
    return keep


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_extraction(args: argparse.Namespace) -> None:
    """Run the full extraction pipeline."""
    base_dir = Path(args.base_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend_fn = BACKENDS[args.backend]
    agencies = args.agencies if args.agencies else None

    # Eagerly load vLLM model so we fail fast on OOM before doing any data work
    if args.backend == "vllm-offline":
        logging.info("Pre-loading vLLM model %s (max_model_len=%d, tp=%d)...",
                      args.model, args.max_model_len, args.tp)
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        from vllm import LLM
        cache_key = (args.model, args.max_model_len, args.tp)
        if cache_key not in _VLLM_CACHE:
            llm_kwargs = {"max_model_len": args.max_model_len}
            if args.tp > 1:
                llm_kwargs["tensor_parallel_size"] = args.tp
            llm = LLM(model=args.model, **llm_kwargs)
            tokenizer = llm.get_tokenizer()
            _VLLM_CACHE[cache_key] = (llm, tokenizer)
        logging.info("vLLM model loaded successfully.")

    # Load tokenizer and compute prompt overhead
    logging.info("Loading tokenizer for %s...", args.model)
    tokenizer = _get_tokenizer(args.backend, args.model)
    prompt_overhead = _compute_prompt_overhead(tokenizer, args.backend)
    max_output_tokens = 8192
    max_doc_tokens = args.max_model_len - prompt_overhead - max_output_tokens
    logging.info(
        "Token budget: max_model_len=%d - overhead=%d - output=%d = %d tokens per doc chunk",
        args.max_model_len, prompt_overhead, max_output_tokens, max_doc_tokens,
    )
    if max_doc_tokens < 500:
        logging.error("Not enough token budget for document text. Increase --max-model-len.")
        return

    # Load existing results for resume
    done_ids: Set[str] = set()
    if output_path.exists() and not args.overwrite:
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["document_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        logging.info("Resuming: %d documents already processed", len(done_ids))

    # Load debug document IDs if specified
    debug_doc_ids: Optional[Set[str]] = None
    if args.debug_list:
        debug_list_path = Path(args.debug_list)
        if debug_list_path.exists():
            debug_doc_ids = set(
                line.strip() for line in debug_list_path.read_text().splitlines()
                if line.strip()
            )
            logging.info("Debug list: %d document IDs loaded from %s", len(debug_doc_ids), args.debug_list)
        else:
            logging.error("Debug list file not found: %s", args.debug_list)
            return

    # Prepare lock file directory
    lock_dir = output_path.parent / ".extraction_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    output_lock_path = str(output_path) + ".lock"

    for doc_type in args.doc_types:
        logging.info("=== Processing %s ===", doc_type)

        docs = load_documents(base_dir, doc_type, agencies)
        if docs.empty:
            logging.warning("No documents found for %s", doc_type)
            continue

        # Filter to debug list if specified
        if debug_doc_ids is not None:
            docs = docs[docs["Document ID"].astype(str).isin(debug_doc_ids)]
            logging.info("%s: filtered to %d docs from debug list", doc_type, len(docs))

        # Debug mode: limit to N docs per agency
        if args.debug and debug_doc_ids is None:
            sampled = docs.groupby("Agency ID").head(args.debug).reset_index(drop=True)
            logging.info("%s: debug mode, %d -> %d docs", doc_type, len(docs), len(sampled))
            docs = sampled

        # Filter already-done (re-read done_ids each time for parallel safety)
        if output_path.exists() and not args.overwrite:
            done_ids = set()
            with filelock.FileLock(output_lock_path, timeout=30):
                with open(output_path) as f:
                    for line in f:
                        try:
                            done_ids.add(json.loads(line)["document_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            logging.info("Resuming: %d documents already processed", len(done_ids))
        docs = docs[~docs["Document ID"].astype(str).isin(done_ids)]
        logging.info("%s: %d documents to process", doc_type, len(docs))

        if docs.empty:
            continue

        # Shuffle for parallel execution — each worker gets a different order
        docs = docs.sample(frac=1, random_state=None).reset_index(drop=True)

        # Process documents one at a time with per-document lock files
        processed = 0
        skipped_locked = 0
        skipped_done = 0

        for doc_idx, (_, row) in enumerate(tqdm(
            docs.iterrows(), total=len(docs), desc=f"{doc_type} (pid={os.getpid()})"
        )):
            doc_id = str(row["Document ID"])

            # Check if another worker already completed this doc
            if doc_id in done_ids:
                skipped_done += 1
                continue

            # Try to acquire per-document lock
            processing_flag = lock_dir / f"{doc_id}.processing"
            try:
                fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
            except FileExistsError:
                skipped_locked += 1
                continue
            _active_processing_files.add(processing_flag)

            try:
                text = str(row["canonical_text"])
                chunks = chunk_document(text, tokenizer, max_doc_tokens)
                row_dict = dict(row)

                # Build prompts for all chunks
                prompts = [
                    USER_PROMPT_TEMPLATE.format(input_text=chunk)
                    for chunk in chunks
                ]

                # Process in batches
                chunk_results: Dict[int, tuple] = {}
                failed_chunk_indices: List[int] = []

                for batch_start in range(0, len(prompts), args.batch_size):
                    batch_prompts = prompts[batch_start:batch_start + args.batch_size]
                    batch_indices = list(range(batch_start, batch_start + len(batch_prompts)))

                    try:
                        responses = backend_fn(
                            batch_prompts, args.model,
                            concurrency=args.concurrency,
                            vllm_base_url=args.vllm_base_url,
                            max_model_len=args.max_model_len,
                            tp=args.tp,
                        )
                    except Exception as e:
                        logging.error("Batch failed for %s: %s", doc_id, e)
                        failed_chunk_indices.extend(batch_indices)
                        continue

                    for ci, resp_text in zip(batch_indices, responses):
                        parsed = parse_llm_response(resp_text)
                        if parsed is None:
                            failed_chunk_indices.append(ci)
                        else:
                            chunk_results[ci] = (parsed, resp_text)

                # Retry failed chunks
                for retry_round in range(1, args.max_retries + 1):
                    if not failed_chunk_indices:
                        break
                    still_failed = []
                    retry_prompts = [prompts[i] for i in failed_chunk_indices]
                    try:
                        responses = backend_fn(
                            retry_prompts, args.model,
                            concurrency=args.concurrency,
                            vllm_base_url=args.vllm_base_url,
                            max_model_len=args.max_model_len,
                            tp=args.tp,
                        )
                        for ci, resp_text in zip(failed_chunk_indices, responses):
                            parsed = parse_llm_response(resp_text)
                            if parsed is None:
                                still_failed.append(ci)
                            else:
                                chunk_results[ci] = (parsed, resp_text)
                    except Exception:
                        still_failed = failed_chunk_indices
                    failed_chunk_indices = still_failed

                # Assemble results
                all_parsed = []
                all_raw = []
                for ci in range(len(chunks)):
                    if ci in chunk_results:
                        parsed, raw = chunk_results[ci]
                        all_parsed.extend(parsed)
                        all_raw.append(raw)
                    else:
                        all_raw.append("[PARSE_FAILED]")

                # Deduplication
                all_parsed = _deduplicate_responses(all_parsed)

                # Second-pass completeness check
                if args.second_pass and all_parsed:
                    extra = _run_second_pass(
                        backend_fn, args.model, text, all_parsed,
                        concurrency=args.concurrency,
                        vllm_base_url=args.vllm_base_url,
                        max_model_len=args.max_model_len,
                        tp=args.tp,
                    )
                    if extra:
                        logging.info(
                            "Second pass found %d additional responses for %s",
                            len(extra), doc_id,
                        )
                        all_parsed.extend(extra)
                        all_parsed = _deduplicate_responses(all_parsed)

                record = {
                    "document_id": doc_id,
                    "agency_id": row_dict["Agency ID"],
                    "docket_id": row_dict.get("Docket ID", ""),
                    "doc_type": doc_type,
                    "posted_date": row_dict.get("Posted Date", ""),
                    "n_chunks": len(chunks),
                    "n_responses": len(all_parsed),
                    "responses": all_parsed,
                    "raw_outputs": all_raw,
                }

                # Write with file lock for parallel safety
                with filelock.FileLock(output_lock_path, timeout=60):
                    with open(output_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                done_ids.add(doc_id)
                processed += 1

            except Exception as e:
                logging.error("Failed on %s: %s", doc_id, e)
            finally:
                # Always clean up processing flag
                processing_flag.unlink(missing_ok=True)
                _active_processing_files.discard(processing_flag)

        logging.info(
            "%s: done — %d processed, %d skipped (locked), %d skipped (already done)",
            doc_type, processed, skipped_locked, skipped_done,
        )

    logging.info("Output written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract government responses to public comments from regulatory documents"
    )
    parser.add_argument(
        "--base-dir", type=str, default=str(DEFAULT_BASE_DIR),
        help="Root of bulk_downloads data",
    )
    parser.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES,
    )
    parser.add_argument(
        "--agencies", nargs="+", default=None,
        help="Only process these agencies (e.g. --agencies EPA FAA CMS)",
    )
    parser.add_argument(
        "--backend", choices=list(BACKENDS.keys()), default="vllm",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default="comment_responses.jsonl",
        help="Output JSONL path (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Documents per batch (default: 32)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=30,
        help="Max concurrent API requests (default: 30)",
    )
    parser.add_argument(
        "--vllm-base-url", default="http://127.0.0.1:8002/v1",
        help="vLLM server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=16384,
        help="Max model context length (default: 16384). Documents are chunked "
             "to fit within this budget after accounting for prompt overhead.",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size for vLLM offline (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retry rounds for chunks that produce unparseable JSON (default: 3)",
    )
    parser.add_argument(
        "--debug", type=int, nargs="?", const=100, default=None,
        help="Debug mode: limit to N documents per agency (default: 100 if flag used without value)",
    )
    parser.add_argument(
        "--debug-list", type=str, default=None,
        help="Path to a text file with one Document ID per line. "
             "Only process these specific documents.",
    )
    parser.add_argument(
        "--second-pass", action="store_true",
        help="Run a completeness-check second pass on each document to find "
             "comment-response pairs missed by the first extraction.",
    )

    args = parser.parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
