"""
Usage:
    python data/bulk_downloads/scripts/comments_extract_claims.py --mode offline --batch-size 5120
"""
from __future__ import annotations

import argparse
import ast
import atexit
import json
import logging
import os
import random
import re
import signal
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

PROMPT_TEMPLATE_HEADER = (
    "Extract all distinct claims and substantive arguments from the following public comment to a "
    "government agency.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON array of strings: [\"claim1\", \"claim2\", ...]\n"
    "- Each claim must stand alone — understandable without reading the original comment AND "
    "without reading the other claims in your list. Every claim must explicitly name its subject "
    "(the agency, program, rule, system, docket, or regulation it is about) rather than relying "
    "on pronouns or anaphoric references such as \"it\", \"this\", \"the proposal\", \"the overhaul\", "
    "\"the rule\", or \"the system\". Repeat the subject in every claim even if it feels redundant.\n"
    "  - BAD (depends on a prior claim): \"Could lead to racial profiling and community targeting\".\n"
    "  - GOOD (stands alone): \"DHS's proposed CARIER system overhaul could lead to racial profiling and community targeting\".\n"
    "- Include enough context (agency name, program name, regulation, docket number, geographic area) "
    "so the claim stands alone.\n"
    "- Claims should closely reflect what the commenter actually says — do not editorialize or "
    "add normative framing beyond what the comment contains.\n"
    "- Not every sentence is a claim. Focus on substantive factual assertions, policy arguments, "
    "and evidence cited by the commenter. Skip greetings, thanks, and filler.\n"
    "- If the commenter cites specific data, regulations, costs, or standards, include those "
    "details in the claim.\n"
    "- If the comment contains no substantive claims (e.g., just \"I support this rule\"), "
    "return exactly: []\n"
    "- Do not include any text before or after the JSON array\n"
    "- Do not include booleans, objects, or any other values\n"
)

# Legacy template without few-shot examples (used as fallback)
PROMPT_TEMPLATE = (
    PROMPT_TEMPLATE_HEADER + "\n"
    "<comment>{comment}</comment>\n\n"
    "Your response:\n"
)


def _load_few_shot_examples() -> dict[str, list[dict]]:
    """Load per-agency few-shot examples from JSONL files.

    Returns dict mapping lowercase agency code to list of example dicts,
    each with 'comment_summary' and 'claims' keys.
    """
    examples_dir = Path(__file__).resolve().parent / "data"
    examples_by_agency: dict[str, list[dict]] = {}

    for pattern in ["example_comments_with_claims_all.jsonl",
                     "example_comments_with_claims.jsonl",
                     "example_comments_with_claims_batch*.jsonl"]:
        for fpath in sorted(examples_dir.glob(pattern)):
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        agency = rec.get("agency", "").lower().strip()
                        claims = rec.get("claims", [])
                        # Prefer full_text for few-shot (shows real comment),
                        # fall back to comment_summary
                        text = rec.get("full_text", "") or rec.get("comment_summary", "")
                        if not agency or not text:
                            continue
                        # Truncate for prompt budget — few-shot examples
                        # shouldn't dominate the context window
                        if len(text) > 1500:
                            text = text[:1500] + "..."
                        examples_by_agency.setdefault(agency, []).append({
                            "comment_text": text,
                            "claims": claims,
                        })
            except Exception:
                logging.debug("Could not load examples from %s", fpath)

    return examples_by_agency


# Global cache — loaded once at import time.
_FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {}


def _get_few_shot_examples(agency_id: str, max_examples: int = 2) -> str:
    """Build few-shot example text for a given agency.

    Selects up to max_examples from the agency's pool (preferring one
    technical + one generic if available). Falls back to a small set of
    cross-agency defaults if no agency-specific examples exist.
    """
    global _FEW_SHOT_EXAMPLES
    if not _FEW_SHOT_EXAMPLES:
        _FEW_SHOT_EXAMPLES = _load_few_shot_examples()

    agency_key = agency_id.lower().strip()
    pool = _FEW_SHOT_EXAMPLES.get(agency_key, [])

    # Fall back to a generic cross-agency set
    if not pool:
        pool = (
            _FEW_SHOT_EXAMPLES.get("epa", [])[:1]
            + _FEW_SHOT_EXAMPLES.get("blm", [])[:1]
        )

    # Pick up to max_examples, preferring diverse types
    selected = []
    seen_types = set()
    for ex in pool:
        if len(selected) >= max_examples:
            break
        # Simple diversity: skip if we already have one with same claim count bucket
        bucket = "empty" if not ex["claims"] else ("short" if len(ex["claims"]) <= 2 else "long")
        if bucket not in seen_types or len(selected) < max_examples:
            selected.append(ex)
            seen_types.add(bucket)

    if not selected:
        return ""

    parts = ["\nHere are examples:\n"]
    for ex in selected:
        parts.append(f"<comment>{ex['comment_text']}</comment>\n")
        parts.append(json.dumps(ex["claims"]) + "\n")

    return "\n".join(parts)


def build_prompt(comment: str, agency_id: str = "") -> str:
    """Build the full extraction prompt with agency-specific few-shot examples."""
    few_shot = _get_few_shot_examples(agency_id)
    return (
        PROMPT_TEMPLATE_HEADER
        + few_shot + "\n"
        "Now extract claims from this comment:\n\n"
        "<comment>" + comment + "</comment>\n\n"
        "Your response:\n"
    )

PROMPT_FIX_TEMPLATE = (
    "This answer contains a poorly formatted list of strings that I cannot parse with json. "
    "Please copy the list and fix it so I can parse it. Just copy and fix the list. "
    "Do not add any extra text. I will be trying to parse this. "
    "<list>{claims}</list>. Your response:"
)

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_API_KEY = "EMPTY"

# Track .processing files created by this process so we can clean them up on exit.
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


def iter_input_files(base_dir: Path) -> Iterable[Path]:
    gz = list(base_dir.glob("*/*/public_submission_all_text.csv.gz"))
    plain = list(base_dir.glob("*/*/public_submission_all_text.csv"))
    seen = set()
    result = []
    for p in gz + plain:
        key = str(p).replace(".csv.gz", "").replace(".csv", "")
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _strip_think_blocks(raw: str) -> str:
    """Strip <think>...</think> reasoning blocks (Qwen3-style).

    - Paired <think>...</think> blocks are removed entirely.
    - If an unclosed <think> remains (model ran out of tokens mid-reasoning),
      truncate at that <think> since whatever follows is unusable.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = cleaned[: m.start()]
    return cleaned


def _iter_json_arrays(text: str):
    """Yield every bracket-balanced substring that parses as a JSON list."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape_next = False
        j = i
        matched = False
        while j < n:
            c = text[j]
            if escape_next:
                escape_next = False
            elif c == "\\" and in_string:
                escape_next = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : j + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, list):
                                yield result
                        except json.JSONDecodeError:
                            pass
                        matched = True
                        break
            j += 1
        i = j + 1 if matched else i + 1


def try_extract_json_list(raw: str) -> list | None:
    # Strip reasoning-model thinking blocks first (e.g. Qwen3 <think>...</think>)
    cleaned = _strip_think_blocks(raw).strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Prefer the LAST non-empty JSON array in the cleaned output. Reasoning
    # models sometimes emit a placeholder [] before thinking, then the real
    # array after; taking the last non-empty array handles both orderings.
    last_nonempty = None
    last_any = None
    for result in _iter_json_arrays(cleaned):
        last_any = result
        if result:
            last_nonempty = result
    if last_nonempty is not None:
        return last_nonempty
    return last_any


def _parse_claims(raw: str) -> Tuple[Optional[List[str]], bool]:
    if raw is None:
        return None, False
    raw = raw.strip()
    if not raw:
        return None, False
    extracted = try_extract_json_list(raw)
    if extracted is None:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None, False
    else:
        parsed = extracted
    if not isinstance(parsed, list):
        return None, False
    if not all(isinstance(item, str) for item in parsed):
        return None, False
    return parsed, True


def _gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_name(0) or ""


def _select_model(default_model: str, override: Optional[str]) -> str:
    if override:
        return override
    name_upper = _gpu_name().upper()
    if "B200" in name_upper:
        return "meta-llama/Llama-3.3-70B-Instruct"
    if "H200" in name_upper:
        return "nvidia/Llama-3.1-70B-Instruct-FP8"
    return default_model


def _batch(iterable: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def _hard_clamp_prompt(prompt: str, *, tokenizer, max_input_tokens: int) -> str:
    if tokenizer is None:
        return prompt
    # Pre-truncate by characters to avoid tokenizer hang on huge texts.
    max_chars = max_input_tokens * 6
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
    try:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
    except Exception:
        return prompt
    if len(tokens) <= max_input_tokens:
        return prompt
    logging.warning(
        "Prompt exceeded token budget (%d > %d); hard-clamping.",
        len(tokens),
        max_input_tokens,
    )
    return tokenizer.decode(tokens[:max_input_tokens], skip_special_tokens=True)


def _fit_comment_to_budget(
    comment: str, *, tokenizer, max_input_tokens: int
) -> str:
    # Estimate prompt template overhead in chars (~400 chars → ~100 tokens).
    # Use ~2 chars/token as a pessimistic ratio for the comment portion.
    template_overhead_chars = len(PROMPT_TEMPLATE) - len("{comment}")
    char_budget = max(500, (max_input_tokens * 2) - template_overhead_chars)
    if tokenizer is None:
        logging.debug("No tokenizer; truncating comment to %d chars", char_budget)
        return (comment or "")[:char_budget]
    comment = comment or ""
    # Pre-truncate by characters before tokenizing to avoid spending minutes
    # tokenizing absurdly long texts (e.g., 57M chars from concatenation bugs).
    # ~4 chars/token is conservative; this just prevents tokenizer hang.
    max_chars = max_input_tokens * 6
    if len(comment) > max_chars:
        logging.debug("Pre-truncating comment from %d to %d chars", len(comment), max_chars)
        comment = comment[:max_chars]
    try:
        comment_tokens = tokenizer.encode(comment, add_special_tokens=True)
    except Exception:
        return comment[:char_budget]
    if not comment_tokens:
        return ""

    def prompt_len_for(k: int) -> int:
        candidate = tokenizer.decode(comment_tokens[:k], skip_special_tokens=True)
        # Use the full prompt template (with few-shot examples) for accurate token budgeting
        prompt = build_prompt(candidate, agency_id="")
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    # Fast path if already fits
    if prompt_len_for(len(comment_tokens)) <= max_input_tokens:
        return comment

    low, high = 0, len(comment_tokens)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if prompt_len_for(mid) <= max_input_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best < len(comment_tokens) and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "Truncated comment from %d tokens to %d tokens to fit budget.",
            len(comment_tokens),
            best,
        )
    return tokenizer.decode(comment_tokens[:best], skip_special_tokens=True)


def _fit_fix_to_budget(
    claims: str, *, tokenizer, max_input_tokens: int
) -> str:
    # Fix prompts can be large; budget them similarly to comment prompts.
    template_overhead_chars = len(PROMPT_FIX_TEMPLATE) - len("{claims}")
    char_budget = max(200, (max_input_tokens * 2) - template_overhead_chars)
    if tokenizer is None:
        logging.debug("No tokenizer; truncating fix prompt claims to %d chars", char_budget)
        return (claims or "")[:char_budget]
    claims = claims or ""
    try:
        claims_tokens = tokenizer.encode(claims, add_special_tokens=True)
    except Exception:
        return claims[:char_budget]
    if not claims_tokens:
        return ""

    def prompt_len_for(k: int) -> int:
        candidate = tokenizer.decode(claims_tokens[:k], skip_special_tokens=True)
        prompt = PROMPT_FIX_TEMPLATE.format(claims=candidate)
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    if prompt_len_for(len(claims_tokens)) <= max_input_tokens:
        return claims

    low, high = 0, len(claims_tokens)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if prompt_len_for(mid) <= max_input_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best < len(claims_tokens) and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "Truncated fix claims from %d tokens to %d tokens to fit budget.",
            len(claims_tokens),
            best,
        )
    return tokenizer.decode(claims_tokens[:best], skip_special_tokens=True)


def _prepare_prompts(
    texts: List[str], *, tokenizer, max_input_tokens: int,
    agency_id: str = "",
) -> List[str]:
    prompts = []
    for text in texts:
        truncated = _fit_comment_to_budget(
            text or "",
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
        )
        prompt = build_prompt(truncated, agency_id=agency_id)
        prompts.append(
            _hard_clamp_prompt(
                prompt,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
            )
        )
    return prompts


def _prepare_fix_prompts(
    texts: List[str], *, tokenizer, max_input_tokens: int
) -> List[str]:
    prompts = []
    for text in texts:
        truncated = _fit_fix_to_budget(
            text or "",
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
        )
        prompt = PROMPT_FIX_TEMPLATE.format(claims=truncated)
        prompts.append(
            _hard_clamp_prompt(
                prompt,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
            )
        )
    return prompts


def _consolidate_partial_to_csv(partial_path: Path, output_path: Path, existing_claims: Optional[pd.DataFrame]) -> None:
    """Read partial JSONL, merge with existing_claims, write final CSV, unlink partial."""
    if not partial_path.exists():
        if existing_claims is not None and not existing_claims.empty:
            existing_claims.to_csv(output_path, index=False)
            logging.info("No partial to consolidate; re-wrote %s (%d rows)", output_path, len(existing_claims))
        return
    rows: List[dict] = []
    with open(partial_path, "r") as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    new_output = pd.DataFrame(rows)
    if existing_claims is not None and not existing_claims.empty:
        output = pd.concat([existing_claims, new_output], ignore_index=True)
        logging.info("Merged %d existing + %d new = %d total rows", len(existing_claims), len(new_output), len(output))
    else:
        output = new_output
    output.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(output))
    try:
        partial_path.unlink()
        logging.info("Removed partial checkpoint %s", partial_path)
    except Exception as exc:
        logging.warning("Could not remove partial %s: %s", partial_path, exc)


def process_file(
    csv_path: Path,
    *,
    mode: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    max_input_tokens: int,
    temperature: float,
    generate_batch,
    tokenizer,
    overwrite: bool,
    output_suffix: str = "",
) -> None:
    logging.info("Processing %s", csv_path)
    mapper_path = csv_path.parent / "public_submission_all_text__dedup_mapper.csv.gz"
    if not mapper_path.exists():
        mapper_path = csv_path.parent / "public_submission_all_text__dedup_mapper.csv"
    if not mapper_path.exists():
        logging.warning("Missing mapper %s; skipping", mapper_path)
        return
    output_path = csv_path.parent / f"public_submission_all_text__claims{output_suffix}.csv.gz"
    if not output_path.exists():
        # Fall back to uncompressed claims file
        alt_path = csv_path.parent / f"public_submission_all_text__claims{output_suffix}.csv"
        if alt_path.exists():
            output_path = alt_path
    if output_path.exists() and not overwrite:
        # Check if existing claims cover all document IDs in the all_text file.
        # Claims stores one row per cluster, so we expand cluster_uids back to
        # doc IDs via the mapper, and treat any doc not in the mapper as a
        # singleton (covered if its doc ID appears directly in claims).
        try:
            all_text_df = pd.read_csv(csv_path, usecols=["Document ID", "canonical_text"])
            all_text_ids = set(
                all_text_df.dropna(subset=["Document ID", "canonical_text"])["Document ID"]
                .astype(str)
                .unique()
            )
            claims_uids = set(
                pd.read_csv(output_path, usecols=["cluster_uid"])["cluster_uid"]
                .dropna()
                .astype(str)
                .unique()
            )
            mapper_df = pd.read_csv(mapper_path, usecols=["document_id", "cluster_uid"])
            mapper_df["document_id"] = mapper_df["document_id"].astype(str)
            mapper_df["cluster_uid"] = mapper_df["cluster_uid"].astype(str)
            # Doc IDs covered = mapper docs whose cluster is done + singleton docs whose ID is a done cluster_uid
            covered_via_mapper = set(
                mapper_df.loc[mapper_df["cluster_uid"].isin(claims_uids), "document_id"]
            )
            # Docs not in mapper are singletons; covered if their doc ID is in claims as a cluster_uid
            unmapped_ids = all_text_ids - set(mapper_df["document_id"])
            covered_singletons = unmapped_ids & claims_uids
            covered_ids = covered_via_mapper | covered_singletons
            missing = all_text_ids - covered_ids
            if not missing:
                logging.info("Skipping %s (claims already cover all %d doc IDs)", csv_path, len(all_text_ids))
                return
            logging.info(
                "Resuming %s: claims cover %d/%d doc IDs (%d missing)",
                csv_path, len(covered_ids), len(all_text_ids), len(missing),
            )
        except Exception as exc:
            logging.warning("Could not verify claims completeness for %s (%s); re-processing", csv_path, exc)

    # Atomically claim this folder so other processes skip it.
    processing_flag = csv_path.parent / ".processing"
    try:
        fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        logging.info("Skipping %s (another process is working on it)", csv_path)
        return
    _active_processing_files.add(processing_flag)

    try:
        _process_file_inner(
            csv_path,
            mapper_path=mapper_path,
            output_path=output_path,
            mode=mode,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            temperature=temperature,
            generate_batch=generate_batch,
            tokenizer=tokenizer,
            overwrite=overwrite,
            output_suffix=output_suffix,
        )
    finally:
        processing_flag.unlink(missing_ok=True)
        _active_processing_files.discard(processing_flag)


def _process_file_inner(
    csv_path: Path,
    *,
    mapper_path: Path,
    output_path: Path,
    mode: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    max_input_tokens: int,
    temperature: float,
    generate_batch,
    tokenizer,
    overwrite: bool = False,
    output_suffix: str = "",
) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    mapper = pd.read_csv(mapper_path, low_memory=False)

    if "Document ID" not in df.columns or "canonical_text" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_path}")
    if "document_id" not in mapper.columns or "cluster_uid" not in mapper.columns:
        raise ValueError(f"Missing required columns in {mapper_path}")

    df = df.dropna(subset=["Document ID", "canonical_text"]).copy()
    df["Document ID"] = df["Document ID"].astype(str)
    mapper["document_id"] = mapper["document_id"].astype(str)

    merged = df.merge(mapper, left_on="Document ID", right_on="document_id", how="left")
    if merged.empty:
        logging.warning("No merged rows for %s; skipping", csv_path)
        return

    # Docs not in the mapper are treated as singleton clusters.
    unmapped = merged["cluster_uid"].isna()
    if unmapped.any():
        merged.loc[unmapped, "document_id"] = merged.loc[unmapped, "Document ID"]
        merged.loc[unmapped, "cluster_uid"] = merged.loc[unmapped, "Document ID"]
        # Fill agency_id and docket_id from the all_text CSV columns.
        if "Agency ID" in merged.columns:
            merged.loc[unmapped, "agency_id"] = merged.loc[unmapped, "Agency ID"]
        if "Docket ID" in merged.columns:
            merged.loc[unmapped, "docket_id"] = merged.loc[unmapped, "Docket ID"]
        logging.info("%d docs not in mapper; treating as singleton clusters", unmapped.sum())

    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    logging.info('Sorting + groupby on %d merged rows...', len(merged))
    reps = merged.sort_values("text_len", ascending=False).groupby("cluster_uid", as_index=False).head(1)
    logging.info('Built %d cluster representatives', len(reps))

    # Resume support: load existing claims and only process new cluster_uids.
    existing_claims: Optional[pd.DataFrame] = None
    if output_path.exists() and not overwrite:
        try:
            existing_claims = pd.read_csv(output_path, low_memory=False)
            done_uids = set(existing_claims["cluster_uid"].dropna().astype(str).unique())
            reps = reps[~reps["cluster_uid"].astype(str).isin(done_uids)].copy()
            logging.info(
                "Resuming %s: %d clusters already done, %d remaining",
                csv_path, len(done_uids), len(reps),
            )
            if reps.empty:
                logging.info("All clusters already processed for %s", csv_path)
                return
        except Exception as exc:
            logging.warning("Could not load existing claims for resume (%s); starting fresh", exc)
            existing_claims = None

    # Partial JSONL checkpoint: load any cluster_uids already written to partial file.
    partial_path = csv_path.parent / f"public_submission_all_text__claims{output_suffix}.partial.jsonl"
    partial_done_uids: set = set()
    if partial_path.exists():
        n_read = 0
        try:
            with open(partial_path, "r") as pf:
                for line in pf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    uid = rec.get("cluster_uid")
                    if uid is not None and uid != "":
                        partial_done_uids.add(str(uid))
                        n_read += 1
            logging.info(
                "Loaded %d done cluster_uids from partial %s (%d valid lines)",
                len(partial_done_uids), partial_path, n_read,
            )
            if partial_done_uids:
                reps = reps[~reps["cluster_uid"].astype(str).isin(partial_done_uids)].copy()
                logging.info("After partial resume: %d clusters remaining", len(reps))
                if reps.empty:
                    logging.info("All remaining clusters already in partial; consolidating now")
                    _consolidate_partial_to_csv(partial_path, output_path, existing_claims)
                    return
        except Exception as exc:
            logging.warning("Could not read partial %s (%s); ignoring partial", partial_path, exc)
            partial_done_uids = set()

    reps = reps.reset_index(drop=True)
    logging.info('Extracting %d texts to list...', len(reps))
    texts = reps["canonical_text"].tolist()
    logging.info('Texts extracted')

    # Extract agency_id for few-shot example selection
    agency_id = ""
    if "agency_id" in reps.columns:
        agency_id = str(reps["agency_id"].dropna().iloc[0]) if not reps["agency_id"].dropna().empty else ""
    elif "Agency ID" in reps.columns:
        agency_id = str(reps["Agency ID"].dropna().iloc[0]) if not reps["Agency ID"].dropna().empty else ""

    logging.info('Preparing %d prompts (tokenizing)...', len(texts))
    prompts = _prepare_prompts(
        texts, tokenizer=tokenizer, max_input_tokens=max_input_tokens,
        agency_id=agency_id,
    )
    logging.info('Prepared %d prompts', len(prompts))

    # Per-batch main+fix+append loop with JSONL checkpointing.
    # Partial file is append-only; each line is a complete claims row.
    logging.info('Converting reps to records...')
    reps_records = reps.to_dict("records")
    input_chars_list = reps["canonical_text"].str.len().tolist()
    total = len(prompts)
    logging.info('Ready to process %d prompts in batches of %d', total, batch_size)
    with open(partial_path, "a", buffering=1) as partial_f:
        pbar = tqdm(total=total, desc=f"{csv_path.parent.name} claims")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = prompts[start:end]
            batch_raw = generate_batch(batch)

            batch_parsed: List[Optional[List[str]]] = []
            batch_ok: List[bool] = []
            for raw in batch_raw:
                parsed, ok = _parse_claims(raw)
                batch_parsed.append(parsed)
                batch_ok.append(ok)

            batch_fix_raw: List[str] = [""] * len(batch_raw)
            batch_fix_parsed: List[Optional[List[str]]] = [None] * len(batch_raw)
            batch_fix_ok: List[bool] = [False] * len(batch_raw)
            fix_indices_local = [i for i, ok in enumerate(batch_ok) if not ok]
            if fix_indices_local:
                fix_prompts = _prepare_fix_prompts(
                    [batch_raw[i] for i in fix_indices_local],
                    tokenizer=tokenizer,
                    max_input_tokens=max_input_tokens,
                )
                fix_results = generate_batch(fix_prompts)
                for idx_local, raw_fix in zip(fix_indices_local, fix_results):
                    batch_fix_raw[idx_local] = raw_fix
                    parsed, ok = _parse_claims(raw_fix)
                    batch_fix_parsed[idx_local] = parsed
                    batch_fix_ok[idx_local] = ok

            for i in range(len(batch)):
                global_idx = start + i
                rec = reps_records[global_idx]
                row = {
                    "agency_id": rec.get("agency_id") if rec.get("agency_id") is not None else "",
                    "docket_id": rec.get("docket_id") if rec.get("docket_id") is not None else "",
                    "cluster_id": rec.get("cluster_id") if rec.get("cluster_id") is not None else "",
                    "cluster_uid": rec.get("cluster_uid") if rec.get("cluster_uid") is not None else "",
                    "document_id": rec.get("document_id") if rec.get("document_id") is not None else "",
                    "model": model,
                    "mode": mode,
                    "input_chars": int(input_chars_list[global_idx]) if input_chars_list[global_idx] is not None else 0,
                    "claims_raw": batch_raw[i],
                    "claims_parsed_json": json.dumps(batch_parsed[i], ensure_ascii=False) if batch_parsed[i] is not None else "",
                    "parse_ok": bool(batch_ok[i]),
                    "claims_fix_raw": batch_fix_raw[i],
                    "claims_fix_parsed_json": json.dumps(batch_fix_parsed[i], ensure_ascii=False) if batch_fix_parsed[i] is not None else "",
                    "fix_parse_ok": bool(batch_fix_ok[i]),
                }
                partial_f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            partial_f.flush()
            pbar.update(len(batch))
        pbar.close()

    # Consolidate partial JSONL into final CSV and unlink partial.
    _consolidate_partial_to_csv(partial_path, output_path, existing_claims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--safety-margin", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing claims files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to output filename, e.g. '_v2' produces __claims_v2.csv.gz",
    )
    parser.add_argument(
        "--vllm-progress",
        action="store_true",
        help="Show vLLM internal progress bars.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Run only a shard of files: 'K/N' means take every N-th file starting at index K (0-based).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization (default 0.95 on B200/H200, 0.9 otherwise).",
    )
    parser.add_argument(
        "--dir-filter",
        type=str,
        default=None,
        help="Path to file with one agency/year dir per line; only those dirs will be processed.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = _select_model(DEFAULT_MODEL, args.model)
    max_input_tokens = args.max_model_len - args.max_tokens - args.safety_margin
    if max_input_tokens <= 0:
        raise ValueError("max-model-len must exceed max-tokens + safety-margin")
    if args.max_tokens <= 0:
        logging.warning(
            "max-tokens is %d; zero output tokens leaves no headroom for off-by-one.",
            args.max_tokens,
        )
    if args.mode == "offline":
        if not args.vllm_progress:
            os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for offline mode") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected for offline mode")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is not installed for offline mode") from exc

        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        llm_kwargs = {"max_model_len": args.max_model_len}
        if any(tag in _gpu_name().upper() for tag in ("B200", "H200")):
            llm_kwargs.update(
                {
                    "dtype": "auto",
                    "gpu_memory_utilization": args.gpu_memory_utilization if args.gpu_memory_utilization is not None else 0.95,
                    "kv_cache_dtype": "fp8",
                }
            )
            logging.info("Using B200/H200 vLLM settings: %s", llm_kwargs)
        llm = LLM(model=model, **llm_kwargs)
        try:
            tokenizer = llm.get_tokenizer()
        except Exception as exc:
            logging.warning(
                "Could not access tokenizer (%s); falling back to char truncation.",
                exc,
            )
            tokenizer = None

        if tokenizer is not None:
            logging.info("Tokenizer loaded successfully: %s", type(tokenizer).__name__)
        else:
            logging.warning(
                "Running WITHOUT tokenizer – using char-based truncation. "
                "Token budget violations may occur for long inputs."
            )

        def generate_batch(batch: List[str]) -> List[str]:
            outputs = llm.generate(batch, sampling_params)
            results = []
            for out in outputs:
                if not out.outputs:
                    results.append("")
                    continue
                results.append(out.outputs[0].text or "")
            return results
    else:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for online mode token budgeting") from exc
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        def _call_one(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return getattr(response.choices[0].message, "content", "") or ""

        def generate_batch(batch: List[str]) -> List[str]:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                return list(pool.map(_call_one, batch))

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _signal_handler)

    base_dir = Path("data/bulk_downloads")
    files = list(iter_input_files(base_dir))
    if not files:
        logging.warning("No input files found under %s", base_dir)
        return

    if args.dir_filter:
        allowed = {line.strip() for line in Path(args.dir_filter).read_text().splitlines() if line.strip()}
        keep = []
        for p in files:
            rel = str(p.relative_to(base_dir))
            parent = str(p.parent.relative_to(base_dir))
            if rel in allowed or parent in allowed:
                keep.append(p)
        logging.info("--dir-filter: %d files selected (from %d)", len(keep), len(files))
        files = keep

    if args.shard:
        k, n = map(int, args.shard.split("/"))
        files = sorted(files)  # deterministic order for sharding
        files = files[k::n]
        logging.info("Shard %d/%d: processing %d files (pid=%d)", k, n, len(files), os.getpid())
    else:
        random.shuffle(files)
        logging.info("Processing %d file(s) in random order (pid=%d)", len(files), os.getpid())

    for csv_path in tqdm(files, desc="Files"):
        process_file(
            csv_path,
            mode=args.mode,
            model=model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            max_input_tokens=max_input_tokens,
            temperature=args.temperature,
            generate_batch=generate_batch,
            tokenizer=tokenizer,
            overwrite=args.overwrite,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
