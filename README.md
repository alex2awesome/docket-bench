# DOCKET & DOCKETBench: Code and Data Submission

Code and data for "DOCKET: A Comprehensive Dataset of U.S. Federal Rulemaking and DOCKETBench, a Pluralism Benchmark for LLMs."

## Repository Structure

```
scripts/
  ingestion/          # 16 agency-specific scraping pipelines (regulations.gov + 15 external portals)
  matching/           # 3-stage matching pipeline (bi-encoder -> cross-encoder -> LLM judge)
  benchmark/          # DOCKETBench generation, evaluation, and pluralism measurement
  analysis/           # Analysis and visualization scripts
  validation/         # Nader-FCC validation pipeline

data/                 # (gitignored, download separately)
  raw_text/           # *_all_text.csv.gz per agency/period
  dedup/              # MinHash dedup mapper and cluster files
  benchmark/          # 558-docket benchmark population, provision lists, org-type lookup
  deontic_units/      # Extracted provisions, matched claims, analysis datasets
  derived/            # Clustered taxonomy, evidence patterns, frame labels
  benchmark_runs/     # Per-model generation outputs, pair judgments, match results
  validation/         # Nader-FCC validation report and metrics
  feature_analysis_summary.csv.gz  # 8.27M comment features

latex/                # Paper source (neurips_2026.tex)
```

## Pipeline Overview

The pipeline proceeds in stages, each with its own scripts:

### Stage 1: Data Ingestion (`scripts/ingestion/`)
Scrapes comments and documents from 16 federal data sources. Entry point: `bulk_downloader.py` for regulations.gov; agency-specific `fetch_*.py` for external portals.

### Stage 2: Deduplication (`scripts/`)
MinHash-based near-duplicate detection across 10.79M raw comments.
- `minhash_comment_deduping.py` — MinHash algorithm
- `comment_dedup.py` — orchestration

### Stage 3: Extraction & Matching (`scripts/matching/`)
Three-stage cascade: bi-encoder retrieval, ModernBERT cross-encoder, LLM judge.
- `comments_extract_claims.py` — extract claims from comments
- `extract_responses_from_gov.py` — extract agency responses from final rules  
- `match_comments_and_create_indexes.py` — build retrieval indexes and run matching
- `match_rules_to_proposals.py` — match proposed provisions to final rule provisions
- `train_cross_encoder.py`, `train_per_agency_ce.py` — train ModernBERT cross-encoders

### Stage 4: Benchmark Generation & Evaluation (`scripts/benchmark/`)
- `run_benchmark_per_model.py` — main benchmark runner (claim-list generation per model)
- `run_claim_list_variants.py` — RAG, few-shot, tool-calling claim-list variants
- `run_full_comment_variants.py` — full-comment generation (vanilla, persona, RAG, few-shot, tool)
- `sk3_extract_v2.py` — extract framings/structures from full comments (Llama-3.3-70B on GPU)
- `sk3_cluster_taxonomy.py` — cluster real framings/structures into canonical groups
- `sk3_match_v2.py` — exhaustive frame/style matching with Llama judge
- `run_claim_pair_judging.py` — claim pair judging with gpt-4o-mini
- `match_via_gpt4omini.py` — frame/style matching via OpenAI API
- `compute_pluralism_results.py` — compute precision, recall, distributional faithfulness

### Stage 5: Validation (`scripts/validation/`)
- `nader_validate.py` — validate against Handan-Nader FCC-RIF ground truth (66.8% recall)

## Requirements

- Python 3.10+
- GPU cluster with NVIDIA B200s (or equivalent) for Llama-3.3-70B inference via vLLM
- OpenAI API key (for gpt-4o-mini, gpt-5-mini, gpt-5)
- Anthropic API key (for validation with Claude Sonnet)

Key Python packages:
```
openai, anthropic, vllm, sentence-transformers, 
retriv, torch, transformers, pandas, numpy, scipy,
datasketch (MinHash), playwright (scraping)
```

## Reproducing Results

### 1. Data Collection
```bash
# Regulations.gov bulk download (requires manual captcha solving)
python scripts/ingestion/bulk_downloader.py

# External portals (e.g., FCC)
python scripts/ingestion/fetch_fcc_ecfs.py
```

### 2. Deduplication
```bash
python scripts/minhash_comment_deduping.py --input data/raw_text/ --output data/dedup/
```

### 3. Extraction & Matching
```bash
# Extract claims from comments
OPENAI_API_KEY=... python scripts/matching/comments_extract_claims.py

# Build indexes and run matching cascade
python scripts/matching/match_comments_and_create_indexes.py
```

### 4. Benchmark Evaluation
```bash
# Generate claims (vanilla + persona) per model
OPENAI_API_KEY=... python scripts/benchmark/run_benchmark_per_model.py \
    --model gpt-5-mini --provider openai

# Run claim pair judging
OPENAI_API_KEY=... python scripts/benchmark/run_claim_pair_judging.py

# Compute pluralism metrics
python scripts/benchmark/compute_pluralism_results.py
```

## Compute & Cost

- **GPU:** 8x NVIDIA B200 for embedding, cross-encoder training, and Llama-3.3-70B inference
- **API cost:** ~$340 total across all pipeline stages (extraction, matching, audit, benchmark generation)
- Cost breakdown per stage is documented in Appendix D of the paper

## Citation

```bibtex
@inproceedings{anonymous2026docket,
  title={DOCKET: ...},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
# docket-neurips-submission
