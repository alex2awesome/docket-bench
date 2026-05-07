# Comment-Matching Pipeline Validation Report
## FCC Docket 17-108 (Restoring Internet Freedom) vs. Handan-Nader Ground Truth

Generated: 2026-04-15

## 1. Setup

| Component | Description |
|---|---|
| **Rule text** | FCC-17-166A1 (Restoring Internet Freedom order), 539 pages |
| **Response extraction** | 72 passages extracted via gpt-5-nano |
| **Claim extraction** | Llama-3.1-70B-FP8, claims_v2 level |
| **Dense retrieval** | nvidia/llama-embed-nemotron-8b, top-200 |
| **Cross-encoder** | FCC-specific ModernBERT (fine-tuned) |
| **CE threshold** | 0.5 (calibrated) |
| **Corpus** | 8,660 submissions -> 10,235 claims -> 8,686 dedup clusters |

## 2. Ground Truth

| Metric | Value |
|---|---|
| Nader cited unique submissions | **307** |
| Nader unique citation IDs (footnotes) | 1,883 |
| Nader total citation rows | 2,060 |
| Nader unique order passages (points) | 200 |
| Our extracted response passages | 72 |

## 3. Headline Recall Metrics

| Metric | Count | Rate |
|---|---|---|
| **Recall (submission-level, no expansion)** | **205 / 307** | **66.8%** |
| Recall (of in-pipeline only) | 205 / 232 | 88.4% |
| **Recall (with cluster expansion)** | **218 / 307** | **71.0%** |
| Recall (with Nader near-dup expansion) | 220 / 334 | 65.9% |

## 4. Pipeline Output Summary

| Metric | Value |
|---|---|
| Total submissions in pipeline | 8,660 |
| Total claims extracted | 10,235 |
| Total dedup clusters | 8,686 |
| **Total matched submissions** | **2,153** |
| Total matched claims | 2,876 |
| Total matched clusters | 2,174 |
| Total matched (claim, response) pairs | 3,657 |
| Responses with >= 1 match | 64 / 72 |
| Match rate (submissions) | 24.9% |

## 5. Error Source Breakdown

Of the 102 Nader-cited submissions not directly matched:

| Error Source | Count | % of Missed |
|---|---|---|
| Not in dedup mapper (unknown doc) | 1 | 1.0% |
| Dedup-merged, cluster rep not matched | 61 | 59.8% |
| Dedup-merged, recoverable via expansion | 13 | 12.7% |
| In pipeline, retrieval gap (never top-200) | 0 | 0.0% |
| In pipeline, CE threshold rejection | 27 | 26.5% |
| In pipeline, cluster-mate recoverable | 0 | 0.0% |
| **Total missed** | **102** | |

## 6. Novel Match Confidence

We matched 2,153 unique submissions total. 205 overlap with Nader's 307 cited. 
The remaining 1,948 are 'novel' matches not in Nader's ground truth.

| Score Metric | Nader Cited (matched) | Novel Matches |
|---|---|---|
| N (claims) | 593 | 2283 |
| CE mean | 0.916 | 0.881 |
| CE median | 0.975 | 0.946 |
| CE std | 0.120 | 0.137 |
| Dense mean | 0.640 | 0.631 |
| Dense median | 0.646 | 0.637 |
| Novel with CE > Nader median | | 33.7% |

Novel matches have comparable score distributions to ground-truth matches, 
suggesting they are genuine matches that Nader's footnote-based method did not capture 
(the FCC order cannot cite every relevant comment).

## 7. CE Threshold Sensitivity

| CE Threshold | Recall (of 307) | Total Matched Subs | Matched Pairs |
|---|---|---|---|
| >= 0.3 | 214 / 307 (69.7%) | 2,385 | 4,168 |
| >= 0.4 | 208 / 307 (67.8%) | 2,275 | 3,916 |
| >= 0.5 **(current)** | 205 / 307 (66.8%) | 2,153 | 3,657 |
| >= 0.6 | 202 / 307 (65.8%) | 1,992 | 3,358 |
| >= 0.7 | 199 / 307 (64.8%) | 1,836 | 3,061 |
| >= 0.8 | 195 / 307 (63.5%) | 1,686 | 2,762 |
| >= 0.9 | 186 / 307 (60.6%) | 1,423 | 2,262 |

## 8. Comparison to Pilot

| Metric | Pilot (mpnet + comment-level) | Current (nemotron-8b + claims_v2 + FCC CE) |
|---|---|---|
| Recall | 25.0% (79/316) | **66.8%** (205/307) |
| Recall (with expansion) | -- | **71.0%** (218/307) |
| Retrieval reach | 31.0% | 100.0% |
| Total matched subs | 287 | **2,153** |
| Match rate | 3.4% | 24.9% |
| Dense embedder | all-mpnet-base-v2 (384 tok) | nemotron-8b (4096 tok) |
| Matching level | comment | claims_v2 (per-claim) |
| Cross-encoder | none (LLM threshold) | FCC ModernBERT (fine-tuned) |
| Retrieval k | 10 | 200 |

## 9. Key Takeaways

1. **2.7x recall improvement** over pilot (66.8% vs 25.0%), 
   driven by claims-level matching, nemotron-8b embeddings, and FCC cross-encoder.
2. **Cluster expansion adds 4.2 percentage points** (from 66.8% to 71.0%), 
   recovering 13 submissions merged during dedup.
3. **1,948 novel matches** beyond Nader's 307 cited, 
   with score distributions comparable to ground-truth matches.
4. **Primary error source**: 62 of 102 missed (61%) are lost to dedup merging 
   where the cluster representative was never matched.
5. **CE threshold at 0.5 is well-calibrated**: lowering to 0.3 gains only marginal recall
   while substantially increasing false positives.
