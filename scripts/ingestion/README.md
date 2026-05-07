# External Source Ingestion

Scripts for pulling public comments from systems that are NOT Regulations.gov.

See `INGESTION_PLAN.md` for the prioritized list of systems and the research notes at `../../running-research-notes.md` for background.

## Output Location

All scripts write to `../../external_sources/{source_name}/`:
- `comments.csv.gz` — normalized per-comment records
- `dockets.csv.gz` — per-docket metadata (where applicable)
- `metadata.json` — source metadata, fetch timestamp, counts
- `raw/` — raw downloaded files (CSVs, PDFs, HTML dumps)

## Common Schema

All ingestion scripts output comments in this schema (matching Regulations.gov where possible):
- `source` — e.g., "fcc_ecfs", "faca", "sec_comments"
- `comment_id`
- `docket_id`
- `agency_id`
- `submitter_name`
- `submitter_org`
- `posted_date`
- `comment_text`
- `attachment_urls`
- `raw_metadata` — JSON blob with source-specific fields

## Available Scripts

### Tier 1: Public API / Bulk Download

| Script | System | Requires API key? | Status |
|--------|--------|-------------------|--------|
| `fetch_fcc_ecfs.py` | FCC ECFS | **YES** — free, instant at https://api.data.gov/signup/ | Ready (needs key; handles 10k offset cap via date-sharding or `--downloadplan`) |
| `fetch_faca_database.py` | FACA Database (GSA) | No | Ready |
| `fetch_state_ag_letters.py` | State AG letters (Nolette DB) | No | Ready |

### Tier 2: HTML Scraping

| Script | System | Requires API key? | Status |
|--------|--------|-------------------|--------|
| `fetch_sec_comments.py` | SEC rulemaking comments | No | Ready |
| `fetch_fdic_comments.py` | FDIC public comments | No | Ready |
| `fetch_fed_reserve.py` | Federal Reserve Board | No | Ready (brute-force ID probing; 2025+ only) |
| `fetch_fec_fosers.py` | FEC FOSERS | No | Ready (session-based JSP scraping) |

### Tier 3: NEPA/Project Portals

| Script | System | Requires API key? | Status |
|--------|--------|-------------------|--------|
| `fetch_blm_eplanning.py` | BLM ePlanning | No | Ready (projects only, no per-comment) |
| `fetch_nps_pepc.py` | NPS PEPC | No | Ready (projects/docs, no per-comment) |
| `fetch_fs_cara.py` | Forest Service CARA | No | Ready (includes per-letter data) ⭐ |
| `fetch_usace_rrs.py` | USACE RRS | No | Ready (metadata only, no per-comment data) |

### Deferred

- USPS — in-person inspection only at HQ library
- Tribal consultation archives — scattered PDFs, no API
- NIST guidance drafts — email-only submission, comments not publicly posted
- TSA Security Directives — not public
- FERC eFiling — possible but large scope

## Setup

```bash
# Install dependencies
pip install pandas requests beautifulsoup4

# Get FCC API key (free, instant; gateway is api.data.gov)
# Sign up at https://api.data.gov/signup/
# Save to ~/.fcc-api-key or export FCC_API_KEY=...
```

## Usage Examples

```bash
# FACA database bulk download
python fetch_faca_database.py

# State AG letters
python fetch_state_ag_letters.py

# SEC comments for specific file numbers
python fetch_sec_comments.py --file-numbers S7-14-19 S7-03-22

# FDIC comments for all RINs
python fetch_fdic_comments.py --list-all

# FCC comments for a specific docket (requires API key)
python fetch_fcc_ecfs.py --docket 17-108                    # shard by 30-day windows (safe)
python fetch_fcc_ecfs.py --docket 21-450 --downloadplan     # single-shot, no sharding

# Federal Reserve comments (brute-force ID probing, 2025+ only)
python fetch_fed_reserve.py --years 2025 2026

# FEC FOSERS comments (session-based JSP scraping)
python fetch_fec_fosers.py --years 2020 2021 2022 2023 2024 2025

# USACE public notices and pending permits (metadata only)
python fetch_usace_rrs.py
```

## Per-Comment Data Availability

| Source | Per-comment text exposed? | Notes |
|--------|---------------------------|-------|
| FCC ECFS | Yes | Full API + attachment URLs |
| FACA | N/A | Advisory committee meeting/member data, not "comments" |
| State AG letters | Yes | PDFs of signed letters |
| SEC comments | Yes | Direct `.htm`/PDF links |
| FDIC comments | Yes | Per-RIN comment pages |
| Fed Reserve | Yes (PDFs only) | Submitter name only for first 10 per proposal (SSR limitation) |
| FEC FOSERS | Yes (PDFs only) | Submitter name + role in HTML; text in PDFs |
| Forest Service CARA | Yes ⭐ | Individual letters with full metadata |
| BLM ePlanning | **No** | Only bundled staff PDFs |
| NPS PEPC | **No** | Only bundled staff PDFs |
| USACE RRS | **No** | Write-only comment API; metadata only |

## Merging with Regulations.gov Data

After ingestion, the `comments.csv.gz` files can be merged with our existing
Regulations.gov data by matching on `docket_id` + `posted_date` or by using
Federal Register Number (RIN) as a common key.

For agencies with dual-channel submission (OCC, FDIC, Fed):
- Regulations.gov has some comments
- External source has others
- Dedupe by `(submitter_name, docket_id, date)` tuple to merge
