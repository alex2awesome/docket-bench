# Non-Regulations.gov Comment Ingestion Plan

## Priority Tiers

### Tier 1: Easy Wins (Public API or Bulk Download)
1. **FCC ECFS** — Public JSON API at `https://publicapi.fcc.gov/ecfs/`
   - Free API key required: https://www.fcc.gov/ecfs/help/public_api
   - Very high volume (net neutrality 2017 had 21.7M comments)
   - Documentation: https://www.fcc.gov/ecfs/public-api-docs.html
2. **FACA Database** — Bulk CSV download at data.gov
   - No API key needed
   - URL: https://catalog.data.gov/dataset/federal-advisory-committee-act-faca-database-complete-raw
   - ~1,000 committees, 7,400 meetings/year, 72,000 members
3. **State AG Letters (Nolette DB)** — HTML scraping
   - URL: https://attorneysgeneral.org/letters-and-formal-comments/
   - No API key needed
   - Covers 2017-present, thousands of letters

### Tier 2: HTML Scraping (Well-Structured)
4. **SEC Comments** — Predictable URL structure
   - `sec.gov/comments/[file-number]/[file-number].htm`
5. **FDIC Public Comments** — Per-RIN pages
   - `fdic.gov/federal-register-publications/comments-rin-XXXX-XXXX`
6. **FEC FOSERS** — Rulemaking system (legacy JSP, session-based scraping)
   - `sers.fec.gov/fosers/`
   - No API; requires JSESSIONID bootstrap + `showselected` POST before
     `ruledata.htm` works. PDFs are at `showpdf.htm?docid=N` (no session needed).
   - As of 2026-04-04: no rulemakings open for comment.
7. **Federal Reserve Proposals** — Blazor Server SPA, no REST API
   - `federalreserve.gov/apps/proposals/`
   - SSR returns only 10 newest items per list view; URL params ignored.
   - Solution: brute-force probe `FR-{year}-{nnnn}-01` proposal IDs and
     `{proposal_id}-C{n}` comment IDs (both are dense-ish integer sequences).
   - Every comment is a direct PDF download.
   - Coverage starts at 2025 (pre-2025 comments are on Regulations.gov).

### Tier 3: NEPA/Project-Level Portals

Research findings: Only **CARA** exposes individual per-commenter letters. BLM ePlanning and NPS PEPC only publish comments as bundled staff PDFs (correspondence tables, scoping reports).

8. **BLM ePlanning** — NEPA Register, `eplanning.blm.gov`
   - Project enumeration: `POST /searchresults/` with `download=true` returns 67,835 projects in one call
   - **Per-comment data NOT exposed**
9. **NPS PEPC** — `parkplanning.nps.gov`
   - ColdFusion URLs (`parks.cfm`, `parkHome.cfm?parkID=N`, `documentsList.cfm?projectID=N`)
   - ~400 parks, 110,000+ projects since 2004
   - **Per-comment data NOT exposed** — only bundled PDFs
10. **Forest Service CARA** — `cara.fs2c.usda.gov` ⭐ **Best source for NEPA comments**
    - **Exposes individual letters** with author, org, date, attachments
    - Reading Room: `/Public/ReadingRoom?Project={ID}`
    - Letter detail: `/Public/Letter/{letterId}?project={ID}`
    - No browseable project index — must get project IDs from `fs.usda.gov/r{RR}/{forest}/projects`
    - Example: project 65356 has 10,100 letters

### Tier 4: Harder Systems
11. **PRC eDockets** — ArkCase system with undocumented REST
12. **FERC eFiling** — Entirely separate system
13. **EPA TCOTS** — APEX UI, no API
14. **USACE RRS** — Public notices metadata available via JSON API, but
    comments are write-only (`addPublicNoticeComment`). No read endpoint
    exists, no comment counts, no authorship exposed. Same class as BLM/NPS —
    rich project metadata, zero public comment text. RRS script collects
    metadata for the ~230 active public notices + ~1,200 pending individual
    permits from `permits.ops.usace.army.mil/orm-public-api/`.

### Deferred (FOIA or inaccessible)
- USPS — In-person inspection only
- Tribal consultation archives — Scattered PDFs
- Federal Subsistence Board — 5-year rolling window

## API Keys Needed

| System | API Key? | How to Get |
|--------|----------|------------|
| **FCC ECFS** | YES | Free, instant signup at https://api.data.gov/signup/ (gateway is api.data.gov; FCC help page just redirects there) |
| **Regulations.gov** (already have) | YES | api.data.gov key |
| All others | NO | Scraping, no key needed |

**Action required from user:** Register for FCC ECFS API key (free, instant).

## Output Schema

All ingestion scripts will write to `data/bulk_downloads/external_sources/{source}/` with:
- `comments.csv.gz` — per-comment records with common columns
- `dockets.csv.gz` — per-docket metadata
- `metadata.json` — source metadata, fetch timestamp, counts

Common columns (match Regulations.gov format where possible):
- `source` (e.g., "fcc_ecfs", "faca")
- `comment_id`
- `docket_id`
- `agency_id`
- `submitter_name`
- `submitter_org`
- `posted_date`
- `comment_text`
- `attachment_urls`
- `raw_metadata` (JSON blob)
