"""Scrape Federal Reserve Board public comments on rulemaking proposals.

https://www.federalreserve.gov/apps/proposals/

The Fed comment portal is a Blazor Server SPA with NO REST/JSON API. The
server-side prerender always returns only the 10 newest items per list view
regardless of URL params, so stateless enumeration via HTML is not possible.

HOWEVER, two things make scraping feasible without a headless browser:

1. Every proposal lives at a predictable ID: `FR-{year}-{nnnn}-01`.
2. Every comment is a direct PDF at
   `/apps/proposals/comments/{PROPOSAL_ID}-C{N}` with N a dense-ish integer.

Strategy:
  a) Brute-force probe `FR-{year}-{nnnn}-01/details` for each year to
     discover live proposals.
  b) Read the `Comments (N)` tab-pill to size the comment ID search space.
  c) HEAD-probe every `FR-...-C{n}` to find live comment IDs (cheap).
  d) GET the PDF bytes and extract text with pypdf.
  e) Parse the first ~10 submitter rows from the SSR HTML (all we get).

Coverage: The Fed's new comment app starts at 2025. Pre-2025 Fed comments
are posted to Regulations.gov in the usual way.

Usage:
    python fetch_fed_reserve.py                # discover + fetch everything from 2025+
    python fetch_fed_reserve.py --years 2025 2026
    python fetch_fed_reserve.py --proposal FR-2025-0077-01   # single proposal
"""

import argparse
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_comments, save_dockets, save_metadata, http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://www.federalreserve.gov/apps/proposals"
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; stanford-regulations-research/1.0; "
        "+https://github.com/; research-scraper)"
    ),
}
POLITE_DELAY = 0.35


def fetch_proposal_details(proposal_id: str) -> dict | None:
    """Fetch /apps/proposals/{pid}/details and parse metadata.

    Returns None if the proposal doesn't exist.
    """
    url = f"{BASE}/{proposal_id}/details"
    try:
        resp = http_get(url, headers=UA_HEADERS, timeout=30)
    except Exception as e:
        logger.debug("details fetch failed for %s: %s", proposal_id, e)
        return None

    html = resp.text
    if "Proposal not found" in html or "Page not found" in html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.main-content") or soup

    title_h5 = main.select_one("h5")
    title_text = title_h5.get_text(" ", strip=True) if title_h5 else ""
    label_match = re.search(r"\[([^\]]+)\]\s*$", title_text)
    internal_label = label_match.group(1) if label_match else None
    title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title_text).strip()

    doc_type = None
    b = main.find("b")
    if b:
        doc_type = b.get_text(strip=True)

    ps = main.find_all("p")
    abstract = ps[0].get_text(" ", strip=True) if ps else ""

    due_match = re.search(r"Comments Due:\s*(\d\d/\d\d/\d{4})", html)
    comments_due = due_match.group(1) if due_match else None

    count_match = re.search(r"Comments\s*\((\d+)\)", html)
    n_comments = int(count_match.group(1)) if count_match else 0

    fr_link = soup.find("a", href=re.compile(r"federalregister\.gov"))
    fr_url = fr_link["href"] if fr_link else None

    return {
        "proposal_id": proposal_id,
        "internal_label": internal_label,
        "title": title,
        "doc_type": doc_type,
        "abstract": abstract,
        "comments_due": comments_due,
        "n_comments": n_comments,
        "federal_register_url": fr_url,
    }


def discover_proposals(years: list[int], max_seq: int = 200, gap_tolerance: int = 30) -> list[dict]:
    """Probe FR-{year}-{nnnn}-01 until `gap_tolerance` consecutive misses."""
    found = []
    for year in years:
        consec_miss = 0
        logger.info("Discovering proposals for %d", year)
        for n in range(1, max_seq + 1):
            pid = f"FR-{year}-{n:04d}-01"
            meta = fetch_proposal_details(pid)
            if meta is None:
                consec_miss += 1
                if consec_miss >= gap_tolerance:
                    logger.info("  giving up at n=%d after %d misses", n, gap_tolerance)
                    break
                continue
            consec_miss = 0
            logger.info("  %s: %s (%d comments)", pid, meta["title"][:60], meta["n_comments"])
            found.append(meta)
            time.sleep(POLITE_DELAY)
    return found


COMMENT_ROW_RE = re.compile(
    r'<div[^>]*class="[^"]*comment-row[^"]*"[^>]*>'
    r'.*?<a[^>]*href="(?:[^"]*?/)?comments/([A-Z0-9-]+)"[^>]*>\s*([^<]*?)\s*</a>'
    r'.*?Posted\s*\|\s*([A-Za-z]+\s+\d+,\s*\d{4})',
    re.DOTALL,
)


def parse_comment_rows(html: str) -> dict[str, dict]:
    """Parse SSR comment rows (returns {comment_id: {display, posted}})."""
    rows = {}
    for m in COMMENT_ROW_RE.finditer(html):
        cid = m.group(1)
        display = m.group(2)
        posted = m.group(3)
        rows[cid] = {"display": display, "posted": posted}
    return rows


def parse_display_name(display: str) -> tuple[str | None, str | None, str | None]:
    """Parse '{Org}, {Name} (PDF)' → (org, name, format)."""
    if not display:
        return None, None, None
    fmt_match = re.search(r"\(([^)]+)\)\s*$", display)
    fmt = fmt_match.group(1) if fmt_match else None
    core = re.sub(r"\s*\([^)]+\)\s*$", "", display).strip()
    if core.lower() == "anonymous":
        return None, None, fmt
    # Mask ", et. al.," so it isn't split on
    tmp = core.replace(", et. al.,", "§§§ET§§§")
    parts = tmp.rsplit(",", 1)
    if len(parts) == 2:
        org = parts[0].replace("§§§ET§§§", ", et. al.,").strip()
        name = parts[1].replace("§§§ET§§§", ", et. al.,").strip()
        return (org or None), (name or None), fmt
    return None, core, fmt


def head_probe(url: str) -> tuple[int, dict]:
    """HEAD with redirects; returns (status, headers)."""
    import requests
    try:
        r = requests.head(url, headers=UA_HEADERS, allow_redirects=True, timeout=30)
        return r.status_code, dict(r.headers)
    except requests.RequestException:
        return 0, {}


def fetch_comment_pdf(cid: str) -> tuple[bytes | None, str | None, str | None]:
    """Download a single comment PDF.

    Returns (pdf_bytes, extracted_text, filename).
    """
    url = f"{BASE}/comments/{cid}"
    try:
        resp = http_get(url, headers=UA_HEADERS, timeout=60)
    except Exception as e:
        logger.debug("pdf fetch failed %s: %s", cid, e)
        return None, None, None

    if "pdf" not in resp.headers.get("content-type", "").lower():
        return None, None, None

    cd = resp.headers.get("content-disposition", "")
    fn_match = re.search(r'filename=([^;]+)', cd)
    filename = fn_match.group(1).strip().strip('"') if fn_match else None

    text = None
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(resp.content))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        logger.debug("pdf parse failed %s: %s", cid, e)

    return resp.content, text, filename


def scrape_proposal_comments(meta: dict, raw_pdf_dir: Path) -> list[dict]:
    """Walk comment ID space for a proposal, fetching every PDF."""
    pid = meta["proposal_id"]
    hint = meta["n_comments"]
    if hint == 0:
        return []

    # Fetch SSR comment list page to grab the (at most) 10 visible rows' submitter metadata
    try:
        list_html = http_get(f"{BASE}/{pid}/comments", headers=UA_HEADERS, timeout=30).text
    except Exception as e:
        logger.warning("Failed to fetch comments list for %s: %s", pid, e)
        list_html = ""
    visible_rows = parse_comment_rows(list_html)

    # Determine the probe range. Comment IDs are sparse integers that DON'T
    # start at C1 — they're assigned sequentially across ALL proposals system-wide.
    # Use visible IDs on the initial page to find the local range, then probe
    # a wide window around it.
    visible_ids = []
    for vid in visible_rows.keys():
        m = re.search(r"-C0*(\d+)$", vid)
        if m:
            visible_ids.append(int(m.group(1)))

    if visible_ids:
        # Visible IDs on the page are the NEWEST N comments. Older comments
        # extend downward from the minimum visible ID. Probe from C1 (or as
        # far back as needed to cover `hint` comments) up to max_visible+buffer.
        max_visible = max(visible_ids)
        # Go back at least hint*2 IDs to account for sparse IDs + other proposals'
        # interleaved numbering, but never below C1.
        start_n = max(1, max_visible - max(hint * 3, 500))
        end_n = max_visible + 50
    else:
        # Fallback: hint-based bounds
        start_n = 1
        end_n = max(500, hint * 3 + 100)

    logger.info("  scanning %s-C%d..C%d (hint=%d, visible_rows=%d)",
                pid, start_n, end_n, hint, len(visible_rows))

    results = []
    consecutive_misses = 0
    max_consec_miss = 200  # Comment IDs are sparse; allow longer gaps

    for n in range(start_n, end_n + 1):
        if consecutive_misses >= max_consec_miss and len(results) >= hint:
            break
        # Comment IDs use zero-padding: C01..C99 (2 digits), then C100+ (no padding)
        cid = f"{pid}-C{n:02d}" if n < 100 else f"{pid}-C{n}"
        status, headers = head_probe(f"{BASE}/comments/{cid}")
        # HEAD may return empty content-type (Cloudflare strips it).
        # Treat any non-404 response as a potential hit.
        if status == 404 or status == 0:
            consecutive_misses += 1
            continue

        pdf_bytes, text, filename = fetch_comment_pdf(cid)
        if pdf_bytes is None:
            continue

        # Save PDF to disk for future reference
        if filename:
            (raw_pdf_dir / filename).write_bytes(pdf_bytes)
        else:
            (raw_pdf_dir / f"{cid}.pdf").write_bytes(pdf_bytes)

        row = visible_rows.get(cid, {})
        display = row.get("display")
        org, name, _fmt = parse_display_name(display) if display else (None, None, None)

        # Extract upload timestamp from filename (MMDDYYYYHHMM suffix)
        posted = row.get("posted")
        if not posted and filename:
            ts_match = re.search(r"-(\d{2})(\d{2})(\d{4})\d{4}\.pdf$", filename)
            if ts_match:
                mm, dd, yyyy = ts_match.groups()
                posted = f"{yyyy}-{mm}-{dd}"

        results.append({
            "comment_id": cid,
            "docket_id": pid,
            "agency_id": "FRS",
            "submitter_name": name,
            "submitter_org": org,
            "posted_date": posted,
            "comment_text": text or "",
            "attachment_urls": f"{BASE}/comments/{cid}",
            "raw_metadata": {
                "filename": filename,
                "display": display,
                "proposal_title": meta["title"],
                "proposal_type": meta["doc_type"],
                "internal_label": meta["internal_label"],
            },
        })
        time.sleep(POLITE_DELAY)

    logger.info("  %s: fetched %d comments (hint was %d)", pid, len(results), hint)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026],
                        help="Years to scan for proposals")
    parser.add_argument("--proposal", help="Single proposal ID (skips discovery)")
    parser.add_argument("--max-seq", type=int, default=200)
    parser.add_argument("--skip-pdfs", action="store_true",
                        help="Discover + list only, do not download PDFs")
    args = parser.parse_args()

    output_dir = get_output_dir("fed_reserve")
    raw_pdf_dir = output_dir / "raw" / "pdfs"
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    if args.proposal:
        meta = fetch_proposal_details(args.proposal)
        if not meta:
            logger.error("Proposal %s not found", args.proposal)
            return
        proposals = [meta]
    else:
        proposals = discover_proposals(args.years, max_seq=args.max_seq)

    logger.info("Discovered %d proposals, %d total comments",
                len(proposals),
                sum(p["n_comments"] for p in proposals))

    df_dockets = pd.DataFrame(proposals)
    save_dockets(df_dockets, "fed_reserve")

    if args.skip_pdfs:
        save_metadata("fed_reserve", {
            "source_url": BASE,
            "n_proposals": len(proposals),
            "total_comments": sum(p["n_comments"] for p in proposals),
            "mode": "discovery_only",
        })
        return

    all_comments = []
    for meta in proposals:
        if meta["n_comments"] == 0:
            continue
        try:
            rows = scrape_proposal_comments(meta, raw_pdf_dir)
            all_comments.extend(rows)
        except Exception as e:
            logger.error("Failed proposal %s: %s", meta["proposal_id"], e)

    if all_comments:
        df = pd.DataFrame(all_comments)
        df["raw_metadata"] = df["raw_metadata"].apply(lambda d: pd.io.json.dumps(d) if hasattr(pd.io.json, "dumps") else str(d))
        save_comments(df, "fed_reserve")

    save_metadata("fed_reserve", {
        "source_url": BASE,
        "n_proposals": len(proposals),
        "n_comments": len(all_comments),
        "years_scanned": args.years,
        "notes": (
            "Federal Reserve proposals app is a Blazor Server SPA with no REST API. "
            "Proposals and comments scraped by brute-forcing sequential IDs "
            "(FR-{year}-{nnnn}-01 for proposals, {pid}-C{n} for comments). "
            "Submitter name/org only available for the first ~10 comments per proposal "
            "(SSR limitation). For later comments, parse name from PDF cover letter "
            "or use content-disposition filename's MMDDYYYYHHMM suffix for posted_date."
        ),
    })


if __name__ == "__main__":
    main()
