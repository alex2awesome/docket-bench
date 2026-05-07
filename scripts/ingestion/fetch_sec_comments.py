"""Scrape SEC rulemaking comments from sec.gov/comments.

SEC posts comments at predictable URLs:
    https://www.sec.gov/comments/[file-number]/[file-number].htm

File number prefixes:
    S7-XX-YY — SEC rulemakings (e.g., S7-14-19)
    SR-NNN   — Self-regulatory organization filings
    4-NNN    — Rulemaking petitions
    PCAOB-NN — PCAOB rules

Usage:
    python fetch_sec_comments.py --file-numbers S7-03-22 S7-14-19
    python fetch_sec_comments.py --list-recent
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, http_get, save_comments, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEC_BASE = "https://www.sec.gov"
SEC_HEADERS = {
    "User-Agent": "regulations-demo-research research@example.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def _parse_comment_table(soup: BeautifulSoup, file_number: str,
                          letter_type: str = "") -> list[dict]:
    """Parse comment rows from all tables on a single SEC comments page."""
    records = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            date = cells[0].get_text(strip=True) if len(cells) > 0 else ""
            submitter = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            link = tr.find("a")
            pdf_url = link["href"] if link and link.get("href") else ""
            if pdf_url and not pdf_url.startswith("http"):
                pdf_url = SEC_BASE + pdf_url
            records.append({
                "file_number": file_number,
                "letter_type": letter_type,
                "posted_date": date,
                "submitter": submitter,
                "pdf_url": pdf_url,
            })
    return records


def fetch_comment_page(file_number: str) -> tuple[list[dict], list[dict]]:
    """Fetch comments for a single file number.

    SEC comment pages have a two-level structure:
    1. Top-level page lists named individual comments (~30) in HTML tables,
       PLUS links to "Letter Type" sub-pages (A-Z, AA-AZ, etc.)
    2. Each sub-page is NOT a table — it's the template text of a mass form
       letter. The link text on the main page shows the count of signers.

    Returns (individual_comments, form_letters).
    """
    fn_lower = file_number.lower()
    fn_compact = fn_lower.replace("-", "")
    urls_to_try = [
        f"{SEC_BASE}/comments/{fn_lower}/{fn_lower}.htm",
        f"{SEC_BASE}/comments/{fn_lower}/{fn_compact}.htm",
    ]

    base_url = None
    resp = None
    for url in urls_to_try:
        try:
            resp = http_get(url, headers=SEC_HEADERS)
            base_url = url
            break
        except Exception as e:
            logger.debug("Failed %s: %s", url, e)

    if not resp:
        logger.warning("No comment page found for %s", file_number)
        return [], []

    soup = BeautifulSoup(resp.text, "html.parser")
    records = _parse_comment_table(soup, file_number, letter_type="named")

    # SEC uses Drupal-style ?page=N pagination. Find the "Last" page link.
    import time
    max_page = 0
    # Try title="Go to last page" or li.pager-last > a
    last_li = soup.find("li", class_=re.compile(r"last|pager-last"))
    last_page_link = last_li.find("a") if last_li else None
    if not last_page_link:
        last_page_link = soup.find("a", attrs={"title": re.compile(r"last", re.IGNORECASE)})
    if not last_page_link:
        # Fallback: find highest ?page=N in any link
        all_pages = re.findall(r"\?page=(\d+)", str(soup))
        if all_pages:
            max_page = max(int(p) for p in all_pages)
    if last_page_link and last_page_link.get("href"):
        page_match = re.search(r"page=(\d+)", last_page_link["href"])
        if page_match:
            max_page = int(page_match.group(1))
    if max_page > 0:
        logger.info("  %s has %d pages of named comments", file_number, max_page + 1)
        for page_num in range(1, max_page + 1):
            try:
                page_resp = http_get(f"{base_url}?page={page_num}",
                                     headers=SEC_HEADERS, timeout=120)
                page_soup = BeautifulSoup(page_resp.text, "html.parser")
                page_records = _parse_comment_table(page_soup, file_number, letter_type="named")
                records.extend(page_records)
                time.sleep(0.3)
            except Exception as e:
                logger.warning("  page %d failed: %s", page_num, e)
                break

    # Discover "Letter Type" sub-pages and their counts from anchor text
    type_pattern = re.compile(
        rf"/comments/{re.escape(fn_lower)}/{re.escape(fn_compact)}-type([a-z]+)\.htm",
        re.IGNORECASE,
    )
    type_pages = []
    for a in soup.find_all("a", href=True):
        m = type_pattern.search(a["href"])
        if m:
            letter_type = m.group(1)
            # The anchor text is "{TYPE_LABEL}: {count}" or just the count
            anchor_text = a.get_text(strip=True)
            # The count is usually in a sibling or the anchor text itself
            # Look for a number near this link
            count_text = anchor_text
            # Also check the parent element for a count
            parent_text = a.parent.get_text(strip=True) if a.parent else ""
            count_match = re.search(r"([\d,]+)", parent_text)
            count = int(count_match.group(1).replace(",", "")) if count_match else 0
            type_pages.append({
                "letter_type": letter_type,
                "href": a["href"],
                "count": count,
                "anchor_text": anchor_text,
            })

    # Dedupe by letter_type
    seen_types = set()
    unique_type_pages = []
    for tp in type_pages:
        if tp["letter_type"] not in seen_types:
            seen_types.add(tp["letter_type"])
            unique_type_pages.append(tp)

    form_letters = []
    if unique_type_pages:
        logger.info("  %s has %d letter-type sub-pages", file_number, len(unique_type_pages))

    import time
    for tp in sorted(unique_type_pages, key=lambda x: x["letter_type"]):
        sub_url = tp["href"] if tp["href"].startswith("http") else SEC_BASE + tp["href"]
        template_text = ""
        try:
            sub_resp = http_get(sub_url, headers=SEC_HEADERS, timeout=120)
            sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
            # Extract template text from the body. Some pages use <p> tags,
            # others have bare text nodes. Use full body text minus headings.
            body = sub_soup.find("body")
            if body:
                # Remove the heading (h2) which just repeats the subject/type
                for h in body.find_all(["h1", "h2", "h3"]):
                    h.decompose()
                template_text = body.get_text("\n", strip=True).strip()
            time.sleep(0.3)
        except Exception as e:
            logger.warning("    type %s fetch failed: %s", tp["letter_type"], e)

        form_letters.append({
            "file_number": file_number,
            "letter_type": tp["letter_type"].upper(),
            "signer_count": tp["count"],
            "template_text": template_text,
            "url": sub_url,
        })
        logger.info("    type %s: %d signers, %d chars template",
                     tp["letter_type"].upper(), tp["count"], len(template_text))

    total_form = sum(fl["signer_count"] for fl in form_letters)
    logger.info("Found %d individual + %d form-letter types (%d total signers) for %s",
                len(records), len(form_letters), total_form, file_number)
    return records, form_letters


def normalize(rows: list[dict]) -> pd.DataFrame:
    """Normalize to common comments schema."""
    records = []
    for i, row in enumerate(rows):
        fn = row.get("file_number", "")
        records.append({
            "comment_id": f"sec_{fn}_{i}",
            "docket_id": fn,
            "agency_id": "SEC",
            "submitter_name": row.get("submitter", ""),
            "submitter_org": "",
            "posted_date": row.get("posted_date", ""),
            "comment_text": "",  # Would need to fetch each PDF
            "attachment_urls": row.get("pdf_url", ""),
            "raw_metadata": str(row),
        })
    return pd.DataFrame(records)


def enumerate_file_numbers(start_year: int = 2016, end_year: int = 2026) -> list[str]:
    """Enumerate all S7-* file numbers from the SEC rulemaking-activity index.

    Iterates each year's page and extracts S7-XX-YY patterns from the rendered
    HTML. Paginates when a year's result list has more than one page.
    """
    all_fns = set()
    for year in range(start_year, end_year + 1):
        for page in range(0, 20):
            url = (
                f"{SEC_BASE}/rules-regulations/rulemaking-activity"
                f"?search=&rulemaking_status=All&division_office=All&year={year}&page={page}"
            )
            try:
                resp = http_get(url, headers=SEC_HEADERS, timeout=60)
            except Exception:
                break
            fns = set(re.findall(r"\bS7-\d{2}-\d{2}\b", resp.text))
            if not fns:
                break
            new = fns - all_fns
            all_fns.update(fns)
            if page > 0 and not new:
                break
        logger.info("  %d: %d unique so far", year, len(all_fns))
    return sorted(all_fns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-numbers", nargs="+", default=None,
                        help="SEC file numbers to fetch (e.g., S7-03-22)")
    parser.add_argument("--file-numbers-file", default=None,
                        help="Text file with one file number per line")
    parser.add_argument("--enumerate-years", action="store_true",
                        help="Auto-enumerate S7 file numbers from the rulemaking-activity index")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-processed file numbers (default: True)")
    args = parser.parse_args()

    # Determine the list of file numbers to process
    if args.file_numbers:
        file_numbers = args.file_numbers
    elif args.file_numbers_file:
        with open(args.file_numbers_file) as f:
            file_numbers = [line.strip() for line in f if line.strip()]
    elif args.enumerate_years:
        logger.info("Enumerating SEC file numbers %d-%d from rulemaking-activity index",
                    args.start_year, args.end_year)
        file_numbers = enumerate_file_numbers(args.start_year, args.end_year)
        logger.info("Enumerated %d file numbers", len(file_numbers))
        # Save the discovered list for future reference
        (get_output_dir("sec_comments") / "enumerated_file_numbers.txt").write_text(
            "\n".join(file_numbers)
        )
    else:
        logger.error("Provide --file-numbers, --file-numbers-file, or --enumerate-years")
        return

    done = load_done_units("sec_comments") if args.resume else set()
    logger.info("Total file numbers: %d, already done: %d, to process: %d",
                len(file_numbers), len(done), len(file_numbers) - len(done))

    output_dir = get_output_dir("sec_comments")
    form_letters_path = output_dir / "form_letters.csv.gz"

    for fn in file_numbers:
        if fn in done:
            continue
        try:
            individual, form_letters = fetch_comment_page(fn)
        except Exception as e:
            logger.warning("Failed %s: %s", fn, e)
            continue

        # Save individual named comments
        if individual:
            df = normalize(individual)
            append_comments(df, "sec_comments")

        # Save form letter templates + counts
        if form_letters:
            df_fl = pd.DataFrame(form_letters)
            if form_letters_path.exists():
                try:
                    existing = pd.read_csv(form_letters_path, dtype=str)
                    df_fl = pd.concat([existing, df_fl.astype(str)], ignore_index=True)
                    df_fl = df_fl.drop_duplicates(subset=["file_number", "letter_type"])
                except Exception:
                    pass
            df_fl.to_csv(form_letters_path, index=False, compression="gzip")

        mark_done("sec_comments", fn)

    save_metadata("sec_comments", {
        "file_numbers_count": len(file_numbers),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "notes": (
            "SEC comments scraped from sec.gov/comments/[file-number]/[file-number].htm. "
            "Two output files: comments.csv.gz (individual named comments with PDF links) "
            "and form_letters.csv.gz (mass form-letter templates with signer counts). "
            "File numbers enumerated from /rules-regulations/rulemaking-activity per year. "
            "Resumable via done_units.txt checkpoint."
        ),
    })


if __name__ == "__main__":
    main()
