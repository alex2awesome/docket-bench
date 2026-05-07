"""Scrape FEC FOSERS (Searchable Electronic Rulemaking System).

https://sers.fec.gov/fosers/

FOSERS is a legacy Java/Tomcat JSP application with NO JSON/REST API.
All data is extracted via session-based HTML scraping.

Scraping protocol:
  1. GET /fosers/ → bootstrap JSESSIONID + AWSALB cookies.
  2. POST /fosers/;jsessionid={JS} with btnSubmit=findbyyear to list REGs per year.
  3. POST /fosers/;jsessionid={JS} with btnSubmit=showselected to set server-side
     "current rule" state (required or ruledata.htm returns a 133-byte error page).
  4. GET /fosers/ruledata.htm;jsessionid={JS}?ruleNumber=REG+YYYY-NN to get the
     document tree including Comments section.
  5. GET /fosers/showpdf.htm?docid={NNN} to download a comment PDF (no session
     required for this endpoint).

Coverage: years 1977-present. Activity is concentrated 2002+. Each year has 0-9 REGs.
As of 2026-04-04, NO rulemakings are currently open for comment.

Rate limits: AWS ALB/WAF trips after ~6-10 rapid requests → 403 Forbidden.
Safe throttle is ~1.5s between requests with exponential backoff on 403.

Usage:
    python fetch_fec_fosers.py                          # all years, all rules
    python fetch_fec_fosers.py --years 2020 2021 2022   # specific years
    python fetch_fec_fosers.py --rule "REG 2024-01"     # single rule
    python fetch_fec_fosers.py --skip-pdfs              # metadata only
"""

import argparse
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_comments, save_dockets, save_metadata, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://sers.fec.gov/fosers"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/",
}
THROTTLE = 1.5
MAX_RETRIES = 5


REG_LIST_RE = re.compile(
    r"ruleNumber\.value='(REG \d{4}-\d+[A-Za-z]?)'"
    r"[^\"]*\"[^>]*>\s*<strong>\s*(REG \d{4}-\d+[A-Za-z]?)\s+([^<]+?)</a></strong>",
    re.DOTALL,
)

# <li> for one document. Matches docid, label, optional date, optional note, then
# the following ul.ul_associatedPlayer block with entity rows.
DOC_LI_RE = re.compile(
    r'<a href="showpdf\.htm\?docid=(\d+)" target=image>([^<]+)</a>'
    r'(?:(?:&nbsp;)|\s)*?(\d{2}/\d{2}/\d{4})?'
    r'(?:.*?<BR>\s*([^<\n]+?)\s*)?'
    r'\s*</li>\s*'
    r'<ul class="ul_associatedPlayer">(.*?)</ul>',
    re.DOTALL,
)

ENTITY_LI_RE = re.compile(
    r'<li>\s*([^<]+?),&nbsp;&nbsp;(\w[\w/]*)\s*</li>',
    re.DOTALL,
)


class FosersClient:
    def __init__(self, throttle: float = THROTTLE):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.throttle = throttle
        self.jsid = None
        self._bootstrap()

    def _bootstrap(self):
        logger.info("Bootstrapping FOSERS session")
        resp = self.session.get(f"{BASE}/", timeout=30)
        resp.raise_for_status()
        self.jsid = self.session.cookies.get("JSESSIONID")
        if not self.jsid:
            raise RuntimeError("No JSESSIONID after bootstrap")
        logger.info("  JSESSIONID=%s...", self.jsid[:16])

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(MAX_RETRIES):
            time.sleep(self.throttle)
            try:
                resp = self.session.request(method, url, timeout=60, **kwargs)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning("Request failed %s: %s", url, e)
                time.sleep(5 * (2 ** attempt))
                continue

            if resp.status_code == 403:
                backoff = 5 * (3 ** attempt)
                logger.warning("403 on %s — sleeping %ds and re-bootstrapping", url, backoff)
                time.sleep(backoff)
                # Session may have expired; try re-bootstrapping once
                if attempt == MAX_RETRIES // 2:
                    self._bootstrap()
                continue

            if resp.status_code >= 500:
                time.sleep(5 * (2 ** attempt))
                continue

            resp.raise_for_status()

            # FOSERS returns a 133-byte "Error Page" HTML when session state is wrong
            if len(resp.content) == 133 and b"Error Page" in resp.content:
                raise RuntimeError(f"FOSERS session state lost on {url}")

            return resp
        raise RuntimeError(f"exhausted retries for {url}")

    def list_year(self, year: int) -> list[dict]:
        """Return list of {reg, title} for all REGs in a year."""
        data = {
            "btnSubmit": "findbyyear",
            "searchType": "B",
            "showPending": "0",
            "showByRegYear": str(year),
            "displayPageNo": "1",
            "sortOrder": "2",
            "keywords": "",
            "ruleNumber": "",
            "title": "",
            "resultsPerPage": "0",
        }
        resp = self._request(
            "POST",
            f"{BASE}/;jsessionid={self.jsid}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        rules = []
        for m in REG_LIST_RE.finditer(resp.text):
            rules.append({
                "reg_number": m.group(1),
                "title": m.group(3).strip(),
                "year": year,
            })
        return rules

    def list_open(self) -> list[dict]:
        data = {
            "btnSubmit": "find",
            "searchType": "B",
            "showPending": "1",
            "showByRegYear": "",
            "displayPageNo": "1",
            "sortOrder": "2",
            "keywords": "",
            "ruleNumber": "",
            "title": "",
            "resultsPerPage": "0",
        }
        resp = self._request(
            "POST",
            f"{BASE}/;jsessionid={self.jsid}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if "No Rulemakings are open" in resp.text:
            return []
        return [
            {"reg_number": m.group(1), "title": m.group(3).strip()}
            for m in REG_LIST_RE.finditer(resp.text)
        ]

    def get_rule_details(self, reg_number: str) -> str:
        """Fetch ruledata.htm for a rule. Requires setting server-side state first."""
        # Step 1: showselected to set server-side state
        data = {
            "btnSubmit": "showselected",
            "ruleNumber": reg_number,
            "displayDoc": "",
            "displayPageNo": "1",
            "resultsPerPage": "0",
        }
        self._request(
            "POST",
            f"{BASE}/;jsessionid={self.jsid}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # Step 2: GET ruledata.htm
        resp = self._request(
            "GET",
            f"{BASE}/ruledata.htm;jsessionid={self.jsid}",
            params={"ruleNumber": reg_number},
        )
        return resp.text

    def download_pdf(self, docid: str) -> bytes | None:
        resp = self._request("GET", f"{BASE}/showpdf.htm", params={"docid": docid})
        if "pdf" not in resp.headers.get("Content-Type", "").lower():
            return None
        return resp.content


def parse_rule_documents(reg_number: str, html: str) -> list[dict]:
    """Walk ruledata.htm and yield one dict per comment document.

    Uses BeautifulSoup rather than regex to handle the inconsistent whitespace
    and HTML formatting in FOSERS's JSP output.
    """
    from bs4 import BeautifulSoup as BS

    if "All Documents" not in html:
        return []

    # Slice to just the All Documents section (skip the Entities roster below)
    parts = html.split("All Documents", 1)
    if len(parts) < 2:
        return []
    docs_section = parts[1]
    if "Entities" in docs_section:
        docs_section = docs_section.split("Entities", 1)[0]

    soup = BS(docs_section, "html.parser")
    comments = []

    for li in soup.find_all("li"):
        # Each comment <li> contains a showpdf.htm link
        link = li.find("a", href=re.compile(r"showpdf\.htm\?docid=\d+"))
        if not link:
            continue
        docid_match = re.search(r"docid=(\d+)", link["href"])
        if not docid_match:
            continue
        docid = docid_match.group(1)
        label = link.get_text(strip=True)

        # Only keep Comments and Hearing Testimony
        if "Comment" not in label and "Hearing" not in label:
            continue

        # Extract date and note from the <li> text
        li_text = li.get_text(" ", strip=True)
        date_match = re.search(r"(\d{2}/\d{2}/\d{4})", li_text)
        date = date_match.group(1) if date_match else ""

        # Note is text after a <BR> tag (e.g., "Late Comment", "307 Comments from Individuals...")
        br = li.find("br")
        note = ""
        if br and br.next_sibling:
            note_text = br.next_sibling
            if isinstance(note_text, str):
                note = note_text.strip()

        # Entity list is the next <ul class="ul_associatedPlayer"> sibling
        entity_ul = li.find_next_sibling("ul", class_="ul_associatedPlayer")
        commenters = []
        reps = []
        if entity_ul:
            for entity_li in entity_ul.find_all("li"):
                text = entity_li.get_text(strip=True)
                # Format: "Name,\xa0\xa0Role" (with &nbsp;)
                text = text.replace("\xa0", " ")
                parts = re.split(r",\s{2,}", text, maxsplit=1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    role = parts[1].strip()
                    if role == "Commenter":
                        commenters.append(name)
                    else:
                        reps.append((name, role))

        # Heuristic: orgs don't contain ", " in their name; people do (Last, First)
        orgs = [n for n in commenters if ", " not in n]
        people = [n for n in commenters if ", " in n]

        comments.append({
            "comment_id": f"{reg_number.replace(' ', '_')}_doc{docid}",
            "docket_id": reg_number,
            "agency_id": "FEC",
            "submitter_name": "; ".join(people) or None,
            "submitter_org": "; ".join(orgs) or None,
            "posted_date": date or None,
            "comment_text": "",
            "attachment_urls": f"{BASE}/showpdf.htm?docid={docid}",
            "raw_metadata": {
                "docid": docid,
                "label": label,
                "note": note,
                "representatives": [f"{n} ({r})" for n, r in reps],
            },
        })

    return comments


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF. Returns empty string on failure."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        logger.debug("pdf text extraction failed: %s", e)
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2000, 2027)),
                        help="Years to scan (default: 2000-2026)")
    parser.add_argument("--rule", help="Single REG number (e.g., 'REG 2024-01')")
    parser.add_argument("--skip-pdfs", action="store_true",
                        help="Skip PDF downloads, just save metadata")
    parser.add_argument("--throttle", type=float, default=THROTTLE)
    args = parser.parse_args()

    output_dir = get_output_dir("fec_fosers")
    raw_pdf_dir = output_dir / "raw" / "pdfs"
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    client = FosersClient(throttle=args.throttle)

    # Discover rules
    if args.rule:
        rules = [{"reg_number": args.rule, "title": "", "year": None}]
    else:
        rules = []
        for year in args.years:
            try:
                year_rules = client.list_year(year)
                logger.info("Year %d: %d REGs", year, len(year_rules))
                rules.extend(year_rules)
            except Exception as e:
                logger.warning("Failed to list %d: %s", year, e)

    if not rules:
        logger.warning("No rules found")
        return

    df_dockets = pd.DataFrame(rules)
    save_dockets(df_dockets, "fec_fosers")

    done = load_done_units("fec_fosers")
    logger.info("Rules total: %d, already done: %d", len(rules), len(done))

    # Scrape comments per rule, writing incrementally per REG
    for rule in rules:
        reg = rule["reg_number"]
        if reg in done:
            continue
        try:
            html = client.get_rule_details(reg)
        except Exception as e:
            logger.error("Failed details for %s: %s", reg, e)
            continue

        comments = parse_rule_documents(reg, html)
        logger.info("  %s: %d comments", reg, len(comments))

        if not args.skip_pdfs:
            for c in comments:
                docid = c["raw_metadata"]["docid"]
                pdf_path = raw_pdf_dir / f"{docid}.pdf"
                if pdf_path.exists():
                    pdf_bytes = pdf_path.read_bytes()
                else:
                    try:
                        pdf_bytes = client.download_pdf(docid)
                    except Exception as e:
                        logger.warning("    pdf fail docid=%s: %s", docid, e)
                        pdf_bytes = None
                    if pdf_bytes:
                        pdf_path.write_bytes(pdf_bytes)
                if pdf_bytes:
                    c["comment_text"] = extract_pdf_text(pdf_bytes)

        for c in comments:
            c["raw_metadata"]["rule_title"] = rule.get("title", "")

        if comments:
            import json
            df = pd.DataFrame(comments)
            df["raw_metadata"] = df["raw_metadata"].apply(json.dumps)
            append_comments(df, "fec_fosers")
        mark_done("fec_fosers", reg)

    save_metadata("fec_fosers", {
        "source_url": BASE,
        "n_rules": len(rules),
        "years_scanned": args.years if not args.rule else None,
        "notes": (
            "FEC FOSERS is a legacy JSP app with no API. Scraped via session-based "
            "HTML parsing. Each rule has 0-N Comments PDFs (submitter name + role "
            "exposed in HTML). PDF text extracted via pypdf. "
            "Pre-2010 filings are often scanned and need OCR. "
            "Rate limit: ~1.5s/request, exponential backoff on 403."
        ),
    })


if __name__ == "__main__":
    main()
