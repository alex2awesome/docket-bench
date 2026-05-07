"""Scrape Forest Service CARA (Comment Analysis and Response Application).

CARA at https://cara.fs2c.usda.gov is the Forest Service's exclusive system for
public comments on NEPA projects. Unlike BLM ePlanning and NPS PEPC, CARA
exposes INDIVIDUAL per-commenter letters with author name, organization, date,
and attachment PDFs.

CARA has no browseable project index — project IDs must come from Forest
Service forest-level project pages (fs.usda.gov/r{RR}/{forest}/projects).

Each project's reading room:
    https://cara.fs2c.usda.gov/Public/ReadingRoom?Project={ID}

Each letter:
    https://cara.fs2c.usda.gov/Public/Letter/{letterId}?project={ID}

Letter files:
    /Public/DownloadCommentFile?LetterId={id}&IsLetter={True|False}
    /Public/DownloadCommentFiles?letterId={id}&project={ID}  (bulk ZIP)

Usage:
    python fetch_fs_cara.py --projects 65356 43545
    python fetch_fs_cara.py --projects-file cara_projects.txt
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_output_dir, save_metadata, http_get, save_comments, append_comments, load_done_units, mark_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CARA_BASE = "https://cara.fs2c.usda.gov"
CARA_HEADERS = {"User-Agent": "regulations-demo-research research@example.edu"}


def fetch_reading_room_page(project_id: str, page: int = 1, page_size: int = 25) -> list[dict]:
    """Fetch a single page of the reading room table for a project.

    NOTE: CARA's grid uses a fixed 25-per-page Telerik tGrid. Larger sizes
    may be ignored. Table columns: [_, Download, Author, Organization, Date,
    Size, Reading Room Keys]. Rows either link to /Public/Letter/{id}
    (commenter opted in to public letter detail) or /Public/DownloadCommentFile
    (download-only). We capture both.
    """
    url = f"{CARA_BASE}/Public/ReadingRoom"
    params = {
        "Project": project_id,
        "List-size": page_size,
        "List-page": page,
    }
    logger.info("Fetching ReadingRoom project=%s page=%d", project_id, page)
    resp = http_get(url, headers=CARA_HEADERS, params=params)
    soup = BeautifulSoup(resp.text, "html.parser")

    records = []
    current_phase = ""
    # Find the main data table (it's the one with > 20 rows)
    main_table = None
    for t in soup.find_all("table"):
        if len(t.find_all("tr")) > 10:
            main_table = t
            break
    if main_table is None:
        return records

    for tr in main_table.find_all("tr"):
        cells = tr.find_all("td")

        # Phase group header row ("Notice of Availability", etc.)
        if len(cells) == 1:
            text = cells[0].get_text(strip=True)
            if text and text != "":
                current_phase = text
            continue

        if len(cells) < 5:
            continue

        # Either a Letter/{id} link or a DownloadCommentFile link
        detail_link = tr.find("a", href=re.compile(r"/Public/Letter/\d+"))
        download_link = tr.find("a", href=re.compile(r"/Public/DownloadCommentFile", re.IGNORECASE))

        letter_id = None
        detail_url = None
        download_url = None

        if detail_link:
            m = re.search(r"/Public/Letter/(\d+)", detail_link["href"])
            if m:
                letter_id = m.group(1)
                detail_url = CARA_BASE + detail_link["href"]
        if download_link:
            m = re.search(r"letterid=(\d+)", download_link["href"], re.IGNORECASE)
            if m and not letter_id:
                letter_id = m.group(1)
            download_url = CARA_BASE + download_link["href"].replace("&amp;", "&")

        if not letter_id:
            continue

        # Column order: [expand], [download-icon], Author, Organization, Date, Size, Keys
        author = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        org = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        date = cells[4].get_text(strip=True) if len(cells) > 4 else ""
        size = cells[5].get_text(strip=True) if len(cells) > 5 else ""

        records.append({
            "project_id": project_id,
            "letter_id": letter_id,
            "phase": current_phase,
            "author": author,
            "organization": org,
            "date_submitted": date,
            "size": size,
            "detail_url": detail_url or "",
            "download_url": download_url or "",
        })
    return records


def get_total_pages(project_id: str, page_size: int = 25) -> int:
    """Get total page count for a project's reading room.

    CARA's grid is fixed at 25 rows/page. Uses pagination URLs in the HTML
    (they expose all page numbers) or falls back to Total Letters / 25.
    """
    url = f"{CARA_BASE}/Public/ReadingRoom"
    params = {"Project": project_id, "List-size": page_size, "List-page": 1}
    resp = http_get(url, headers=CARA_HEADERS, params=params)
    html = resp.text
    # Pagination links expose all page numbers directly
    pages = re.findall(r"List-page=(\d+)", html)
    if pages:
        return max(int(p) for p in pages)
    m = re.search(r"Total Letters:\s*(\d+)", html)
    if m:
        total = int(m.group(1))
        return (total + page_size - 1) // page_size
    return 1


def fetch_all_letters_for_project(project_id: str, max_pages: int = 10000) -> list[dict]:
    """Fetch all letters for a project across all pages."""
    all_records = []
    total_pages = get_total_pages(project_id)
    logger.info("Project %s has ~%d pages (25/page)", project_id, total_pages)
    consecutive_empty = 0
    for page in range(1, min(total_pages + 1, max_pages + 1)):
        records = fetch_reading_room_page(project_id, page=page)
        if not records:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break
            continue
        consecutive_empty = 0
        all_records.extend(records)
        time.sleep(0.5)  # rate limit
    return all_records


def fetch_letter_detail(letter_id: str, project_id: str) -> dict:
    """Fetch detail page for a single letter (author, org, date, attachments)."""
    url = f"{CARA_BASE}/Public/Letter/{letter_id}"
    params = {"project": project_id}
    resp = http_get(url, headers=CARA_HEADERS, params=params)
    soup = BeautifulSoup(resp.text, "html.parser")

    detail = {"letter_id": letter_id, "project_id": project_id}
    for label_id, field in [
        ("Letter_Sender_FullName", "author"),
        ("Letter_Sender_Organization_Name", "organization"),
        ("Letter_DateSubmitted", "date_submitted"),
    ]:
        label = soup.find("label", {"for": label_id})
        if label:
            # The value is typically in the following sibling element
            val_el = label.find_next_sibling()
            if val_el:
                detail[field] = val_el.get_text(strip=True)

    # Attached files
    attachments = []
    doc_list = soup.find(id="DocumentList")
    if doc_list:
        for a in doc_list.find_all("a", href=re.compile(r"DownloadCommentFile")):
            href = a["href"]
            if not href.startswith("http"):
                href = CARA_BASE + href
            attachments.append(href)
    detail["attachments"] = attachments
    return detail


def normalize(letters: list[dict]) -> pd.DataFrame:
    """Normalize to common comments schema."""
    records = []
    for l in letters:
        # Build attachment URL from download_url, detail_url, or attachments list
        urls = []
        if l.get("download_url"):
            urls.append(l["download_url"])
        if l.get("detail_url"):
            urls.append(l["detail_url"])
        if isinstance(l.get("attachments"), list):
            urls.extend(l["attachments"])
        elif l.get("attachments"):
            urls.append(l["attachments"])
        # Fallback: build from letter_id + project_id directly
        if not urls and l.get("letter_id") and l.get("project_id"):
            urls.append(f"{CARA_BASE}/Public/DownloadCommentFile?letterid={l['letter_id']}&project={l['project_id']}")

        records.append({
            "comment_id": f"cara_{l.get('letter_id')}",
            "docket_id": f"cara_project_{l.get('project_id')}",
            "agency_id": "USDA_FS",
            "submitter_name": l.get("author", ""),
            "submitter_org": l.get("organization", ""),
            "posted_date": l.get("date_submitted", ""),
            "comment_text": "",  # full text is in the PDF attachment
            "attachment_urls": ";".join(urls),
            "raw_metadata": str(l),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", help="Project IDs to scrape")
    parser.add_argument("--projects-file", help="File with one project ID per line")
    parser.add_argument("--fetch-details", action="store_true",
                        help="Also fetch per-letter detail pages (slow)")
    args = parser.parse_args()

    projects = []
    if args.projects:
        projects = args.projects
    elif args.projects_file:
        with open(args.projects_file) as f:
            projects = [line.strip() for line in f if line.strip()]

    if not projects:
        logger.error("Provide --projects or --projects-file")
        logger.info("To find project IDs, visit https://www.fs.usda.gov/r{RR}/{forest}/projects")
        return

    done = load_done_units("fs_cara")
    logger.info("Total projects: %d, already done: %d", len(projects), len(done))

    for project_id in projects:
        if project_id in done:
            logger.info("Skipping already-done project %s", project_id)
            continue
        try:
            records = fetch_all_letters_for_project(project_id)
            logger.info("Project %s: %d letters", project_id, len(records))
        except Exception as e:
            logger.error("Failed on project %s: %s", project_id, e)
            continue

        if args.fetch_details:
            for i, l in enumerate(records):
                try:
                    detail = fetch_letter_detail(l["letter_id"], l["project_id"])
                    l.update(detail)
                    time.sleep(0.3)
                except Exception as e:
                    logger.debug("Letter detail failed %s: %s", l["letter_id"], e)

        if records:
            df = normalize(records)
            append_comments(df, "fs_cara")
        mark_done("fs_cara", project_id)

    save_metadata("fs_cara", {
        "projects": projects,
        "notes": (
            "Forest Service CARA per-letter scrape. Per-project resume via "
            "done_units.txt checkpoint. CARA has no browseable project index — "
            "pass project IDs via --projects or --projects-file."
        ),
    })


if __name__ == "__main__":
    main()
