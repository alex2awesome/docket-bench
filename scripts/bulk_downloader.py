#!/usr/bin/env python3
"""Automate the Regulations.gov bulk download form with Playwright."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from playwright.async_api import Locator, Page, TimeoutError, async_playwright

DOC_TYPES = ("Notice", "Proposed Rule", "Public Submission", "Rule", "Supporting & Related Material")
FORM_URL = "https://www.regulations.gov/bulkdownload"
RECAPTCHA_SITEKEY = "6Lca0dsUAAAAABVCyKnSp7IGvvXCwBIOC-V0ruV9"
CAPSOLVER_ENDPOINT = "https://api.capsolver.com"
CAPSOLVER_BAD_REQUEST_DELAY = 30.0
CAPSOLVER_BAD_REQUEST_MAX_RETRIES = 5
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
AGENCY_LIST_PATH = SCRIPT_DIR / "agency_list.csv"
BLANK_RESULTS_PATH = SCRIPT_DIR / "blank_results.csv"
BLANK_RESULTS_HEADERS = ["agency_name", "year", "file_name"]


def sanitize_doc_type(doc_type: str) -> str:
    clean = doc_type.strip().lower().replace("&", "and")
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    return clean or "unknown"



def _parse_year_input(value: str) -> list[int]:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Year cannot be empty.")
    if "-" in value:
        start_str, end_str = value.split("-", 1)
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid year range format.") from exc
        if start > end:
            raise argparse.ArgumentTypeError("Year range must be ascending (start <= end).")
        return list(range(start, end + 1))
    try:
        year = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Year must be an integer or range (YYYY-YYYY).") from exc
    return [year]


def _normalize_email_list(args: argparse.Namespace) -> list[str]:
    emails = []
    if args.emails:
        emails = [email.strip() for email in args.emails if email.strip()]
    if not emails and args.email:
        emails = [args.email.strip()]
    if not emails:
        raise argparse.ArgumentTypeError("At least one email address must be provided.")
    return emails


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Regulations.gov bulk download requests for all supported document types."
        )
    )
    parser.add_argument(
        "year",
        nargs="?",
        help="4-digit year or inclusive range (e.g., 2020 or 2017-2023) to request.",
    )
    parser.add_argument(
        "agencies",
        nargs="*",
        metavar="agency",
        help="Zero or more agency names/IDs to feed into the Regulations.gov typeahead.",
    )
    parser.add_argument(
        "--agency-folder",
        help="Override the filesystem folder name (defaults to lowercased agency argument).",
    )
    parser.add_argument(
        "--email",
        default="user@example.edu",
        help="Notification email address (defaults to the user's address).",
    )
    parser.add_argument(
        "--doc-types",
        nargs="*",
        choices=DOC_TYPES,
        default=list(DOC_TYPES),
        help="Optional subset of document types to submit.",
    )
    parser.add_argument(
        "--base-url",
        default=FORM_URL,
        help="Override the bulk download form URL (mainly for testing environments).",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Optional Playwright slow motion delay (ms) between actions.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headlessly. The default keeps it visible as requested.",
    )
    parser.add_argument(
        "--browser-channel",
        choices=[
            "chrome",
            "chrome-beta",
            "chrome-dev",
            "chrome-canary",
            "msedge",
            "msedge-beta",
            "msedge-dev",
            "msedge-canary",
        ],
        help="Launch Chromium via a specific installed browser channel (e.g., chrome).",
    )
    parser.add_argument(
        "--user-agent",
        help="Override the browser User-Agent string for each request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Default wait timeout (seconds) while searching for form controls.",
    )
    parser.add_argument(
        "--capsolver-key-file",
        default=str(Path("~/.cap-solver-key.txt").expanduser()),
        help="Path to the CapSolver API key file (default: ~/.cap-solver-key.txt).",
    )
    parser.add_argument(
        "--capsolver-sitekey",
        default=RECAPTCHA_SITEKEY,
        help="Override the reCAPTCHA sitekey if Regulations.gov changes it.",
    )
    parser.add_argument(
        "--capsolver-disable",
        action="store_true",
        help="Skip CapSolver automation and always prompt for manual captcha solving.",
    )
    parser.add_argument(
        "--manual-captcha-disable",
        action="store_true",
        help="Disable manual CAPTCHA prompts; only automated solving will be attempted.",
    )
    parser.add_argument(
        "--emails",
        nargs="+",
        help="Optional list of email addresses to rotate through per submission.",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=1.0,
        help="Minimum random delay (seconds) to wait before typing/filling.",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=3.0,
        help="Maximum random delay (seconds) to wait before typing/filling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--manual-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for manual CAPTCHA confirmation before retrying automation.",
    )
    parser.add_argument(
        "--capsolver-retries",
        type=int,
        default=5,
        help="Number of extra CapSolver attempts to perform after manual timeouts.",
    )
    args = parser.parse_args()
    args.agencies = [agency.strip() for agency in args.agencies if agency.strip()]
    args.years = _parse_year_input(args.year) if args.year else None
    args.emails = _normalize_email_list(args)
    if args.agencies and args.years is None:
        parser.error("A year or year range must be provided when agencies are specified.")
    if args.min_delay < 0 or args.max_delay < 0:
        parser.error("Delay values must be non-negative.")
    if args.min_delay > args.max_delay:
        parser.error("--min-delay cannot exceed --max-delay.")
    return args


@dataclass
class FormFieldGroup:
    """Container describing a logical group of inputs such as the 3-part dates."""

    label: str
    values: Sequence[str]


@dataclass
class AgencyJob:
    display_name: str
    search_term: str
    years: list[int]
    folder: str
    head_department: str | None = None


class BulkDownloader:
    """Orchestrates the multi-step form automation for each document type."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._setup_logging()
        self.capsolver_key = self._load_capsolver_key()
        self._email_list = self.args.emails
        self._email_index = 0
        self._manual_captcha_enabled = not self.args.manual_captcha_disable
        self._auto_agency_mode = not bool(self.args.agencies)
        self._blank_results = self._load_blank_results()
        self._jobs = self._build_agency_jobs()
        if not self._jobs:
            raise ValueError(
                "No agencies to process. Provide agencies on the command line or populate agency_list.csv."
            )

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"\s+", "_", value.strip().lower())
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        return slug or value.lower()

    def _blank_results_key(self, job: AgencyJob, year: int, doc_type: str) -> tuple[str, int, str]:
        return (job.display_name.lower(), year, sanitize_doc_type(doc_type))

    def _build_agency_jobs(self) -> list[AgencyJob]:
        if self.args.agencies:
            if self.args.agency_folder and len(self.args.agencies) != 1:
                raise ValueError("--agency-folder can only be used with a single agency input.")
            jobs: list[AgencyJob] = []
            for agency in self.args.agencies:
                folder = (
                    self.args.agency_folder.strip()
                    if self.args.agency_folder
                    else self._slugify(agency)
                )
                if not folder:
                    raise ValueError("Derived agency folder name cannot be empty.")
                jobs.append(
                    AgencyJob(
                        display_name=agency,
                        search_term=agency,
                        years=list(self.args.years or []),
                        folder=folder,
                        head_department=None,
                    )
                )
            return jobs
        return self._load_agencies_from_csv()

    def _load_blank_results(self) -> set[tuple[str, int, str]]:
        results: set[tuple[str, int, str]] = set()
        if not BLANK_RESULTS_PATH.exists():
            return results
        with BLANK_RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                agency = (row.get("agency_name") or "").strip().lower()
                year_str = (row.get("year") or "").strip()
                file_name = (row.get("file_name") or "").strip()
                if not agency or not year_str or not file_name:
                    continue
                try:
                    year = int(year_str)
                except ValueError:
                    continue
                results.add((agency, year, sanitize_doc_type(file_name.replace(".csv", ""))))
        return results

    def _record_blank_result(self, job: AgencyJob, year: int, doc_type: str) -> None:
        key = self._blank_results_key(job, year, doc_type)
        if key in self._blank_results:
            return
        self._blank_results.add(key)
        if not BLANK_RESULTS_PATH.exists():
            with BLANK_RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=BLANK_RESULTS_HEADERS)
                writer.writeheader()
        with BLANK_RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BLANK_RESULTS_HEADERS)
            writer.writerow(
                {
                    "agency_name": job.display_name,
                    "year": year,
                    "file_name": f"{sanitize_doc_type(doc_type)}.csv",
                }
            )

    def _load_agencies_from_csv(self) -> list[AgencyJob]:
        if not AGENCY_LIST_PATH.exists():
            logging.error(
                "Agency list file %s not found; unable to load agencies.", AGENCY_LIST_PATH
            )
            return []
        jobs: list[AgencyJob] = []
        with AGENCY_LIST_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("agency name") or "").strip()
                if not name:
                    continue
                search_term = (row.get("agency search term") or name).strip() or name
                years_field = (row.get("years") or "").strip()
                if self.args.years:
                    years = list(self.args.years)
                elif years_field:
                    try:
                        years = _parse_year_input(years_field)
                    except argparse.ArgumentTypeError as exc:  # noqa: PERF203
                        logging.warning(
                            "Skipping '%s' due to invalid years '%s': %s", name, years_field, exc
                        )
                        continue
                else:
                    logging.warning(
                        "Skipping '%s' because no years specified in CSV and no CLI year provided.",
                        name,
                    )
                    continue
                folder = self._slugify(name)
                head_department = (row.get("head department") or "").strip()
                jobs.append(
                    AgencyJob(
                        display_name=name,
                        search_term=search_term,
                        years=years,
                        folder=folder,
                        head_department=head_department or None,
                    )
                )
        return jobs

    def _load_capsolver_key(self) -> str | None:
        if self.args.capsolver_disable:
            return None
        key_path = Path(self.args.capsolver_key_file).expanduser()
        if not key_path.exists():
            logging.info(
                "CapSolver key file not found at %s; continuing with manual CAPTCHA solving.",
                key_path,
            )
            return None
        try:
            content = key_path.read_text(encoding="utf-8").strip()
            if not content:
                logging.warning("CapSolver key file %s is empty.", key_path)
                return None
            logging.info("Loaded CapSolver API key from %s.", key_path)
            return content
        except Exception as exc:
            logging.error("Unable to read CapSolver key file %s: %s", key_path, exc)
            return None

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=self.args.log_level.upper(),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    async def run(self) -> None:
        async with async_playwright() as playwright:
            launch_kwargs = {"headless": self.args.headless, "slow_mo": self.args.slow_mo}
            if self.args.browser_channel:
                launch_kwargs["channel"] = self.args.browser_channel
            browser = await playwright.chromium.launch(**launch_kwargs)
            context_kwargs = {"viewport": {"width": 1400, "height": 900}}
            if self.args.user_agent:
                context_kwargs["user_agent"] = self.args.user_agent
            context = await browser.new_context(**context_kwargs)
            context.set_default_timeout(self.args.timeout * 1000)
            page = await context.new_page()
            try:
                for job in self._jobs:
                    logging.info("Processing agency '%s' using search term '%s'.", job.display_name, job.search_term)
                    for year in job.years:
                        for doc_type in self.args.doc_types:
                            await self._submit_for_doc_type(page, doc_type, year, job)
            finally:
                await page.close()
                await context.close()
                await browser.close()

    async def _submit_for_doc_type(
        self, page: Page, doc_type: str, year: int, job: AgencyJob
    ) -> None:
        blank_key = self._blank_results_key(job, year, doc_type)
        if self._auto_agency_mode and blank_key in self._blank_results:
            logging.info(
                "Skipping '%s' (%s) for %s; recorded previously as no data.",
                doc_type,
                year,
                job.display_name,
            )
            return
        if self._result_exists(job, year, doc_type):
            logging.info(
                "Skipping '%s' (%s) for %s; result already exists on disk.",
                doc_type,
                year,
                job.display_name,
            )
            return
        attempt = 1
        while True:
            logging.info(
                "Preparing submission for '%s' (%s) [%s attempt %s].",
                doc_type,
                year,
                job.display_name,
                attempt,
            )
            await page.goto(self.args.base_url, wait_until="networkidle")
            await self._select_agency_mode(page)
            await self._fill_agency(page, job.search_term)
            await self._select_document_type(page, doc_type)
            await self._fill_date_ranges(
                page,
                [
                    FormFieldGroup("Posted Start Date", ("01", "01", str(year))),
                    FormFieldGroup("Posted End Date", ("12", "31", str(year))),
                ],
            )
            await self._fill_email(page, self._next_email())
            captcha_ready = await self._prompt_for_recaptcha(page, doc_type)
            if not captcha_ready:
                logging.warning(
                    "CAPTCHA unresolved for '%s' (%s, %s); reloading the form before retrying.",
                    doc_type,
                    year,
                    job.display_name,
                )
                await page.goto("about:blank")
                attempt += 1
                continue
            await self._submit_request(page, doc_type)
            if await self._no_results_message_present(page):
                logging.info(
                    "No data available for '%s' (%s, %s); moving on to the next request.",
                    doc_type,
                    year,
                    job.display_name,
                )
                self._record_blank_result(job, year, doc_type)
                await page.goto("about:blank")
                return
            if not await self._service_error_present(page):
                await page.goto("about:blank")
                return
            await self._wait_for_service_recovery(page, doc_type, year)
            attempt += 1

    def _result_exists(self, job: AgencyJob, year: int, doc_type: str) -> bool:
        year_dir = f"{job.folder}_{year}_{year + 1}"
        target_dir = BASE_DIR / job.folder / year_dir
        filename = f"{sanitize_doc_type(doc_type)}.csv.gz"
        candidate = target_dir / filename
        return candidate.exists()

    async def _select_agency_mode(self, page: Page) -> None:
        agency_radio = page.locator("input#agency")
        await agency_radio.wait_for(state="attached")
        if not await agency_radio.is_checked():
            await agency_radio.check()
        logging.debug("Agency filter mode selected.")

    async def _fill_agency(self, page: Page, agency: str) -> None:
        input_box = page.locator("input[name='agencyId']")
        await input_box.wait_for(state="visible")
        await self._prepare_field(input_box)
        await input_box.fill("")
        await self._human_pause()
        typed_value = agency.strip()
        if " " not in typed_value and typed_value.isupper() and len(typed_value) <= 4:
            typed_value = f"{typed_value} "
        logging.debug("Typing agency query: %s", typed_value)
        type_delay = random.randint(60, 110)
        await input_box.type(typed_value, delay=type_delay)
        await self._human_pause()
        suggestion = page.locator(".tt-menu .tt-suggestion").first
        try:
            await suggestion.wait_for(state="visible", timeout=5000)
            selection_text = (await suggestion.inner_text()).strip()
            logging.info("Picking agency suggestion: %s", selection_text)
            await self._human_pause()
            await input_box.press("ArrowDown")
            await self._human_pause()
            await input_box.press("Enter")
            await self._human_pause()
        except TimeoutError:
            logging.warning("No typeahead suggestions surfaced; submitting the typed value.")
            await self._human_pause()
            await input_box.press("Enter")
            await self._human_pause()

    async def _select_document_type(self, page: Page, document_type: str) -> None:
        logging.debug("Selecting document type %s", document_type)
        select_box = page.locator("select[name='documentType']")
        if await select_box.count():
            await select_box.select_option(label=document_type)
            return
        combobox = page.get_by_role("combobox", name=re.compile("Document Type", re.I))
        if await combobox.count():
            await combobox.first.click()
            await self._click_dropdown_option(page, document_type)
            return
        dropdown_button = page.get_by_role("button", name=re.compile("Document Type", re.I))
        if not await dropdown_button.count():
            dropdown_button = page.locator(
                "div:has(:text('Document Type')) button, "
                "div:has(label:has-text('Document Type')) .dropdown-toggle"
            )
        if await dropdown_button.count():
            await dropdown_button.first.click()
            await self._click_dropdown_option(page, document_type)
            return
        raise RuntimeError("Unable to locate the Document Type dropdown control.")

    async def _click_dropdown_option(self, page: Page, document_type: str) -> None:
        option = page.get_by_role("option", name=re.compile(document_type, re.I))
        if await option.count():
            await option.first.click()
            return
        generic_option = page.locator(f"text={document_type}").first
        await generic_option.wait_for(state="visible")
        await generic_option.click()

    async def _fill_date_ranges(self, page: Page, field_groups: Iterable[FormFieldGroup]) -> None:
        for group in field_groups:
            filled = await self._fill_three_part_date(page, group.label, group.values)
            if not filled:
                raise RuntimeError(f"Unable to locate date inputs for '{group.label}'.")

    async def _fill_three_part_date(
        self, page: Page, label_text: str, values: Sequence[str]
    ) -> bool:
        logging.debug("Filling %s with %s", label_text, "-".join(values))
        normalized = self._normalize_label(label_text)
        camel = normalized[:1].lower() + normalized[1:]
        # Try single date inputs first (Regulations.gov uses <input type="date"> controls).
        for selector in self._single_date_selectors(label_text, normalized, camel):
            locator = page.locator(selector).first
            if await locator.count():
                await self._fill_single_date_input(locator, values)
                return True
        # Next, see if the label's "for" attribute points directly to the input.
        label = page.locator(f"label:has-text('{label_text}')").first
        if await label.count():
            target_id = await label.get_attribute("for")
            if target_id:
                target_locator = page.locator(f"#{target_id}")
                if await target_locator.count():
                    await self._fill_single_date_input(target_locator.first, values)
                    return True
        selector_sets = []
        for base in (normalized, camel):
            selector_sets.extend(
                [
                    [
                        f"input[name='{base}Month']",
                        f"input[name='{base}Day']",
                        f"input[name='{base}Year']",
                    ],
                    [
                        f"input[name*='{base}'][name*='Month']",
                        f"input[name*='{base}'][name*='Day']",
                        f"input[name*='{base}'][name*='Year']",
                    ],
                    [
                        f"input[id*='{base}'][id*='Month']",
                        f"input[id*='{base}'][id*='Day']",
                        f"input[id*='{base}'][id*='Year']",
                    ],
                    [
                        f"input[id*='{base}'][aria-label*='Month' i]",
                        f"input[id*='{base}'][aria-label*='Day' i]",
                        f"input[id*='{base}'][aria-label*='Year' i]",
                    ],
                ]
            )
        selector_sets.append(
            [
                f"input[aria-label*='{label_text}' i][aria-label*='Month' i]",
                f"input[aria-label*='{label_text}' i][aria-label*='Day' i]",
                f"input[aria-label*='{label_text}' i][aria-label*='Year' i]",
            ]
        )
        for selectors in selector_sets:
            locators = [page.locator(sel).first for sel in selectors]
            if all([await locator.count() for locator in locators]):
                for locator, value in zip(locators, values):
                    await self._fill_with_pause(locator, value)
                return True
        # Fallback: locate inputs near the label text (works when fields share a parent row).
        label_locator = page.get_by_text(label_text, exact=False)
        if await label_locator.count():
            anchor = label_locator.first
            container = anchor.locator(
                "xpath=ancestor::div[contains(@class,'row-form-bulkdownload')][1]"
            )
            if not await container.count():
                container = anchor.locator("xpath=ancestor::div[1]")
            inputs = container.locator(
                "input:not([type='radio']):not([type='checkbox']):not([type='hidden'])"
            )
            visible_inputs = []
            count_inputs = await inputs.count()
            for idx in range(count_inputs):
                candidate = inputs.nth(idx)
                if await candidate.is_enabled():
                    visible_inputs.append(candidate)
            if len(visible_inputs) == 1:
                await self._fill_single_date_input(visible_inputs[0], values)
                return True
            if len(visible_inputs) >= len(values):
                for idx, value in enumerate(values):
                    input_box = visible_inputs[idx]
                    await self._fill_with_pause(input_box, value)
                return True
        # Final fallback: global order of split date inputs (first 3 -> start, next 3 -> end).
        date_inputs = page.locator("input[type='date'], input[placeholder*='MM/DD' i]")
        count_dates = await date_inputs.count()
        if count_dates:
            if "Start" in label_text or count_dates == 1:
                target = date_inputs.nth(0)
            else:
                target = date_inputs.nth(min(1, count_dates - 1))
            await self._fill_single_date_input(target, values)
            return True
        splitted_inputs = page.locator(
            "input[placeholder*='MM' i], input[placeholder*='DD' i], input[placeholder*='YYYY' i]"
        )
        count = await splitted_inputs.count()
        if count >= len(values):
            start_idx = 0 if "Start" in label_text else len(values)
            for offset, value in enumerate(values):
                idx = start_idx + offset
                if idx >= count:
                    break
                input_box = splitted_inputs.nth(idx)
                await self._fill_with_pause(input_box, value)
            return True
        return False

    def _normalize_label(self, label_text: str) -> str:
        return re.sub(r"\s+", "", label_text.strip())

    def _single_date_selectors(
        self, label_text: str, normalized: str, camel: str
    ) -> Sequence[str]:
        label_lower = label_text.lower()
        selectors = [
            f"input[name='{normalized}']",
            f"input[name='{camel}']",
            f"input[id='{normalized}']",
            f"input[id='{camel}']",
            f"input[name*='{normalized}' i]",
            f"input[name*='{camel}' i]",
            f"input[id*='{normalized}' i]",
            f"input[id*='{camel}' i]",
        ]
        if "start" in label_lower:
            selectors.extend(
                [
                    "input[name='postedDateFrom']",
                    "input[id='postedDateFrom']",
                    "input[name*='DateFrom' i]",
                    "input[name*='Start' i]",
                    "input[id*='DateFrom' i]",
                    "input[id*='Start' i]",
                ]
            )
        if "end" in label_lower:
            selectors.extend(
                [
                    "input[name='postedDateTo']",
                    "input[id='postedDateTo']",
                    "input[name*='DateTo' i]",
                    "input[name*='End' i]",
                    "input[id*='DateTo' i]",
                    "input[id*='End' i]",
                ]
            )
        # Remove duplicates while preserving order.
        return list(dict.fromkeys(selectors))

    async def _human_pause(self, multiplier: float = 1.0) -> None:
        min_delay = self.args.min_delay * multiplier
        max_delay = self.args.max_delay * multiplier
        if max_delay <= 0:
            return
        delay = random.uniform(min_delay, max_delay)
        if delay > 0:
            await asyncio.sleep(delay)

    async def _mouse_jitter(self, locator: Locator) -> None:
        try:
            await locator.hover()
        except Exception:
            return
        try:
            handle = await locator.element_handle()
            if not handle:
                return
            box = await handle.bounding_box()
            if not box:
                return
            width = max(box["width"], 4.0)
            height = max(box["height"], 4.0)
            for _ in range(random.randint(1, 2)):
                offset_x = random.uniform(1.0, max(width - 1.0, 1.0))
                offset_y = random.uniform(1.0, max(height - 1.0, 1.0))
                await handle.hover(position={"x": offset_x, "y": offset_y})
                await asyncio.sleep(random.uniform(0.05, 0.15))
        except Exception:
            return

    async def _prepare_field(self, locator: Locator, click: bool = True) -> None:
        await locator.scroll_into_view_if_needed()
        await self._mouse_jitter(locator)
        if click:
            await locator.click()
        await self._human_pause()

    async def _fill_with_pause(self, locator: Locator, value: str, click: bool = True) -> None:
        await self._prepare_field(locator, click=click)
        await locator.fill(value)
        await self._human_pause()

    async def _fill_single_date_input(self, locator: Locator, values: Sequence[str]) -> None:
        month, day, year = values
        input_type = await locator.get_attribute("type")
        value = self._format_single_input_date(month, day, year, input_type)
        await self._fill_with_pause(locator, value)

    def _format_single_input_date(
        self, month: str, day: str, year: str, input_type: str | None
    ) -> str:
        month = month.zfill(2)
        day = day.zfill(2)
        if (input_type or "").lower() == "date":
            return f"{year}-{month}-{day}"
        return f"{month}/{day}/{year}"

    async def _fill_email(self, page: Page, email: str) -> None:
        email_input = page.locator("input[name='emailAddress']")
        await email_input.wait_for(state="visible")
        await self._fill_with_pause(email_input, email)
        logging.debug("Email populated with %s", email)

    def _next_email(self) -> str:
        email = self._email_list[self._email_index]
        self._email_index = (self._email_index + 1) % len(self._email_list)
        return email

    async def _prompt_for_recaptcha(self, page: Page, doc_type: str) -> bool:
        if not self.capsolver_key:
            if not self._manual_captcha_enabled:
                logging.error(
                    "Manual CAPTCHA solving disabled and CapSolver unavailable for '%s'.",
                    doc_type,
                )
                return False
            manual_ack = await self._await_manual_captcha(
                page, doc_type, timeout=self.args.manual_timeout
            )
            if not manual_ack:
                logging.error(
                    "Manual CAPTCHA confirmation not received for '%s'; aborting submission.",
                    doc_type,
                )
                return False
            return True
        max_attempts = 1 + max(0, self.args.capsolver_retries)
        manual_prompt_remaining = self._manual_captcha_enabled
        for attempt in range(1, max_attempts + 1):
            solved = await self._solve_recaptcha_with_capsolver(page, doc_type)
            if solved:
                return True
            logging.warning(
                "CapSolver attempt %s/%s failed for '%s'.", attempt, max_attempts, doc_type
            )
            if manual_prompt_remaining:
                manual_ack = await self._await_manual_captcha(
                    page, doc_type, timeout=self.args.manual_timeout
                )
                manual_prompt_remaining = False
                if manual_ack:
                    logging.info(
                        "Manual CAPTCHA confirmation received for '%s'; continuing submission.",
                        doc_type,
                    )
                    return True
                logging.info(
                    "Manual CAPTCHA confirmation not received for '%s'; will keep retrying CapSolver "
                    "without additional prompts.",
                    doc_type,
                )
            else:
                logging.debug(
                    "Skipping manual CAPTCHA prompt for '%s' due to prior timeout.", doc_type
                )
        logging.error(
            "CAPTCHA could not be solved automatically and no manual confirmation was received "
            "for '%s'.",
            doc_type,
        )
        return False

    async def _await_manual_captcha(
        self, page: Page, doc_type: str, timeout: float | None = None
    ) -> bool:
        recaptcha_frame = page.locator("#bulkdownload-recaptcha iframe").first
        if await recaptcha_frame.count():
            await recaptcha_frame.scroll_into_view_if_needed()
        prompt = (
            f"Please solve the reCAPTCHA for '{doc_type}' in the browser window, "
            "then press Enter here to continue."
        )
        try:
            if timeout and timeout > 0:
                await asyncio.wait_for(asyncio.to_thread(input, prompt + "\n>> "), timeout=timeout)
            else:
                await asyncio.to_thread(input, prompt + "\n>> ")
            return True
        except asyncio.TimeoutError:
            logging.info(
                "Manual CAPTCHA confirmation not received for '%s' within %.0fs.",
                doc_type,
                timeout or 0,
            )
            return False

    async def _submit_request(self, page: Page, doc_type: str) -> None:
        submit_button = page.get_by_role("button", name=re.compile("^Submit$", re.I))
        if not await submit_button.count():
            submit_button = page.locator("button.btn.btn-primary:has-text('Submit')")
        await submit_button.wait_for(state="visible")
        handle = await submit_button.element_handle()
        if handle:
            try:
                await page.wait_for_function("el => !el.disabled", arg=handle)
            except TimeoutError:
                logging.warning("Submit button did not enable automatically; forcing enable.")
                await page.evaluate(
                    "(el) => { if (el) { el.disabled = false; el.removeAttribute('disabled'); } }",
                    handle,
                )
        await self._mouse_jitter(submit_button)
        await self._human_pause()
        await submit_button.click()
        await self._human_pause(multiplier=1.5)
        logging.info("Submitted request for '%s'. Waiting for confirmation...", doc_type)
        try:
            confirmation = page.get_by_text("submitted", exact=False)
            await confirmation.wait_for(timeout=10000)
        except TimeoutError:
            logging.warning("Confirmation message not detected; check the browser for details.")

    async def _ensure_submit_enabled(self, page: Page) -> None:
        await page.evaluate(
            """() => {
                const button = document.querySelector("button.btn.btn-primary[type='submit']");
                if (button) {
                    button.disabled = false;
                    button.removeAttribute('disabled');
                }
            }"""
        )

    async def _service_error_present(self, page: Page) -> bool:
        error_text = page.get_by_text("We're sorry, an error has occurred", exact=False)
        try:
            await error_text.wait_for(timeout=1000)
            logging.error("Regulations.gov reported a service error after submission.")
            return True
        except TimeoutError:
            return False

    async def _no_results_message_present(self, page: Page) -> bool:
        empty_text = page.get_by_text(
            "The selected combination of bulk data download filters didn’t produce any results.",
            exact=False,
        )
        try:
            await empty_text.wait_for(timeout=1000)
            return True
        except TimeoutError:
            return False

    async def _wait_for_service_recovery(self, page: Page, doc_type: str, year: int) -> None:
        logging.warning(
            "Service error detected for '%s' (%s). Waiting 180 seconds before retrying...",
            doc_type,
            year,
        )
        while True:
            await asyncio.sleep(180)
            try:
                await page.goto(self.args.base_url, wait_until="networkidle")
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Failed to reload Regulations.gov: %s. Retrying in 180 seconds.", exc
                )
                continue
            if not await self._service_error_present(page):
                logging.info("Service recovered. Retrying '%s' (%s).", doc_type, year)
                return
            logging.warning(
                "Service still down for '%s' (%s). Will check again in 180 seconds.",
                doc_type,
                year,
            )

    async def _solve_recaptcha_with_capsolver(self, page: Page, doc_type: str) -> bool:
        logging.info("Attempting to solve reCAPTCHA for '%s' via CapSolver.", doc_type)
        create_payload = {
            "clientKey": self.capsolver_key,
            "task": {
                "type": "NoCaptchaTaskProxyless",
                "websiteURL": self.args.base_url,
                "websiteKey": self.args.capsolver_sitekey,
            },
        }
        response = await self._capsolver_request("createTask", create_payload)
        if not response or "taskId" not in response:
            logging.error("CapSolver createTask failed: %s", response)
            return False
        task_id = response["taskId"]
        logging.debug("CapSolver task created: %s", task_id)
        start = time.time()
        while True:
            await asyncio.sleep(3.0)
            result = await self._capsolver_request(
                "getTaskResult", {"clientKey": self.capsolver_key, "taskId": task_id}
            )
            if not result:
                logging.warning("CapSolver returned empty response while polling.")
                continue
            status = result.get("status")
            if status == "ready":
                solution = result.get("solution", {})
                token = solution.get("gRecaptchaResponse") or solution.get("response")
                if not token:
                    logging.error("CapSolver did not provide a gRecaptchaResponse.")
                    return False
                await self._inject_recaptcha_token(page, token)
                elapsed = time.time() - start
                logging.info("CapSolver solved CAPTCHA in %.1fs.", elapsed)
                return True
            if status == "failed":
                logging.error("CapSolver reported failure: %s", result)
                return False
            logging.debug("CapSolver status: %s. Waiting to poll again...", status)

    async def _inject_recaptcha_token(self, page: Page, token: str) -> None:
        logging.debug("Injecting reCAPTCHA token into the page.")
        await page.evaluate(
            """(token) => {
                const fields = document.querySelectorAll('textarea[name="g-recaptcha-response"], #g-recaptcha-response');
                fields.forEach(el => {
                    el.value = token;
                    el.innerHTML = token;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                });
                window.___lastSolvedRecaptchaToken = token;
                if (window.grecaptcha && window.grecaptcha.get) {
                    try {
                        const clients = window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients;
                        if (clients) {
                            Object.values(clients).forEach(client => {
                                const cb = client && client.callback;
                                if (typeof cb === 'function') {
                                    cb(token);
                                }
                                const l = client && client.l;
                                if (l && typeof l.callback === 'function') {
                                    l.callback(token);
                                }
                            });
                        }
                    } catch (err) {
                        console.warn('Unable to trigger grecaptcha callback', err);
                    }
                }
            }""",
            token,
        )
        await self._ensure_submit_enabled(page)

    async def _capsolver_request(self, endpoint: str, payload: dict) -> dict | None:
        url = f"{CAPSOLVER_ENDPOINT}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await asyncio.to_thread(_read_urlopen_json, request)
                return response
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and endpoint == "getTaskResult":
                    if attempt >= CAPSOLVER_BAD_REQUEST_MAX_RETRIES:
                        logging.error(
                            "CapSolver request to %s hit HTTP 400 too many times; giving up after %s attempts.",
                            endpoint,
                            CAPSOLVER_BAD_REQUEST_MAX_RETRIES,
                        )
                        break
                    logging.warning(
                        "CapSolver request to %s returned HTTP 400 (attempt %s/%s); sleeping %.0fs before retrying.",
                        endpoint,
                        attempt,
                        CAPSOLVER_BAD_REQUEST_MAX_RETRIES,
                        CAPSOLVER_BAD_REQUEST_DELAY,
                    )
                    await asyncio.sleep(CAPSOLVER_BAD_REQUEST_DELAY)
                    continue
                logging.error("CapSolver request to %s failed: %s", endpoint, exc)
            except urllib.error.URLError as exc:
                logging.error("CapSolver request to %s failed: %s", endpoint, exc)
            except Exception as exc:
                logging.error("Unexpected CapSolver error: %s", exc)
            break
        return None


def _read_urlopen_json(request: urllib.request.Request) -> dict:
    with urllib.request.urlopen(request, timeout=60) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data or "{}")


def main() -> None:
    args = parse_args()
    downloader = BulkDownloader(args)
    try:
        asyncio.run(downloader.run())
    except KeyboardInterrupt:
        logging.warning("Bulk downloader interrupted by user.")


if __name__ == "__main__":
    main()
def sanitize_doc_type(doc_type: str) -> str:
    clean = doc_type.strip().lower().replace("&", "and")
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    return clean or "unknown"
