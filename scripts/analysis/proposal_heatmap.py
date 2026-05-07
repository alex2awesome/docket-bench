"""Generate proposal heatmap HTML showing comment engagement on deontic units.

Fuzzy-matches extracted deontic units back to positions in proposal text,
then renders an HTML heatmap where color intensity = comment engagement.

Usage:
    python proposal_heatmap.py                          # all qualifying dockets
    python proposal_heatmap.py --docket EPA-HQ-OW-2017-0300  # specific docket
    python proposal_heatmap.py --max-text-len 30000     # only short proposals
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import html as html_lib
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BULK_DIR = SCRIPTS_DIR.parent
DEONTIC_DIR = BULK_DIR / "deontic_units"
OUTPUT_DIR = SCRIPTS_DIR / "data" / "proposal_heatmaps"


def load_analysis_units() -> dict[str, list[dict]]:
    """Load deontic units with engagement metrics, grouped by docket_id."""
    path = DEONTIC_DIR / "analysis_dataset_audited_v3.jsonl"
    docket_units = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            docket_units[d["docket_id"]].append(d)
    return dict(docket_units)


def load_proposal_text(docket_id: str) -> tuple[str, str] | None:
    """Find and load proposal text for a docket. Returns (doc_id, text) or None."""
    for agency_dir in sorted(BULK_DIR.iterdir()):
        if not agency_dir.is_dir() or agency_dir.name in ("scripts", "old", "data", "deontic_units") \
                or agency_dir.name.startswith("_"):
            continue
        for yd in sorted(agency_dir.iterdir()):
            if not yd.is_dir():
                continue
            for ext in [".csv.gz", ".csv"]:
                fp = yd / f"proposed_rule_all_text{ext}"
                if fp.exists():
                    try:
                        df = pd.read_csv(fp, low_memory=False)
                        matches = df[df["Docket ID"] == docket_id]
                        if len(matches) > 0:
                            # Pick the one with longest canonical_text
                            best = None
                            best_len = 0
                            for _, row in matches.iterrows():
                                txt = str(row.get("canonical_text", ""))
                                if len(txt) > best_len:
                                    best = (row["Document ID"], txt)
                                    best_len = len(txt)
                            if best and best_len > 100:
                                return best
                    except Exception:
                        pass
    return None


def clean_proposal_text(text: str) -> str:
    """Remove page headers/boilerplate from proposal text."""
    # Remove <<COMMENT N>> and <<PAGE N>> markers
    text = re.sub(r"<<COMMENT \d+>>", "", text)
    text = re.sub(r"<<PAGE \d+>>", "", text)
    # Remove Federal Register header lines (page numbers, date lines)
    text = re.sub(r"\n\d{4,5}\s*\n", "\n", text)
    text = re.sub(r"Federal Register / Vol\. \d+.*?/ (Proposed|Final) Rules?\s*\n", "", text)
    return text.strip()


def find_unit_spans(proposal_text: str, units: list[dict]) -> list[dict]:
    """Find character spans where each unit text appears in the proposal.

    Uses substring matching first, then falls back to fuzzy matching.
    Returns list of dicts with 'start', 'end', 'unit' keys.
    """
    spans = []
    text_lower = proposal_text.lower()

    for unit in units:
        unit_text = unit["unit_text"].strip()
        if not unit_text or len(unit_text) < 10:
            continue

        # Try exact substring match first
        ut_lower = unit_text.lower()
        idx = text_lower.find(ut_lower)
        if idx >= 0:
            spans.append({
                "start": idx,
                "end": idx + len(unit_text),
                "unit": unit,
                "match_type": "exact",
            })
            continue

        # Try matching first 60 chars (units may be truncated or slightly edited)
        prefix = ut_lower[:min(60, len(ut_lower))]
        idx = text_lower.find(prefix)
        if idx >= 0:
            # Extend to full unit length
            end = min(idx + len(unit_text), len(proposal_text))
            spans.append({
                "start": idx,
                "end": end,
                "unit": unit,
                "match_type": "prefix",
            })
            continue

        # Try matching by section reference
        section = unit.get("unit_section", "")
        if section:
            # Normalize section for search
            sec_patterns = [
                section,
                section.replace("§ ", "§"),
                section.replace("§ ", ""),
                re.sub(r"\s+", " ", section),
            ]
            for pat in sec_patterns:
                idx = text_lower.find(pat.lower())
                if idx >= 0:
                    # Found the section reference; search nearby for unit content
                    # Look in a window around the section reference
                    window_start = max(0, idx - 200)
                    window_end = min(len(proposal_text), idx + 2000)
                    window = text_lower[window_start:window_end]

                    # Try to find key phrases from unit text
                    words = [w for w in ut_lower.split() if len(w) > 4][:5]
                    if words:
                        best_pos = None
                        for w in words:
                            wpos = window.find(w)
                            if wpos >= 0:
                                abs_pos = window_start + wpos
                                if best_pos is None or abs_pos < best_pos:
                                    best_pos = abs_pos
                        if best_pos is not None:
                            spans.append({
                                "start": best_pos,
                                "end": min(best_pos + len(unit_text), len(proposal_text)),
                                "unit": unit,
                                "match_type": "section_fuzzy",
                            })
                            continue

        # Last resort: sliding window fuzzy match on key phrases
        key_phrase = ut_lower[:40]
        best_ratio = 0
        best_start = -1
        step = 20
        for i in range(0, len(text_lower) - len(key_phrase), step):
            candidate = text_lower[i:i+len(key_phrase)]
            ratio = SequenceMatcher(None, key_phrase, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio >= 0.7 and best_start >= 0:
            spans.append({
                "start": best_start,
                "end": min(best_start + len(unit_text), len(proposal_text)),
                "unit": unit,
                "match_type": "fuzzy",
            })

    # Sort by start position
    spans.sort(key=lambda x: x["start"])
    return spans


def generate_heatmap_html(
    docket_id: str,
    proposal_text: str,
    spans: list[dict],
    units: list[dict],
) -> str:
    """Generate an HTML heatmap of the proposal with comment engagement overlay."""

    # Compute max claims for color scaling
    max_claims = max((s["unit"]["n_total_claims"] for s in spans), default=1)
    if max_claims == 0:
        max_claims = 1

    # Build character-level engagement map
    char_claims = [0] * len(proposal_text)
    char_unit_info = [None] * len(proposal_text)

    for span in spans:
        claims = span["unit"]["n_total_claims"]
        for i in range(span["start"], min(span["end"], len(proposal_text))):
            if claims > char_claims[i]:
                char_claims[i] = claims
                char_unit_info[i] = span["unit"]

    # Build HTML segments
    segments = []
    i = 0
    while i < len(proposal_text):
        claims = char_claims[i]
        unit_info = char_unit_info[i]

        # Find contiguous run of same claims level
        j = i + 1
        while j < len(proposal_text) and char_claims[j] == claims and char_unit_info[j] == unit_info:
            j += 1

        text_chunk = proposal_text[i:j]
        escaped = html_lib.escape(text_chunk)

        if claims > 0 and unit_info:
            # Heatmap coloring: intensity based on claim count
            intensity = min(claims / max_claims, 1.0)
            # Red channel: 255, Green/Blue decrease with intensity
            r = 255
            g = int(255 - intensity * 200)
            b = int(255 - intensity * 220)
            alpha = 0.15 + intensity * 0.55

            section = unit_info.get("unit_section", "")
            n_claims = unit_info["n_total_claims"]
            n_clusters = unit_info.get("n_total_clusters", 0)
            survival = unit_info.get("survival_status", "")
            unit_type = unit_info.get("unit_type", "")
            summary = html_lib.escape(unit_info.get("unit_summary", "")[:200])

            tooltip = f"{section} | {n_claims} claims ({n_clusters} clusters) | {unit_type} | {survival}"
            if summary:
                tooltip += f"\n{summary}"

            # Border styling for survival status
            border = ""
            if survival == "removed":
                border = "border-bottom: 2px solid #c62828;"
            elif survival == "modified":
                border = "border-bottom: 2px solid #f57c00;"
            elif survival == "survived":
                border = ""

            segments.append(
                f'<span class="unit" style="background: rgba({r},{g},{b},{alpha:.2f}); {border}" '
                f'title="{html_lib.escape(tooltip)}" '
                f'data-claims="{n_claims}" data-section="{html_lib.escape(section)}">'
                f'{escaped}</span>'
            )
        else:
            segments.append(f'<span class="plain">{escaped}</span>')

        i = j

    body = "".join(segments)

    # Stats
    total_units = len(units)
    matched_units = len(spans)
    engaged_units = sum(1 for s in spans if s["unit"]["n_total_claims"] > 0)
    total_claims = sum(u["n_total_claims"] for u in units)

    # Legend entries
    legend_items = []
    thresholds = [1, 5, 20, 50, 100, 500]
    for t in thresholds:
        if t <= max_claims:
            intensity = min(t / max_claims, 1.0)
            r, g, b = 255, int(255 - intensity * 200), int(255 - intensity * 220)
            alpha = 0.15 + intensity * 0.55
            legend_items.append(
                f'<span style="background: rgba({r},{g},{b},{alpha:.2f}); padding: 2px 8px;">{t}+</span>'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Proposal Heatmap: {html_lib.escape(docket_id)}</title>
<style>
  body {{
    font-family: 'Times New Roman', Times, serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.6;
    font-size: 11px;
    color: #222;
  }}
  h1 {{ font-size: 18px; margin-bottom: 4px; }}
  .stats {{ font-size: 12px; color: #555; margin-bottom: 12px; }}
  .legend {{
    font-size: 11px; margin-bottom: 16px; padding: 8px;
    background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }}
  .legend b {{ margin-right: 4px; }}
  .proposal-text {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 10px;
    line-height: 1.5;
    border: 1px solid #ccc;
    padding: 12px;
    background: #fff;
  }}
  .unit {{
    cursor: help;
    border-radius: 1px;
  }}
  .unit:hover {{
    outline: 1px solid rgba(200,0,0,0.5);
  }}
  .plain {{ color: #444; }}
  .survival-legend {{
    font-size: 11px; margin-top: 4px;
  }}
  .survival-legend span {{
    margin-right: 12px;
  }}
</style>
</head>
<body>
<h1>Proposal Heatmap: {html_lib.escape(docket_id)}</h1>
<div class="stats">
  {matched_units}/{total_units} units located in text |
  {engaged_units} units with comments |
  {total_claims} total comment claims |
  Max claims on single unit: {max_claims}
</div>
<div class="legend">
  <b>Claims:</b>
  <span style="background: rgba(255,255,255,0.15); padding: 2px 8px; border: 1px solid #ddd;">0</span>
  {"".join(legend_items)}
</div>
<div class="survival-legend">
  <span style="border-bottom: 2px solid #c62828; padding-bottom: 1px;">removed</span>
  <span style="border-bottom: 2px solid #f57c00; padding-bottom: 1px;">modified</span>
  <span>survived (no underline)</span>
</div>
<div class="proposal-text">{body}</div>
</body>
</html>"""

    return html


def process_docket(docket_id: str, units: list[dict], output_dir: Path) -> dict:
    """Process a single docket: load proposal, match units, generate heatmap."""
    result = load_proposal_text(docket_id)
    if result is None:
        return {"status": "no_text", "docket": docket_id}

    doc_id, raw_text = result
    proposal_text = clean_proposal_text(raw_text)

    if len(proposal_text) < 100:
        return {"status": "too_short", "docket": docket_id}

    spans = find_unit_spans(proposal_text, units)

    if len(spans) == 0:
        return {"status": "no_spans", "docket": docket_id}

    html = generate_heatmap_html(docket_id, proposal_text, spans, units)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{docket_id}.html"
    out_path.write_text(html)

    engaged = sum(1 for s in spans if s["unit"]["n_total_claims"] > 0)
    match_types = defaultdict(int)
    for s in spans:
        match_types[s["match_type"]] += 1

    return {
        "status": "ok",
        "docket": docket_id,
        "text_len": len(proposal_text),
        "n_units": len(units),
        "n_matched": len(spans),
        "n_engaged": engaged,
        "coverage": len(spans) / max(len(units), 1),
        "total_claims": sum(u["n_total_claims"] for u in units),
        "max_claims": max((s["unit"]["n_total_claims"] for s in spans), default=0),
        "match_types": dict(match_types),
        "path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate proposal heatmaps")
    parser.add_argument("--docket", type=str, default=None, help="Process specific docket")
    parser.add_argument("--max-text-len", type=int, default=50000, help="Max proposal text length")
    parser.add_argument("--min-claims", type=int, default=50, help="Min total claims")
    parser.add_argument("--min-coverage", type=float, default=0.5, help="Min unit coverage")
    parser.add_argument("--max-dockets", type=int, default=None, help="Max dockets to process")
    args = parser.parse_args()

    logger.info("Loading analysis units...")
    docket_units = load_analysis_units()
    logger.info("Loaded units for %d dockets", len(docket_units))

    if args.docket:
        dockets = [args.docket]
    else:
        # Filter dockets by criteria
        dockets = []
        for dk, units in docket_units.items():
            total_claims = sum(u["n_total_claims"] for u in units)
            engaged = sum(1 for u in units if u["n_total_claims"] > 0)
            coverage = engaged / max(len(units), 1)
            if total_claims >= args.min_claims and len(units) >= 5 and coverage >= args.min_coverage:
                dockets.append(dk)
        logger.info("Found %d qualifying dockets", len(dockets))

    if args.max_dockets:
        dockets = dockets[:args.max_dockets]

    results = []
    for i, dk in enumerate(dockets):
        units = docket_units.get(dk, [])
        logger.info("[%d/%d] Processing %s (%d units)...", i+1, len(dockets), dk, len(units))
        r = process_docket(dk, units, OUTPUT_DIR)
        results.append(r)

        if r["status"] == "ok":
            logger.info("  -> %s: %d/%d matched (%.0f%%), %d claims, text=%dK",
                        dk, r["n_matched"], r["n_units"], r["coverage"]*100,
                        r["total_claims"], r["text_len"]//1000)
        else:
            logger.info("  -> %s: %s", dk, r["status"])

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    logger.info("\nProcessed %d dockets, %d successful", len(results), len(ok))

    if ok:
        # Print best candidates for paper (short, high coverage, interesting)
        ok.sort(key=lambda x: x["text_len"])
        logger.info("\nBest candidates for paper (shortest with high coverage):")
        for r in ok[:20]:
            logger.info("  %s: %dK chars, %d/%d units matched (%.0f%%), %d claims",
                        r["docket"], r["text_len"]//1000, r["n_matched"], r["n_units"],
                        r["coverage"]*100, r["total_claims"])


if __name__ == "__main__":
    main()
