"""Score structural features of public comments using LLM.

Per-comment features are produced in two layers:

1. LLM features (stored as JSON in `llm_features_json`): commenter identity,
   format/letterhead signals, legal sophistication (inc. procedural objections
   and litigation threats), evidence quality (inc. peer-reviewed vs government
   vs media citation splits), experiential/situated-knowledge content,
   framing alignment with agency mandate, engagement depth (inc. requested
   action), impact claims (cost/benefit/small-biz/EJ/etc.), and campaign
   signals.

2. Submission-mode features (derived directly from raw CSV columns, no LLM):
   `submission_mode`, `has_typed_comment`, `has_content_files`,
   `has_attachment_files`, `attachment_count`. These distinguish "commenter
   uploaded a document" from "commenter typed into web form" — regulations.gov
   often renders typed comments as PDFs for display, so the authoritative
   signal is whether Content Files / Attachment Files is populated, not
   whether the displayed result is a PDF.

Also passes through timing columns (Posted Date, Received Date, Comment
Start/Due Date) for downstream within-comment-period timing analysis.

Usage:
    python score_comment_features.py --gpu 1
    python score_comment_features.py --gpu 1 --max-dirs 5  # test on 5 dirs
    python score_comment_features.py --gpu 1 --overwrite   # regenerate
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import re
import signal
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BULK_DIR = SCRIPTS_DIR.parent

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert analyst of U.S. federal regulatory comments. Your task is to analyze a public comment submitted during a federal rulemaking process and extract structured features about the comment and commenter.

Return ONLY a valid JSON object. Do not include any text before or after the JSON. Do not wrap it in markdown code blocks."""

USER_PROMPT = """Analyze this public comment and return a JSON object with the following fields:

{{
  "commenter": {{
    "derived_name": "Full name of the commenter if identifiable from signature, sign-off, or header in the comment text. null if anonymous or not stated.",
    "derived_organization": "Organization, company, agency, or institution the commenter represents or is affiliated with, as stated in the comment text. null if none mentioned.",
    "org_type": "Classify the commenter. One of: individual (private citizen with no organizational affiliation), trade_association (industry group like API, PhRMA, NAM), corporation (specific company), law_firm (legal practice submitting on behalf of clients), ngo_advocacy (nonprofit advocacy org like Sierra Club, ACLU, EFF), academic (university, research institution, or professor), government (federal, state, local, or tribal government entity), labor_union (union or worker organization), professional_society (e.g. AMA, IEEE, ABA), coalition (joint submission by multiple organizations), other, unknown",
    "title_role": "Job title, position, or role if stated (e.g. 'Vice President of Regulatory Affairs', 'Professor of Economics', 'City Manager'). null if not mentioned.",
    "credentials": "Professional degrees, certifications, licenses, or domain expertise explicitly mentioned (e.g. 'PhD in Environmental Science', 'Licensed Professional Engineer', 'Board-certified physician'). null if none mentioned.",
    "is_regulated_entity": "true if the commenter or their organization would be directly regulated by the proposed rule, false if not, null if unclear",
    "state": "U.S. state if explicitly mentioned by the commenter as their location. null if not stated."
  }},
  "format": {{
    "is_form_letter": "true ONLY if the comment appears to be a mass-produced template from an organized campaign where many people submit identical or near-identical boilerplate text (e.g. 'I urge you to protect...' form campaigns). false for ALL unique comments, even if they use formal letter format. Government agency letters, law firm briefs, trade association comments, and detailed organizational letters are NOT form letters even if they follow a formal template.",
    "has_letterhead_signals": "true if the comment shows signs of being on official letterhead: organization name/address block at top, formal date line, formal salutation (Dear Secretary/Administrator/Director), official sign-off block with printed name and title. false for casual emails, web form submissions, or comments without formal letter structure.",
    "has_attachments_referenced": "true if the comment text references attached documents, appendices, exhibits, or supplementary materials (e.g. 'see attached', 'Appendix A', 'enclosed report'). false otherwise.",
    "comment_length_category": "Estimate the word count and classify. One of: very_short (<100 words), short (100-500 words), medium (500-2000 words), long (2000-5000 words), very_long (>5000 words)"
  }},
  "legal_sophistication": {{
    "score": "0-5 integer. 0=no legal content at all. 1=vague mention of law or rights without specifics ('this violates our rights'). 2=cites a specific statute, regulation, or CFR section by name or number. 3=analyzes how legal requirements apply to the proposed rule. 4=detailed legal argument with citations to case law, legislative history, or administrative precedent. 5=comprehensive legal brief with multiple citations, structured legal analysis, and counter-arguments.",
    "cites_statutes": "true if the comment cites specific federal or state statutes by name or section number (e.g. 'Clean Air Act Section 111', '42 U.S.C. §7411', 'ADA Title II'). false otherwise.",
    "cites_case_law": "true if the comment cites specific court decisions (e.g. 'Chevron v. NRDC', 'Sierra Club v. EPA, 884 F.3d 1185'). false otherwise.",
    "cites_cfr": "true if the comment references specific Code of Federal Regulations sections (e.g. '40 CFR Part 60', '§60.5397a'). false otherwise.",
    "references_specific_rule_provisions": "true if the comment addresses specific sections, provisions, or language of the proposed rule being commented on. false if the comment only makes general arguments.",
    "references_prior_agency_action": "true if cites prior agency rulemakings, guidance documents, Federal Register notices, or the agency's own precedent actions. false otherwise.",
    "makes_procedural_objection": "true if raises Administrative Procedure Act or process objections: inadequate notice, failure to consider alternatives, arbitrary-and-capricious claims, rushed timeline, missing analysis, improper delegation. false otherwise.",
    "threatens_litigation": "true if the comment implicitly or explicitly signals willingness to challenge the rule in court (e.g. 'we reserve all legal rights', 'this rule would not survive judicial review', 'we will seek review in the D.C. Circuit'). false otherwise."
  }},
  "evidence_quality": {{
    "score": "0-5 integer. 0=pure opinion with no supporting evidence ('I think this is bad'). 1=anecdotal evidence or personal experience only ('In my 20 years of farming...'). 2=references external data, studies, or reports without providing them ('studies show that...'). 3=provides specific original data, measurements, or quantitative evidence from the commenter's own experience or analysis. 4=detailed quantitative analysis with specific numbers, tables, or statistical evidence. 5=comprehensive evidence package with methodology, peer-reviewed citations, and original analysis.",
    "cites_scientific_studies": "true if the comment cites specific peer-reviewed papers, scientific reports, or named research studies (e.g. 'Alvarez et al., Science, 2018'). false for vague references like 'research shows'.",
    "cites_peer_reviewed_articles": "true ONLY if the citation is specific enough to identify a peer-reviewed journal article (author + year, or article title + journal). false for vague references or for citations to non-peer-reviewed sources.",
    "cites_government_reports": "true if cites specific government reports or technical documents (GAO, CBO, CRS, NIH, CDC MMWR, EIA, BLS, agency regulatory impact analyses, etc.). false otherwise.",
    "cites_news_or_media": "true if cites specific news articles, journalism, or media reports. false otherwise.",
    "provides_economic_data": "true if the comment includes specific dollar amounts, cost estimates, employment figures, revenue impacts, or other quantitative economic data. false otherwise.",
    "includes_cost_benefit_analysis": "true if the comment explicitly weighs costs against benefits of the proposed rule, with at least some quantification. false otherwise.",
    "references_real_world_examples": "true if the comment describes specific real-world scenarios, case studies, or concrete examples from the commenter's direct experience or industry knowledge. false otherwise.",
    "references_tables_or_figures": "true if the comment references attached or embedded tables, figures, charts, graphs, or appendices containing quantitative content. false otherwise."
  }},
  "experiential_content": {{
    "first_person_experiential": "true if the commenter writes in first person about direct experience ('I have seen', 'in my 20 years as a nurse', 'we have observed at our facility'). false if purely third-person analytical.",
    "narrative_personal_experience": "true if the comment contains lay or personal situated narrative about the commenter's own life, family, or community ('my daughter's asthma', 'the flooding in my neighborhood', 'as a small farmer watching my fields'). This captures 'situated knowledge' from non-expert commenters. false for purely technical or organizational framing.",
    "technical_case_study": "true if the comment presents specific organizational, industry, or professional case analysis ('at our plant we measured X', 'our 2019 study of 50 facilities'). Distinct from lay narrative — this is expert/institutional knowledge. false otherwise.",
    "mentions_specific_incident": "true if references a specific named incident, event, or date ('the 2017 Hurricane Harvey response', 'the 2019 E. coli outbreak in Oregon'). false for generic mentions."
  }},
  "framing": {{
    "primary_frame": "The dominant framing of the comment's argument. One of: health (public health, disease, medical outcomes), safety (accidents, worker safety, product safety), economic_cost_benefit (costs, jobs, competitiveness, ROI, market effects), legal_constitutional (statutory authority, constitutional rights, APA, takings), scientific_technical (evidence, methodology, engineering, measurement), environmental (pollution, ecosystems, climate, wildlife), equity_justice (fairness, disparate impact, environmental justice, civil rights), religious_moral (ethical or values-based appeals), national_security (defense, foreign threats, critical infrastructure), procedural (process objections, transparency), other, unknown",
    "secondary_frame": "Secondary framing using the same enum, or null if only one frame is used.",
    "explicitly_invokes_agency_mandate": "true if the comment explicitly invokes the agency's statutory mandate or mission (e.g. 'EPA's duty under the Clean Air Act to protect public health', 'OSHA's mandate to ensure safe workplaces', 'FDA's obligation to ensure drug safety'). false otherwise.",
    "uses_agency_technical_vocabulary": "true if the comment uses specialized terminology native to the agency's regulatory domain (e.g. 'BSER', 'NAAQS', 'LCOE', 'MCL', 'REMS', 'RFS', 'LDAR', 'NSPS'). false if written entirely in lay language."
  }},
  "engagement_depth": {{
    "addresses_specific_provisions": "true if the comment engages with specific numbered sections, defined terms, or particular requirements of the proposed rule. false if the comment only makes general arguments about the rule's overall direction.",
    "proposes_alternative_language": "true if the comment suggests specific alternative regulatory text, modified requirements, different thresholds, exemptions, or concrete changes to the rule. false if it only states opposition or support without proposing alternatives.",
    "number_of_distinct_issues": "Integer estimate of how many separate substantive issues or topics the comment raises. A one-sentence comment = 0-1, a typical individual comment = 1-3, a detailed organizational comment might raise 5-20+.",
    "tone": "Overall tone. One of: supportive, opposed, mixed, neutral, constructive_criticism",
    "requested_action": "The commenter's top-level request. One of: support_as_proposed, modify, oppose_and_withdraw, delay_or_extend, request_information, none_stated",
    "requests_comment_period_extension": "true if the comment explicitly asks the agency to extend the comment period, reopen the record, or hold additional hearings. false otherwise."
  }},
  "impact_claims": {{
    "claims_direct_regulatory_cost": "true if the commenter claims the rule will impose specific direct costs on them or their organization. false otherwise.",
    "claims_direct_benefit": "true if the commenter claims specific direct benefits from the rule accruing to them or their organization. false otherwise.",
    "mentions_small_business_impact": "true if raises small-business impact concerns (RFA, SBREFA, SBA, 'small operators', 'family businesses'). false otherwise.",
    "mentions_worker_or_employment_impact": "true if raises impacts on workers, jobs, wages, or employment levels. false otherwise.",
    "mentions_consumer_impact": "true if raises impacts on consumers, prices, or access to products/services. false otherwise.",
    "mentions_rural_or_geographic_impact": "true if raises impacts on rural communities, specific regions, or geographic inequity. false otherwise.",
    "mentions_environmental_justice": "true if invokes environmental justice, disadvantaged communities, or disparate impact on marginalized groups. false otherwise.",
    "mentions_public_health_impact": "true if raises impacts on public health outcomes (morbidity, mortality, exposure, disease burden). false otherwise."
  }},
  "campaign_signals": {{
    "appears_organized_campaign": "true if the comment shows signs of being part of an organized campaign beyond simple form-letter duplication: mentions of a sponsoring organization, call-to-action framing, generic template with a customized fill-in section. false for clearly independent submissions.",
    "sponsor_organization_mentioned": "Name of the campaign sponsor or organizer if explicitly referenced (e.g. 'submitted via Sierra Club action', 'as requested by the National Association of Manufacturers'). null if none mentioned.",
    "signatory_count_if_stated": "Integer count of co-signers or signatory organizations if the comment is a coalition/sign-on letter and either states a count or names them explicitly. null if solo or not determinable."
  }},
  "summary": "One sentence summary capturing the commenter's main point and who they are."
}}

COMMENT:
{comment_text}"""

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": USER_PROMPT.format(comment_text="""Dear Administrator Regan,

On behalf of the American Petroleum Institute (API) and its over 600 member companies, we submit these comments on EPA's proposed rule "Standards of Performance for New, Reconstructed, and Modified Sources and Emissions Guidelines for Existing Sources: Oil and Natural Gas Sector Climate Review" (86 Fed. Reg. 63110). API has engaged extensively with EPA on methane regulations since the original 2012 NSPS and appreciates the opportunity to comment.

While API supports the goal of reducing methane emissions, we have significant concerns about the proposed monitoring requirements in §60.5397a. Our analysis of GHGRP data from 2015-2020 shows that the proposed OOOOa standards would impose costs of approximately $1.2 billion annually across the sector, with marginal abatement costs exceeding $1,500/ton of methane for small operators.

We recommend EPA adopt performance-based standards rather than prescriptive monitoring requirements, consistent with the approach in Section 111 of the Clean Air Act (42 U.S.C. §7411). The D.C. Circuit's decision in Sierra Club v. EPA, 884 F.3d 1185 (2018) supports flexibility in compliance pathways.

Sincerely,
Frank J. Macchiarola
Senior Vice President, Policy, Economics and Regulatory Affairs
American Petroleum Institute"""),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "commenter": {
                "derived_name": "Frank J. Macchiarola",
                "derived_organization": "American Petroleum Institute (API)",
                "org_type": "trade_association",
                "title_role": "Senior Vice President, Policy, Economics and Regulatory Affairs",
                "credentials": None,
                "is_regulated_entity": True,
                "state": None
            },
            "format": {
                "is_form_letter": False,
                "has_letterhead_signals": True,
                "has_attachments_referenced": False,
                "comment_length_category": "medium"
            },
            "legal_sophistication": {
                "score": 4,
                "cites_statutes": True,
                "cites_case_law": True,
                "cites_cfr": True,
                "references_specific_rule_provisions": True,
                "references_prior_agency_action": True,
                "makes_procedural_objection": False,
                "threatens_litigation": False
            },
            "evidence_quality": {
                "score": 3,
                "cites_scientific_studies": False,
                "cites_peer_reviewed_articles": False,
                "cites_government_reports": True,
                "cites_news_or_media": False,
                "provides_economic_data": True,
                "includes_cost_benefit_analysis": True,
                "references_real_world_examples": True,
                "references_tables_or_figures": False
            },
            "experiential_content": {
                "first_person_experiential": True,
                "narrative_personal_experience": False,
                "technical_case_study": True,
                "mentions_specific_incident": False
            },
            "framing": {
                "primary_frame": "economic_cost_benefit",
                "secondary_frame": "legal_constitutional",
                "explicitly_invokes_agency_mandate": False,
                "uses_agency_technical_vocabulary": True
            },
            "engagement_depth": {
                "addresses_specific_provisions": True,
                "proposes_alternative_language": True,
                "number_of_distinct_issues": 3,
                "tone": "constructive_criticism",
                "requested_action": "modify",
                "requests_comment_period_extension": False
            },
            "impact_claims": {
                "claims_direct_regulatory_cost": True,
                "claims_direct_benefit": False,
                "mentions_small_business_impact": True,
                "mentions_worker_or_employment_impact": False,
                "mentions_consumer_impact": False,
                "mentions_rural_or_geographic_impact": False,
                "mentions_environmental_justice": False,
                "mentions_public_health_impact": False
            },
            "campaign_signals": {
                "appears_organized_campaign": False,
                "sponsor_organization_mentioned": None,
                "signatory_count_if_stated": None
            },
            "summary": "API trade association opposes prescriptive monitoring requirements in EPA methane rule, citing $1.2B annual costs and recommending performance-based standards."
        }),
    },
    {
        "role": "user",
        "content": USER_PROMPT.format(comment_text="""I am strongly opposed to this rule. The government has no business telling us what to do with our land. This is just more government overreach and bureaucratic nonsense. Stop wasting our tax dollars on unnecessary regulations. I've lived in Wyoming my whole life and we don't need Washington telling us how to manage our resources."""),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "commenter": {
                "derived_name": None,
                "derived_organization": None,
                "org_type": "individual",
                "title_role": None,
                "credentials": None,
                "is_regulated_entity": False,
                "state": "Wyoming"
            },
            "format": {
                "is_form_letter": False,
                "has_letterhead_signals": False,
                "has_attachments_referenced": False,
                "comment_length_category": "very_short"
            },
            "legal_sophistication": {
                "score": 1,
                "cites_statutes": False,
                "cites_case_law": False,
                "cites_cfr": False,
                "references_specific_rule_provisions": False,
                "references_prior_agency_action": False,
                "makes_procedural_objection": False,
                "threatens_litigation": False
            },
            "evidence_quality": {
                "score": 1,
                "cites_scientific_studies": False,
                "cites_peer_reviewed_articles": False,
                "cites_government_reports": False,
                "cites_news_or_media": False,
                "provides_economic_data": False,
                "includes_cost_benefit_analysis": False,
                "references_real_world_examples": True,
                "references_tables_or_figures": False
            },
            "experiential_content": {
                "first_person_experiential": True,
                "narrative_personal_experience": True,
                "technical_case_study": False,
                "mentions_specific_incident": False
            },
            "framing": {
                "primary_frame": "legal_constitutional",
                "secondary_frame": "other",
                "explicitly_invokes_agency_mandate": False,
                "uses_agency_technical_vocabulary": False
            },
            "engagement_depth": {
                "addresses_specific_provisions": False,
                "proposes_alternative_language": False,
                "number_of_distinct_issues": 1,
                "tone": "opposed",
                "requested_action": "oppose_and_withdraw",
                "requests_comment_period_extension": False
            },
            "impact_claims": {
                "claims_direct_regulatory_cost": False,
                "claims_direct_benefit": False,
                "mentions_small_business_impact": False,
                "mentions_worker_or_employment_impact": False,
                "mentions_consumer_impact": False,
                "mentions_rural_or_geographic_impact": True,
                "mentions_environmental_justice": False,
                "mentions_public_health_impact": False
            },
            "campaign_signals": {
                "appears_organized_campaign": False,
                "sponsor_organization_mentioned": None,
                "signatory_count_if_stated": None
            },
            "summary": "Individual from Wyoming opposes rule as government overreach without engaging specific provisions."
        }),
    },
    {
        "role": "user",
        "content": USER_PROMPT.format(comment_text="""The undersigned environmental organizations — Sierra Club, Earthjustice, Natural Resources Defense Council, and Clean Air Task Force — respectfully submit these joint comments on the proposed amendments to 40 CFR Part 60, Subpart OOOOa.

We support EPA's proposal to strengthen methane emission standards but urge the agency to go further. Peer-reviewed studies (Alvarez et al., Science, 2018; Rutherford et al., Nature Communications, 2021) demonstrate that actual methane emissions are 60% higher than EPA inventory estimates. The IEA's 2022 Global Methane Tracker confirms significant under-reporting.

Specifically, we recommend:
1. Quarterly LDAR surveys for all well sites (not just large sites as proposed in §60.5397a(g)(1))
2. Adoption of continuous monitoring technologies per the Stanford/EDF Mobile Monitoring Protocol
3. Elimination of the proposed exemption for low-production wells in §60.5397a(g)(3), which we estimate would leave 300,000 tons/year of methane uncontrolled

These recommendations are consistent with EPA's obligations under Section 111(b)(1)(A) of the Clean Air Act to set standards reflecting the "best system of emission reduction" (BSER) that has been "adequately demonstrated." See Portland Cement Ass'n v. Ruckelshaus, 486 F.2d 375 (D.C. Cir. 1973).

Attached please find our detailed technical analysis (Appendix A) and economic modeling results (Appendix B)."""),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "commenter": {
                "derived_name": None,
                "derived_organization": "Sierra Club, Earthjustice, Natural Resources Defense Council, Clean Air Task Force (joint)",
                "org_type": "coalition",
                "title_role": None,
                "credentials": None,
                "is_regulated_entity": False,
                "state": None
            },
            "format": {
                "is_form_letter": False,
                "has_letterhead_signals": True,
                "has_attachments_referenced": True,
                "comment_length_category": "medium"
            },
            "legal_sophistication": {
                "score": 4,
                "cites_statutes": True,
                "cites_case_law": True,
                "cites_cfr": True,
                "references_specific_rule_provisions": True,
                "references_prior_agency_action": True,
                "makes_procedural_objection": False,
                "threatens_litigation": False
            },
            "evidence_quality": {
                "score": 4,
                "cites_scientific_studies": True,
                "cites_peer_reviewed_articles": True,
                "cites_government_reports": True,
                "cites_news_or_media": False,
                "provides_economic_data": True,
                "includes_cost_benefit_analysis": False,
                "references_real_world_examples": True,
                "references_tables_or_figures": True
            },
            "experiential_content": {
                "first_person_experiential": False,
                "narrative_personal_experience": False,
                "technical_case_study": False,
                "mentions_specific_incident": False
            },
            "framing": {
                "primary_frame": "environmental",
                "secondary_frame": "scientific_technical",
                "explicitly_invokes_agency_mandate": True,
                "uses_agency_technical_vocabulary": True
            },
            "engagement_depth": {
                "addresses_specific_provisions": True,
                "proposes_alternative_language": True,
                "number_of_distinct_issues": 3,
                "tone": "constructive_criticism",
                "requested_action": "modify",
                "requests_comment_period_extension": False
            },
            "impact_claims": {
                "claims_direct_regulatory_cost": False,
                "claims_direct_benefit": False,
                "mentions_small_business_impact": False,
                "mentions_worker_or_employment_impact": False,
                "mentions_consumer_impact": False,
                "mentions_rural_or_geographic_impact": False,
                "mentions_environmental_justice": False,
                "mentions_public_health_impact": False
            },
            "campaign_signals": {
                "appears_organized_campaign": False,
                "sponsor_organization_mentioned": None,
                "signatory_count_if_stated": 4
            },
            "summary": "Environmental coalition supports stronger methane standards, cites peer-reviewed studies showing 60% undercount, proposes three specific amendments to monitoring requirements with legal and technical analysis."
        }),
    },
]


# ---------------------------------------------------------------------------
# Tokenizer / truncation
# ---------------------------------------------------------------------------

_tokenizer = None
_tokenizer_model = None


def _get_tokenizer(model: str = "meta-llama/Llama-3.3-70B-Instruct"):
    global _tokenizer, _tokenizer_model
    if _tokenizer is None or _tokenizer_model != model:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        _tokenizer_model = model
    return _tokenizer


def truncate_text(text: str, head_tokens: int = 4096, tail_tokens: int = 2048,
                  tokenizer_model: str = "meta-llama/Llama-3.3-70B-Instruct") -> str:
    """Truncate comment text using head + tail strategy.

    Takes the first head_tokens and last tail_tokens, joined with '...',
    to capture both the introduction (org identity, "on behalf of") and
    the closing (signature, credentials, "sincerely").
    """
    tok = _get_tokenizer(tokenizer_model)
    ids = tok.encode(str(text), add_special_tokens=False, truncation=False)
    max_tokens = head_tokens + tail_tokens
    if len(ids) <= max_tokens:
        return str(text)
    head = tok.decode(ids[:head_tokens], skip_special_tokens=True)
    tail = tok.decode(ids[-tail_tokens:], skip_special_tokens=True)
    return head + "\n\n...[truncated]...\n\n" + tail


# ---------------------------------------------------------------------------
# Submission-mode features (derived from raw CSV columns, no LLM)
# ---------------------------------------------------------------------------

def _nonempty(val) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


def _count_urls(val) -> int:
    """Attachment Files / Content Files are pipe- or newline-separated URL lists."""
    if not _nonempty(val):
        return 0
    s = str(val)
    parts = re.split(r"[|\n,;\s]+", s)
    return sum(1 for p in parts if p.strip().lower().startswith(("http://", "https://")))


def compute_submission_features(row: pd.Series) -> dict:
    """Classify how the comment was actually submitted, from raw columns.

    regulations.gov renders many typed comments as PDFs for display, so the
    authoritative signal of "commenter uploaded a document" is whether
    `Content Files` or `Attachment Files` is populated — NOT whether the
    displayed result is a PDF.
    """
    comment_val = row.get("Comment")
    content_val = row.get("Content Files")
    attach_val = row.get("Attachment Files")

    has_typed = _nonempty(comment_val)
    has_content = _nonempty(content_val)
    has_attach = _nonempty(attach_val)
    attach_count = _count_urls(attach_val) + _count_urls(content_val)

    if has_typed and (has_content or has_attach):
        mode = "text_plus_attachment"
    elif has_typed:
        mode = "text_only"
    elif has_content or has_attach:
        mode = "attachment_only"
    else:
        mode = "unknown"

    return {
        "submission_mode": mode,
        "has_typed_comment": has_typed,
        "has_content_files": has_content,
        "has_attachment_files": has_attach,
        "attachment_count": attach_count,
    }


# ---------------------------------------------------------------------------
# Cluster-level representative selection (MinHash clusters from upstream dedup)
# ---------------------------------------------------------------------------

def select_cluster_representatives(df: pd.DataFrame, dirpath: Path) -> pd.DataFrame:
    """Reduce df to one row per MinHash cluster using the upstream dedup mapper.

    Expects `public_submission_all_text__dedup_mapper.csv` in dirpath with
    columns: agency_id, docket_id, document_id, cluster_id, cluster_uid.
    Docs not in the mapper are singletons and kept as their own cluster
    (cluster_uid = SINGLE::<document_id>). Within each multi-doc cluster,
    the alphabetically-first document_id is selected as the representative
    (stable, reproducible across runs).

    Adds two columns to the returned frame: `cluster_uid`, `cluster_size`.
    Returns the input df unchanged (plus synthetic single-member cluster
    metadata) if the mapper file is absent.
    """
    mapper_path = dirpath / "public_submission_all_text__dedup_mapper.csv"
    if not mapper_path.exists():
        logger.warning("%s: no dedup mapper, treating every comment as its own cluster",
                       dirpath.name)
        out = df.copy()
        out["cluster_uid"] = "SINGLE::" + out.get("Document ID", "").astype(str)
        out["cluster_size"] = 1
        return out

    try:
        mapper = pd.read_csv(mapper_path, low_memory=False,
                             usecols=["document_id", "cluster_uid"])
    except Exception as e:
        logger.warning("%s: failed to read mapper (%s); falling back to per-comment",
                       dirpath.name, e)
        out = df.copy()
        out["cluster_uid"] = "SINGLE::" + out.get("Document ID", "").astype(str)
        out["cluster_size"] = 1
        return out

    # Represent each cluster by its alphabetically-first doc_id.
    mapper = mapper.sort_values("document_id")
    cluster_meta = mapper.groupby("cluster_uid", as_index=False).agg(
        rep_doc_id=("document_id", "first"),
        cluster_size=("document_id", "size"),
    )

    # Attach cluster_uid to every row of df.
    doc_to_cluster = dict(zip(mapper["document_id"], mapper["cluster_uid"]))
    df_aug = df.copy()
    doc_col = "Document ID" if "Document ID" in df_aug.columns else None
    if doc_col is None:
        logger.warning("%s: df has no Document ID column; cannot apply cluster dedup",
                       dirpath.name)
        df_aug["cluster_uid"] = "SINGLE::" + df_aug.index.astype(str)
        df_aug["cluster_size"] = 1
        return df_aug

    df_aug["cluster_uid"] = df_aug[doc_col].map(doc_to_cluster)
    # Singletons fall through with NaN cluster_uid → synthesize SINGLE:: key.
    singleton_mask = df_aug["cluster_uid"].isna()
    df_aug.loc[singleton_mask, "cluster_uid"] = (
        "SINGLE::" + df_aug.loc[singleton_mask, doc_col].astype(str)
    )

    # Pick representative rows: first-by-doc-id within each non-singleton cluster;
    # all singletons kept.
    cluster_rep_ids = set(cluster_meta["rep_doc_id"])
    keep_mask = singleton_mask | df_aug[doc_col].isin(cluster_rep_ids)
    reps = df_aug.loc[keep_mask].copy()

    size_map = dict(zip(cluster_meta["cluster_uid"], cluster_meta["cluster_size"]))
    reps["cluster_size"] = reps["cluster_uid"].map(size_map).fillna(1).astype(int)

    n_multi = int((reps["cluster_size"] > 1).sum())
    n_single = int((reps["cluster_size"] == 1).sum())
    logger.info("%s: cluster-level reps = %d (%d multi-member clusters, %d singletons) "
                "from %d input rows",
                dirpath.name, len(reps), n_multi, n_single, len(df))
    return reps


# ---------------------------------------------------------------------------
# Mass comment deduplication (fallback when cluster-level is not requested)
# ---------------------------------------------------------------------------

def deduplicate_comments(df: pd.DataFrame, text_col: str = "canonical_text") -> pd.DataFrame:
    """Deduplicate mass comment campaigns by keeping first occurrence.

    Uses first 200 chars as fingerprint.
    """
    df = df.copy()
    df["_fingerprint"] = df[text_col].fillna("").astype(str).str[:200].str.strip().str.lower()
    before = len(df)
    df = df.drop_duplicates(subset=["_fingerprint"], keep="first")
    after = len(df)
    if before != after:
        logger.info("Deduplicated: %d -> %d comments (removed %d mass/duplicate comments)",
                     before, after, before - after)
    df = df.drop(columns=["_fingerprint"])
    return df


# ---------------------------------------------------------------------------
# vLLM offline batch scoring
# ---------------------------------------------------------------------------

def load_vllm(model: str, max_model_len: int, tp: int, enforce_eager: bool = False,
              gpu_memory_utilization: float = 0.85):
    """Load vLLM model once."""
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    from vllm import LLM
    logger.info("Loading vLLM model %s (max_model_len=%d, tp=%d, enforce_eager=%s, gpu_util=%.2f)",
                model, max_model_len, tp, enforce_eager, gpu_memory_utilization)
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
    )
    return llm


def score_batch(prompts: List[str], llm) -> List[str]:
    """Score a batch of prompts using a pre-loaded vLLM model."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3072,
        stop=None,
    )

    # Build chat messages for each prompt
    all_messages = []
    for prompt in prompts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": prompt})
        all_messages.append(messages)

    tok = llm.get_tokenizer()
    formatted_prompts = []
    for msgs in all_messages:
        try:
            formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        except Exception:
            formatted_prompts.append(msgs[-1]["content"])

    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outputs]


# ---------------------------------------------------------------------------
# JSON parsing + repair
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Optional[dict]:
    """Try to parse LLM output as JSON."""
    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from surrounding text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


FIX_JSON_PROMPT = """The following text was supposed to be a valid JSON object but has formatting errors. Fix it so it is valid JSON. Return ONLY the corrected JSON, nothing else.

Broken text:
{broken_text}"""


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

_active_processing_files = set()


def _cleanup_processing_files():
    for pf in list(_active_processing_files):
        try:
            pf.unlink(missing_ok=True)
        except Exception:
            pass


atexit.register(_cleanup_processing_files)


def _signal_handler(signum, frame):
    _cleanup_processing_files()
    sys.exit(1)


for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(sig, _signal_handler)


def _json_safe(val):
    """Convert numpy/pandas scalars to JSON-serializable Python types."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    if pd.isna(val):
        return None
    return val


def _build_output_row(src_row, df_cols, year: str, json_str: Optional[str]) -> dict:
    """Assemble one output row — same schema whether from fresh scoring or resume."""
    row = {}
    row["Document ID"] = _json_safe(src_row.get("Document ID", "")) if "Document ID" in df_cols else ""
    row["Docket ID"] = _json_safe(src_row.get("Docket ID", "")) if "Docket ID" in df_cols else ""
    row["Agency ID"] = _json_safe(src_row.get("Agency ID", "")) if "Agency ID" in df_cols else ""
    row["year"] = year
    for col in ["Posted Date", "Received Date", "Postmark Date",
                "Comment Start Date", "Comment Due Date",
                "Comment on Document ID", "Duplicate Comments", "Page Count"]:
        if col in df_cols:
            row[col] = _json_safe(src_row.get(col, ""))
    row.update({k: _json_safe(v) for k, v in compute_submission_features(src_row).items()})
    if "cluster_uid" in df_cols:
        row["cluster_uid"] = _json_safe(src_row.get("cluster_uid", ""))
        row["cluster_size"] = int(src_row.get("cluster_size", 1) or 1)
    row["llm_features_json"] = json_str if json_str else ""
    row["parse_success"] = json_str is not None
    return row


def _load_partial(partial_path: Path) -> dict:
    """Read per-batch JSONL checkpoint into {Document ID: row}. Skips malformed lines."""
    if not partial_path.exists():
        return {}
    out = {}
    with partial_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = r.get("Document ID")
            if doc_id:
                out[str(doc_id)] = r
    return out


def _rewrite_partial(partial_path: Path, rows: dict) -> None:
    """Atomically rewrite partial JSONL from a dict of doc_id -> row."""
    tmp = partial_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(partial_path)


def process_directory(dirpath: Path, args, llm) -> dict:
    """Process all comments in a directory and score features.

    Writes per-batch checkpoints to a JSONL file (one row per batch-produced
    output) so a killed run can resume mid-directory on restart. Partial file
    is removed once the final CSV is written.
    """
    suffix = getattr(args, "output_suffix", "") or ""
    output_path = dirpath / f"public_submission_engineered_features{suffix}.csv"
    partial_path = dirpath / f".feature-scoring-partial{suffix}.jsonl"
    processing_flag = dirpath / f".feature-scoring-processing{suffix}"

    # Skip if already done
    if output_path.exists() and not args.overwrite:
        logger.info("Output exists for %s, skipping.", dirpath.name)
        return {"status": "skipped_existing"}

    # With --overwrite, wipe both partial and final so we start fresh
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    # Skip if being processed by another worker
    if processing_flag.exists():
        logger.info("Directory %s is being processed by another worker, skipping.", dirpath.name)
        return {"status": "skipped_locked"}

    # Find comment file
    comment_file = None
    for ext in [".csv.gz", ".csv"]:
        p = dirpath / f"public_submission_all_text{ext}"
        if p.exists():
            comment_file = p
            break
    if comment_file is None:
        return {"status": "skipped_no_comments"}

    # Acquire lock
    try:
        processing_flag.touch()
        _active_processing_files.add(processing_flag)
    except Exception:
        return {"status": "skipped_lock_failed"}

    try:
        # Load comments
        try:
            df = pd.read_csv(comment_file, low_memory=False)
        except Exception as e:
            logger.error("Failed to read %s: %s", comment_file, e)
            return {"status": "error", "error": str(e)}

        if "canonical_text" not in df.columns:
            logger.warning("No canonical_text column in %s", comment_file)
            return {"status": "skipped_no_text"}

        n_original = len(df)
        logger.info("%s: %d comments loaded", dirpath.name, n_original)

        # Choose scoring targets: cluster reps (via upstream MinHash mapper) or
        # 200-char-fingerprint unique comments.
        if getattr(args, "cluster_level", False):
            df = select_cluster_representatives(df, dirpath)
        else:
            df = deduplicate_comments(df)
        n_deduped = len(df)

        # Smoke-test cap.
        if getattr(args, "max_comments_per_dir", None):
            df = df.head(args.max_comments_per_dir)
            logger.info("%s: capped to %d comments for smoke test", dirpath.name, len(df))

        # Extract year from directory name
        parts = dirpath.name.rsplit("_", 2)
        year = parts[-2] if len(parts) >= 3 else ""

        # Truncate comment text
        df["_truncated_text"] = df["canonical_text"].fillna("").astype(str).apply(
            lambda x: truncate_text(x, head_tokens=4096, tail_tokens=2048,
                                    tokenizer_model=args.model)
        )

        # Resume from partial checkpoint if present
        completed_rows = _load_partial(partial_path)
        df_cols = df.columns
        if completed_rows:
            logger.info("%s: resuming with %d rows already scored from prior run",
                        dirpath.name, len(completed_rows))

        # Identify rows still needing a primary pass
        if "Document ID" in df_cols:
            done_ids = set(completed_rows.keys())
            needed_mask = ~df["Document ID"].astype(str).isin(done_ids)
        else:
            needed_mask = pd.Series([True] * len(df), index=df.index)
        df_todo = df[needed_mask]

        logger.info("%s: scoring %d comments (batch_size=%d; %d skipped as already-done)",
                    dirpath.name, len(df_todo), args.batch_size, len(df) - len(df_todo))

        # Build prompts for the todo set
        todo_prompts = [
            USER_PROMPT.format(comment_text=text)
            for text in df_todo["_truncated_text"]
        ]
        todo_indices = list(df_todo.index)

        # Score in batches; append each batch's outputs to partial JSONL + fsync
        with partial_path.open("a") as f_partial:
            for batch_start in range(0, len(todo_prompts), args.batch_size):
                batch = todo_prompts[batch_start:batch_start + args.batch_size]
                batch_idx = todo_indices[batch_start:batch_start + len(batch)]
                logger.info("  Batch %d-%d / %d",
                            batch_start, batch_start + len(batch), len(todo_prompts))
                try:
                    outputs = score_batch(batch, llm)
                except Exception as e:
                    logger.error("Batch failed at %d: %s", batch_start, e)
                    outputs = [""] * len(batch)

                for idx, raw_out in zip(batch_idx, outputs):
                    src_row = df.loc[idx]
                    result = parse_json_output(raw_out)
                    json_str = json.dumps(result) if result is not None else None
                    row = _build_output_row(src_row, df_cols, year, json_str)
                    completed_rows[str(row["Document ID"])] = row
                    f_partial.write(json.dumps(row) + "\n")
                f_partial.flush()
                os.fsync(f_partial.fileno())

        n_parsed = sum(1 for r in completed_rows.values() if r.get("parse_success"))
        n_failed = len(completed_rows) - n_parsed
        logger.info("%s: %d/%d parsed successfully, %d failed",
                    dirpath.name, n_parsed, len(completed_rows), n_failed)

        # Retry any still-failed rows. Checkpoint after to preserve retry work.
        failed_doc_ids = [doc_id for doc_id, r in completed_rows.items()
                          if not r.get("parse_success")]
        if failed_doc_ids and "Document ID" in df_cols:
            failed_idx_in_df = df[df["Document ID"].astype(str).isin(failed_doc_ids)].index.tolist()
            # Pass 1: original prompt re-run (doesn't depend on prior raw output,
            # so this works on restart too).
            logger.info("Retrying %d failed via original prompt...", len(failed_idx_in_df))
            retry_prompts = [
                USER_PROMPT.format(comment_text=df.at[i, "_truncated_text"])
                for i in failed_idx_in_df
            ]
            try:
                retry_outputs = score_batch(retry_prompts, llm)
                fixed = 0
                for idx, out in zip(failed_idx_in_df, retry_outputs):
                    result = parse_json_output(out)
                    if result is not None:
                        src_row = df.loc[idx]
                        json_str = json.dumps(result)
                        row = _build_output_row(src_row, df_cols, year, json_str)
                        completed_rows[str(row["Document ID"])] = row
                        fixed += 1
                logger.info("Fixed %d/%d via re-run", fixed, len(failed_idx_in_df))
                # Rewrite partial to reflect retry updates (no longer append-only)
                _rewrite_partial(partial_path, completed_rows)
            except Exception as e:
                logger.error("Retry batch failed: %s", e)

        final_failed = sum(1 for r in completed_rows.values() if not r.get("parse_success"))
        logger.info("%s: final results: %d/%d parsed, %d still failed",
                    dirpath.name, len(completed_rows) - final_failed,
                    len(completed_rows), final_failed)

        # Write final CSV (ordered by df index so the output is stable).
        ordered = []
        seen = set()
        for idx in df.index:
            doc_id = str(df.at[idx, "Document ID"]) if "Document ID" in df_cols else None
            if doc_id and doc_id in completed_rows and doc_id not in seen:
                ordered.append(completed_rows[doc_id])
                seen.add(doc_id)
        out_df = pd.DataFrame(ordered)
        out_df.to_csv(output_path, index=False)
        logger.info("%s: saved %d rows to %s", dirpath.name, len(out_df), output_path.name)

        # Clean up partial file now that CSV is durable
        partial_path.unlink(missing_ok=True)

        return {
            "status": "completed",
            "n_original": n_original,
            "n_deduped": n_deduped,
            "n_scored": len(completed_rows),
            "n_parsed": len(completed_rows) - final_failed,
            "n_failed": final_failed,
        }

    except Exception as e:
        logger.error("Failed on %s: %s", dirpath.name, e, exc_info=True)
        return {"status": "error", "error": str(e)}

    finally:
        processing_flag.unlink(missing_ok=True)
        _active_processing_files.discard(processing_flag)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score structural features of public comments")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-dirs", type=int, default=None, help="Max directories to process (for testing)")
    parser.add_argument("--max-comments-per-dir", type=int, default=None,
                        help="Cap comments scored per directory (for smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="vLLM enforce_eager mode (useful for new/FP8 architectures)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM gpu_memory_utilization (lower to share GPU)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to output filename (e.g. '_v2') to preserve prior runs")
    parser.add_argument("--cluster-level", action="store_true",
                        help="Score one representative per MinHash cluster (from public_submission_all_text__dedup_mapper.csv) instead of per 200-char-unique comment. Adds cluster_uid and cluster_size columns to output.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    suffix = args.output_suffix or ""

    # Gather year-dirs grouped by agency. Agency = first-level subdir under BULK_DIR;
    # year-dirs are its children that contain a public_submission_all_text.* file.
    agencies: dict[str, list[Path]] = {}
    for agency_dir in sorted(BULK_DIR.iterdir()):
        if not agency_dir.is_dir() or agency_dir.name == "scripts":
            continue
        year_dirs = []
        for year_dir in sorted(agency_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            has_comments = any(
                (year_dir / f"public_submission_all_text{ext}").exists()
                for ext in [".csv.gz", ".csv"]
            )
            if has_comments:
                year_dirs.append(year_dir)
        if year_dirs:
            agencies[agency_dir.name] = year_dirs

    # Agency size = sum of bytes in public_submission_all_text files. Fast proxy
    # for LLM workload; compressed .gz files count as their on-disk size which
    # roughly tracks uncompressed row count at a fixed compression ratio.
    def agency_size(year_dirs: list[Path]) -> int:
        total = 0
        for yd in year_dirs:
            for ext in [".csv.gz", ".csv"]:
                p = yd / f"public_submission_all_text{ext}"
                if p.exists():
                    total += p.stat().st_size
                    break
        return total

    sorted_agencies = sorted(agencies.items(), key=lambda kv: agency_size(kv[1]))
    total_year_dirs = sum(len(v) for _, v in sorted_agencies)
    logger.info("Found %d agencies covering %d year-dirs with comments",
                len(sorted_agencies), total_year_dirs)
    logger.info("Agency order (smallest first): %s",
                ", ".join(f"{name}[{len(yrs)}d,{agency_size(yrs)//(1024*1024)}MB]"
                          for name, yrs in sorted_agencies[:10])
                + (" ..." if len(sorted_agencies) > 10 else ""))

    # Apply --max-dirs as a global cap across agencies (for testing)
    remaining_cap = args.max_dirs

    # Load vLLM once
    llm = load_vllm(args.model, args.max_model_len, args.tp,
                    enforce_eager=args.enforce_eager,
                    gpu_memory_utilization=args.gpu_memory_utilization)

    # Process agency by agency, with an agency-level lock so a second concurrent
    # worker can safely partition work. Per-dir locks still apply inside.
    processed = 0
    for a_idx, (agency_name, year_dirs) in enumerate(sorted_agencies):
        if remaining_cap is not None and remaining_cap <= 0:
            break
        agency_lock = BULK_DIR / agency_name / f".feature-scoring-agency{suffix}.lock"
        if agency_lock.exists():
            logger.info("Agency %s is being processed by another worker, skipping",
                        agency_name)
            continue
        try:
            agency_lock.touch()
            _active_processing_files.add(agency_lock)
        except Exception as e:
            logger.warning("Could not acquire agency lock for %s: %s", agency_name, e)
            continue

        try:
            logger.info("=== Agency [%d/%d] %s (%d year-dirs, %d MB) ===",
                        a_idx + 1, len(sorted_agencies), agency_name,
                        len(year_dirs), agency_size(year_dirs) // (1024 * 1024))
            for d in year_dirs:
                if remaining_cap is not None and remaining_cap <= 0:
                    break
                processed += 1
                logger.info("--- [dir %d/%d] Processing %s ---",
                            processed, total_year_dirs, d.name)
                result = process_directory(d, args, llm)
                logger.info("Result for %s: %s", d.name, json.dumps(result))
                if remaining_cap is not None:
                    remaining_cap -= 1
        finally:
            agency_lock.unlink(missing_ok=True)
            _active_processing_files.discard(agency_lock)


if __name__ == "__main__":
    main()
