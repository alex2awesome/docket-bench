"""Build a single flat CSV summarizing all features + DIME + claim labels across dirs.

Output: scripts/data/feature_analysis_summary.csv.gz
"""
import json
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BULK = Path(__file__).resolve().parent.parent

def resolve(d, b):
    for ext in ('.csv.gz', '.csv'):
        p = d / f'{b}{ext}'
        if p.exists(): return p
    return None

def parse_features(s):
    if not isinstance(s, str): return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict): return obj[0]
        return obj if isinstance(obj, dict) else None
    except: return None

def sg(f, *keys, default=None):
    for k in keys:
        if isinstance(f, dict): f = f.get(k, default)
        else: return default
    return f

def extract_doc_id(claim_id):
    parts = str(claim_id).split('__')
    return parts[3] if len(parts) >= 4 else None

FLAT_COLS = [
    ('org_type', ('commenter','org_type')),
    ('is_regulated_entity', ('commenter','is_regulated_entity')),
    ('is_form_letter', ('format','is_form_letter')),
    ('has_letterhead', ('format','has_letterhead_signals')),
    ('length_cat', ('format','comment_length_category')),
    ('legal_score', ('legal_sophistication','score')),
    ('cites_statutes', ('legal_sophistication','cites_statutes')),
    ('cites_case_law', ('legal_sophistication','cites_case_law')),
    ('cites_peer_reviewed', ('evidence_quality','cites_peer_reviewed_articles')),
    ('provides_economic_data', ('evidence_quality','provides_economic_data')),
    ('includes_cba', ('evidence_quality','includes_cost_benefit_analysis')),
    ('evidence_score', ('evidence_quality','score')),
    ('threatens_litigation', ('legal_sophistication','threatens_litigation')),
    ('makes_procedural_objection', ('legal_sophistication','makes_procedural_objection')),
    ('first_person_experiential', ('experiential_content','first_person_experiential')),
    ('narrative_personal', ('experiential_content','narrative_personal_experience')),
    ('technical_case_study', ('experiential_content','technical_case_study')),
    ('primary_frame', ('framing','primary_frame')),
    ('invokes_agency_mandate', ('framing','explicitly_invokes_agency_mandate')),
    ('uses_agency_vocab', ('framing','uses_agency_technical_vocabulary')),
    ('addresses_specific_provisions', ('engagement_depth','addresses_specific_provisions')),
    ('proposes_alternative', ('engagement_depth','proposes_alternative_language')),
    ('n_issues', ('engagement_depth','number_of_distinct_issues')),
    ('tone', ('engagement_depth','tone')),
    ('requested_action', ('engagement_depth','requested_action')),
    ('claims_regulatory_cost', ('impact_claims','claims_direct_regulatory_cost')),
    ('claims_benefit', ('impact_claims','claims_direct_benefit')),
    ('mentions_small_biz', ('impact_claims','mentions_small_business_impact')),
    ('mentions_ej', ('impact_claims','mentions_environmental_justice')),
    ('mentions_public_health', ('impact_claims','mentions_public_health_impact')),
    ('organized_campaign', ('campaign_signals','appears_organized_campaign')),
]

def main():
    OUT = BULK / "scripts/data/feature_analysis_summary.csv.gz"
    frames = []
    dirs_loaded = 0

    for agency_dir in sorted(BULK.iterdir()):
        if not agency_dir.is_dir() or agency_dir.name in ('scripts','old','data') or agency_dir.name.startswith('_'):
            continue
        for yd in sorted(agency_dir.iterdir()):
            if not yd.is_dir(): continue
            fp = yd / 'public_submission_engineered_features_v2_clusters.csv'
            if not fp.exists(): continue

            try:
                feats = pd.read_csv(fp)
                feats = feats[feats.parse_success == True].copy()
                feats['_f'] = feats['llm_features_json'].apply(parse_features)
                feats = feats[feats._f.notna()].copy()
                for col, keys in FLAT_COLS:
                    feats[col] = feats._f.apply(lambda f, k=keys: sg(f, *k))
                feats = feats.drop(columns=['_f', 'llm_features_json', 'parse_success'])
            except Exception as e:
                logger.warning("SKIP features %s: %s", yd.name, e)
                continue

            # DIME
            dp = yd / 'dime_matches_enriched.csv'
            if dp.exists():
                try:
                    dime = pd.read_csv(dp, low_memory=False)
                    want = ['Document ID','cfscore','analysis_tier','confidence_score','n_candidates']
                    have = [c for c in want if c in dime.columns]
                    if 'Document ID' in have:
                        feats = feats.merge(dime[have], on='Document ID', how='left')
                except Exception as e:
                    logger.warning("SKIP dime %s: %s", yd.name, e)

            # Claim labels
            cp = resolve(yd, 'public_submission_all_text__claims_comment_labels')
            if cp is not None:
                try:
                    cl = pd.read_csv(cp, low_memory=False)
                    cl['_doc_id'] = cl['doc_id'].apply(extract_doc_id)
                    agg = cl.groupby('_doc_id').agg(
                        any_claim_matched=('matched', lambda s: (s=='yes').any()),
                        n_claims=('matched','count'),
                        n_claims_matched=('matched', lambda s: (s=='yes').sum()),
                    ).reset_index().rename(columns={'_doc_id':'Document ID'})
                    feats = feats.merge(agg, on='Document ID', how='left')
                except Exception as e:
                    logger.warning("SKIP claims %s: %s", yd.name, e)

            feats['agency'] = agency_dir.name
            feats['dir_name'] = yd.name
            frames.append(feats)
            dirs_loaded += 1
            if dirs_loaded % 50 == 0:
                logger.info("loaded %d dirs...", dirs_loaded)

    logger.info("Total: %d dirs, concatenating...", dirs_loaded)
    data = pd.concat(frames, ignore_index=True)
    logger.info("Rows: %d, Columns: %s", len(data), list(data.columns))
    data.to_csv(OUT, index=False, compression='gzip')
    logger.info("Saved to %s (%.1f MB)", OUT, OUT.stat().st_size / 1024 / 1024)

if __name__ == "__main__":
    main()
