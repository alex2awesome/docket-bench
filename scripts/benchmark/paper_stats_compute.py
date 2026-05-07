"""
Compute and distill all paper-relevant statistics into one JSON + a handful of
CSVs that the analysis notebook reads from.

Outputs (to data/derived/paper_stats/):
  paper_stats.json                       — all scalar numbers
  per_agency_volume.csv                  — comment count by agency, year
  per_year_volume.csv                    — comment count by year
  engagement_features.csv                — engagement-with-rule features (Finding 1)
  stakeholder_frame_matrix.csv           — bucket × frame % matrix (Finding 2)
  specialist_validation_by_org.csv       — labeled specialist rate by org_type (Finding 3)
  specialist_validation_by_legal_score.csv
  specialist_full_corpus_by_org.csv      — predicted specialist rate over 8M
  cluster_size_distribution.csv          — cluster size histogram
  clusters_per_provision_distribution.csv  — Match 2 YES clusters per provision (Finding 4)

Run from repo root or anywhere; all paths are absolute.
"""
from __future__ import annotations
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get('DOCKET_BASE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
BULK = BASE / 'data' / 'bulk_downloads'
DU = BULK / 'deontic_units'
FEATURES = BASE / 'data' / 'feature_analysis_summary.csv.gz'
SPECIALIST_LABELS_FULL = BASE / 'data' / 'derived' / 'specialist_labels.csv'
SPECIALIST_VALIDATION = DU / 'specialist_proxy_validation.jsonl'
MATCH2 = DU / 'match2_llm_labeled.jsonl'

OUT_DIR = BASE / 'data' / 'derived' / 'paper_stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Org-type and frame harmonization
# ---------------------------------------------------------------------------
ORG_BUCKETS = {
    'industry':   ['corporation', 'trade_association', 'chamber_of_commerce',
                   'small business', 'small_business'],
    'labor':      ['labor_union'],
    'advocacy':   ['ngo_advocacy', 'coalition'],
    'expert':     ['academic', 'professional_society'],
    'gov':        ['government', 'tribal government', 'tribal government entity'],
    'lawfirm':    ['law_firm'],
    'individual': ['individual'],
}
ORG_REV = {ot: bucket for bucket, ots in ORG_BUCKETS.items() for ot in ots}

# Map enum-drift frame values back to the 12 canonical ones from the prompt
FRAME_MAP = {
    'public_health': 'health',
    'public health': 'health',
    'moral_religious': 'religious_moral',
    'moral': 'religious_moral',
    'public_safety': 'safety',
    'national_security': 'safety',
    'consumer_protection': 'other',
    'consumer': 'other',
    'consumer_impact': 'other',
    'transparency_accountability': 'procedural',
    'animal_welfare': 'environmental',
    'human_rights': 'equity_justice',
    'individual_rights': 'equity_justice',
    'humanitarian': 'equity_justice',
    'cultural': 'other',
    'technical': 'scientific_technical',
    'professional_society': 'other',
    'privacy': 'legal_constitutional',
}
CANONICAL_FRAMES = [
    'environmental', 'health', 'equity_justice', 'economic_cost_benefit',
    'legal_constitutional', 'religious_moral', 'safety', 'scientific_technical',
    'procedural', 'educational', 'other', 'unknown',
]

ENGAGEMENT_FEATURES = [
    'addresses_specific_provisions', 'cites_statutes', 'invokes_agency_mandate',
    'proposes_alternative', 'uses_agency_vocab', 'provides_economic_data',
    'cites_peer_reviewed', 'cites_case_law', 'includes_cba',
    'threatens_litigation', 'first_person_experiential', 'narrative_personal',
    'technical_case_study', 'is_regulated_entity',
]


def to_bool(s) -> int:
    if pd.isna(s):
        return 0
    return int(str(s).strip().lower() in {'true', '1', 'yes', 't'})


# ---------------------------------------------------------------------------
# Load full feature CSV once
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    logger.info('Loading feature_analysis_summary.csv.gz ...')
    cols = (['Document ID', 'Docket ID', 'Agency ID', 'agency', 'dir_name',
             'org_type', 'primary_frame', 'requested_action',
             'cluster_uid', 'cluster_size',
             'is_form_letter', 'length_cat', 'legal_score',
             'year', 'Posted Date']
            + ENGAGEMENT_FEATURES)
    df = pd.read_csv(FEATURES, low_memory=False, usecols=cols)
    logger.info('  rows: %d', len(df))
    df['legal_score_num'] = pd.to_numeric(df['legal_score'], errors='coerce')
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    df['org_bucket'] = df['org_type'].map(ORG_REV)
    df['frame_canon'] = df['primary_frame'].map(lambda x: FRAME_MAP.get(x, x))
    return df


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------
def topline(df: pd.DataFrame) -> dict:
    out = {
        'n_comments': int(len(df)),
        'n_dockets': int(df['Docket ID'].nunique()),
        'n_agencies': int(df['agency'].nunique()),
        'n_agency_years': int(df.groupby(['agency', 'dir_name']).ngroups),
        'year_min': float(df['year_num'].min()) if df['year_num'].notna().any() else None,
        'year_max': float(df['year_num'].max()) if df['year_num'].notna().any() else None,
        'mean_comments_per_docket': float(df.groupby('Docket ID').size().mean()),
        'median_comments_per_docket': float(df.groupby('Docket ID').size().median()),
    }
    return out


def per_agency_year(df: pd.DataFrame):
    # per agency
    pa = df.groupby('agency').size().rename('n_comments').reset_index()
    pa = pa.sort_values('n_comments', ascending=False)
    pa.to_csv(OUT_DIR / 'per_agency_volume.csv', index=False)
    # per year
    py = df.groupby('year_num').size().rename('n_comments').reset_index()
    py = py.dropna()
    py['year_num'] = py['year_num'].astype(int)
    py.to_csv(OUT_DIR / 'per_year_volume.csv', index=False)
    return {'per_agency_top5': pa.head(5).to_dict('records'),
            'per_year_range': py.to_dict('records')}


def engagement_features(df: pd.DataFrame) -> dict:
    # Category labels for legend
    ANECDOTAL = {'first_person_experiential', 'narrative_personal'}
    CATEGORY = {c: ('anecdotal' if c in ANECDOTAL else 'technical')
                for c in ENGAGEMENT_FEATURES}

    rows = []
    bool_cols = {c: df[c].apply(to_bool) for c in ENGAGEMENT_FEATURES}
    for c in ENGAGEMENT_FEATURES:
        n = int(bool_cols[c].sum())
        rows.append({'feature': c, 'n': n, 'pct': 100 * n / len(df),
                     'category': CATEGORY[c]})

    # Compute UNION of "other technical" features (everything technical NOT in top 6)
    eng_sorted = sorted([(c, bool_cols[c].sum()) for c in ENGAGEMENT_FEATURES
                         if CATEGORY[c] == 'technical'], key=lambda x: -x[1])
    top_6_tech = {c for c, _ in eng_sorted[:6]}
    other_tech_cols = [c for c in ENGAGEMENT_FEATURES
                       if CATEGORY[c] == 'technical' and c not in top_6_tech]
    if other_tech_cols:
        union_mat = np.column_stack([bool_cols[c].values for c in other_tech_cols])
        n_union = int((union_mat.max(axis=1) > 0).sum())
        rows.append({
            'feature': 'technical_other_union',
            'n': n_union,
            'pct': 100 * n_union / len(df),
            'category': 'technical',
            'components': ','.join(other_tech_cols),
        })

    eng = pd.DataFrame(rows).sort_values('pct', ascending=False)
    eng.to_csv(OUT_DIR / 'engagement_features.csv', index=False)

    out = {
        'mean_legal_score': float(df['legal_score_num'].mean()),
        'legal_score_dist': df['legal_score_num'].value_counts(dropna=False).sort_index().to_dict(),
        'features': eng.set_index('feature')[['n', 'pct', 'category']].to_dict('index'),
        'top_6_technical': sorted(list(top_6_tech)),
        'other_technical_components': other_tech_cols,
    }
    return out


def stakeholder_frame_matrix(df: pd.DataFrame) -> dict:
    sub = df[df['org_bucket'].notna() & df['frame_canon'].isin(CANONICAL_FRAMES)]
    pivot = pd.crosstab(sub['org_bucket'], sub['frame_canon'], normalize='index') * 100
    # Only keep canonical columns that are present
    cols = [c for c in CANONICAL_FRAMES if c in pivot.columns]
    pivot = pivot[cols]
    pivot.to_csv(OUT_DIR / 'stakeholder_frame_matrix.csv')

    counts = pd.crosstab(sub['org_bucket'], sub['frame_canon'])
    counts.to_csv(OUT_DIR / 'stakeholder_frame_counts.csv')

    # Headline ratios (max within row, between-row max for each frame)
    headlines = {}
    for bucket in pivot.index:
        top = pivot.loc[bucket].sort_values(ascending=False).head(3)
        headlines[bucket] = {f: float(v) for f, v in top.items()}
    return {
        'n_used': int(len(sub)),
        'matrix_shape': list(pivot.shape),
        'top_frame_per_bucket': headlines,
    }


def specialist_validation(df_features: pd.DataFrame) -> dict:
    """Concordance between LLM judgments (5,987) and silver labels (org_type, legal_score)."""
    if not SPECIALIST_VALIDATION.exists():
        logger.warning('  specialist_proxy_validation.jsonl missing — skipping')
        return {}

    rows = [json.loads(l) for l in open(SPECIALIST_VALIDATION)]
    rows = [r for r in rows if r.get('parse_ok') and r.get('is_specialist') is not None]
    logger.info('  specialist validation rows: %d', len(rows))

    by_ot = defaultdict(lambda: {'spec': 0, 'tot': 0})
    by_ls = defaultdict(lambda: {'spec': 0, 'tot': 0})
    for r in rows:
        ot = r.get('org_type', 'unknown')
        ls = int(float(r.get('legal_score') or 0))
        is_spec = bool(r.get('is_specialist'))
        by_ot[ot]['tot'] += 1
        by_ot[ot]['spec'] += int(is_spec)
        by_ls[ls]['tot'] += 1
        by_ls[ls]['spec'] += int(is_spec)

    ot_rows = [{'org_type': k, 'n': v['tot'], 'specialist_rate': v['spec'] / max(v['tot'], 1)}
               for k, v in by_ot.items()]
    ot_df = pd.DataFrame(ot_rows).sort_values('specialist_rate', ascending=False)
    ot_df.to_csv(OUT_DIR / 'specialist_validation_by_org.csv', index=False)

    ls_rows = [{'legal_score': k, 'n': v['tot'], 'specialist_rate': v['spec'] / max(v['tot'], 1)}
               for k, v in sorted(by_ls.items())]
    ls_df = pd.DataFrame(ls_rows)
    ls_df.to_csv(OUT_DIR / 'specialist_validation_by_legal_score.csv', index=False)

    return {
        'n_validation_labels': len(rows),
        'by_org_type': ot_df.to_dict('records'),
        'by_legal_score': ls_df.to_dict('records'),
    }


def specialist_full_corpus(df_features: pd.DataFrame) -> dict:
    if not SPECIALIST_LABELS_FULL.exists():
        logger.warning('  specialist_labels.csv missing — skipping')
        return {}
    logger.info('Loading specialist_labels.csv ...')
    labels = pd.read_csv(SPECIALIST_LABELS_FULL)
    m = labels.merge(df_features[['Document ID', 'org_type', 'length_cat', 'org_bucket']],
                     on='Document ID', how='inner')
    logger.info('  joined: %d', len(m))

    # By org_type
    by_org = m.groupby('org_type').agg(
        n=('predicted_specialist', 'count'),
        n_specialist=('predicted_specialist', 'sum'),
        spec_rate=('predicted_specialist', 'mean'),
        mean_p=('p_specialist', 'mean'),
    ).reset_index().sort_values('spec_rate', ascending=False)
    by_org.to_csv(OUT_DIR / 'specialist_full_corpus_by_org.csv', index=False)

    # By bucket (with % of all specialist content)
    mb = m.dropna(subset=['org_bucket'])
    by_bucket = mb.groupby('org_bucket').agg(
        n=('predicted_specialist', 'count'),
        n_specialist=('predicted_specialist', 'sum'),
        spec_rate=('predicted_specialist', 'mean'),
    ).reset_index()
    total_spec = by_bucket['n_specialist'].sum()
    by_bucket['pct_of_all_specialist'] = 100 * by_bucket['n_specialist'] / max(total_spec, 1)
    by_bucket = by_bucket.sort_values('n_specialist', ascending=False)
    by_bucket.to_csv(OUT_DIR / 'specialist_full_corpus_by_bucket.csv', index=False)

    # By length category — monotonic gradient
    LEN_ORDER = ['very_short', 'short', 'medium', 'long', 'very_long']
    ml = m[m['length_cat'].isin(LEN_ORDER)]
    by_len = ml.groupby('length_cat').agg(
        n=('predicted_specialist', 'count'),
        n_specialist=('predicted_specialist', 'sum'),
        spec_rate=('predicted_specialist', 'mean'),
    ).reindex(LEN_ORDER).reset_index()
    by_len['pct_of_corpus'] = 100 * by_len['n'] / by_len['n'].sum()
    by_len.to_csv(OUT_DIR / 'specialist_by_length.csv', index=False)

    # Length × bucket cross-tab (rate)
    if len(mb) > 0:
        mb_len = mb[mb['length_cat'].isin(LEN_ORDER)].copy()
        mb_len['len_short'] = mb_len['length_cat'].map(
            lambda x: 'very_short_short' if x in ['very_short', 'short']
            else 'medium_long_verylong')
        xt = mb_len.groupby(['org_bucket', 'len_short'])['predicted_specialist'].mean().unstack() * 100
        xt = xt.round(1)
        xt.to_csv(OUT_DIR / 'specialist_length_x_bucket.csv')

    individuals = m[m['org_type'] == 'individual']
    medium_long = m[m['length_cat'].isin(['medium', 'long', 'very_long'])]
    return {
        'n_scored_corpus': int(len(m)),
        'overall_specialist_rate': float(m['predicted_specialist'].mean()),
        'overall_specialist_count': int(m['predicted_specialist'].sum()),
        'individual_n': int(len(individuals)),
        'individual_specialist_rate': float(individuals['predicted_specialist'].mean()),
        'individual_specialist_count': int(individuals['predicted_specialist'].sum()),
        'medium_long_n': int(len(medium_long)),
        'medium_long_specialist_rate': float(medium_long['predicted_specialist'].mean()),
        'medium_long_specialist_count': int(medium_long['predicted_specialist'].sum()),
        'by_bucket_pct_of_all_specialist': by_bucket.set_index('org_bucket')['pct_of_all_specialist'].to_dict(),
        'by_length': by_len.set_index('length_cat')['spec_rate'].to_dict(),
    }


def cluster_distribution(df: pd.DataFrame) -> dict:
    """Cluster size distribution from cluster_uid + cluster_size in feature CSV."""
    cluster_df = df[['cluster_uid', 'cluster_size']].drop_duplicates('cluster_uid')
    cluster_df['cluster_size'] = pd.to_numeric(cluster_df['cluster_size'], errors='coerce')
    cluster_df = cluster_df.dropna(subset=['cluster_size'])
    cluster_df['cluster_size'] = cluster_df['cluster_size'].astype(int)

    # Bin counts: 1, 2-5, 6-50, 51-500, 501-5000, 5000+
    bins = [0, 1, 5, 50, 500, 5000, 10**9]
    labels = ['1 (singleton)', '2–5', '6–50', '51–500', '501–5,000', '5,001+']
    cluster_df['size_bucket'] = pd.cut(cluster_df['cluster_size'], bins=bins, labels=labels)
    bucket_counts = cluster_df['size_bucket'].value_counts().reindex(labels)
    bucket_counts.to_csv(OUT_DIR / 'cluster_size_distribution.csv', header=['n_clusters'])

    # Comment volume in each bucket
    cluster_df['comment_volume'] = cluster_df['cluster_size']
    bucket_volume = cluster_df.groupby('size_bucket', observed=True)['comment_volume'].sum().reindex(labels)
    pd.DataFrame({
        'n_clusters': bucket_counts,
        'comment_volume': bucket_volume,
    }).to_csv(OUT_DIR / 'cluster_size_distribution.csv')

    n_total_clusters = int(len(cluster_df))
    n_singleton = int((cluster_df['cluster_size'] == 1).sum())
    n_volume_total = int(cluster_df['cluster_size'].sum())
    n_volume_singleton = int(cluster_df[cluster_df['cluster_size'] == 1]['cluster_size'].sum())

    return {
        'n_total_clusters': n_total_clusters,
        'n_singleton_clusters': n_singleton,
        'pct_singleton_clusters': 100 * n_singleton / max(n_total_clusters, 1),
        'n_total_comments_in_clusters': n_volume_total,
        'pct_volume_in_singletons': 100 * n_volume_singleton / max(n_volume_total, 1),
    }


def clusters_per_provision() -> dict:
    """How many distinct comment clusters touch each proposed provision (match2 YES)."""
    if not MATCH2.exists():
        logger.warning('  match2_llm_labeled.jsonl missing — skipping')
        return {}
    logger.info('Scanning match2_llm_labeled.jsonl ...')
    clusters_per_unit = defaultdict(set)
    n_yes = 0
    for line in open(MATCH2):
        if '"llm_label": 1' not in line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('llm_label') != 1:
            continue
        n_yes += 1
        cu = r.get('cluster_uid')
        uid = r.get('unit_id')
        if cu and uid:
            clusters_per_unit[uid].add(cu)
    sizes = np.array([len(v) for v in clusters_per_unit.values()])
    pd.DataFrame({
        'unit_id': list(clusters_per_unit.keys()),
        'n_clusters': sizes,
    }).to_csv(OUT_DIR / 'clusters_per_provision.csv', index=False)

    # Also save a binned histogram for fast plotting
    bins = [0, 1, 2, 5, 10, 30, 100, 10**9]
    labels = ['1', '2', '3–5', '6–10', '11–30', '31–100', '100+']
    binned = pd.cut(sizes, bins=bins, labels=labels)
    pd.Series(binned).value_counts().reindex(labels).to_csv(
        OUT_DIR / 'clusters_per_provision_distribution.csv', header=['n_provisions'])

    return {
        'n_match2_yes_pairs': int(n_yes),
        'n_provisions_with_match': int(len(sizes)),
        'mean_clusters_per_provision': float(sizes.mean()),
        'median_clusters_per_provision': float(np.median(sizes)),
        'p95_clusters_per_provision': float(np.percentile(sizes, 95)),
        'max_clusters_per_provision': int(sizes.max()),
    }


def requested_action(df: pd.DataFrame) -> dict:
    """What are commenters asking for? Top-level distribution + by stakeholder bucket."""
    CANONICAL = ['oppose_and_withdraw', 'modify', 'support_as_proposed',
                 'none_stated', 'delay_or_extend', 'request_information']
    d = df.copy()
    d['action_canon'] = d['requested_action'].where(
        d['requested_action'].isin(CANONICAL), 'other')
    # overall
    overall = d['action_canon'].value_counts(normalize=False)
    overall_pct = d['action_canon'].value_counts(normalize=True) * 100
    out_overall = pd.DataFrame({'n': overall, 'pct': overall_pct}).reindex(
        CANONICAL + ['other']).fillna(0)
    out_overall.to_csv(OUT_DIR / 'requested_action_overall.csv')

    # by bucket
    sub = d.dropna(subset=['org_bucket'])
    pivot = pd.crosstab(sub['org_bucket'], sub['action_canon'], normalize='index') * 100
    keep_cols = [c for c in CANONICAL + ['other'] if c in pivot.columns]
    pivot = pivot[keep_cols]
    pivot.to_csv(OUT_DIR / 'requested_action_by_bucket.csv')

    return {
        'n_used': int(len(d)),
        'overall_pct': out_overall['pct'].to_dict(),
        'overall_n': out_overall['n'].astype(int).to_dict(),
        'top_action_per_bucket': {
            b: {a: float(v) for a, v in pivot.loc[b].sort_values(ascending=False).head(3).items()}
            for b in pivot.index
        },
    }


def matching_pipeline_stats() -> dict:
    """Coverage of the matching pipeline."""
    out = {}
    # Match 2: total pairs, yes rate
    if MATCH2.exists():
        n_total = 0
        n_yes = 0
        for line in open(MATCH2):
            n_total += 1
            if '"llm_label": 1' in line:
                n_yes += 1
        out['match2_total_pairs'] = n_total
        out['match2_yes_pairs'] = n_yes
        out['match2_yes_rate'] = n_yes / max(n_total, 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_features()

    stats = {}

    logger.info('Computing topline ...')
    stats['topline'] = topline(df)

    logger.info('Computing per-agency / per-year ...')
    stats['per_agency_year'] = per_agency_year(df)

    logger.info('Computing engagement features (Finding 1) ...')
    stats['engagement'] = engagement_features(df)

    logger.info('Computing stakeholder × frame matrix (Finding 2) ...')
    stats['stakeholder_frame'] = stakeholder_frame_matrix(df)

    logger.info('Computing specialist validation (Finding 3) ...')
    stats['specialist_validation'] = specialist_validation(df)
    stats['specialist_full_corpus'] = specialist_full_corpus(df)

    logger.info('Computing cluster distribution ...')
    stats['cluster_distribution'] = cluster_distribution(df)

    logger.info('Computing clusters per provision (Finding 4) ...')
    stats['provision_targeting'] = clusters_per_provision()

    logger.info('Computing requested_action distribution ...')
    stats['requested_action'] = requested_action(df)

    logger.info('Computing matching-pipeline stats ...')
    stats['matching_pipeline'] = matching_pipeline_stats()

    out_path = OUT_DIR / 'paper_stats.json'
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info('Wrote %s (%d top-level keys)', out_path, len(stats))


if __name__ == '__main__':
    main()
