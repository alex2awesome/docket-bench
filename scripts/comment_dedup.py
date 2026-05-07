"""
MinHash LSH pipeline for clustering near-duplicate regulatory comments.

Pipeline:
1. Preprocess comments (strip metadata, normalize)
2. Shingle into word n-grams
3. MinHash + LSH to find near-duplicate clusters
4. Sample representatives from each cluster for downstream claim extraction

Requirements:
    pip install datasketch
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

logger = logging.getLogger(__name__)

MIN_TOKENS = 80  # roughly 100 words


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_comment(text: str) -> str:
    """
    Clean a raw comment extracted from a regulations.gov PDF.
    Strips common header/footer patterns, normalizes whitespace,
    and lowercases for shingling.
    
    Customize the patterns here based on what your PDF parser produces.
    """
    # Remove common PDF extraction artifacts
    text = re.sub(r'\f', ' ', text)                    # form feeds
    text = re.sub(r'-\n', '', text)                     # hyphenated line breaks

    # Strip parsing markers
    text = re.sub(r'<<COMMENT \d+>>', '', text)
    text = re.sub(r'<<PAGE \d+>>', '', text)
    text = re.sub(r'^\s*(true|false)\s*\d+\s*', '', text, flags=re.IGNORECASE)

    # Strip "See Attached" stub comments
    text = re.sub(
        r'^\s*see\s+attached\s*$',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Strip full address blocks (multiline: name, street, city/state/zip)
    text = re.sub(r'(?m)^.*\d+\s+\d+\w*\s+Street.*$', '', text)
    text = re.sub(r'(?m)^.*,\s*(VA|DC|NM|WV)\s+\d{5}.*$', '', text)

    # Strip date lines
    text = re.sub(
        r'(?m)^.*(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s+\d{4}.*$',
        '',
        text,
    )

    # Strip VIA/RE/Docket lines
    text = re.sub(r'(?im)^VIA\s+.*$', '', text)
    text = re.sub(r'(?im)^RE:.*$', '', text)
    text = re.sub(r'(?im)^Docket.*$', '', text)

    # Strip Dear/Sincerely blocks
    text = re.sub(r'(?i)dear\s+[\w\s.,/]+[,:]\s*', '', text)
    text = re.sub(
        r'(?is)(sincerely|respectfully|regards|thank you)[\s,].*$',
        '',
        text,
    )

    # Strip common agency references that appear in every comment
    text = re.sub(
        r"(?i)mine safety and health administration['’]?s?\s*"
        r"\(MSHA( or Administration)?\)",
        "",
        text,
    )

    # Strip From/Sent/To email headers
    text = re.sub(
        r'(?im)^(From|Sent|To|Cc|Subject|Attachments):\s*.*$',
        '',
        text,
    )

    # Strip URLs and citations
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'doi:\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s+Fed\.?\s*Reg\.?\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s+C\.?F\.?R\.?\s*§?\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\w+\s+\d{1,2},?\s+\d{4}\)', '', text)
    
    # Strip typical comment metadata lines (customize these)
    # e.g., "Comment from John Doe, received 03/15/2024"
    text = re.sub(
        r'(?i)^(comment\s+(from|by|submitted).*?(received|dated).*?\d{4})\s*',
        '', text
    )
    
    # Normalize whitespace
    text = re.sub(r'\n', ' ', text)                     # remaining newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()


# ---------------------------------------------------------------------------
# 2. Shingling
# ---------------------------------------------------------------------------

def shingle(text: str, k: int = 5) -> set[tuple[str, ...]]:
    """
    Generate word k-gram shingles from text.
    
    Args:
        text: Preprocessed (lowercased) comment text.
        k: Shingle size. 5 is good for near-duplicate detection.
           Use 3 for shorter comments or looser matching.
    
    Returns:
        Set of k-word tuples.
    """
    tokens = text.split()
    if len(tokens) < k:
        # For very short comments, fall back to individual tokens
        return set(tuple(tokens[i:i+max(1, len(tokens))]) 
                   for i in range(len(tokens)))
    return set(tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1))


# ---------------------------------------------------------------------------
# 3. Shingle filtering
# ---------------------------------------------------------------------------

def filter_common_shingles(
    shingle_sets: dict[str, set[tuple[str, ...]]],
    max_doc_freq: float = 0.2,
) -> dict[str, set[tuple[str, ...]]]:
    """Remove shingles that appear in too many documents."""
    n_docs = len(shingle_sets)
    if n_docs < 2:
        return shingle_sets

    threshold = max(1, int(n_docs * max_doc_freq))
    logger.info("[filter_common_shingles] counting doc freq across %d docs (threshold=%d) ...", n_docs, threshold)
    t0 = time.time()

    # Count document frequency of each shingle
    doc_freq = Counter()
    for shingles in tqdm(shingle_sets.values(), desc="count shingle doc_freq"):
        for s in shingles:
            doc_freq[s] += 1

    logger.info("[filter_common_shingles] counted %d unique shingles (%.1fs)", len(doc_freq), time.time() - t0)

    # Filter
    common = {s for s, count in doc_freq.items() if count > threshold}
    logger.info("[filter_common_shingles] removing %d common shingles (>%d docs) ...", len(common), threshold)
    t0 = time.time()

    result = {
        doc_id: shingles - common
        for doc_id, shingles in tqdm(shingle_sets.items(), desc="filter shingles")
    }
    logger.info("[filter_common_shingles] done (%.1fs)", time.time() - t0)
    return result


# ---------------------------------------------------------------------------
# 4. MinHash creation
# ---------------------------------------------------------------------------

def create_minhash(shingle_set: set[tuple[str, ...]], num_perm: int = 128) -> MinHash:
    """
    Create a MinHash signature from a set of shingles.
    """
    m = MinHash(num_perm=num_perm)
    for s in shingle_set:
        # Join the tuple back to a string for hashing
        m.update(' '.join(s).encode('utf8'))
    return m


# ---------------------------------------------------------------------------
# 5. Cluster dataclass
# ---------------------------------------------------------------------------

@dataclass
class CommentCluster:
    """A cluster of near-duplicate comments."""
    cluster_id: int
    member_ids: list[str] = field(default_factory=list)
    representative_id: Optional[str] = None   # chosen representative
    size: int = 0

    def __repr__(self):
        return (f"Cluster(id={self.cluster_id}, size={self.size}, "
                f"representative={self.representative_id})")


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

class CommentDeduplicator:
    """
    Clusters regulatory comments by near-duplicate text similarity
    using MinHash LSH.
    
    Usage:
        dedup = CommentDeduplicator(threshold=0.8, shingle_k=5, num_perm=128)
        
        # Add comments (id, raw text)
        for comment in comments:
            dedup.add_comment(comment['id'], comment['text'])
        
        # Run clustering
        clusters = dedup.cluster()
        
        # Get representatives for claim extraction
        reps = dedup.get_representatives(sample_n=3)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        shingle_k: int = 5,
        num_perm: int = 128,
        max_doc_freq: float | None = 0.2,
    ):
        """
        Args:
            threshold: Jaccard similarity threshold for LSH.
                0.8+ = strict (catches obvious form letters)
                0.5-0.7 = loose (catches paraphrased variants)
            shingle_k: Word n-gram size for shingling.
            num_perm: Number of permutations for MinHash accuracy.
        """
        self.threshold = threshold
        self.shingle_k = shingle_k
        self.num_perm = num_perm
        self.max_doc_freq = max_doc_freq

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        self.comments: dict[str, str] = {}           # id -> raw text
        self.preprocessed: dict[str, str] = {}        # id -> cleaned text
        self.minhashes: dict[str, MinHash] = {}       # id -> MinHash
        self.clusters: list[CommentCluster] = []
        self._singletons: list[str] = []              # unclustered comment ids

    def add_comment(self, comment_id: str, raw_text: str) -> None:
        """Add a single comment to the index."""
        self.comments[comment_id] = raw_text
        
        cleaned = preprocess_comment(raw_text)
        self.preprocessed[comment_id] = cleaned

        if len(cleaned.split()) < MIN_TOKENS:
            if comment_id not in self._singletons:
                self._singletons.append(comment_id)
            return
        
        shingles = shingle(cleaned, k=self.shingle_k)
        
        if not shingles:
            # Empty after preprocessing — skip
            return
        
        mh = create_minhash(shingles, self.num_perm)
        self.minhashes[comment_id] = mh
        
        try:
            self.lsh.insert(comment_id, mh)
        except ValueError:
            # Duplicate key — already inserted
            pass

    def add_comments(self, comments: list[dict]) -> None:
        """
        Batch add comments.

        Args:
            comments: List of dicts with 'id' and 'text' keys.
        """
        if not self.max_doc_freq:
            for c in tqdm(comments, desc="add_comments (no doc_freq)"):
                self.add_comment(c['id'], c['text'])
            return

        n = len(comments)
        logger.info("[add_comments] Phase 1/4: preprocessing + shingling %d comments ...", n)
        t0 = time.time()
        shingle_sets: dict[str, set[tuple[str, ...]]] = {}
        for c in tqdm(comments, desc="preprocess+shingle"):
            comment_id = c['id']
            raw_text = c['text']

            self.comments[comment_id] = raw_text

            cleaned = preprocess_comment(raw_text)
            self.preprocessed[comment_id] = cleaned

            if len(cleaned.split()) < MIN_TOKENS:
                if comment_id not in self._singletons:
                    self._singletons.append(comment_id)
                continue

            shingles = shingle(cleaned, k=self.shingle_k)
            shingle_sets[comment_id] = shingles

        logger.info(
            "[add_comments] Phase 1/4 done: %d shingle sets, %d singletons (%.1fs)",
            len(shingle_sets), len(self._singletons), time.time() - t0,
        )

        logger.info("[add_comments] Phase 2/4: filtering common shingles (max_doc_freq=%.2f) ...", self.max_doc_freq)
        t0 = time.time()
        shingle_sets = filter_common_shingles(
            shingle_sets,
            max_doc_freq=self.max_doc_freq,
        )
        logger.info("[add_comments] Phase 2/4 done (%.1fs)", time.time() - t0)

        logger.info("[add_comments] Phase 3/4: creating MinHash signatures for %d docs ...", len(shingle_sets))
        t0 = time.time()
        empty_after_filter = 0
        for comment_id, shingles in tqdm(shingle_sets.items(), desc="minhash+lsh"):
            if not shingles:
                empty_after_filter += 1
                continue

            mh = create_minhash(shingles, self.num_perm)
            self.minhashes[comment_id] = mh

            try:
                self.lsh.insert(comment_id, mh)
            except ValueError:
                # Duplicate key — already inserted
                pass

        logger.info(
            "[add_comments] Phase 3/4 done: %d minhashes created, %d empty after filter (%.1fs)",
            len(self.minhashes), empty_after_filter, time.time() - t0,
        )

    def cluster(self) -> list[CommentCluster]:
        """
        Run clustering via connected components on LSH neighbors.

        Rather than just grouping direct LSH hits, this finds connected
        components — so if A~B and B~C, all three end up in the same
        cluster even if A and C aren't directly similar. This handles
        gradual template drift across a campaign.

        Returns:
            List of CommentCluster objects.
        """
        all_ids = list(self.minhashes.keys())
        logger.info("[cluster] Phase 4a/5: querying LSH for %d documents ...", len(all_ids))
        t0 = time.time()

        # Union-Find for efficiency
        parent = {cid: cid for cid in all_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Query LSH for each comment and union with neighbors
        n_unions = 0
        for cid in tqdm(all_ids, desc="LSH query + union-find"):
            neighbors = self.lsh.query(self.minhashes[cid])
            for neighbor_id in neighbors:
                if neighbor_id != cid:
                    union(cid, neighbor_id)
                    n_unions += 1

        logger.info(
            "[cluster] Phase 4a/5 done: %d union operations (%.1fs)",
            n_unions, time.time() - t0,
        )

        # Group by root
        logger.info("[cluster] Phase 4b/5: grouping connected components ...")
        t0 = time.time()
        groups = defaultdict(list)
        for cid in all_ids:
            groups[find(cid)].append(cid)

        # Build cluster objects
        short_text_singletons = [
            sid for sid in self._singletons if sid not in self.minhashes
        ]
        self.clusters = []
        self._singletons = list(short_text_singletons)
        cluster_id = 0

        for root, members in groups.items():
            if len(members) == 1:
                if members[0] not in self._singletons:
                    self._singletons.append(members[0])
            else:
                cluster = CommentCluster(
                    cluster_id=cluster_id,
                    member_ids=members,
                    size=len(members),
                )
                cluster.representative_id = max(
                    members, key=lambda cid: len(self.preprocessed.get(cid, ''))
                )
                self.clusters.append(cluster)
                cluster_id += 1

        logger.info(
            "[cluster] Phase 4b/5 done: %d multi-doc clusters, %d singletons (%.1fs)",
            len(self.clusters), len(self._singletons), time.time() - t0,
        )

        logger.info("[cluster] Phase 5/5: validating clusters ...")
        t0 = time.time()
        self.validate_clusters(min_similarity=0.5)
        logger.info(
            "[cluster] Phase 5/5 done: %d clusters after validation (%.1fs)",
            len(self.clusters), time.time() - t0,
        )

        return self.clusters

    def validate_clusters(self, min_similarity: float = 0.7) -> None:
        total_members = sum(c.size for c in self.clusters)
        logger.info(
            "[validate] %d clusters, %d total members to validate",
            len(self.clusters), total_members,
        )
        total_ejected = 0
        for cluster in tqdm(self.clusters, desc="validate clusters"):
            rep_mh = self.minhashes[cluster.representative_id]
            validated = [cluster.representative_id]
            ejected = []
            for mid in cluster.member_ids:
                if mid == cluster.representative_id:
                    continue
                if self.minhashes[mid].jaccard(rep_mh) >= min_similarity:
                    validated.append(mid)
                else:
                    ejected.append(mid)
            cluster.member_ids = validated
            cluster.size = len(validated)
            total_ejected += len(ejected)
            for mid in ejected:
                if mid not in self._singletons:
                    self._singletons.append(mid)
        logger.info("[validate] ejected %d members across all clusters", total_ejected)

    def get_representatives(
        self, sample_n: int = 3
    ) -> list[dict]:
        """
        Get representative comments for downstream claim extraction.
        
        For each cluster, returns up to `sample_n` members (including
        the primary representative). Also returns all singletons.
        
        Args:
            sample_n: Number of samples per cluster. For tight clusters
                (threshold >= 0.8), 1-3 is usually sufficient. For looser
                clusters, consider 5-10.
        
        Returns:
            List of dicts:
                {
                    'id': comment_id,
                    'text': raw_text,
                    'cluster_id': int or None (None = singleton),
                    'cluster_size': int,
                    'is_representative': bool
                }
        """
        results = []

        for cluster in self.clusters:
            # Always include the primary representative
            rep_id = cluster.representative_id
            results.append({
                'id': rep_id,
                'text': self.comments[rep_id],
                'cluster_id': cluster.cluster_id,
                'cluster_size': cluster.size,
                'is_representative': True,
            })

            # Sample additional members (skip the representative)
            others = [m for m in cluster.member_ids if m != rep_id]
            import random
            sampled = random.sample(others, min(sample_n - 1, len(others)))
            for sid in sampled:
                results.append({
                    'id': sid,
                    'text': self.comments[sid],
                    'cluster_id': cluster.cluster_id,
                    'cluster_size': cluster.size,
                    'is_representative': False,
                })

        # All singletons go through claim extraction individually
        for sid in self._singletons:
            results.append({
                'id': sid,
                'text': self.comments[sid],
                'cluster_id': None,
                'cluster_size': 1,
                'is_representative': True,
            })

        return results

    def pairwise_similarity(self, id_a: str, id_b: str) -> float:
        """Compute exact Jaccard similarity between two comments (for debugging)."""
        return self.minhashes[id_a].jaccard(self.minhashes[id_b])

    def summary(self) -> dict:
        """Return a summary of clustering results."""
        cluster_sizes = [c.size for c in self.clusters]
        return {
            'total_comments': len(self.comments),
            'num_clusters': len(self.clusters),
            'num_singletons': len(self._singletons),
            'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': (
                sorted(cluster_sizes)[len(cluster_sizes) // 2]
                if cluster_sizes else 0
            ),
            'comments_in_clusters': sum(cluster_sizes),
            'reduction_ratio': (
                1 - (len(self.clusters) + len(self._singletons)) 
                / len(self.comments)
                if self.comments else 0
            ),
        }


# ---------------------------------------------------------------------------
# 7. Example usage / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Simulated regulatory comments
    comments = [
        # --- Campaign 1: near-identical form letters ---
        {
            'id': 'c001',
            'text': (
                'Dear Administrator, I strongly oppose the proposed rule '
                'regarding emissions standards for power plants. This rule '
                'would devastate our local economy and lead to significant '
                'job losses in the coal industry. I urge you to withdraw '
                'this proposal immediately. Sincerely, John Smith'
            ),
        },
        {
            'id': 'c002',
            'text': (
                'Dear Administrator, I strongly oppose the proposed rule '
                'regarding emissions standards for power plants. This rule '
                'would devastate our local economy and lead to significant '
                'job losses in the coal industry. I urge you to withdraw '
                'this proposal immediately. Sincerely, Jane Doe'
            ),
        },
        {
            'id': 'c003',
            'text': (
                'Dear Administrator, I strongly oppose the proposed rule '
                'regarding emissions standards for power plants. This rule '
                'would devastate our local economy and lead to significant '
                'job losses in the coal industry. As a miner for 20 years, '
                'I have seen regulations destroy communities. I urge you to '
                'withdraw this proposal immediately. Sincerely, Bob Johnson'
            ),
        },
        # --- Campaign 2: different template ---
        {
            'id': 'c004',
            'text': (
                'To whom it may concern, as a healthcare professional I am '
                'writing to express my support for the proposed air quality '
                'standards. The scientific evidence clearly shows that '
                'particulate matter causes respiratory disease. Stronger '
                'regulations will save lives and reduce healthcare costs. '
                'Please finalize this rule as proposed.'
            ),
        },
        {
            'id': 'c005',
            'text': (
                'To whom it may concern, as a healthcare professional I am '
                'writing to express my support for the proposed air quality '
                'standards. The scientific evidence clearly shows that '
                'particulate matter causes respiratory disease. Stronger '
                'regulations will save lives and reduce healthcare costs. '
                'Please finalize this rule as proposed.'
            ),
        },
        # --- Unique / substantive comment ---
        {
            'id': 'c006',
            'text': (
                'We submit these comments on behalf of the Environmental '
                'Defense Coalition regarding Docket No. EPA-2024-0042. '
                'Section 3.2 of the proposed rule fails to account for '
                'cumulative impacts on environmental justice communities. '
                'We recommend the agency conduct a supplemental analysis '
                'using the EJSCREEN tool and establish a 50-mile buffer '
                'zone around designated EJ communities. We attach our '
                'technical report as Appendix A.'
            ),
        },
    ]

    # Run the pipeline
    dedup = CommentDeduplicator(threshold=0.8, shingle_k=5, num_perm=128)
    dedup.add_comments(comments)
    clusters = dedup.cluster()

    # Print results
    print("=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)
    print()
    
    summary = dedup.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    for cluster in clusters:
        print(f"Cluster {cluster.cluster_id} ({cluster.size} comments):")
        print(f"  Representative: {cluster.representative_id}")
        print(f"  Members: {cluster.member_ids}")
        # Show pairwise similarities within cluster
        for i, a in enumerate(cluster.member_ids):
            for b in cluster.member_ids[i+1:]:
                sim = dedup.pairwise_similarity(a, b)
                print(f"  Jaccard({a}, {b}) = {sim:.3f}")
        print()

    print(f"Singletons ({len(dedup._singletons)}): {dedup._singletons}")
    print()

    # Get representatives for claim extraction
    reps = dedup.get_representatives(sample_n=3)
    print("=" * 60)
    print("REPRESENTATIVES FOR CLAIM EXTRACTION")
    print("=" * 60)
    for r in reps:
        label = "REP" if r['is_representative'] else "sample"
        cluster_label = f"cluster {r['cluster_id']}" if r['cluster_id'] is not None else "singleton"
        print(f"  [{label}] {r['id']} ({cluster_label}, size={r['cluster_size']})")
