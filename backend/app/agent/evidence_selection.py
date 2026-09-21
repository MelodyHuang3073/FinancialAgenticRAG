"""
Merging the evidence of several sub-queries into the fixed-size windows the
pipeline hands on (top CONTEXT_CHUNK_LIMIT to the PoT sandbox, top
EVIDENCE_PROMPT_CAP to the LLM prompt).

BM25 scores depend on the query text, so a score from one sub-query is not
comparable with a score from another (e.g. a passage that is the best match
for its own sub-query can score 30 while an average match for a different
sub-query scores 100+). Ranking the merged pool by raw score alone therefore
lets one sub-query crowd out the others. Every sub-query keeps its own top
`per_subquery` passages (ranked by the score of THAT query, where scores are
comparable); the remaining slots are filled by raw score as before.
"""
from typing import Any, Dict, List

#: Passages each sub-query is guaranteed to contribute to a window.
PER_SUBQUERY_QUOTA = 2


def select_with_quota(
    items: List[Dict[str, Any]],
    limit: int,
    per_subquery: int = PER_SUBQUERY_QUOTA,
) -> List[Dict[str, Any]]:
    """Return at most `limit` items, highest relevance_score first.

    Items carry the index of the sub-query that retrieved them in
    `subquery_idx`; items without it (or a pool that already fits in the
    window) behave exactly like the plain "sort by score, cut to limit".
    """
    ranked = sorted(items, key=lambda x: x.get("relevance_score") or 0, reverse=True)
    if len(ranked) <= limit:
        return ranked

    reserved: List[Dict[str, Any]] = []
    taken_per_subquery: Dict[Any, int] = {}
    for item in ranked:  # ranked order = best-scoring first within each sub-query
        sq = item.get("subquery_idx")
        if sq is None:
            continue
        if taken_per_subquery.get(sq, 0) < per_subquery:
            taken_per_subquery[sq] = taken_per_subquery.get(sq, 0) + 1
            reserved.append(item)
    reserved = reserved[:limit]  # already in score order

    reserved_ids = {id(item) for item in reserved}
    rest = [item for item in ranked if id(item) not in reserved_ids]
    picked = reserved + rest[: limit - len(reserved)]
    return sorted(picked, key=lambda x: x.get("relevance_score") or 0, reverse=True)
