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
from typing import Any, Dict, List, Optional, Set

#: Passages each sub-query is guaranteed to contribute to a window.
PER_SUBQUERY_QUOTA = 2

#: Passages the original-question sub-query is guaranteed (its whole top-k).
ORIGINAL_QUESTION_QUOTA = 5


def select_with_quota(
    items: List[Dict[str, Any]],
    limit: int,
    per_subquery: int = PER_SUBQUERY_QUOTA,
    pinned_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Return at most `limit` items, highest relevance_score first.

    Items carry the index of the sub-query that retrieved them in
    `subquery_idx`; items without it (or a pool that already fits in the
    window) behave exactly like the plain "sort by score, cut to limit".

    `pinned_ids` (item "id" values) are guaranteed a slot in the window
    before the per-subquery quota or raw-score fill runs -- for evidence
    a downstream computation (e.g. the PoT sandbox) actually used, so it
    can't rank just outside the window while still backing the answer.
    A pool that already fits within `limit` needs no pinning (everything
    is included either way).
    """
    ranked = sorted(items, key=lambda x: x.get("relevance_score") or 0, reverse=True)
    if len(ranked) <= limit:
        return ranked

    pinned: List[Dict[str, Any]] = []
    if pinned_ids:
        pinned = [item for item in ranked if item.get("id") in pinned_ids][:limit]
    pinned_obj_ids = {id(item) for item in pinned}
    remaining_pool = [item for item in ranked if id(item) not in pinned_obj_ids]
    remaining_limit = limit - len(pinned)

    # The sub-query that is the user's own question is the one query certain to
    # carry the user's wording, so every passage it retrieved keeps a slot.
    original_sqs = {i.get("subquery_idx") for i in remaining_pool if i.get("original_question_hit")}

    reserved: List[Dict[str, Any]] = []
    taken_per_subquery: Dict[Any, int] = {}
    for item in remaining_pool:  # ranked order = best-scoring first within each sub-query
        sq = item.get("subquery_idx")
        if sq is None:
            continue
        quota = ORIGINAL_QUESTION_QUOTA if sq in original_sqs else per_subquery
        if taken_per_subquery.get(sq, 0) < quota:
            taken_per_subquery[sq] = taken_per_subquery.get(sq, 0) + 1
            reserved.append(item)
    # Original-question passages first, so a crowded window never cuts them.
    reserved = (
        [i for i in reserved if i.get("subquery_idx") in original_sqs]
        + [i for i in reserved if i.get("subquery_idx") not in original_sqs]
    )[:remaining_limit]

    reserved_ids = {id(item) for item in reserved}
    rest = [item for item in remaining_pool if id(item) not in reserved_ids]
    picked = pinned + reserved + rest[: remaining_limit - len(reserved)]
    return sorted(picked, key=lambda x: x.get("relevance_score") or 0, reverse=True)
