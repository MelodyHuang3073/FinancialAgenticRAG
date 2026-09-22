"""Per-sub-query evidence quota merge (app/agent/evidence_selection.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.evidence_selection import select_with_quota


def _hit(i, score, sq):
    return {"id": i, "relevance_score": score, "subquery_idx": sq}


def test_quota_keeps_each_subquery_top_items():
    # sub-query 1 scores are low but it must still keep its own top 2
    items = [_hit(f"a{i}", 100 - i, 0) for i in range(10)] + [_hit("b0", 30, 1), _hit("b1", 20, 1), _hit("b2", 10, 1)]
    picked = select_with_quota(items, 6, per_subquery=2)
    ids = {h["id"] for h in picked}
    assert {"b0", "b1"} <= ids
    assert len(picked) == 6
    assert picked == sorted(picked, key=lambda h: h["relevance_score"], reverse=True)


def test_quota_is_plain_top_n_when_pool_fits_or_untagged():
    items = [{"id": i, "relevance_score": s} for i, s in enumerate([5, 9, 1, 7])]
    assert [h["id"] for h in select_with_quota(items, 10)] == [1, 3, 0, 2]
    assert [h["id"] for h in select_with_quota(items, 2)] == [1, 3]
