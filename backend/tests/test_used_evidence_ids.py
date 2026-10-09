"""pot_reasoner._collect_used_evidence_ids(): maps a formula's resolved
variables back to the evidence_list id(s) that actually produced them, so
callers can pin that evidence into the LLM's prompt window (see
evidence_selection.select_with_quota's pinned_ids)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.pot_reasoner import _collect_used_evidence_ids


def _evidence(n):
    return [{"id": f"ev{i}"} for i in range(n)]


def test_maps_evidence_index_to_evidence_id():
    meta = {
        "revenue": {"evidence_index": 2},
        "cost_of_revenue": {"evidence_index": 0},
    }
    ids = _collect_used_evidence_ids(meta, _evidence(5))
    assert set(ids) == {"ev2", "ev0"}


def test_free_text_and_composite_matches_have_no_index_and_are_skipped():
    # Both store evidence_index=None (free-text match, or a composite sum
    # across several rows with no single source) -- see
    # _extract_formula_guided's meta docstring.
    meta = {
        "revenue": {"evidence_index": None, "source": "free-text"},
        "cost_of_revenue": {"evidence_index": None, "source": "composite"},
    }
    assert _collect_used_evidence_ids(meta, _evidence(5)) == []


def test_missing_evidence_index_key_defaults_safely():
    # e.g. a manually-constructed meta override elsewhere in pot_reasoner
    # (the adjusted-EBIT-from-EBITDA-reconciliation override) that never
    # set this key at all.
    meta = {"ebit": {"source": "derived-from-ebitda-reconciliation", "is_approximate": True}}
    assert _collect_used_evidence_ids(meta, _evidence(5)) == []


def test_out_of_range_index_is_ignored_not_crashed_on():
    meta = {"revenue": {"evidence_index": 99}}
    assert _collect_used_evidence_ids(meta, _evidence(3)) == []


def test_duplicate_evidence_index_across_placeholders_deduplicated():
    meta = {
        "revenue_new": {"evidence_index": 1},
        "revenue_old": {"evidence_index": 1},
    }
    assert _collect_used_evidence_ids(meta, _evidence(3)) == ["ev1"]


def test_empty_meta_returns_empty_list():
    assert _collect_used_evidence_ids({}, _evidence(5)) == []
