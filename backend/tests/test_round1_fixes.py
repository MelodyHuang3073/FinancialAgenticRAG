"""Unit tests for the 2026-09 fix round: evidence quota selection, PoT skip shapes,
company-key filtering, classifier change verbs / Foot Locker entity."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.evidence_selection import select_with_quota
from app.agent.orchestrator import FinAgentRAGOrchestrator
from app.agent.pot_reasoner import _no_calculation_path
from app.agent.question_classifier import FinanceBenchClassifier


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


def test_no_calculation_path_shapes():
    assert _no_calculation_path("which of jpm's business segments had the lowest net revenue in 2021 q1?")
    assert _no_calculation_path("by how many percentage points did pepsico raise full year guidance in respect of core eps growth?")
    assert _no_calculation_path("how did jnj's us sales growth compare to international sales growth in fy2022?")
    assert _no_calculation_path("were there any potential events that increased net income in 2019?")
    # ordinary numeric questions keep their PoT path
    assert not _no_calculation_path("what is the fy2019 cash conversion cycle for general mills?")
    assert not _no_calculation_path("was there any drop in cash & cash equivalents between fy 2023 and q2 of fy2024?")


def test_company_key():
    key = FinAgentRAGOrchestrator._company_key
    assert key("JOHNSON_JOHNSON_2022Q4_EARNINGS") == "JOHNSONJOHNSON"
    assert key("PEPSICO_2023_8K_dated-2023-05-30") == "PEPSICO"
    assert key("3M_2018_10K") == "3M"
    assert key("company") == ""


def test_classifier_change_verbs_and_foot_locker():
    clf = FinanceBenchClassifier()
    assert clf.classify("Did Pfizer grow its PPNE between FY20 and FY21?")["calc_type"] == "change"
    assert clf.classify("Was there any drop in Cash & Cash equivalents between FY 2023 and Q2 of FY2024?")["calc_type"] == "change"
    # a "which ... increase the most" selection question keeps no calc type
    assert clf.classify("In which segment did sales proportionally increase the most?")["calc_type"] == ""
    assert clf.classify("Does Foot Locker's new CEO have previous CEO experience?")["entity"] == "Foot Locker"
