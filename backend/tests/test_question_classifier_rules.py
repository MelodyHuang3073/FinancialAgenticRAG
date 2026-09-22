"""FinanceBenchClassifier rules: change verbs, Foot Locker entity, "how much ...
USD" -> NUMERIC, and the leverage/deleverage narrative bridge for "expense as
a percent of ..." questions (app/agent/question_classifier.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.question_classifier import FinanceBenchClassifier


def test_classifier_change_verbs_and_foot_locker():
    clf = FinanceBenchClassifier()
    assert clf.classify("Did Pfizer grow its PPNE between FY20 and FY21?")["calc_type"] == "change"
    assert clf.classify("Was there any drop in Cash & Cash equivalents between FY 2023 and Q2 of FY2024?")["calc_type"] == "change"
    # a "which ... increase the most" selection question keeps no calc type
    assert clf.classify("In which segment did sales proportionally increase the most?")["calc_type"] == ""
    assert clf.classify("Does Foot Locker's new CEO have previous CEO experience?")["entity"] == "Foot Locker"


def test_how_much_usd_question_is_numeric():
    c = FinanceBenchClassifier().classify(
        "How much does Pfizer expect to pay to spin off Upjohn in the future in USD million?"
    )
    assert c["answer_mode"] == "NUMERIC"


def test_cost_share_of_sales_question_gets_leverage_bridge_query():
    c = FinanceBenchClassifier().classify(
        "Did Ulta Beauty's wages expense as a percent of net sales increase or decrease in FY2023?"
    )
    assert any("deleverage" in q.lower() for q in c["retrieval_queries"])
