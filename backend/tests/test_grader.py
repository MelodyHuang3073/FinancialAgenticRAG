"""test_financebench_qa.py's grading helpers: a short gold answer with no
numbers (e.g. "Yes, the gain on completion of Consumer Healthcare JV
Transaction") is checked for its named terms instead of staying an
unassertable INFO result."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_financebench_qa import _check_contains_facts, _key_terms


def test_grader_checks_named_terms_of_short_gold():
    gold = "Yes, the gain on completion of Consumer Healthcare JV Transaction"
    assert "jv" in _key_terms(gold)
    assert _check_contains_facts(gold, "Yes - a one-time gain of $8,107 million from the Consumer Healthcare JV transaction") is True
    assert _check_contains_facts(gold, "Yes - a one-time gain of $8,107 million in 2019") is False
    # long descriptive gold stays informational
    long_gold = " ".join(["Word"] * 30)
    assert _check_contains_facts(long_gold, "anything") is None


def test_grader_accepts_percent_for_bare_decimal_ratio_gold():
    # Coca-Cola FY2017 ROA: gold states the plain decimal, model states it as a percent
    assert _check_contains_facts("0.01", "The FY2017 ROA for Coca-Cola was approximately 1.02%.") is True
    # Coca-Cola FY2022 dividend payout ratio: same shape
    assert _check_contains_facts("0.8", "Coca-Cola's FY2022 dividend payout ratio was about 80%.") is True
    # a gold already stated as a percent is unaffected (not routed through the new path)
    assert _check_contains_facts("39.7%", "Coca-Cola's FY2021 COGS margin was 39.7%.") is True
