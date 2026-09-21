"""Unit tests for the second fix round (manual-test findings #4-#19)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.pot_reasoner import _RATIO_DIRECTION_QUERY_RE
from app.agent.question_classifier import FinanceBenchClassifier
from app.rag.parser import FinancialFileParser
from test_financebench_qa import _check_contains_facts, _key_terms


def test_ratio_direction_question_has_no_calculation_path():
    q1 = "Did Ulta Beauty's wages expense as a percent of net sales increase or decrease in FY2023?".lower()
    q2 = "Did JnJ's net earnings as a percent of sales increase in Q2 of FY2023 compared to Q2 of FY2022?".lower()
    assert _RATIO_DIRECTION_QUERY_RE.search(q1)
    assert _RATIO_DIRECTION_QUERY_RE.search(q2)
    # a computed 3-year average must keep its calculation path
    q3 = "What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard?".lower()
    assert not _RATIO_DIRECTION_QUERY_RE.search(q3)


def test_how_much_usd_question_is_numeric():
    c = FinanceBenchClassifier().classify(
        "How much does Pfizer expect to pay to spin off Upjohn in the future in USD million?"
    )
    assert c["answer_mode"] == "NUMERIC"


def test_page_unit_only_when_single_unit():
    p = FinancialFileParser()
    assert p._page_unit("(in millions, except per share) 2022 2021") == "USD millions"
    assert p._page_unit("$ million 2022 $ million 2023") == "USD millions"
    assert p._page_unit("no unit stated anywhere") == ""
    assert p._page_unit("(in millions) and also in billions") == ""


def test_text_header_split_into_columns():
    names = FinancialFileParser._split_text_header(
        "By Business Segment Organic sales Acquisitions Divestitures Translation Total sales change", 5
    )
    assert names == ["Organic sales", "Acquisitions", "Divestitures", "Translation", "Total sales change"]
    assert FinancialFileParser._split_text_header("Net sales for 2022 were higher", 3) == []


def test_tier3_keeps_subheadings_inside_one_table_with_year_header():
    layout = "\n".join([
        "A reconciliation of operating income is as follows:",
        "   (dollars in millions)",
        "   Years Ended December 31,      2021     2020     2019",
        "   Operating Income",
        "   Total reportable segments     $ 33,392 $ 32,629 $ 32,723",
        "   Corporate and other              (449)  (1,472)  (1,403)",
        "   Reconciling items:",
        "     Severance charges              (209)    (221)    (204)",
        "     Other components of pension charges (Note 11) (769) (817) (813)",
        "   Consolidated operating income  32,448   28,798   30,378",
    ])
    tables, _ = FinancialFileParser()._layout_text_to_markdown_and_prose(layout)
    assert len(tables) == 1
    assert tables[0].splitlines()[0] == "| Line Item | 2021 | 2020 | 2019 |"
    assert "Other components of pension charges (Note 11)" in tables[0]


def test_grader_checks_named_terms_of_short_gold():
    gold = "Yes, the gain on completion of Consumer Healthcare JV Transaction"
    assert "jv" in _key_terms(gold)
    assert _check_contains_facts(gold, "Yes - a one-time gain of $8,107 million from the Consumer Healthcare JV transaction") is True
    assert _check_contains_facts(gold, "Yes - a one-time gain of $8,107 million in 2019") is False
    # long descriptive gold stays informational
    long_gold = " ".join(["Word"] * 30)
    assert _check_contains_facts(long_gold, "anything") is None


def test_cost_share_of_sales_question_gets_leverage_bridge_query():
    c = FinanceBenchClassifier().classify(
        "Did Ulta Beauty's wages expense as a percent of net sales increase or decrease in FY2023?"
    )
    assert any("deleverage" in q.lower() for q in c["retrieval_queries"])
