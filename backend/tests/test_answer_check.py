from app.agent.answer_check import check_answer
from app.agent.pot_reasoner import _check_stock_signs

EV = [{"company": "ACME_2023_10K", "content": "Revenue 100"}]


def _check(answer, entity="ACME", query="What was ACME revenue?"):
    return check_answer(query, answer, EV, None, entity)


def test_clean_answer_passes():
    assert _check("ACME revenue was $100 million.") is None


def test_opening_non_answer_is_flagged():
    assert _check("The filing does not provide quarterly dividend data.")["failure"] == "non_answer"
    assert _check("I cannot determine the answer from the evidence.")["failure"] == "non_answer"
    assert _check("")["failure"] == "non_answer"


def test_caveat_after_a_verdict_is_not_flagged():
    assert _check("Yes, growth slows. The filing does not disclose the prior-year figure.") is None


def test_event_that_did_not_happen_is_an_answer():
    assert _check("ACME did not report any acquisitions in the period.") is None


def test_other_company_named_as_source_is_flagged():
    assert _check("Globex's 10-K shows revenue of $100.")["failure"] == "wrong_company"


def test_generic_subject_and_own_company_are_not_flagged():
    assert _check("The Company's report shows revenue of $100.") is None
    assert _check("ACME's 10-K shows revenue of $100.") is None


def test_negative_balance_sheet_stock_is_implausible():
    assert _check_stock_signs({"accounts_receivable": -7.2, "revenue": 100.0})
    assert _check_stock_signs({"net_income": -5.0, "shareholders_equity": -3.0}) is None
