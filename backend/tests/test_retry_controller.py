"""The Tri-Check verifier's structured failure type and the orchestrator's
single retry controller (_plan_retry) that maps it to the next retrieval."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.verifier import TriCheckSelfVerifier
from app.agent.orchestrator import FinAgentRAGOrchestrator
from app.agent.financial_formula_library import detect_formula

_EVIDENCE = [{"content": "Line Item: Total revenue | 2023: 5,000", "company": "ACME_2023_10K"}]


def test_deliberately_skipped_calculation_is_not_a_numeric_failure():
    res = TriCheckSelfVerifier().verify(
        "Which segment had the highest revenue in 2023?", _EVIDENCE,
        {"answer_mode": "NUMERIC", "success": True, "result_value": None, "calculation_skipped": True},
    )
    assert res["checks"]["nu_num"]["applicable"] is False
    assert res["decision"] == "ACCEPT"
    assert res["failure_type"] is None


def test_failure_type_orders_no_evidence_first_then_implausible():
    v = TriCheckSelfVerifier()
    empty = v.verify("What is gross margin in 2023?", [], {"answer_mode": "NUMERIC", "success": False})
    assert empty["failure_type"] == "no_evidence"
    implausible = v.verify(
        "What is gross margin in 2023?", _EVIDENCE,
        {"answer_mode": "NUMERIC", "success": True, "result_value": 250.0,
         "implausible_result_reason": "a margin cannot exceed 100%"},
    )
    assert implausible["decision"] == "REJECT"
    assert implausible["failure_type"] == "implausible_result"


def _orchestrator():
    orch = FinAgentRAGOrchestrator.__new__(FinAgentRAGOrchestrator)
    from app.agent.refiner import QueryRefiner
    orch.refiner = QueryRefiner()
    return orch


def test_plan_retry_maps_each_failure_to_one_action():
    orch = _orchestrator()
    classification = {"retrieval_queries": ["Acme revenue 2023"], "target_metrics": ["revenue"]}
    formula = detect_formula("What is the gross margin?")

    def plan(failure, formula_entry=formula, **extra):
        return orch._plan_retry(
            "What is Acme's 2023 gross margin?", {"failure_type": failure, **extra},
            formula_entry, "Acme", ["2023"], classification, 1, [],
        )

    assert plan("no_evidence")["action"] == "broaden"
    targeted = plan("implausible_result")
    assert targeted["action"] == "targeted" and targeted["queries"]
    assert plan("missing_years", missing_years=["2022"])["action"] == "targeted"
    assert plan("implausible_result", formula_entry=None)["action"] == "refine"
    assert plan("entity_mismatch")["action"] == "refine"


def test_same_label_rows_prefer_the_line_items_primary_statement():
    from app.agent.pot_reasoner import _pick_best_in_group, _on_home_statement
    note_row = {"item": "Cost of goods sold", "year": "2023", "val": 900.0,
                "canonical": "cost_of_revenue", "section": "balance_sheet"}
    statement_row = {"item": "Cost of goods sold", "year": "2023", "val": 300.0,
                     "canonical": "cost_of_revenue", "section": "income_statement"}
    assert _on_home_statement("cost_of_revenue", "income_statement")
    assert not _on_home_statement("cost_of_revenue", "balance_sheet")
    # Same label, same year: the income-statement row wins over the larger
    # same-labeled row from another table.
    assert _pick_best_in_group([note_row, statement_row], "cost_of_revenue", "2023") is statement_row
