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


def _rows(*items):
    return [{"content": f"Company: ACME | Line Item: {name} | 2023: {val}", "company": "ACME",
             "type": "table_row", "id": str(i)} for i, (name, val) in enumerate(items)]


def test_inferred_formula_does_not_override_a_reliable_deterministic_result():
    from app.agent.pot_reasoner import ProgramOfThoughtReasoner
    reasoner = ProgramOfThoughtReasoner()
    evidence = _rows(("Total assets", "1,000"))
    query = "What were Acme's total assets in 2023?"
    inferred = {"formula_key": "llm_extracted", "stated_in_question": False, "unit": "",
                "formula_expr": "op0 / op1", "result_label": "x",
                "required_vars": {"op0": ["total assets"], "op1": ["revenue"]}}
    baseline = reasoner.generate_and_execute(query, evidence, "Acme", library_formula_entry=None)
    with_inferred = reasoner.generate_and_execute(
        query, evidence, "Acme", library_formula_entry=None, dynamic_formula_entry=inferred)
    assert baseline["result_value"] == 1000.0
    assert with_inferred["result_value"] == baseline["result_value"]


def test_inferred_formula_is_used_when_the_deterministic_path_finds_nothing():
    from app.agent.pot_reasoner import ProgramOfThoughtReasoner
    reasoner = ProgramOfThoughtReasoner()
    evidence = _rows(("Widget repurchases fourth quarter", "25"), ("Widget repurchases full year", "100"))
    query = "What share of Acme's widget repurchases happened in the fourth quarter of 2023?"
    inferred = {"formula_key": "llm_extracted", "stated_in_question": False, "unit": "%",
                "formula_expr": "q4 / total", "result_label": "Q4 share",
                "required_vars": {"q4": ["widget repurchases fourth quarter"],
                                  "total": ["widget repurchases full year"]}}
    without = reasoner.generate_and_execute(query, evidence, "Acme", library_formula_entry=None)
    with_inferred = reasoner.generate_and_execute(
        query, evidence, "Acme", library_formula_entry=None, dynamic_formula_entry=inferred)
    assert with_inferred["result_value"] == 25.0
    assert without["result_value"] != 25.0


def test_incurred_cost_directive_fires_only_for_initiative_questions_with_the_sentence():
    from app.agent.accounting_conventions import incurred_cost_directive as _incurred_cost_directive
    ev = ("We expect to incur costs of approximately $700 million in connection with "
          "separating a unit, of which approximately 90% has been incurred since inception.")
    assert "MUST be YES" in _incurred_cost_directive("Is the company separating any business?", ev)
    how_much = _incurred_cost_directive("How much more will it pay to separate the unit?", ev)
    assert "777.78" in how_much and "77.78" in how_much and "MUST use" in how_much
    assert _incurred_cost_directive("What is the dividend per share?", ev) == ""
    assert _incurred_cost_directive("Is the company separating any business?", "No such sentence.") == ""


def test_original_question_hits_keep_their_slots_in_the_prompt_window():
    from app.agent.evidence_selection import select_with_quota
    pool = []
    # Eight sub-queries, each retrieving two strong passages, would fill a 16-slot window by quota alone.
    for sq in range(8):
        for k in range(2):
            pool.append({"id": f"s{sq}_{k}", "subquery_idx": sq, "relevance_score": 60 - sq - k})
    # The original question's third-best hit scores higher than several quota winners
    for k in range(4):
        pool.append({"id": f"o{k}", "subquery_idx": 99, "relevance_score": 45 - k, "original_question_hit": True})
    ids = {i["id"] for i in select_with_quota(pool, 16)}
    assert {"o0", "o1", "o2", "o3"} <= ids


def test_leverage_directive_lists_directions_only_for_percent_of_sales_questions():
    from app.agent.accounting_conventions import leverage_directive as _leverage_directive
    ev = ("SG&A declined as a percent of net sales, primarily due to leverage of marketing expenses, "
          "partially offset by deleverage of store payroll and\nbenefits due to wage investments.")
    d = _leverage_directive("Did the cost as a percent of net sales increase in FY2023?", ev)
    assert '"marketing expenses" fell' in d and '"store payroll and benefits" rose' in d
    assert _leverage_directive("What was the dividend per share?", ev) == ""
    assert _leverage_directive("Did the cost as a percent of net sales increase?", "No such idiom here.") == ""


def test_question_shape_directives_fire_only_for_their_shapes():
    from app.agent.answer_directives import (
        ranking_directive, fiscal_year_directive, question_directives,
    )
    assert "BOTH" in ranking_directive("Which product category performed the best in the domestic market?")
    assert "BOTH" not in ranking_directive("Which category had the highest growth rate?")
    assert "finest-level" in ranking_directive("Which region had the biggest drop in revenue?")
    assert ranking_directive("What was net income in 2023?") == ""
    # a word between "worst" and "performance" is still a best/worst question
    assert "BOTH" in ranking_directive("Which region had the worst topline performance last year?")
    assert "BOTH" not in ranking_directive("Which region had the worst sales growth performance?")
    ev = "52 weeks ended January 28, 2023. During fiscal 2022, the Company repurchased shares."
    assert "fiscal 2022" in fiscal_year_directive("What share of repurchases was in Q4 of FY2023?", ev)
    assert fiscal_year_directive("What share was in Q4 of FY2023?", "year ended December 31, 2023") == ""
    assert question_directives("What was revenue in 2023?", ev) == ""


def test_acceleration_questions_ask_for_every_guidance_basis():
    from app.agent.answer_directives import acceleration_directive
    assert "EACH basis" in acceleration_directive("Is growth in adjusted EPS expected to accelerate next year?")
    assert "EACH basis" in acceleration_directive("Will sales growth slow down in 2024?")
    assert acceleration_directive("What was net income in 2023?") == ""


def test_customer_concentration_sentence_becomes_the_headline_fact():
    from app.agent.answer_directives import customer_concentration_directive
    ev = ("Revenues from the Acme government (including allied sales through the Acme government), "
          "primarily recorded at the Defense unit, represented 40%, 49% and 51% of consolidated "
          "revenues for 2022, 2021 and 2020. Revenue by location: Home $120 of $300.")
    out = customer_concentration_directive("Who are the primary customers of Acme as of FY2022?", ev)
    assert "represented 40%" in out and "MUST name" in out
    assert customer_concentration_directive("What was net income?", ev) == ""
    assert customer_concentration_directive("Who are the primary customers?", "No such sentence here.") == ""


class _StubStore:
    """A corpus store whose every search returns one new passage, so a retry always finds something."""

    def __init__(self):
        self.corpus = [{"id": "seed", "company": "ACME_2023_10K", "content": "ACME revenue", "type": "text_note"}]
        self.uploaded_files = [{"filename": "ACME_2023_10K.pdf", "company": "ACME_2023_10K", "passage_count": 1}]
        self.parent_map = {}
        self.searches = 0

    def search(self, query, top_k=5, exclude_ids=None, **kwargs):
        self.searches += 1
        return [{
            "id": f"hit{self.searches}", "company": "ACME_2023_10K", "table_name": "ACME_2023_10K.pdf (Page 3)",
            "content": f"ACME operating margin changed because of mix, passage {self.searches}",
            "type": "text_note", "relevance_score": 5.0,
        }]

    def get_parent_content(self, parent_id):
        return ""


def test_non_numeric_question_retries_after_a_rejected_verification(monkeypatch):
    from app.agent import llm_client, decomposer
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(llm_client.LLMAnswerGenerator, "_get_client", lambda self: None)
    monkeypatch.setattr(decomposer.QueryDecomposer, "_get_llm_client", lambda self: None)
    orch = FinAgentRAGOrchestrator(corpus_store=_StubStore())
    real, calls = orch.verifier.verify, []

    def verify(query, evidence, pot, entity=None):
        calls.append(query)
        result = real(query, evidence, pot, entity=entity)
        if len(calls) < 3:
            result = {**result, "decision": "REJECT", "failure_type": "no_relevant_data"}
        return result

    monkeypatch.setattr(orch.verifier, "verify", verify)
    result = orch.process_query("What drove the change in ACME's operating margin?")
    assert result["answer_mode"] != "NUMERIC"
    assert len(calls) == 3 and result["total_iterations"] == 3
