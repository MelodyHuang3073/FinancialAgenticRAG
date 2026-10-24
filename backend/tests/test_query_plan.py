"""Query planning (FinAgentRAGOrchestrator._plan_queries): base queries, the
question itself, and the optional LLM topic query, for numeric and non-numeric
questions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.orchestrator import FinAgentRAGOrchestrator
from app.agent.financial_formula_library import detect_formula

QUERY = "What was ACME's gross profit margin in 2023?"
FORMULA = detect_formula(QUERY)


class _StubDecomposer:
    def __init__(self, topic="filing section words"):
        self.topic = topic

    def decompose(self, query, target_metrics=None, years=None, entity=None):
        return [{"step": 1, "type": "retrieval", "query": "llm part", "source": "llm"}], None

    def suggest_narrative_topic_query(self, query, entity):
        return self.topic


def _plan(answer_mode, formula, classification=None, query=QUERY, topic="filing section words"):
    orch = FinAgentRAGOrchestrator.__new__(FinAgentRAGOrchestrator)
    orch.decomposer = _StubDecomposer(topic)
    cls = {"target_metrics": [], "calc_type": "", "entity": "ACME", "years": ["2023"],
           "retrieval_queries": [query]}
    cls.update(classification or {})
    return orch._plan_queries(query, cls, answer_mode, formula, "ACME", ["2023"])


def _sources(plan):
    return [sq.get("source") for sq in plan.sub_questions]


def test_formula_numeric_uses_formula_variables_only():
    assert FORMULA is not None
    plan = _plan("NUMERIC", FORMULA, {"calc_type": "ratio", "target_metrics": ["gross_margin"]})
    assert set(_sources(plan)) == {"formula"}


def test_formula_numeric_without_matched_metric_gets_topic_query_but_not_the_question():
    plan = _plan("NUMERIC", FORMULA)
    assert _sources(plan)[-1] == "narrative_topic"
    assert "original_question" not in _sources(plan)


def test_no_formula_numeric_gets_llm_split_question_and_topic_query():
    plan = _plan("NUMERIC", None)
    assert _sources(plan) == ["llm", "original_question", "narrative_topic"]


def test_numeric_topic_query_skipped_for_matched_ratio_and_when_llm_returns_nothing():
    matched = {"calc_type": "ratio", "target_metrics": ["gross_margin"]}
    assert "narrative_topic" not in _sources(_plan("NUMERIC", None, matched))
    assert "narrative_topic" not in _sources(_plan("NUMERIC", None, topic=None))


def test_non_numeric_without_formula_searches_classifier_queries_topic_then_question():
    plan = _plan("EXPLANATION", None, {"retrieval_queries": ["classifier query"]})
    assert plan.search_queries == ["classifier query", "ACME filing section words", QUERY]
    assert plan.prefer_narrative is True


def test_non_numeric_with_formula_searches_variables_and_adds_question_only_for_attribution():
    plain = _plan("ASSESSMENT", FORMULA)
    assert plain.prefer_narrative is False
    assert QUERY not in plain.search_queries
    drivers = "What drove the change in gross profit margin in 2023?"
    why = _plan("EXPLANATION", detect_formula(drivers), query=drivers)
    assert why.prefer_narrative is True
    assert why.search_queries[-1] == drivers
