"""Matching a question to one of several uploaded filings: an ordinary lowercase word of the
question must not prefix-match a company name. Company names here are fictional."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.orchestrator import FinAgentRAGOrchestrator
from app.rag.corpus_store import FinancialCorpusStore


def _orchestrator(*companies):
    store = FinancialCorpusStore()
    store.uploaded_files = [{"company": c} for c in companies]
    return FinAgentRAGOrchestrator(corpus_store=store)


def test_the_word_net_does_not_select_a_company_starting_with_net():
    orch = _orchestrator("NETWORKCO_2025_10K", "9Q_2025_10K")
    q = "Did 9Q's net income grow at the same rate as its net sales in FY2025?"
    assert orch._match_entity_to_corpus("company", q) == "9Q_2025_10K"


def test_an_acronym_still_matches_the_start_of_a_company_name():
    orch = _orchestrator("NETWORKCO_2025_10K", "ZEBRATRADING_2025_10K")
    assert orch._match_entity_to_corpus("company", "What was the ZEB gross margin in FY2025?") == "ZEBRATRADING_2025_10K"


def test_a_company_named_in_full_is_still_selected():
    orch = _orchestrator("NETWORKCO_2025_10K", "9Q_2025_10K")
    assert orch._match_entity_to_corpus("company", "What was NetworkCo's revenue in FY2025?") == "NETWORKCO_2025_10K"
