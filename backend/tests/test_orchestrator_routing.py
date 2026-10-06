"""FinAgentRAGOrchestrator's file-routing helpers (app/agent/orchestrator.py):
the company-key extracted from a corpus doc name, and the tie-break that picks
the most recent filing for a forward-looking separation/spin-off cost
question naming no year at all."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.orchestrator import FinAgentRAGOrchestrator


def test_company_key():
    key = FinAgentRAGOrchestrator._company_key
    assert key("ACME_WIDGETS_2022Q4_EARNINGS") == "ACMEWIDGETS"
    assert key("BETACORP_2023_8K_dated-2023-05-30") == "BETACORP"
    assert key("company") == ""


def test_separation_cost_question_routes_to_most_recent_filing():
    """No year is named, so the plain-annual-10-K default must not win when the
    question is a forward-looking separation/spin-off cost question."""
    class _FakeVS:
        uploaded_files = [
            {"company": "ACME_2021_10K"},
            {"company": "Acme_2023Q2_10Q"},
        ]
        corpus = []

    orch = FinAgentRAGOrchestrator.__new__(FinAgentRAGOrchestrator)
    orch.vector_store = _FakeVS()
    resolved = orch._match_entity_to_corpus(
        "Acme", "How much more will Acme expect to spend on separating its unit, in USD million?"
    )
    assert resolved == "Acme_2023Q2_10Q"
