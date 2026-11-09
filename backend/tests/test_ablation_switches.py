"""Tests of the ablation switches (app/ablation.py, EXPERIMENT_PLAN.md section 4).

The switch is read once when the process starts, so each check runs a small program in a child process with
ABLATE_ROW set, on a tiny synthetic filing. No model call is made: the answer call is recorded, the question
decomposer replaced by a counter.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PRELUDE = textwrap.dedent('''
    import json, sys
    from types import SimpleNamespace
    sys.path.insert(0, %r)
    from app import ablation
    from app.rag.corpus_store import FinancialCorpusStore
    from app.tools.lexical_retriever import LexicalFinancialRetriever
    from app.agent import orchestrator as om

    ROWS = [
        {"id": "r1", "company": "ACME_2022_10K", "table_name": "Income statement", "period": "2022",
         "page_number": "1", "content": "Company: ACME | Line Item: Revenue | 2022: 1,000 | 2021: 900",
         "type": "table_row", "parent_id": "p1", "parent_content": "| Line Item | 2022 | 2021 |\\n|---|---|---|\\n| Revenue | 1,000 | 900 |",
         "section": "income_statement"},
        {"id": "r2", "company": "ACME_2022_10K", "table_name": "Income statement", "period": "2022",
         "page_number": "1", "content": "Company: ACME | Line Item: Net income | 2022: 100 | 2021: 80",
         "type": "table_row", "parent_id": "p1", "parent_content": "| Line Item | 2022 | 2021 |\\n|---|---|---|\\n| Net income | 100 | 80 |",
         "section": "income_statement"},
        {"id": "t1", "company": "ACME_2022_10K", "table_name": "MD&A", "period": "2022", "page_number": "9",
         "content": "Revenue grew because of higher volume in the retail segment during 2022.",
         "type": "text_note", "parent_id": "", "parent_content": "", "section": "mdna"},
    ]

    class Recorder:
        def __init__(self):
            self.kwargs = None
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        def _create(self, **kw):
            self.kwargs = kw
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="-"))], usage=None)

    calls = {"decompose": 0, "topic": 0}
    def fake_decompose(self, query, **kw):
        calls["decompose"] += 1
        return [{"step": 1, "type": "retrieval", "query": query + " decomposed", "target_metric": "",
                 "target_year": "", "source": "llm"}], None
    def fake_topic(self, query, entity=None):
        calls["topic"] += 1
        return "topic terms"
    om.QueryDecomposer.decompose = fake_decompose
    om.QueryDecomposer.suggest_narrative_topic_query = fake_topic

    def orchestrator():
        store = FinancialCorpusStore()
        store.corpus = (ablation.plain_entries("ACME_2022_10K", [r["content"] for r in ROWS])
                        if ablation.off("C1") else [dict(r) for r in ROWS])
        store.uploaded_files = [{"filename": "ACME_2022_10K.pdf", "company": "ACME_2022_10K", "passage_count": 3}]
        store.retriever = LexicalFinancialRetriever(store.corpus)
        orch = om.FinAgentRAGOrchestrator(corpus_store=store)
        rec = Recorder()
        orch.llm_generator._client, orch.llm_generator._model = rec, "recorded"
        return orch, rec
''' % BACKEND)


def run_row(row, body):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("OPENAI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    if row is None:
        env.pop("ABLATE_ROW", None)
    else:
        env["ABLATE_ROW"] = str(row)
    out = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(body)], env=env, cwd=BACKEND,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_row_selects_components():
    expected = {None: [], 0: [], 1: ["C1"], "2a": ["C2", "C3"], "2b": ["C2", "C3"], 3: ["C4"], 4: ["C5"], 5: ["C6", "C7"]}
    for row, off in expected.items():
        assert run_row(row, 'print(json.dumps(ablation.describe()["components_off"]))') == off


@pytest.mark.parametrize("value", ["2", "6", "-1", "all", "1,2", "C1"])
def test_unknown_value_stops_the_process(value):
    env = dict(os.environ, ABLATE_ROW=value)
    out = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {BACKEND!r}); import app.ablation"],
                         env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "ABLATE_ROW" in out.stderr


def test_vocabulary_tables_and_classifier_follow_c3():
    body = '''
        from app.agent.financial_vocabulary import QUESTION_METRICS, LINE_ITEM_ALIASES, COMPOSITE_LINE_ITEM_ALIASES, LINE_ITEM_STATEMENT
        from app.agent.question_classifier import FinanceBenchClassifier
        c = FinanceBenchClassifier().classify("Did ACME return to profitability and what drove the change?")
        print(json.dumps({"tables": [len(t) for t in (QUESTION_METRICS, LINE_ITEM_ALIASES, COMPOSITE_LINE_ITEM_ALIASES, LINE_ITEM_STATEMENT)],
                          "mode": c["answer_mode"], "strategy": c["retrieval_strategy"], "hint": c["statement_type_hint"]}))
    '''
    full, a, b = run_row(0, body), run_row("2a", body), run_row("2b", body)
    assert all(n > 0 for n in full["tables"]) and full["mode"] != "NUMERIC"
    assert a["tables"] == b["tables"] == [0, 0, 0, 0]
    assert (a["mode"], a["strategy"], a["hint"]) == ("NUMERIC", "HYBRID", None)
    assert (b["mode"], b["strategy"], b["hint"]) == ("EXPLANATION", "HYBRID", None)


def test_prompt_follows_c1_and_c2():
    body = '''
        orch, rec = orchestrator()
        orch.process_query("What was ACME's revenue in 2022?")
        m = rec.kwargs["messages"]
        print(json.dumps({"roles": [x["role"] for x in m], "text": m[-1]["content"]}))
    '''
    full, c1, c2 = run_row(0, body), run_row(1, body), run_row("2a", body)
    assert full["roles"] == ["system", "user"] and "[ACME_2022_10K / Income statement]" in full["text"]
    assert "ANSWERING PRINCIPLES" in full["text"]
    # row 1: same prompt text around the evidence, but the evidence is the plain chunk, no label
    assert "ANSWERING PRINCIPLES" in c1["text"] and "[ACME_2022_10K /" not in c1["text"]
    # row 2: the Naive RAG prompt -- one user message, a context block, then the question
    assert c2["roles"] == ["user"] and c2["text"].startswith("Context:\n")
    assert "ANSWERING PRINCIPLES" not in c2["text"] and "Answer Mode" not in c2["text"]
    assert c2["text"].rstrip().split("Question:")[-1].strip().startswith("What was ACME's revenue")


def test_c4_off_does_not_decompose():
    body = '''
        orch, rec = orchestrator()
        res = orch.process_query("What was ACME's revenue in 2022?")
        steps = [s for s in res["execution_trace"] if s.get("type") == "decomposition"]
        print(json.dumps({"calls": calls, "queries": [q["query"] for q in steps[0]["sub_questions"]]}))
    '''
    full, off = run_row(0, body), run_row(3, body)
    assert full["calls"]["decompose"] + full["calls"]["topic"] >= 1 or len(full["queries"]) > 1
    assert off["calls"] == {"decompose": 0, "topic": 0}
    assert off["queries"] == ["What was ACME's revenue in 2022?"]


def test_c5_off_is_one_plain_search_of_at_most_16():
    body = '''
        orch, rec = orchestrator()
        searched = []
        real = orch.corpus_store.search_bm25
        orch.corpus_store.search_bm25 = lambda q, **kw: (searched.append(q), real(q, **kw))[1]
        orch.corpus_store.search = lambda *a, **k: (_ for _ in ()).throw(AssertionError("system retriever used"))
        res = orch.process_query("What was ACME's revenue in 2022?")
        print(json.dumps({"searches": len(searched), "limit": orch.CONTEXT_CHUNK_LIMIT,
                          "evidence": [e["id"] for e in res["evidence_sources"]]}))
    '''
    off = run_row(4, body)
    assert off["searches"] >= 1 and off["limit"] == 16 and len(off["evidence"]) <= 16


def test_c6_c7_off_remove_pot_and_verification_from_prompt():
    body = '''
        orch, rec = orchestrator()
        orch.pot_reasoner.generate_and_execute = lambda *a, **k: (_ for _ in ()).throw(AssertionError("PoT ran"))
        orch.verifier.verify = lambda *a, **k: (_ for _ in ()).throw(AssertionError("verifier ran"))
        res = orch.process_query("What was ACME's revenue in 2022?")
        text = rec.kwargs["messages"][-1]["content"]
        print(json.dumps({"iterations": res["total_iterations"], "computed": "\\nComputed value:" in text,
                          "verification": "Verification summary:" in text, "principle5": "Computed values" in text}))
    '''
    off = run_row(5, body)
    assert off == {"iterations": 1, "computed": False, "verification": False, "principle5": True}


def test_search_bm25_ranks_by_content_and_skips_excluded():
    body = '''
        orch, rec = orchestrator()
        hits = orch.corpus_store.search_bm25("net income 100", top_k=2)
        skipped = orch.corpus_store.search_bm25("net income 100", top_k=2, exclude_ids=[hits[0]["id"]])
        print(json.dumps({"best": hits[0]["id"], "n": len(hits), "skipped_has_best": hits[0]["id"] in [h["id"] for h in skipped],
                          "scored": all("relevance_score" in h for h in hits)}))
    '''
    r = run_row(None, body)
    assert r == {"best": "r2", "n": 2, "skipped_has_best": False, "scored": True}
