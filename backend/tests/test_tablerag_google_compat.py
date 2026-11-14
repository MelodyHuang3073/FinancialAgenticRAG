"""Offline tests of the Google TableRAG adapter (no API call). Skipped when the third-party source is absent."""
import os
import sys
import types

import pytest

SRC = os.environ.get("TABLERAG_SRC", r"C:\Users\y9205\financial_agentic_rag_runs\tablerag_google\src\table_rag")
pytestmark = pytest.mark.skipif(not os.path.isdir(SRC), reason="TableRAG source not downloaded")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# The packages live in the separate TableRAG venv, not in the main backend venv.
for _pkg in ("langchain_classic", "langchain_community", "rank_bm25", "faiss"):
    pytest.importorskip(_pkg)


@pytest.fixture(scope="module")
def tr():
    from baselines.tablerag_google import compat
    return compat.load(SRC)


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch):
    import app.agent.usage_tracker as ut
    monkeypatch.setattr(ut, "record_usage", lambda *a, **k: None)


class _Completions:
    def __init__(self, text):
        self.text, self.kwargs = text, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                                      completion_tokens_details=types.SimpleNamespace(reasoning_tokens=2))
        msg = types.SimpleNamespace(content=self.text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def test_import_and_context_limit(tr, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    import agent.model as m
    model = m.Model("gpt-5-mini")
    assert model.context_limit == 400000
    assert model.get_token_count("hello world") > 0


def test_query_drops_unsupported_params_and_cuts_at_stop(tr, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    import agent.model as m
    model = m.Model("gpt-5-mini")
    comp = _Completions("Thought: x\nAction: ```python\n1+1\n```\nObservation: made up")
    model.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    out = model.query(prompt="p", temperature=0.8, top_p=0.95, stop=["Observation:"], max_tokens=128)
    assert "made up" not in out and out.rstrip().endswith("```")
    assert not ({"temperature", "top_p", "stop", "max_tokens"} & set(comp.kwargs))
    assert comp.kwargs["max_completion_tokens"] >= 1000


def test_solver_loop_runs_on_small_table(tr, monkeypatch):
    import pandas as pd
    import agent.model as m
    from agent import TableAgent
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    replies = iter(["Action: ```python\nprint(df['v'].sum())\n```", "Final Answer: 7"])
    monkeypatch.setattr(m.Model, "query", lambda self, prompt, **k: next(replies))
    agent = TableAgent("gpt-5-mini", "bm25", task="wtq", agent_type="PyReAct", db_dir=os.environ.get("TEMP", "."))
    answer, n_iter, solution = agent.solver_loop(pd.DataFrame({"v": [3, 4]}), "Q\n")
    assert "7" in answer and n_iter == 2 and "Observation: 7" in solution


def test_bm25_retriever_builds_without_embeddings(tr):
    import pandas as pd
    from agent.retriever import Retriever
    r = Retriever("TableRAG", "bm25", "none", top_k=2, db_dir=os.path.join(os.environ.get("TEMP", "."), "tr_db"))
    r.init_retriever("t1", pd.DataFrame({"Item": ["Revenue", "Cost"], "FY2020": [10, 4]}))
    assert r.retrieve_schema("Revenue")


def test_cell_corpus_sees_text_columns(tr):
    """Google's code was written for pandas 2: with pandas 3 the text columns are not 'object', so the cell
    database would hold only min/max rows. The TableRAG venv pins pandas 2.2.x."""
    import pandas as pd
    from agent.retriever import Retriever
    r = Retriever("TableRAG", "bm25", "none", top_k=2, db_dir=os.path.join(os.environ.get("TEMP", "."), "tr_db2"))
    df = pd.DataFrame({"line_item": ["Revenue", "Cost"], "value": ["1,000", "400"], "number": [1000.0, 400.0]})
    docs = [d.page_content for d in r.build_cell_corpus(df)]
    assert any('"cell_value": "Revenue"' in d for d in docs)
