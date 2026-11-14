"""Offline end-to-end test of the TableRAG runner: scripted model replies, BM25 retrieval, no API call."""
import json
import os
import sys

import pytest

SRC = os.environ.get("TABLERAG_SRC", r"C:\Users\y9205\financial_agentic_rag_runs\tablerag_google\src\table_rag")
pytestmark = pytest.mark.skipif(not os.path.isdir(SRC), reason="TableRAG source not downloaded")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
for _pkg in ("langchain_classic", "langchain_community", "rank_bm25", "faiss"):
    pytest.importorskip(_pkg)


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch):
    import app.agent.usage_tracker as ut
    monkeypatch.setattr(ut, "record_usage", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "x")


def _data(tmp_path):
    (tmp_path / "tables").mkdir()
    table = [["page", "line_item", "period", "value", "number", "unit"],
             [5, "Revenue", "2020", "$1,000", 1000.0, "millions"],
             [5, "Revenue", "2019", "900", 900.0, "millions"],
             [5, "Cost of sales", "2020", "(400)", -400.0, "millions"]]
    json.dump({"table_caption": "X", "table_text": table}, open(tmp_path / "tables" / "X.json", "w"))
    with open(tmp_path / "questions_train.jsonl", "w") as fp:
        for i in (1, 2):
            fp.write(json.dumps({"id": f"train-{i}", "doc_name": "X", "table_id": "X", "question": f"What is revenue in 2020? ({i})",
                                 "label": "$1000.00", "question_type": "metrics-generated"}) + "\n")
    open(tmp_path / "questions_out.jsonl", "w").close()
    return str(tmp_path)


def test_runner_end_to_end_and_selection(tmp_path, monkeypatch):
    from baselines.tablerag_google import compat, fin_prompts, run_questions as rq
    compat.load(SRC)
    import prompts
    fin_prompts.register(prompts)
    import agent.model as m
    from agent import TableRAGAgent

    solve_calls = []

    def fake_query(self, prompt, **k):
        if "suggest some column names" in prompt:
            return '["line_item", "value"]'
        if "extract some keywords" in prompt:
            return '["Revenue"]'
        solve_calls.append(prompt)
        if len(solve_calls) % 2 == 1:
            code = "print(df[(df.line_item=='Revenue') & (df.period.astype(str)=='2020')]['number'].iloc[0])"
            return "Thought: read it\nAction: ```python\n" + code + "\n```"
        return "Final Answer: 1000"
    monkeypatch.setattr(m.Model, "query", fake_query)

    data_dir = _data(tmp_path)
    qs = rq.load_questions(data_dir, "all")
    assert [q["id"] for q in qs] == ["train-1", "train-2"] and qs[0]["set"] == "original" and qs[0]["n"] == 1
    assert [q["id"] for q in rq.select(qs, ids="train-2")] == ["train-2"]
    assert [q["id"] for q in rq.select(qs, shard="1/2")] == ["train-2"]
    with pytest.raises(SystemExit):
        rq.select(qs, ids="train-9")

    agent = TableRAGAgent(model_name="gpt-5-mini", retrieve_mode="bm25", embed_model_name="none", task="fin",
                          agent_type="TableRAG", top_k=3, log_dir=str(tmp_path / "log"), db_dir=str(tmp_path / "db"))
    os.makedirs(tmp_path / "log" / "log")
    rec = rq.run_one(agent, data_dir, qs[0])
    assert rec["answer"] == "1000" and rec["n_iter"] == 2 and rec["n_rows"] == 3
    assert rec["input_tokens"] > 0 and "Observation: 1000.0" in rec["solution"]

    out = str(tmp_path / "r.json")
    rq.save(out, [rec])
    assert json.load(open(out))[0]["answer"] == "1000"
