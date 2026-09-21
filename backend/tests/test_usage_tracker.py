"""The token ledger covers every LLM call site and survives parallel processes."""
import os
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app.agent.usage_tracker as ut
from app.agent.decomposer import QueryDecomposer
from app.agent.llm_client import LLMAnswerGenerator


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        usage = types.SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=30),
        )
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def _fake_client(content):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions(content)))


def _use_tmp_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ut, "USAGE_DIR", str(tmp_path / "usage"))
    monkeypatch.setattr(ut, "BASE_DIR", str(tmp_path))


def test_record_and_report(monkeypatch, tmp_path):
    _use_tmp_ledger(monkeypatch, tmp_path)
    ut.record_usage("gpt-5-mini", "answer", types.SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15), 40)
    ut.record_usage("gpt-5-mini", "decomposer", None, 20)  # no usage returned: still a call
    d = ut._read_today()
    assert d["total_tokens"] == 15 and d["calls"] == 2 and d["calls_without_usage"] == 1
    assert "answer" in ut.format_report()


def test_decomposer_and_answer_calls_are_both_recorded(monkeypatch, tmp_path):
    _use_tmp_ledger(monkeypatch, tmp_path)
    dec = QueryDecomposer()
    dec._llm_client = _fake_client('[{"step": 1, "query": "Co revenue 2022", "target_metric": "revenue", "target_year": "2022"}]')
    dec._llm_enabled = True
    dec._llm_model = "gpt-5-mini"
    assert dec._llm_decompose("What was revenue?", "Co", ["revenue"], ["2022"])
    gen = LLMAnswerGenerator()
    gen._client = _fake_client("An answer.")
    gen._model = "gpt-5-mini"
    gen.generate_answer(query="q", answer_mode="EXPLANATION", evidence=[], route_res={})
    d = ut._read_today()
    assert d["by_caller"]["decomposer"]["tokens"] == 150
    assert d["by_caller"]["answer"]["tokens"] == 150
    assert d["total_tokens"] == 300


def test_parallel_processes_do_not_lose_updates(tmp_path):
    usage_dir = str(tmp_path / "usage")
    code = (
        "import sys, types; sys.path.insert(0, %r); import app.agent.usage_tracker as ut; "
        "ut.USAGE_DIR = %r; ut.BASE_DIR = %r; "
        "[ut.record_usage('m', 'answer', types.SimpleNamespace(total_tokens=1)) for _ in range(150)]"
    ) % (os.path.dirname(os.path.dirname(__file__)), usage_dir, str(tmp_path))
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(6)]
    for p in procs:
        assert p.wait() == 0
    n = 0
    for name in os.listdir(usage_dir):
        with open(os.path.join(usage_dir, name), encoding="utf-8") as f:
            n += sum(1 for _ in f)
    assert n == 900
