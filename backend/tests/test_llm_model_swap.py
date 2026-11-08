"""Model-swap runs (gpt-5-nano, gpt-4o-mini): the system picks the model from OPENAI_MODEL, every LLM call uses it
with the parameters its family accepts, every call lands in the usage ledger under that model, and the run tools
(merge check, failed-record removal) catch what they must. No network, no API key, no model call.
"""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "llm_runner"))

import app.agent.decomposer as dec_mod  # noqa: E402
import app.agent.llm_client as llm_mod  # noqa: E402
import app.agent.usage_tracker as ut  # noqa: E402
import drop_failed  # noqa: E402,F401  (imports cleanly)
import merge_check  # noqa: E402
from failed import failure  # noqa: E402

MODELS = ("gpt-5-nano", "gpt-4o-mini")


class _FakeOpenAI:
    """Stands in for openai.OpenAI: records the arguments of every chat completion call."""
    calls = []

    def __init__(self, api_key=None):
        def create(**kw):
            _FakeOpenAI.calls.append(kw)
            usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            msg = types.SimpleNamespace(content="An answer.")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


@pytest.fixture()
def swapped(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "ABLATE_ROW", "ABLATE_POT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ut, "USAGE_DIR", str(tmp_path / "usage"))
    monkeypatch.setattr(ut, "BASE_DIR", str(tmp_path))
    _FakeOpenAI.calls = []

    def use(model):
        monkeypatch.setenv("OPENAI_MODEL", model)
        _FakeOpenAI.calls = []
    return use


def _ledger_models(tmp_path):
    out = set()
    for name in os.listdir(tmp_path / "usage"):
        for line in open(tmp_path / "usage" / name, encoding="utf-8"):
            out.add(json.loads(line)["model"])
    return out


def test_model_families():
    assert llm_mod._is_reasoning_model("gpt-5-nano") and dec_mod._is_reasoning_model("gpt-5-nano")
    assert not llm_mod._is_reasoning_model("gpt-4o-mini") and not dec_mod._is_reasoning_model("gpt-4o-mini")


@pytest.mark.parametrize("model", MODELS)
def test_answer_and_decomposer_calls_use_the_swapped_model(swapped, tmp_path, model):
    swapped(model)
    gen = llm_mod.LLMAnswerGenerator()
    assert gen.generate_answer(query="q", answer_mode="EXPLANATION", evidence=[], route_res={}) == "An answer."
    assert gen._model == model
    dec = dec_mod.QueryDecomposer()
    assert dec._get_llm_client() is not None and dec._llm_model == model
    dec._llm_decompose("What was 3M's FY2018 capital expenditure?", "3M", ["capex"], ["2018"])
    dec.suggest_narrative_topic_query("What legal battles did 3M face?", "3M")
    calls = _FakeOpenAI.calls
    assert len(calls) == 3 and all(c["model"] == model for c in calls)
    if model == "gpt-5-nano":  # reasoning family: no temperature, large completion budget on the decomposer calls
        assert all("temperature" not in c for c in calls)
        assert [c["max_completion_tokens"] for c in calls[1:]] == [dec_mod.REASONING_MAX_COMPLETION_TOKENS] * 2
    else:  # gpt-4o-mini: temperatures are set, small budgets
        assert calls[0]["temperature"] == 0.2 and [c["temperature"] for c in calls[1:]] == [0, 0]
        assert [c["max_completion_tokens"] for c in calls[1:]] == [768, 100]
    assert _ledger_models(tmp_path) == {model}


def test_default_model_is_the_baseline_model_when_nothing_is_set(swapped, monkeypatch):
    swapped("gpt-5-mini")
    monkeypatch.delenv("OPENAI_MODEL")
    gen = llm_mod.LLMAnswerGenerator()
    gen.generate_answer(query="q", answer_mode="EXPLANATION", evidence=[], route_res={})
    assert gen._model == "gpt-5-mini"


def test_failure_shapes():
    assert failure({"model_answer": "A real answer.", "error": None}) is None
    assert failure({"model_answer": "x", "error": "RuntimeError('boom')"}).startswith("error")
    assert failure({"model_answer": "  ", "error": None}) == "empty answer"
    assert "fallback" in failure({"model_answer": "According to the financial report disclosures, ...", "error": None})


def _run_dir(tmp_path, model, bad=False):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"model": model, "code_commit": "abc", "started": "2026-11-12T08:30:00", "limit": 0}))
    for tag, name in (("train", "all_candidates_train.json"), ("out", "all_candidates_out.json")):
        cands = json.load(open(os.path.join(merge_check.EXP, name), encoding="utf-8"))
        rows = [{"doc_name": c["doc"], "question": c["q"], "model_answer": "An answer.", "error": None} for c in cands]
        if bad and tag == "out":
            rows[0]["model_answer"] = "The calculation result for X could not be obtained."
        (run / f"{tag}_0.json").write_text(json.dumps(rows[0::2]))
        (run / f"{tag}_1.json").write_text(json.dumps(rows[1::2]))
    return run


def test_merge_check_accepts_a_complete_run_and_rejects_the_other_cases(tmp_path):
    run = _run_dir(tmp_path, "gpt-5-nano")
    cfg, merged, problems = merge_check.merge(str(run), "gpt-5-nano")
    assert problems == [] and len(merged["train"]) == 150 and len(merged["out"]) == 150
    assert merge_check.merge(str(run), "gpt-4o-mini")[2]  # config says another model
    rows = json.load(open(run / "train_0.json"))
    json.dump(rows[:-1], open(run / "train_0.json", "w"))  # one question missing
    assert any("records, expected" in p for p in merge_check.merge(str(run), "gpt-5-nano")[2])


def test_merge_check_flags_a_fallback_answer_and_drop_failed_removes_only_it(tmp_path):
    run = _run_dir(tmp_path, "gpt-5-nano", bad=True)
    problems = merge_check.merge(str(run), "gpt-5-nano")[2]
    assert len(problems) == 1 and "fallback" in problems[0]
    import subprocess
    script = os.path.join(HERE, "..", "llm_runner", "drop_failed.py")
    subprocess.run([sys.executable, script, "--run-dir", str(run), "--apply"], check=True, capture_output=True)
    assert len(json.load(open(run / "out_0.json"))) == 74 and len(json.load(open(run / "out_1.json"))) == 75
    assert len(json.load(open(run / "out_0.failed.json"))) == 1
    assert len(merge_check.merge(str(run), "gpt-5-nano")[2]) == 1  # one question missing now, as it must re-run


def test_ledger_check_flags_a_call_by_another_model(tmp_path):
    d = tmp_path / "led"
    d.mkdir()
    lines = [{"ts": "2026-11-12 08:31:00", "model": "gpt-5-nano", "caller": "answer", "total_tokens": 100},
             {"ts": "2026-11-12 08:32:00", "model": "gpt-5-mini", "caller": "decomposer", "total_tokens": 7},
             {"ts": "2026-11-12 07:00:00", "model": "gpt-5-mini", "caller": "answer", "total_tokens": 9}]  # before the run
    (d / "2026-11-12_1.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    problems, calls, tokens = merge_check.ledger("gpt-5-nano", "2026-11-12 08:30:00", str(d))
    assert len(problems) == 1 and calls["answer"] == 1 and tokens["answer"] == 100


# ── the runner's --resume ────────────────────────────────────────────────────────────────────────────────────

def _runner(monkeypatch, tmp_path, n_cands=6):
    """run_single_file_subset.main() with the orchestrator replaced by a counting stub (no corpus, no model)."""
    import run_single_file_subset as rs
    qa_path = rs.SETS["train"][0]
    cache = tmp_path / "cache.pkl"
    cache.write_bytes(b"")
    monkeypatch.setitem(rs.SETS, "train", (qa_path, str(cache)))
    qa = json.load(open(qa_path, encoding="utf-8"))
    monkeypatch.setattr(rs.pickle, "load", lambda f: {"uploaded_files": [{"company": q["doc_name"]} for q in qa]})
    monkeypatch.setattr(rs, "_single_store", lambda cached, company: None)
    calls = []

    class Stub:
        def __init__(self, corpus_store=None):
            pass

        def process_query(self, q):
            calls.append(q)
            return {"final_answer": f"answer {len(calls)}"}

    monkeypatch.setattr(rs, "FinAgentRAGOrchestrator", Stub)
    cands = json.load(open(os.path.join(merge_check.EXP, "all_candidates_train.json"), encoding="utf-8"))[:n_cands]
    cand_file = tmp_path / "cands.json"
    cand_file.write_text(json.dumps(cands))
    out = tmp_path / "out.json"

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["run", "--set", "train", "--candidates", str(cand_file), "--out", str(out), *extra])
        rs.main()
    return run, out, calls


def test_runner_without_resume_starts_again_and_overwrites(monkeypatch, tmp_path):
    run, out, calls = _runner(monkeypatch, tmp_path)
    run()
    assert len(calls) == 6 and len(json.load(open(out))) == 6
    run()  # same command again, no --resume: everything runs again
    assert len(calls) == 12 and len(json.load(open(out))) == 6


def test_runner_resume_keeps_the_records_it_has_and_runs_only_the_missing(monkeypatch, tmp_path):
    run, out, calls = _runner(monkeypatch, tmp_path)
    run()
    rows = json.load(open(out))
    rows[0]["model_answer"] = "KEEP"
    missing = rows.pop(3)
    json.dump(rows, open(out, "w"))
    run("--resume")
    assert len(calls) == 7  # 6 first time + exactly one question again
    after = json.load(open(out))
    assert len(after) == 6 and after[0]["model_answer"] == "KEEP"
    assert [r["question"] for r in after].count(missing["question"]) == 1
    run("--resume")  # nothing left to run
    assert len(calls) == 7 and len(json.load(open(out))) == 6
