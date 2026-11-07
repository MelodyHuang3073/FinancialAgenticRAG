"""Offline checks for the runner's restart rule, the shard merge and the launcher (no API call)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import merge_shards as m  # noqa: E402
import run_step as r  # noqa: E402


def _qa():
    return {s: json.load(open(os.path.join(m.TESTS, f), encoding="utf-8")) for s, f in m.QUESTIONS.items()}


def _rows(qa, answer="a"):
    return [{"set": s, "n": n, "question": q["question"], "doc_name": q["doc_name"], "model_answer": answer, "error": None}
            for s, qs in qa.items() for n, q in enumerate(qs, 1)]


def test_a_restart_runs_error_and_empty_rows_again_and_keeps_good_rows(tmp_path):
    p = tmp_path / "shard.json"
    rows = [{"n": 1, "model_answer": "good", "error": None}, {"n": 3, "model_answer": "", "error": "boom"},
            {"n": 5, "model_answer": "   ", "error": None}, {"n": 7, "model_answer": "also good", "error": None}]
    p.write_text(json.dumps(rows), encoding="utf-8")
    assert sorted(r.load_done(str(p))) == [1, 7]
    assert r.load_done(str(tmp_path / "missing.json")) == {} and r.load_done(None) == {}


def test_merge_orders_the_shards_by_question_and_accepts_a_complete_set():
    qa = _qa()
    rows = _rows(qa)
    shard_a, shard_b = rows[::2], rows[1::2]                     # two shards, interleaved as --shard i/2 would
    merged, problems = m.merge([shard_b, shard_a], qa)
    assert problems == [] and [(x["set"], x["n"]) for x in merged] == [(x["set"], x["n"]) for x in rows]
    assert len(merged) == sum(len(v) for v in qa.values())


def test_merge_reports_every_kind_of_problem_and_writes_nothing_by_itself():
    qa = _qa()
    rows = _rows(qa)
    rows[0]["error"] = "api down"
    rows[1]["model_answer"] = ""
    rows[2]["question"] = "a different question"
    dup = dict(rows[3])
    stray = {"set": "original", "n": 9999, "question": "x", "doc_name": "x", "model_answer": "a", "error": None}
    merged, problems = m.merge([rows[:-1], [dup, stray]], qa)    # the last question is missing
    text = "\n".join(problems)
    for needle in ("error api down", "empty answer", "differs from the question file", "appears twice", "missing", "not a question"):
        assert needle in text, needle


def test_the_launcher_starts_two_shards_per_set_and_each_resumes_from_its_own_file():
    src = open(os.path.join(HERE, "launch.ps1"), encoding="utf-8").read()
    assert src.count("Start-Process") == 1 and '@("original", "outsample")' in src and "@(1, 2)" in src
    assert "--shard $i/2" in src and "--out '$RunDir\\$tag.json'" in src and ".done" in src
