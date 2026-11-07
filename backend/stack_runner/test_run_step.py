"""Offline checks for the stacking runner, step 1 (no API call, no key needed)."""
import json
import os
import pickle
import sys

import pytest
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_step as r  # noqa: E402

SYSTEM_PARTS = ("orchestrator", "pot_reasoner", "verifier", "refiner", "decomposer", "question_classifier",
                "lexical_retriever", "financial_formula_library", "financial_vocabulary")


def _row(i, company="ACME", table="T", content=None, parent="PARENT PAGE", typ="table_row", pid="p1"):
    return {"id": f"e{i}", "company": company, "table_name": table, "type": typ, "parent_id": pid,
            "parent_content": parent, "content": content or f"row {i}"}


def test_step_one_pulls_in_no_system_component_beyond_the_evidence_display():
    loaded = [m for m in sys.modules if m.startswith("app.") and m.split(".")[-1] in SYSTEM_PARTS]
    assert loaded == []


def test_only_step_one_exists_and_another_step_stops_instead_of_running(monkeypatch):
    assert sorted(r.STEPS) == [1]
    monkeypatch.setattr(sys, "argv", ["run_step.py", "--step", "2", "--set", "original", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        r.main()
    assert "not implemented" in str(e.value)


def test_prompt_layout_is_step_zeros_around_the_evidence_lines():
    q = "What is X?"
    assert r.build_prompt(["only line"], q) == r.naive.build_prompt(["only line"], q)         # one item: identical
    p = r.build_prompt(["- a", "- b"], q)
    assert p == "Context:\n- a\n- b\n\nQuestion: What is X?"                                 # lines joined by one newline


def test_evidence_line_is_exactly_the_systems_label_plus_the_whole_parent_page_and_nothing_else():
    line = r.evidence_line(_row(1, company="ACME_2020_10K", table="p46 table", parent="THE WHOLE PAGE"))
    assert line == "- [ACME_2020_10K / p46 table] THE WHOLE PAGE"
    # a missing parent page falls back to the parent map, then to the passage itself
    e = _row(2, parent="", pid="px")
    assert r.evidence_line(e, {"px": "FROM MAP"}) == "- [ACME / T] FROM MAP"
    assert r.evidence_line(e, {}) == "- [ACME / T] row 2"


def test_no_tag_line_anywhere_even_when_the_chunk_has_period_section_page_and_repair_fields():
    e = _row(5, content="Company: A | Line Item: Revenue")
    e.update(period="2024", section="income_statement", page_number="46", table_repair_source="mineru")
    line = r.evidence_line(e)
    assert line == "- [ACME / T] PARENT PAGE"
    for word in ("Type:", "Period:", "Section:", "Page:", "Repair:", "2024", "income_statement", "mineru"):
        assert word not in line
    assert r.index_text(e) == "Company: A | Line Item: Revenue"          # BM25 indexes the content only
    assert not hasattr(r, "tag_text")


def test_long_pages_are_cut_the_way_the_system_cuts_them():
    page = "x" * 9000
    line = r.evidence_line(_row(3, parent=page, typ="text_note", content="Content: x" + "y" * 40))
    assert len(line) < 4300 and line.startswith("- [ACME / T] ")
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| r{i} | {i} |" for i in range(80))
    shown = r.evidence_line(_row(4, parent=table))
    assert "r29" in shown and "r30" not in shown and "more rows omitted" in shown


def test_retrieval_is_step_zeros_rank_bm25_on_the_entry_content_and_keeps_duplicate_pages():
    filing = [_row(i, content=f"revenue growth {i}" if i < 4 else f"other words {i}", pid="same") for i in range(8)]
    bm25 = BM25Okapi([r.naive.tokenize(r.index_text(e)) for e in filing])
    prompt, top = r.step1_prompt(filing, bm25, "revenue growth", 3)
    assert top == r.naive.retrieve(bm25, "revenue growth", 3)
    assert set(top) <= {0, 1, 2, 3}
    assert prompt.count("PARENT PAGE") == 3                    # the same page, shown once per hit


def test_the_systems_period_and_section_fields_do_not_change_what_is_retrieved():
    plain = [_row(i, content=f"revenue growth {i}" if i % 2 else f"other words {i}") for i in range(8)]
    tagged = [dict(e, period="2024", section="income_statement", page_number=str(i)) for i, e in enumerate(plain)]
    a = BM25Okapi([r.naive.tokenize(r.index_text(e)) for e in plain])
    b = BM25Okapi([r.naive.tokenize(r.index_text(e)) for e in tagged])
    assert r.naive.retrieve(a, "income statement 2024", 4) == r.naive.retrieve(b, "income statement 2024", 4)


def test_find_filing_needs_exactly_one_match():
    assert r.find_filing(["A_2020_10K", "B_2021_10K"], "A_2020_10K") == "A_2020_10K"
    assert r.find_filing(["A_2020_10K_dated-2020-01-01"], "A_2020_10K") == "A_2020_10K_dated-2020-01-01"
    with pytest.raises(SystemExit):
        r.find_filing(["A_2020_10K", "B_2021_10K"], "C_2022_10K")


def _ctx(entries, model="gpt-5-mini"):
    return {"filing": entries, "model": model, "bm25": BM25Okapi([r.naive.tokenize(r.index_text(e)) for e in entries])}


def test_step_one_sends_exactly_what_step_zero_would_send():
    filing = [_row(i, content=f"Line Item: revenue item {i} | 2024: {i}", parent=f"PAGE {i}", pid=f"p{i}") for i in range(5)]
    messages, _ = r.STEPS[1](_ctx(filing), "revenue item 2", 16)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("Context:\n- [ACME / T] PAGE")
    assert messages[0]["content"].endswith("\n\nQuestion: revenue item 2")


def test_send_returns_text_and_tokens_and_passes_only_model_and_messages():
    class Client:
        def __init__(self):
            self.kwargs = []
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            self.kwargs.append(kw)
            usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 3})()
            msg = type("M", (), {"content": "answer"})()
            return type("R", (), {"usage": usage, "choices": [type("Ch", (), {"message": msg})()]})()

    c = Client()
    msgs = [{"role": "user", "content": "u"}]
    assert r.send(c, "gpt-5-mini", msgs) == ("answer", (11, 3))
    assert c.kwargs == [{"model": "gpt-5-mini", "messages": msgs}]


CACHE = os.path.join(r.TESTS, "original_corpus_cache.pkl")


@pytest.mark.skipif(not os.path.exists(CACHE), reason="corpus cache not present")
def test_first_real_question_builds_a_sixteen_entry_prompt_with_no_tag_line():
    cached = pickle.load(open(CACHE, "rb"))
    q = json.load(open(os.path.join(r.TESTS, "financebench_qa_subset.json"), encoding="utf-8"))[0]
    companies = sorted({e["company"] for e in cached["corpus"] if e.get("company")})
    filing = r.filing_entries(cached["corpus"], r.find_filing(companies, q["doc_name"]))
    assert filing and {e["company"] for e in filing} == {filing[0]["company"]}                 # one filing only
    bm25 = BM25Okapi([r.naive.tokenize(r.index_text(e)) for e in filing])
    prompt, top = r.step1_prompt(filing, bm25, q["question"], 16, cached.get("parent_map"))
    assert len(top) == 16 and prompt.startswith("Context:\n- [") and prompt.endswith("\n\nQuestion: " + q["question"])
    assert prompt.count("\n- [") == 16 and prompt.split("Context:")[0] == ""      # 16 evidence lines, nothing before "Context:"
    # every evidence line is "- [company / table] page": the label is followed directly by the page, no tag line
    for line in prompt.split("\n\nQuestion:")[0].split("\n- [")[1:]:
        label, rest = line.split("] ", 1)
        assert " / " in label and not rest.startswith("(")
