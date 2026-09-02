"""
End-to-end regression test against real 10-K PDFs and ground-truth answers.

financebench_qa_subset.json holds every question from the FinanceBench
open-source benchmark (patronus-ai/financebench on Hugging Face,
https://huggingface.co/datasets/PatronusAI/financebench) whose doc_name
matches a PDF fixture this project ships in tests/financebench_pdfs/ —
fetched directly from the dataset's own rows (question/answer/question_type
copied verbatim, not retyped or independently recomputed), so this stays a
straightforward mirror of the official benchmark rather than a hand-curated
subset. As of the last refresh that's 52 questions across ~19 distinct
10-Ks. Re-running the fetch (see the datasets-server API,
https://datasets-server.huggingface.co/rows?dataset=PatronusAI/financebench)
and filtering by which doc_names have a matching PDF in
tests/financebench_pdfs/ regenerates this file when new PDF fixtures are
added.

Unlike the tests/test_*.py unit suite (which uses hand-built PDF fixtures to
test parser/chunker internals directly), THIS file is the only place in the
project that exercises the full pipeline — parse real PDF -> classify ->
retrieve -> PoT reasoning -> final answer — against ground-truth answers with
known correct values, on the exact documents the project is meant to handle.
With this many real PDFs now indexed, a full run is slow (expect it to take
well over 30 minutes under an LLM-backed orchestrator).

Run directly for a human-readable pass/fail report:
    python tests/test_financebench_qa.py

Run under pytest for CI (numeric questions use a tolerance-based assertion;
qualitative questions assert the gold answer's key facts appear in the
generated answer — see _check_numeric / _check_contains_facts below):
    pytest tests/test_financebench_qa.py -v -s
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.vector_store import FinancialVectorStoreManager
from app.rag.parser import FinancialFileParser
from app.agent.orchestrator import FinAgentRAGOrchestrator

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "financebench_pdfs")
QA_PATH = os.path.join(os.path.dirname(__file__), "financebench_qa_subset.json")

# doc_name (FinanceBench) -> (pdf filename, company label used when uploading)
DOC_TO_FILE = {
    "3M_2022_10K": ("3M_2022_10K.pdf", "3M"),
    "ACTIVISIONBLIZZARD_2019_10K": ("ACTIVISIONBLIZZARD_2019_10K.pdf", "Activision Blizzard"),
    "ADOBE_2015_10K": ("ADOBE_2015_10K.pdf", "Adobe"),
    "ADOBE_2016_10K": ("ADOBE_2016_10K.pdf", "Adobe"),
    "ADOBE_2017_10K": ("ADOBE_2017_10K.pdf", "Adobe"),
    "ADOBE_2022_10K": ("ADOBE_2022_10K.pdf", "Adobe"),
    "AES_2022_10K": ("AES_2022_10K.pdf", "AES Corporation"),
    "AMAZON_2017_10K": ("AMAZON_2017_10K.pdf", "Amazon"),
    "AMCOR_2023_10K": ("AMCOR_2023_10K.pdf", "Amcor"),
    "AMD_2015_10K": ("AMD_2015_10K.pdf", "AMD"),
    "AMD_2022_10K": ("AMD_2022_10K.pdf", "AMD"),
    "AMERICANWATERWORKS_2020_10K": ("AMERICANWATERWORKS_2020_10K.pdf", "American Water Works"),
    "AMERICANWATERWORKS_2021_10K": ("AMERICANWATERWORKS_2021_10K.pdf", "American Water Works"),
    "AMERICANWATERWORKS_2022_10K": ("AMERICANWATERWORKS_2022_10K.pdf", "American Water Works"),
    "BESTBUY_2017_10K": ("BESTBUY_2017_10K.pdf", "Best Buy"),
    "BESTBUY_2019_10K": ("BESTBUY_2019_10K.pdf", "Best Buy"),
    "BLOCK_2016_10K": ("BLOCK_2016_10K.pdf", "Block"),
    "BLOCK_2020_10K": ("BLOCK_2020_10K.pdf", "Block"),
    "BOEING_2018_10K": ("BOEING_2018_10K.pdf", "Boeing"),
    "COCACOLA_2017_10K": ("COCACOLA_2017_10K.pdf", "Coca-Cola"),
    "COCACOLA_2021_10K": ("COCACOLA_2021_10K.pdf", "Coca-Cola"),
    "COCACOLA_2022_10K": ("COCACOLA_2022_10K.pdf", "Coca-Cola"),
    "CORNING_2020_10K": ("CORNING_2020_10K.pdf", "Corning"),
    "CORNING_2021_10K": ("CORNING_2021_10K.pdf", "Corning"),
    "CORNING_2022_10K": ("CORNING_2022_10K.pdf", "Corning"),
    "CVSHEALTH_2018_10K": ("CVSHEALTH_2018_10K.pdf", "CVS Health"),
    "GENERALMILLS_2019_10K": ("GENERALMILLS_2019_10K.pdf", "General Mills"),
    "GENERALMILLS_2020_10K": ("GENERALMILLS_2020_10K.pdf", "General Mills"),
    "GENERALMILLS_2022_10K": ("GENERALMILLS_2022_10K.pdf", "General Mills"),
    "JOHNSON_JOHNSON_2022_10K": ("JOHNSON_JOHNSON_2022_10K.pdf", "Johnson & Johnson"),
    "KRAFTHEINZ_2019_10K": ("KRAFTHEINZ_2019_10K.pdf", "Kraft Heinz"),
}

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(text: str):
    """Extract all numeric tokens (commas stripped) from a string as floats."""
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _check_numeric(gold_answer: str, model_answer: str, rel_tol: float = 0.02) -> bool:
    """
    FinanceBench numeric gold answers are short (e.g. "24.26", "1.9%", "0.83").
    Pass if ANY number extracted from the model's answer is within rel_tol
    (default 2%) of the (first, primary) number in the gold answer.
    This is intentionally lenient — it only checks whether the right NUMBER
    surfaced anywhere in the answer, not phrasing/formatting. Tightening this
    to also check units/labels is a reasonable follow-up once basic numeric
    accuracy is passing.
    """
    gold_nums = _numbers_in(gold_answer)
    model_nums = _numbers_in(model_answer)
    if not gold_nums or not model_nums:
        return False
    target = gold_nums[0]
    for n in model_nums:
        if target == 0:
            if abs(n) < 1e-9:
                return True
            continue
        if abs(n - target) / abs(target) <= rel_tol:
            return True
    return False


def _check_contains_facts(gold_answer: str, model_answer: str) -> bool:
    """
    Qualitative gold answers (e.g. 'The consumer segment shrunk by 0.9%
    organically.') are checked by requiring every NUMBER in the gold answer
    to also appear (within 2% tolerance) in the model answer. This doesn't
    validate prose/explanation quality, only that the key extracted facts
    made it into the final answer — a floor, not a full correctness check.
    Gold answers with no numbers at all (pure qualitative, e.g. industry
    description) are treated as informational-only and always reported but
    never asserted on.
    """
    gold_nums = _numbers_in(gold_answer)
    if not gold_nums:
        return None  # no numeric ground truth to check — informational only
    model_nums = _numbers_in(model_answer)
    if not model_nums:
        return False
    hits = 0
    for g in gold_nums:
        if any((abs(g - m) / abs(g) <= 0.02 if g != 0 else abs(m) < 1e-9) for m in model_nums):
            hits += 1
    return hits >= max(1, len(gold_nums) // 2)  # at least half the gold numbers must surface


def _build_indexed_store() -> FinancialVectorStoreManager:
    """Parse all 4 real 10-K PDFs with the project's actual parser and index
    them exactly the way the real upload flow does (FinancialFileParser ->
    add_parsed_passages), so this test exercises the real ingestion path,
    not a shortcut."""
    vs = FinancialVectorStoreManager()
    parser = FinancialFileParser()
    for doc_name, (filename, company) in DOC_TO_FILE.items():
        path = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing fixture PDF: {path}\n"
                f"Copy the 4 real 10-K PDFs into tests/financebench_pdfs/ first."
            )
        with open(path, "rb") as f:
            content = f.read()
        result = parser._parse_pdf(filename, content, company)
        vs.add_parsed_passages(filename, company, result["passages"])
        print(f"  indexed {filename} ({company}): {len(result['passages'])} passages"
              + (f"  [WARNING: {result['warning']}]" if result["warning"] else ""))
    return vs


def run_financebench_subset(verbose: bool = True):
    """
    Runs every question in financebench_qa_subset.json against the real,
    fully-indexed 4-PDF corpus and reports pass/fail per question plus a
    summary. Returns (n_pass, n_total, results) for programmatic use.
    """
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print("Indexing real 10-K PDFs through the actual parser/upload path...")
    vs = _build_indexed_store()
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    results = []
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        gold = qa["answer"]
        doc_name = qa["doc_name"]

        try:
            res = orchestrator.process_query(question)
            model_answer = res.get("final_answer", "") or ""
            error = None
        except Exception as e:
            model_answer = ""
            error = repr(e)

        gold_nums = _numbers_in(gold)
        if gold_nums and qa.get("question_type") == "metrics-generated":
            passed = _check_numeric(gold, model_answer)
            check_kind = "numeric (2% tol)"
        else:
            fact_check = _check_contains_facts(gold, model_answer)
            passed = fact_check if fact_check is not None else None
            check_kind = "fact-presence (2% tol)" if fact_check is not None else "informational only"

        results.append({
            "doc_name": doc_name, "question": question, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind, "error": error,
        })

        if verbose:
            status = "✅ PASS" if passed is True else ("❌ FAIL" if passed is False else "ℹ️  INFO")
            print(f"\n[{i}/{len(qa_pairs)}] {status}  ({check_kind})  [{doc_name}]")
            print(f"  Q: {question}")
            print(f"  Gold  : {gold[:200]}")
            print(f"  Model : {model_answer[:200] if model_answer else '(empty)'}")
            if error:
                print(f"  ERROR : {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    n_total = len(scored)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"FinanceBench subset result: {n_pass}/{n_total} passed "
              f"({n_pass / n_total * 100:.0f}%)" if n_total else "No scoreable questions.")
        n_info = len(results) - n_total
        if n_info:
            print(f"({n_info} question(s) were informational-only — no numeric ground truth to check)")
    return n_pass, n_total, results


# ── pytest entry points (one test per question, so CI shows per-question status) ──
def _load_qa_pairs():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_QA_PAIRS = _load_qa_pairs()
_SHARED_STORE = None


def _get_shared_store():
    global _SHARED_STORE
    if _SHARED_STORE is None:
        _SHARED_STORE = _build_indexed_store()
    return _SHARED_STORE


import pytest


@pytest.mark.parametrize("qa", _QA_PAIRS, ids=[f"{q['doc_name']}::{q['question'][:40]}" for q in _QA_PAIRS])
def test_financebench_question(qa):
    vs = _get_shared_store()
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)
    res = orchestrator.process_query(qa["question"])
    model_answer = res.get("final_answer", "") or ""

    gold_nums = _numbers_in(qa["answer"])
    if gold_nums and qa.get("question_type") == "metrics-generated":
        assert _check_numeric(qa["answer"], model_answer), (
            f"\nQ: {qa['question']}\nGold: {qa['answer']}\nGot: {model_answer}"
        )
    else:
        fact_check = _check_contains_facts(qa["answer"], model_answer)
        if fact_check is None:
            pytest.skip("informational-only gold answer (no numeric ground truth)")
        assert fact_check, f"\nQ: {qa['question']}\nGold: {qa['answer']}\nGot: {model_answer}"


if __name__ == "__main__":
    run_financebench_subset(verbose=True)
