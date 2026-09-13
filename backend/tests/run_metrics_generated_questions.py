"""
Runs every FinanceBench question tagged question_type="metrics-generated" —
one of the three official categories the benchmark itself splits all 150
questions into (the other two are "domain-relevant" and "novel-generated",
see run_domain_relevant_questions.py / run_novel_generated_questions.py).
This is a full-dataset regression by FinanceBench's own category, distinct
from run_calc_questions.py's separate hand-picked "計算題" set (which draws
from all three categories and tracks a different, curated question list).

"metrics-generated" questions ask for one specific computed/extracted
number (a ratio, a line-item figure, a period-over-period change) — graded
with a 2%-tolerance numeric check (_check_numeric) against the FIRST number
in the gold answer, since these gold answers are short bare figures like
"24.26" or "1.9%".

Pulls the full question set directly from financebench_qa_subset.json
(all 150 FinanceBench questions) rather than a hand-curated list, so this
automatically covers every metrics-generated question as soon as its PDF
fixture is added to tests/financebench_pdfs/ — no list to keep in sync.

Usage:
    python tests/run_metrics_generated_questions.py
Writes metrics_generated_results.json (backend/metrics_generated_results.json)
with full per-question detail; prints a pass/fail table to stdout.
"""
import sys, os, json, time

# See run_extraction_questions.py's own copy of this comment for why this
# is needed: FinanceBench gold answers routinely contain non-ASCII
# characters that crash a plain print() under Windows' default console
# codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import (
    QA_PATH, _build_indexed_store, _check_numeric, _available_doc_names,
)
from app.agent.orchestrator import FinAgentRAGOrchestrator

QUESTION_TYPE = "metrics-generated"
RESULTS_FILENAME = "metrics_generated_results.json"


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = json.load(f)
    qa_pairs = [qa for qa in all_qa if qa.get("question_type") == QUESTION_TYPE]

    available = _available_doc_names()
    runnable = [qa for qa in qa_pairs if qa["doc_name"] in available]
    skipped = [qa for qa in qa_pairs if qa["doc_name"] not in available]

    print(f"{QUESTION_TYPE} question set: {len(qa_pairs)} total "
          f"({len(runnable)} runnable, {len(skipped)} skipped — no PDF fixture)")
    if skipped:
        missing_docs = sorted({qa["doc_name"] for qa in skipped})
        print(f"  Missing PDF fixtures for {len(missing_docs)} doc(s): {', '.join(missing_docs)}")
        for qa in skipped:
            print(f"  SKIP (no PDF): [{qa['doc_name']}] {qa['question'][:70]}")

    t0 = time.time()
    vs = _build_indexed_store()
    print(f"Indexing took {time.time() - t0:.1f}s, total passages: {len(vs.corpus)}")

    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    results = []
    for i, qa in enumerate(runnable, 1):
        q = qa["question"]
        gold = qa["answer"]
        doc = qa["doc_name"]
        t1 = time.time()
        try:
            res = orchestrator.process_query(q)
            model_answer = res.get("final_answer", "") or ""
            error = None
        except Exception as e:
            model_answer = ""
            error = repr(e)
        elapsed = time.time() - t1

        passed = _check_numeric(gold, model_answer)

        results.append({
            "doc_name": doc, "question": q, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": "numeric (2% tol)", "error": error, "elapsed": elapsed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"\n[{i}/{len(runnable)}] {status} ({elapsed:.1f}s) [{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold}")
        print(f"  Model: {model_answer[:250]}")
        if error:
            print(f"  ERROR: {error}")

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"{QUESTION_TYPE.upper()} QUESTIONS RESULT: {n_pass}/{len(results)} passed")
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", RESULTS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
