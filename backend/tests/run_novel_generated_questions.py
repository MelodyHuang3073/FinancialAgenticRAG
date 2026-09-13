"""
Runs every FinanceBench question tagged question_type="novel-generated" —
one of the three official categories the benchmark itself splits all 150
questions into (the other two are "metrics-generated" and
"domain-relevant", see run_metrics_generated_questions.py /
run_domain_relevant_questions.py). This is a full-dataset regression by
FinanceBench's own category; unlike the other two, no existing hand-picked
question list in this project covers this category at all.

"novel-generated" questions are open-ended analytical/interpretive ones
that require reasoning across multiple facts rather than reading a single
line item (e.g. "If we exclude the impact of M&A, which segment has
dragged down 3M's overall growth in 2022?", "Does 3M maintain a stable
trend of dividend distribution?") — graded with the same fact-presence
check (_check_contains_facts) used for domain-relevant questions, since
these gold answers are also prose/reasoning, not a single bare number.

Pulls the full question set directly from financebench_qa_subset.json
(all 150 FinanceBench questions) rather than a hand-curated list, so this
automatically covers every novel-generated question as soon as its PDF
fixture is added to tests/financebench_pdfs/ — no list to keep in sync.

Usage:
    python tests/run_novel_generated_questions.py
Writes novel_generated_results.json (backend/novel_generated_results.json)
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
    QA_PATH, _build_indexed_store, _check_contains_facts, _available_doc_names,
)
from app.agent.orchestrator import FinAgentRAGOrchestrator

QUESTION_TYPE = "novel-generated"
RESULTS_FILENAME = "novel_generated_results.json"


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

        fact_check = _check_contains_facts(gold, model_answer)
        passed = fact_check if fact_check is not None else None
        check_kind = "fact-presence (2% tol)" if fact_check is not None else "informational only"

        results.append({
            "doc_name": doc, "question": q, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind, "error": error, "elapsed": elapsed,
        })
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"\n[{i}/{len(runnable)}] {status} ({elapsed:.1f}s) [{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold}")
        print(f"  Model: {model_answer[:250]}")
        if error:
            print(f"  ERROR: {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"{QUESTION_TYPE.upper()} QUESTIONS RESULT: {n_pass}/{len(scored)} passed"
          + (f" ({len(results) - len(scored)} informational-only, not scored)"
             if len(results) > len(scored) else ""))
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", RESULTS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
