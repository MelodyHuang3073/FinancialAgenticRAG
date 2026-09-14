"""
Runs both official FinanceBench categories currently in active focus --
"domain-relevant" (50q, fact-presence graded) and "metrics-generated"
(50q, 2%-tolerance numeric graded) -- against a SINGLE indexed corpus
build, instead of invoking run_domain_relevant_questions.py and
run_metrics_generated_questions.py separately (each of which re-parses
all ~63 PDFs from scratch on its own). One parse pass, both question
sets, one run.

Usage:
    python tests/run_domain_and_metrics_combined.py
Writes domain_relevant_results.json AND metrics_generated_results.json
(backend/*.json), same format/filenames as the two separate scripts, plus
prints a combined summary at the end.
"""
import sys, os, json, time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import (
    QA_PATH, _build_indexed_store, _check_contains_facts, _check_numeric,
    _available_doc_names,
)
from app.agent.orchestrator import FinAgentRAGOrchestrator

SUITES = [
    ("domain-relevant", _check_contains_facts, "domain_relevant_results.json", "fact-presence (2% tol)"),
    ("metrics-generated", _check_numeric, "metrics_generated_results.json", "numeric (2% tol)"),
]


def run_suite(orchestrator, all_qa, available, question_type, check_fn, results_filename, check_kind_label):
    qa_pairs = [qa for qa in all_qa if qa.get("question_type") == question_type]
    runnable = [qa for qa in qa_pairs if qa["doc_name"] in available]
    skipped = [qa for qa in qa_pairs if qa["doc_name"] not in available]

    print(f"\n{'#' * 70}")
    print(f"# {question_type} question set: {len(qa_pairs)} total "
          f"({len(runnable)} runnable, {len(skipped)} skipped — no PDF fixture)")
    print(f"{'#' * 70}")
    if skipped:
        missing_docs = sorted({qa["doc_name"] for qa in skipped})
        print(f"  Missing PDF fixtures for {len(missing_docs)} doc(s): {', '.join(missing_docs)}")
        for qa in skipped:
            print(f"  SKIP (no PDF): [{qa['doc_name']}] {qa['question'][:70]}")

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

        check_result = check_fn(gold, model_answer)
        passed = check_result if check_result is not None else None

        results.append({
            "doc_name": doc, "question": q, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind_label if check_result is not None else "informational only",
            "error": error, "elapsed": elapsed,
        })
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"\n[{question_type} {i}/{len(runnable)}] {status} ({elapsed:.1f}s) [{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold}")
        print(f"  Model: {model_answer[:250]}")
        if error:
            print(f"  ERROR: {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"{question_type.upper()} QUESTIONS RESULT: {n_pass}/{len(scored)} passed"
          + (f" ({len(results) - len(scored)} informational-only, not scored)"
             if len(results) > len(scored) else ""))
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", results_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return n_pass, len(scored)


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = json.load(f)
    available = _available_doc_names()

    t0 = time.time()
    vs = _build_indexed_store()
    print(f"Indexing took {time.time() - t0:.1f}s, total passages: {len(vs.corpus)}")

    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    summary = []
    for question_type, check_fn, results_filename, check_kind_label in SUITES:
        n_pass, n_scored = run_suite(
            orchestrator, all_qa, available, question_type, check_fn,
            results_filename, check_kind_label,
        )
        summary.append((question_type, n_pass, n_scored))

    print(f"\n{'*' * 70}")
    print("COMBINED SUMMARY")
    for question_type, n_pass, n_scored in summary:
        print(f"  {question_type}: {n_pass}/{n_scored} passed")
    total_pass = sum(p for _, p, _ in summary)
    total_scored = sum(s for _, _, s in summary)
    print(f"  TOTAL: {total_pass}/{total_scored} passed")
    print(f"{'*' * 70}")


if __name__ == "__main__":
    main()
