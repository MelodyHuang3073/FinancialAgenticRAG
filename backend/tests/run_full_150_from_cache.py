"""
Runs the complete original 150-question FinanceBench set (all three
official categories: domain-relevant, metrics-generated, novel-generated)
in one pass, loading the corpus from tests/original_corpus_cache.pkl
instead of re-parsing all 84 PDFs (per [[avoid-double-reparse]] -- build
the cache once with tests/build_original_corpus_cache.py, then reuse it
here instead of also paying for test_financebench_qa.py's own from-scratch
re-parse in the same round).

Writes the exact same three result files the official per-category runners
write (domain_relevant_results.json, metrics_generated_results.json,
novel_generated_results.json), in the exact same per-entry schema, so this
is a drop-in way to refresh all three from one indexed corpus instead of
running run_domain_relevant_questions.py / run_metrics_generated_questions.py
/ run_novel_generated_questions.py separately (which would each re-parse
all 84 PDFs on their own).

Usage:
    python tests/run_full_150_from_cache.py
Requires tests/original_corpus_cache.pkl to exist and be up to date (see
tests/build_original_corpus_cache.py) -- fails fast with a clear message
if it's missing, rather than silently falling back to a slow re-parse.
"""
import sys, os, json, time, pickle

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import QA_PATH, _check_numeric, _check_contains_facts, _available_doc_names
from app.rag.vector_store import FinancialVectorStoreManager
from app.tools.hybrid_retriever import HybridFinancialRetriever
from app.agent.orchestrator import FinAgentRAGOrchestrator

CACHE_PATH = os.path.join(os.path.dirname(__file__), "original_corpus_cache.pkl")

CHECK_FNS = {
    "metrics-generated": _check_numeric,
    "domain-relevant": _check_contains_facts,
    "novel-generated": _check_contains_facts,
}
RESULTS_FILENAMES = {
    "metrics-generated": "metrics_generated_results.json",
    "domain-relevant": "domain_relevant_results.json",
    "novel-generated": "novel_generated_results.json",
}
CHECK_KIND_LABELS = {
    "metrics-generated": "numeric (2% tol)",
    "domain-relevant": "fact-presence (2% tol)",
    "novel-generated": "fact-presence (2% tol)",
}


def _build_indexed_store_from_cache() -> FinancialVectorStoreManager:
    if not os.path.exists(CACHE_PATH):
        raise SystemExit(
            f"Missing {CACHE_PATH}. Run tests/build_original_corpus_cache.py first "
            "(this script deliberately does not fall back to a from-scratch re-parse)."
        )
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    vs = FinancialVectorStoreManager()
    vs.corpus = cached["corpus"]
    vs.parent_map = cached["parent_map"]
    vs.uploaded_files = cached["uploaded_files"]
    vs.retriever = HybridFinancialRetriever(vs.corpus)
    for uf in vs.uploaded_files:
        print(f"  loaded {uf['filename']} ({uf['company']}): {uf['passage_count']} passages")
    return vs


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"Full question set: {len(qa_pairs)} total")

    t0 = time.time()
    vs = _build_indexed_store_from_cache()
    print(f"Loaded cached corpus in {time.time() - t0:.1f}s, total passages: {len(vs.corpus)}")

    # Uses the official benchmark doc_name -> PDF-existence mapping, not the
    # cache's own company metadata (filename-stem derived) -- these can
    # legitimately differ (e.g. benchmark doc_name
    # "AMCOR_2022_8K_dated-2022-07-01" vs the actual file's stem
    # "AMCOR_2022_8K_2022-07-01"), and the retrieval path's own entity
    # matching handles that difference fine; only the "is this question
    # runnable at all" check needs to agree with the official runners.
    available_docs = _available_doc_names()
    runnable = [qa for qa in qa_pairs if qa["doc_name"] in available_docs]
    skipped = [qa for qa in qa_pairs if qa["doc_name"] not in available_docs]
    if skipped:
        missing_docs = sorted({qa["doc_name"] for qa in skipped})
        print(f"  SKIP {len(skipped)} question(s) — no PDF fixture cached for: {', '.join(missing_docs)}")

    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    results_by_type = {k: [] for k in RESULTS_FILENAMES}
    for i, qa in enumerate(runnable, 1):
        q = qa["question"]
        gold = qa["answer"]
        doc = qa["doc_name"]
        qtype = qa["question_type"]
        check_fn = CHECK_FNS[qtype]
        t1 = time.time()
        try:
            res = orchestrator.process_query(q)
            model_answer = res.get("final_answer", "") or ""
            error = None
        except Exception as e:
            model_answer = ""
            error = repr(e)
        elapsed = time.time() - t1

        check_result = check_fn(gold, model_answer) if not error else None
        passed = check_result
        # Match each official runner's own semantics exactly: metrics-generated
        # is never "informational only" (passed is always True/False), the
        # other two can be None (gold has no checkable facts/numbers at all).
        check_kind = CHECK_KIND_LABELS[qtype] if check_result is not None else "informational only"

        results_by_type[qtype].append({
            "doc_name": doc, "question": q, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind, "error": error, "elapsed": elapsed,
        })

        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"\n[{i}/{len(runnable)}] {status} ({elapsed:.1f}s) [{qtype}/{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold[:200]}")
        print(f"  Model: {model_answer[:300]}")
        if error:
            print(f"  ERROR: {error}")

    print(f"\n{'=' * 70}")
    for qtype, filename in RESULTS_FILENAMES.items():
        type_results = results_by_type[qtype]
        scored = [r for r in type_results if r["passed"] is not None]
        n_pass = sum(1 for r in scored if r["passed"])
        print(f"{qtype.upper()}: {n_pass}/{len(scored)} passed"
              + (f" ({len(type_results) - len(scored)} informational-only)"
                 if len(type_results) > len(scored) else ""))
        out_path = os.path.join(os.path.dirname(__file__), "..", filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(type_results, f, ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
