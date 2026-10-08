"""
Runs the out-of-sample generalization set (tests/financebench_outsample_qa.json)
against the real, fully-indexed out-of-sample PDF corpus
(tests/financebench_outsample_pdfs/) -- same real ingestion path as the
official FinanceBench runners (FinancialFileParser.parse_file() ->
add_parsed_passages(), company_name derived from the filename stem exactly
as the real /api/upload-file route does).

This set uses FY2025 10-Ks for companies that also appear in the original
150-question FinanceBench set, so the filings themselves were never seen
in that set -- it measures generalization to new filings, not memorization
of the original 150 questions' own source documents.

Usage:
    python tests/run_outsample_questions.py
Writes outsample_results.json (backend/outsample_results.json) with full
per-question detail; prints a pass/fail table to stdout.
"""
import sys, os, json, time, pickle

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import _check_numeric, _check_contains_facts
from app.rag.parser import FinancialFileParser
from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.orchestrator import FinAgentRAGOrchestrator

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "financebench_outsample_pdfs")
QA_PATH = os.path.join(os.path.dirname(__file__), "financebench_outsample_qa.json")
RESULTS_FILENAME = "outsample_results.json"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "outsample_corpus_cache.pkl")

CHECK_FNS = {
    "metrics-generated": _check_numeric,
    "domain-relevant": _check_contains_facts,
    "novel-generated": _check_contains_facts,
}


def _cache_is_fresh() -> bool:
    """False if the cache is missing or any PDF in the fixtures dir was
    modified after the cache was built (e.g. a source PDF got swapped out)."""
    if not os.path.exists(CACHE_PATH):
        return False
    cache_mtime = os.path.getmtime(CACHE_PATH)
    for filename in os.listdir(FIXTURES_DIR):
        if filename.lower().endswith(".pdf"):
            if os.path.getmtime(os.path.join(FIXTURES_DIR, filename)) > cache_mtime:
                return False
    return True


def _build_indexed_store() -> FinancialVectorStoreManager:
    vs = FinancialVectorStoreManager()
    if _cache_is_fresh():
        print(f"  loading cached corpus from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        vs.corpus = cached["corpus"]
        vs.parent_map = cached["parent_map"]
        vs.uploaded_files = cached["uploaded_files"]
        from app.tools.hybrid_retriever import HybridFinancialRetriever
        vs.retriever = HybridFinancialRetriever(vs.corpus)
        for uf in vs.uploaded_files:
            print(f"  loaded {uf['filename']} ({uf['company']}): {uf['passage_count']} passages")
        return vs

    parser = FinancialFileParser()
    for filename in sorted(os.listdir(FIXTURES_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(FIXTURES_DIR, filename)
        with open(path, "rb") as f:
            content = f.read()
        result = parser.parse_file(filename, content)
        company_name = os.path.splitext(filename)[0]
        vs.add_parsed_passages(filename, company_name, result["passages"])
        print(f"  indexed {filename} ({company_name}): {len(result['passages'])} passages"
              + (f"  [WARNING: {result['warning']}]" if result.get("warning") else ""))
    return vs


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Out-of-sample question set: {len(qa_pairs)} total")

    t0 = time.time()
    vs = _build_indexed_store()
    print(f"Indexing took {time.time() - t0:.1f}s, total passages: {len(vs.corpus)}")

    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    results = []
    for i, qa in enumerate(qa_pairs, 1):
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

        results.append({
            "doc_name": doc, "question": q, "gold": gold, "question_type": qtype,
            "model_answer": model_answer, "passed": passed,
            "error": error, "elapsed": elapsed,
        })

        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"\n[{i}/{len(qa_pairs)}] {status} ({elapsed:.1f}s) [{qtype}/{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold[:200]}")
        print(f"  Model: {model_answer[:300]}")
        if error:
            print(f"  ERROR: {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"OUT-OF-SAMPLE RESULT: {n_pass}/{len(scored)} passed"
          + (f" ({len(results) - len(scored)} informational-only, not scored)"
             if len(results) > len(scored) else ""))
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", RESULTS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
