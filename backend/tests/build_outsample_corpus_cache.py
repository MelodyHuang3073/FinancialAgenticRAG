"""
Parses every PDF in tests/financebench_outsample_pdfs/ through the real
FinancialFileParser.parse_file() (the same public entry point
run_outsample_questions.py and /api/upload-file both use) and pickles the
resulting corpus so future diagnostic runs against this out-of-sample set
don't have to re-parse 31 PDFs from scratch every time (parsing alone takes
close to half an hour).

This cache is independent of the original 150-question FinanceBench corpus
cache -- it only covers the out-of-sample FY2025 filings.

Usage:
    python tests/build_outsample_corpus_cache.py
Writes tests/outsample_corpus_cache.pkl.
"""
import sys
import os
import pickle
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.parser import FinancialFileParser

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "financebench_outsample_pdfs")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "outsample_corpus_cache.pkl")


def main():
    parser = FinancialFileParser()
    corpus = []
    parent_map = {}
    uploaded_files = []

    t0 = time.time()
    for filename in sorted(os.listdir(FIXTURES_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(FIXTURES_DIR, filename)
        with open(path, "rb") as f:
            content = f.read()
        result = parser.parse_file(filename, content)
        company_name = os.path.splitext(filename)[0]
        passages = result["passages"]

        for p in passages:
            pid = p.get("parent_id")
            pc = p.get("parent_content")
            if pid and pc and pid not in parent_map:
                parent_map[pid] = pc
        corpus.extend(passages)
        uploaded_files.append({
            "filename": filename,
            "company": company_name,
            "passage_count": len(passages),
        })
        print(f"  parsed {filename} ({company_name}): {len(passages)} passages"
              + (f"  [WARNING: {result['warning']}]" if result.get("warning") else ""))

    print(f"Parsing took {time.time() - t0:.1f}s, total passages: {len(corpus)}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "corpus": corpus,
            "parent_map": parent_map,
            "uploaded_files": uploaded_files,
        }, f)
    print(f"Wrote cache to {CACHE_PATH}")


if __name__ == "__main__":
    main()
