"""
Parses every PDF the original 150-question FinanceBench suite ships a
fixture for (per DOC_TO_FILE in test_financebench_qa.py) through the real
FinancialFileParser.parse_file() and pickles the resulting corpus, so
future diagnostic runs against the original corpus don't have to
re-parse 84 PDFs from scratch every time.

Independent of the out-of-sample corpus cache (tests/outsample_corpus_cache.pkl).
Local-only artifact -- not pushed to GitHub (see .gitignore).

Usage:
    python tests/build_original_corpus_cache.py
Writes tests/original_corpus_cache.pkl.
"""
import sys
import os
import pickle
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from app.rag.parser import FinancialFileParser
from test_financebench_qa import DOC_TO_FILE, FIXTURES_DIR

CACHE_PATH = os.path.join(os.path.dirname(__file__), "original_corpus_cache.pkl")


def main():
    parser = FinancialFileParser()
    corpus = []
    parent_map = {}
    uploaded_files = []

    t0 = time.time()
    seen_filenames = set()
    for doc_name, (filename, _label) in DOC_TO_FILE.items():
        if filename in seen_filenames:
            continue
        path = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(path):
            print(f"  skip {filename} (no PDF fixture)")
            continue
        seen_filenames.add(filename)
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

    print(f"Parsing took {time.time() - t0:.1f}s, total passages: {len(corpus)}, files: {len(seen_filenames)}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "corpus": corpus,
            "parent_map": parent_map,
            "uploaded_files": uploaded_files,
        }, f)
    print(f"Wrote cache to {CACHE_PATH}")


if __name__ == "__main__":
    main()
