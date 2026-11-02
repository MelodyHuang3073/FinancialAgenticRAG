"""Clean corpus for the plain-RAG control: nothing but text.

Output: {doc_name: [chunk string, ...]} -- one pickle per question set.
Each PDF's text is read as-is (PyMuPDF), pages joined with a newline, then cut into
fixed 1000-character windows with a stride of 800 (200 overlap). There is no page
number, id, type or company field, no header, no table handling and no whitespace
clean-up. The doc_name (the PDF file name) is only the key that says which filing a
question is run against; it is not part of any chunk.

    python backend/clean_baseline/build_corpus.py
writes backend/tests/original_clean_corpus_cache.pkl and outsample_clean_corpus_cache.pkl
(git-ignored by the *_corpus_cache.pkl rule).
"""
import os
import pickle

import pymupdf

TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
SIZE, STEP = 1000, 800
SETS = {"original": "financebench_pdfs", "outsample": "financebench_outsample_pdfs"}


def chunk(text: str):
    out = []
    for start in range(0, len(text), STEP):
        out.append(text[start:start + SIZE])
        if start + SIZE >= len(text):
            break
    return out


def build(pdf_dir: str):
    corpus = {}
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith(".pdf"):
            with pymupdf.open(os.path.join(pdf_dir, fn)) as doc:
                text = "\n".join(page.get_text() for page in doc)
            corpus[os.path.splitext(fn)[0]] = chunk(text)
    return corpus


if __name__ == "__main__":
    for name, folder in SETS.items():
        data = build(os.path.join(TESTS, folder))
        with open(os.path.join(TESTS, f"{name}_clean_corpus_cache.pkl"), "wb") as f:
            pickle.dump(data, f)
        print(name, len(data), "docs", sum(map(len, data.values())), "chunks")
