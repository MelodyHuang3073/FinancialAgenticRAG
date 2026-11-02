"""Plain-RAG control: standard BM25 over a clean corpus, straight to the model.

Nothing of the FinAgent-RAG system is imported or used. For each question:
  1. take the chunks of the question's own filing (doc_name) -- one filing per question;
  2. standard Okapi BM25 (rank_bm25 defaults: k1=1.5, b=0.75, epsilon=0.25), the query and
     the chunks lower-cased and split on whitespace, nothing else;
  3. the 16 highest-scoring chunks, joined with a blank line, are the context;
  4. one model call whose whole prompt is the context and the question.
The answer is stored as returned.

    python backend/clean_baseline/run.py --set original --shard 1/4 --out single_clean_1.json
    python backend/clean_baseline/run.py --set original --dry-run      # no API call
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
SETS = {
    "original": ("financebench_qa_subset.json", "original_clean_corpus_cache.pkl"),
    "outsample": ("financebench_outsample_qa.json", "outsample_clean_corpus_cache.pkl"),
}
TOP_K = 16


def tokenize(text: str):
    return text.lower().split()


def find_doc(corpus, doc_name: str) -> str:
    """The corpus key for a question's doc_name. The question files and the PDF file names
    differ in punctuation or an appended word ("dated"), so compare letters and digits only."""
    def norm(x):
        return re.sub(r"[^a-z0-9]", "", x.lower().replace("dated", ""))
    matches = [k for k in corpus if norm(k) == norm(doc_name)]
    if len(matches) != 1:
        raise SystemExit(f"doc_name {doc_name!r} matches {len(matches)} filings")
    return matches[0]


def retrieve(bm25: BM25Okapi, query: str, k: int = TOP_K):
    """Indices of the k best chunks, best first (ties keep corpus order)."""
    scores = bm25.get_scores(tokenize(query))
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def build_prompt(chunks, question: str) -> str:
    return "Context:\n" + "\n\n".join(chunks) + "\n\nQuestion: " + question


def answer(client, model: str, prompt: str):
    """One call, no system message, no sampling or reasoning settings. Returns (text, usage)."""
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            u = r.usage
            return (r.choices[0].message.content or ""), ((u.prompt_tokens, u.completion_tokens) if u else None)
        except Exception as e:  # rate limit or transient error: wait and retry
            last = e
            time.sleep(10 * (attempt + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--shard", default="1/1", help="i/n: questions whose (number - 1) %% n equals i - 1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--dry-run", action="store_true", help="retrieve and print the first prompt; no API call")
    a = ap.parse_args()
    qa_file, cache_file = SETS[a.set]
    questions = json.load(open(os.path.join(TESTS, qa_file), encoding="utf-8"))
    corpus = pickle.load(open(os.path.join(TESTS, cache_file), "rb"))
    i, n = (int(x) for x in a.shard.split("/"))
    todo = [(k, q) for k, q in enumerate(questions, 1) if (k - 1) % n == i - 1]

    client = None
    if not a.dry_run:
        from openai import OpenAI
        client = OpenAI()
    done = {}
    if a.out and os.path.exists(a.out):
        done = {r["n"]: r for r in json.load(open(a.out, encoding="utf-8"))}
    indexes = {}
    for k, q in todo:
        if k in done:
            continue
        name = find_doc(corpus, q["doc_name"])
        if name not in indexes:
            indexes[name] = BM25Okapi([tokenize(c) for c in corpus[name]])
        top = retrieve(indexes[name], q["question"], a.top_k)
        prompt = build_prompt([corpus[name][j] for j in top], q["question"])
        if a.dry_run:
            print(prompt)
            print("\nretrieved chunk indices:", top)
            return
        t0 = time.time()
        row = {"set": a.set, "n": k, "doc_name": q["doc_name"], "question_type": q["question_type"],
               "question": q["question"], "gold": q["answer"], "model_answer": "", "error": None,
               "retrieved": top, "tokens": None}
        try:
            row["model_answer"], row["tokens"] = answer(client, a.model, prompt)
        except Exception as e:
            row["error"] = str(e)[:300]
        row["elapsed"] = round(time.time() - t0, 1)
        done[k] = row
        print(f"[{a.set} {k}] {row['elapsed']}s tokens={row['tokens']} err={row['error']}", flush=True)
        if a.out:
            json.dump([done[x] for x in sorted(done)], open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
