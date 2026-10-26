"""
Runs a chosen subset of questions in single-file mode: for each question the
corpus store holds only that question's own filing (the grading condition
where one file is uploaded per question), instead of every filing at once.

Usage:
    python tests/run_single_file_subset.py --set train|out --candidates FILE --out FILE [--limit N] [--shard K/N]

--candidates is a JSON list of {"doc": <doc_name>, "q": <question prefix>};
the QA entry is resolved by doc_name and question prefix. Experiment
switches (e.g. ABLATE_POT=1) are read from the environment by the modules
they belong to, so the same script runs every configuration.
Writes --out incrementally (one record per question) so an interrupted run
keeps what it finished. --shard K/N runs every N-th candidate starting at K
(0-based), so N processes can split one list.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.corpus_store import FinancialCorpusStore
from app.tools.lexical_retriever import LexicalFinancialRetriever
from app.agent.orchestrator import FinAgentRAGOrchestrator

HERE = os.path.dirname(__file__)
SETS = {
    "train": (os.path.join(HERE, "financebench_qa_subset.json"),
              os.path.join(HERE, "original_corpus_cache.pkl")),
    "out": (os.path.join(HERE, "financebench_outsample_qa.json"),
            os.path.join(HERE, "outsample_corpus_cache.pkl")),
}


def _norm(name: str) -> str:
    # "dated" is only a filename decoration some doc_names carry (e.g. "..._8K_dated-2022-07-01")
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("dated", ""))


def _company_for(doc_name: str, companies):
    """The cache company whose name matches the benchmark doc_name (the two
    can differ in punctuation or an appended date)."""
    target = _norm(doc_name)
    by_norm = {_norm(c): c for c in companies}
    if target in by_norm:
        return by_norm[target]
    for n, c in by_norm.items():
        if target.startswith(n) or n.startswith(target):
            return c
    raise SystemExit(f"no cached filing matches doc_name {doc_name!r}")


def _single_store(cached, company: str) -> FinancialCorpusStore:
    vs = FinancialCorpusStore()
    vs.corpus = [e for e in cached["corpus"] if e.get("company") == company]
    vs.parent_map = cached["parent_map"]
    vs.uploaded_files = [u for u in cached["uploaded_files"] if u["company"] == company]
    vs.retriever = LexicalFinancialRetriever(vs.corpus)
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="")
    args = ap.parse_args()

    qa_path, cache_path = SETS[args.set]
    qa = json.load(open(qa_path, encoding="utf-8"))
    cached = pickle.load(open(cache_path, "rb"))
    companies = sorted({u["company"] for u in cached["uploaded_files"]})
    cands = json.load(open(args.candidates, encoding="utf-8"))
    if args.limit:
        cands = cands[:args.limit]
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        cands = cands[k::n]

    results, stores = [], {}
    for i, c in enumerate(cands, 1):
        matches = [x for x in qa
                   if x["doc_name"].lower().endswith(c["doc"].lower().split("/")[-1])
                   and x["question"].startswith(c["q"])]
        if len(matches) != 1:
            raise SystemExit(f"candidate {c!r} matched {len(matches)} questions")
        item = matches[0]
        company = _company_for(item["doc_name"], companies)
        if company not in stores:
            stores[company] = FinAgentRAGOrchestrator(corpus_store=_single_store(cached, company))
        t0 = time.time()
        try:
            res = stores[company].process_query(item["question"])
            answer, error = res.get("final_answer", "") or "", None
        except Exception as e:  # keep going; the failure is part of the record
            answer, error = "", repr(e)
        results.append({
            "doc_name": item["doc_name"], "question": item["question"], "gold": item["answer"],
            "question_type": item["question_type"], "model_answer": answer, "error": error,
            "elapsed": time.time() - t0, "mode": "single-file",
            "ablate_pot": os.environ.get("ABLATE_POT") == "1",
        })
        json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{i}/{len(cands)}] {item['doc_name']} ({time.time() - t0:.0f}s)\n  Q: {item['question'][:100]}\n"
              f"  Gold : {item['answer'][:160]}\n  Model: {answer[:260]}\n", flush=True)


if __name__ == "__main__":
    main()
