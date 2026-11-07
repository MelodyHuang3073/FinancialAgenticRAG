"""Merge the shard files of one stacking step into one answers file and check it is complete.

Each shard file is what `run_step.py --out` wrote: rows with `set` (original | outsample) and `n` (the
question's number in its question file). The merged list is in question order, original first, then
outsample. It must hold every question exactly once, with the question and doc_name of the question file,
no error and no empty answer; otherwise this stops with the list of problems and writes nothing.

    python backend/stack_runner/merge_shards.py --out answers_raw.json shard1.json shard2.json ...
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "tests")
QUESTIONS = {"original": "financebench_qa_subset.json", "outsample": "financebench_outsample_qa.json"}


def merge(shards, questions=None):
    """(merged rows, problems) for the given shard row lists."""
    questions = questions or {s: json.load(open(os.path.join(TESTS, f), encoding="utf-8")) for s, f in QUESTIONS.items()}
    seen, problems = {}, []
    for rows in shards:
        for r in rows:
            key = (r.get("set"), r.get("n"))
            if key in seen:
                problems.append(f"{key}: appears twice")
            seen[key] = r
    merged = []
    for s, qs in questions.items():
        for n, q in enumerate(qs, 1):
            r = seen.pop((s, n), None)
            if r is None:
                problems.append(f"{(s, n)}: missing")
                continue
            if r.get("error"):
                problems.append(f"{(s, n)}: error {str(r['error'])[:60]}")
            if not (r.get("model_answer") or "").strip():
                problems.append(f"{(s, n)}: empty answer")
            if r.get("question") != q["question"] or r.get("doc_name") != q["doc_name"]:
                problems.append(f"{(s, n)}: question or doc_name differs from the question file")
            merged.append(r)
    for key in seen:
        problems.append(f"{key}: not a question of the question files")
    return merged, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    merged, problems = merge([json.load(open(p, encoding="utf-8")) for p in a.shards])
    if problems:
        print(len(problems), "problems, nothing written:")
        for p in problems[:50]:
            print(" ", p)
        return 1
    json.dump(merged, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(merged)} questions merged -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
