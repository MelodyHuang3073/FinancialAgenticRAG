"""Merge the four shard files of one ablation row and check the run is complete and really that row.

    python backend/ablation_runner/merge_check.py --row 5 --run-dir <RunDir> --out answers_raw.json

Stops (and writes nothing) when: a question is missing or appears twice; a record has an error, an empty
answer or a fallback-shaped answer (the system's own rule-based synthesis, which means a model call failed);
or a record was produced with another ablation row than --row (every record carries the row it ran with,
and config.json carries the row and the code commit the run was started from). Prints the problems found.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "..", "experiment_results")
FALLBACK = ("According to the financial report disclosures", "The calculation result for",
            "Analysis based on retrieved financial evidence", "Based on retrieved financial data for",
            "could not be obtained")


def merge(run_dir, row):
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8-sig"))
    problems = []
    if str(cfg["ablate_row"]) != row:
        problems.append(f"config.json says row {cfg['ablate_row']}, expected {row}")
    out = {}
    for tag, cand_file in (("train", "all_candidates_train.json"), ("out", "all_candidates_out.json")):
        cands = json.load(open(os.path.join(ABL, cand_file), encoding="utf-8"))
        rows = []
        for i in (0, 1):
            path = os.path.join(run_dir, f"{tag}_{i}.json")
            rows += json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        got = {(r["doc_name"], r["question"]): r for r in rows}
        if len(got) != len(rows):
            problems.append(f"{tag}: a question appears twice")
        if len(rows) != len(cands):
            problems.append(f"{tag}: {len(rows)} records, expected {len(cands)}")
        for r in rows:
            who = f"{tag} {r['doc_name']} / {r['question'][:50]}"
            ans = r.get("model_answer") or ""
            if r.get("error"):
                problems.append(f"{who}: error {r['error'][:60]}")
            elif not ans.strip():
                problems.append(f"{who}: empty answer")
            elif any(f in ans[:200] for f in FALLBACK):
                problems.append(f"{who}: fallback-shaped answer")
            if str((r.get("ablation") or {}).get("ablate_row")) != row:
                problems.append(f"{who}: produced with {r.get('ablation')}")
        out[tag] = rows
    return out, problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", choices=["1", "2a", "2b", "3", "4", "5"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merged, problems = merge(a.run_dir, a.row)
    for p in problems:
        print("PROBLEM", p)
    if problems:
        sys.exit(1)
    json.dump(merged, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"row {a.row}: {sum(len(v) for v in merged.values())} questions, complete -> {a.out}")
