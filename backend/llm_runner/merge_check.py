"""Merge the four shard files of one model-swap run and check the run is complete and really that model.

    python backend/llm_runner/merge_check.py --model gpt-5-nano --run-dir <RunDir> --out answers_raw.json

Stops (and writes nothing) when: a question is missing or appears twice; a record has an error, an empty
answer or a fallback-shaped answer (the system's own rule-based synthesis, which means a model call failed);
config.json carries another model than --model; or the usage ledger lines written since the run started
(backend/.llm_usage; --ledger-since overrides the start time) contain a call by any other model.
Prints the problems found and, when nothing is wrong, the number of calls and tokens per caller.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from failed import failure  # noqa: E402

EXP = os.path.join(HERE, "..", "experiment_results")
LEDGER = os.path.join(HERE, "..", ".llm_usage")


def merge(run_dir, model):
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8-sig"))
    problems = []
    if cfg["model"] != model:
        problems.append(f"config.json says model {cfg['model']}, expected {model}")
    out = {}
    for tag, cand_file in (("train", "all_candidates_train.json"), ("out", "all_candidates_out.json")):
        cands = json.load(open(os.path.join(EXP, cand_file), encoding="utf-8"))
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
            why = failure(r)
            if why:
                problems.append(f"{tag} {r['doc_name']} / {r['question'][:50]}: {why}")
        out[tag] = rows
    return cfg, out, problems


def ledger(model, since, ledger_dir=LEDGER):
    """(problems, calls per caller, tokens per caller) of the ledger lines written at or after `since`."""
    calls, tokens, problems = collections.Counter(), collections.Counter(), []
    for path in sorted(glob.glob(os.path.join(ledger_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ts", "") < since:
                continue
            if r.get("model") != model:
                problems.append(f"ledger {os.path.basename(path)}: a call by model {r.get('model')!r}")
                continue
            calls[r["caller"]] += 1
            tokens[r["caller"]] += r.get("total_tokens") or 0
    return problems, calls, tokens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt-5-nano", "gpt-4o-mini"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger-since", default="", help="'YYYY-MM-DD HH:MM:SS'; default = start time in config.json")
    a = ap.parse_args()
    cfg, merged, problems = merge(a.run_dir, a.model)
    since = a.ledger_since or cfg["started"].replace("T", " ")
    lp, calls, tokens = ledger(a.model, since)
    problems += sorted(set(lp))
    for p in problems:
        print("PROBLEM", p)
    if problems:
        sys.exit(1)
    json.dump(merged, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.model}: {sum(len(v) for v in merged.values())} questions, complete -> {a.out}")
    print("calls", dict(calls), "tokens", dict(tokens), "total", sum(tokens.values()))
