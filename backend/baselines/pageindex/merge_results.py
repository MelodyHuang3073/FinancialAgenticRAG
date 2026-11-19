"""Merge the per-shard files of a PageIndex run into answers.json-style records, an index summary and a usage cross-check.

    python merge_results.py --run-dir <train_run> --out <experiment_results/baseline_pageindex> --ledger-since "YYYY-MM-DD HH:MM:SS"

answers_partial.json has the field names of the other baselines' answers.json (id, set, n, doc_name, question_type, question,
gold, answer, elapsed, error) plus calls and tokens; label/rule stay empty until the answers are graded by reading
(EXPERIMENT_PLAN.md section 9). The tokens recorded per question and per index are compared with the usage ledger
(caller "pageindex") so that lost or doubled records show up.
"""
import argparse
import glob
import json
import os


def read_json(path, default):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default


def merge(run_dir):
    results, index = [], {}
    for f in sorted(glob.glob(os.path.join(run_dir, "results_*.json"))):
        results += read_json(f, [])
    for f in sorted(glob.glob(os.path.join(run_dir, "index_*.json"))):
        index.update(read_json(f, {}))
    results = [r for r in results if not r.get("error")]
    seen, uniq = set(), []
    for r in sorted(results, key=lambda r: r["n"]):
        if r["id"] not in seen:  # a question answered twice (rerun after a crash): keep the first complete one
            seen.add(r["id"])
            uniq.append(r)
    return uniq, index


def _key(rec):
    return (rec["ts"], rec["pid"], rec.get("prompt_tokens"), rec.get("completion_tokens"))


def ledger_lines(ledger_dir, since):
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(ledger_dir, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if '"pageindex"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("caller") != "pageindex" or rec["ts"] < since:
                continue
            if _key(rec) not in seen:
                seen.add(_key(rec))
                out.append(rec)
    return sorted(out, key=lambda r: (r["ts"], r["pid"]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger-since", required=True, help="first ledger timestamp that belongs to this run (earlier lines are other runs)")
    ap.add_argument("--ledger-dir", default=os.path.join(os.path.dirname(os.path.dirname(here)), ".llm_usage"))
    ap.add_argument("--ledger-name", default="train")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "usage_ledger"), exist_ok=True)
    results, index = merge(args.run_dir)
    json.dump(results, open(os.path.join(args.out, "answers_partial.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(index, open(os.path.join(args.out, "index_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # the ledger copy is append-only: lines already saved stay even if the originals are later deleted
    path = os.path.join(args.out, "usage_ledger", f"{args.ledger_name}.jsonl")
    old = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
    have = {_key(r) for r in old}
    merged = old + [r for r in ledger_lines(args.ledger_dir, args.ledger_since) if _key(r) not in have]
    merged.sort(key=lambda r: (r["ts"], r["pid"]))
    with open(path, "w", encoding="utf-8") as fp:
        for r in merged:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec_tokens = sum(r["tokens"] for r in results) + sum(v.get("tokens", 0) for v in index.values())
    led_tokens = sum(r["total_tokens"] or 0 for r in merged)
    summary = {"questions": len(results), "filings_indexed": sum("doc_id" in v for v in index.values()),
               "filings_failed": [k for k, v in index.items() if "doc_id" not in v],
               "tokens_in_results_and_index": rec_tokens, "tokens_in_ledger": led_tokens, "ledger_calls": len(merged),
               "ledger_minus_records": led_tokens - rec_tokens}
    json.dump(summary, open(os.path.join(args.out, "progress.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
