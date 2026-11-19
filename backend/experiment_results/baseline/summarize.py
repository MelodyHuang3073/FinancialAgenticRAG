"""Tables for the full-system baseline from answers.json (label + rule per question), usage_ledger/ and run_windows.json.
    python backend/experiment_results/baseline/summarize.py   (writes summary.json next to it)

The training set and the test set were run one after the other (4 processes each), so the usage ledger is split
into train.jsonl and test.jsonl by the two time windows in run_windows.json; usage and time are reported per set."""
import collections, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
windows = json.load(open(os.path.join(HERE, "run_windows.json"), encoding="utf-8"))
LAB = "CSPW"
SETS = (("train", "original"), ("test", "outsample"))


def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {"n": len(sel), **{k: c[k] for k in LAB}}


out = {"all": tab(rows)}
for name, key in SETS:
    out[name] = tab([r for r in rows if r["set"] == key])
out["by_type"] = {t: tab([r for r in rows if r["question_type"] == t]) for t in sorted({r["question_type"] for r in rows})}
out["by_type_and_set"] = {f"{n}/{t}": tab([r for r in rows if r["set"] == k and r["question_type"] == t])
                          for n, k in SETS for t in sorted({r["question_type"] for r in rows})}
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"]} for r in rows if r["label"] != "C"]
out["errors"] = sum(bool(r["error"]) for r in rows)
out["per_set"] = {}
for name, key in SETS:
    sel = [r for r in rows if r["set"] == key]
    e = [r["elapsed"] for r in sel]
    c = collections.Counter()
    by_caller = collections.defaultdict(collections.Counter)
    for line in open(os.path.join(HERE, "usage_ledger", f"{name}.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            c[k] += d.get(k, 0)
            by_caller[d["caller"]][k] += d.get(k, 0)
        c["calls"] += 1
        by_caller[d["caller"]]["calls"] += 1
    out["per_set"][name] = {
        "questions": len(sel),
        "wall_seconds": windows[name]["wall_seconds"],
        "seconds_per_question": {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": round(max(e), 1)},
        "usage": dict(c),
        "usage_by_caller": {k: dict(v) for k, v in by_caller.items()},
        "tokens_per_question": round(c["total_tokens"] / len(sel)),
    }
tot = sum(v["usage"]["total_tokens"] for v in out["per_set"].values())
out["total_tokens"] = tot
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
