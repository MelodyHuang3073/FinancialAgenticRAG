"""Tables for this baseline run from answers.json (label + rule per question) and usage_ledger/.
    python backend/experiment_results/baseline/summarize.py   (writes summary.json next to it)"""
import collections, glob, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
LAB = "CSPW"
def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {k: c[k] for k in LAB}
out = {"all": {"n": len(rows), **tab(rows)}}
for name, key in (("train", "original"), ("out", "outsample")):
    sel = [r for r in rows if r["set"] == key]
    out[name] = {"n": len(sel), **tab(sel)}
out["by_type"] = {t: {"n": sum(r["question_type"] == t for r in rows), **tab([r for r in rows if r["question_type"] == t])}
                  for t in sorted({r["question_type"] for r in rows})}
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"]} for r in rows if r["label"] != "C"]
out["errors"] = sum(bool(r["error"]) for r in rows)
e = [r["elapsed"] for r in rows]
out["seconds_per_question"] = {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": max(e)}
use = collections.defaultdict(lambda: collections.Counter())
for f in glob.glob(os.path.join(HERE, "usage_ledger", "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        u = use[d["caller"]]
        u["calls"] += 1; u["prompt_tokens"] += d["prompt_tokens"]; u["completion_tokens"] += d["completion_tokens"]
        u["total_tokens"] += d["total_tokens"]
out["usage"] = {k: dict(v) for k, v in use.items()}
out["usage"]["total_tokens"] = sum(v["total_tokens"] for v in use.values())
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
