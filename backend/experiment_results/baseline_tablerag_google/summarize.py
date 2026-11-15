"""Tables for the Google TableRAG baseline from answers.json (label + rule per question) and usage_ledger/.
    python backend/experiment_results/baseline_tablerag_google/summarize.py   (writes summary.json next to it)"""
import collections, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
LAB = "CSPW"


def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {"n": len(sel), **{k: c[k] for k in LAB},
            "with_final_answer": sum(r["final_answer"] for r in sel),
            "C_among_final_answers": sum(r["final_answer"] and r["label"] == "C" for r in sel),
            "used_all_5_rounds": sum(r["n_rounds"] >= 5 for r in sel),
            "empty_answer": sum(not r["answer"].strip() for r in sel)}


out = {"all": tab(rows)}
for name, key in (("train", "original"), ("out", "outsample")):
    out[name] = tab([r for r in rows if r["set"] == key])
out["by_type"] = {t: tab([r for r in rows if r["question_type"] == t]) for t in sorted({r["question_type"] for r in rows})}
out["by_type_and_set"] = {f"{s}/{t}": tab([r for r in rows if r["set"] == s and r["question_type"] == t])
                          for s in ("original", "outsample") for t in sorted({r["question_type"] for r in rows})}
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"], "final_answer": r["final_answer"]}
                      for r in rows if r["label"] != "C"]
out["errors"] = sum(bool(r["error"]) for r in rows)
e = [r["elapsed"] for r in rows]
out["seconds_per_question"] = {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": round(max(e), 1)}
use = {}
for name in ("train", "out", "rerun", "trials"):
    c = collections.Counter()
    for line in open(os.path.join(HERE, "usage_ledger", f"{name}.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        c["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            c[k] += d.get(k, 0)
        c["capped_replies"] += d["completion_tokens"] >= 4000
    use[name] = dict(c)
out["usage"] = use
out["usage"]["final_runs_total_tokens"] = use["train"]["total_tokens"] + use["out"]["total_tokens"] + use["rerun"]["total_tokens"]
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
