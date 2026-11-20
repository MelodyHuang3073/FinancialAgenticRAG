"""Tables for stacking step 0 (Naive RAG) from answers.json (labels were read by hand, one answer at a time; run on 2026-11-20).
label = C/S/P/W by EXPERIMENT_PLAN.md section 9; rule = the decision-table row or 9.4 rule it was judged by.
The training set and the test set were run one after the other (4 processes each); run_windows.json holds the two windows.
Tokens come from each answer's own `tokens` field (this runner does not use the usage ledger).
    python backend/experiment_results/stack_step0/summarize.py"""
import collections, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
windows = json.load(open(os.path.join(HERE, "run_windows.json"), encoding="utf-8"))
def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {k: c[k] for k in "CSPW"}
SETS = (("train", "original"), ("test", "outsample"))
out = {"all": {"n": len(rows), **tab(rows)}}
for name, key in SETS:
    sel = [r for r in rows if r["set"] == key]
    out[name] = {"n": len(sel), **tab(sel)}
out["by_type"] = {t: {"n": sum(r["question_type"] == t for r in rows), **tab([r for r in rows if r["question_type"] == t])} for t in sorted({r["question_type"] for r in rows})}
def wk(sel):
    c = collections.Counter(r.get("w_kind") for r in sel if r["label"] == "W")
    return {"W": c["lack"] + c["wrong"], "lack_or_ask_back": c["lack"], "gave_wrong_answer": c["wrong"]}
out["w_breakdown"] = {n: wk([r for r in rows if r["set"] == k]) for n, k in SETS}
out["per_set"] = {}
for name, key in SETS:
    sel = [r for r in rows if r["set"] == key]
    p = sum(r["tokens"][0] for r in sel); c = sum(r["tokens"][1] for r in sel); e = [r["elapsed"] for r in sel]
    out["per_set"][name] = {"calls": len(sel), "errors": sum(bool(r["error"]) for r in sel), "prompt_tokens": p, "completion_tokens": c,
                            "total_tokens": p + c, "tokens_per_question": (p + c) // len(sel), "wall_seconds": windows[name]["wall_seconds"],
                            "seconds_per_question": {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": max(e)}}
out["total_tokens"] = sum(v["total_tokens"] for v in out["per_set"].values())
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
