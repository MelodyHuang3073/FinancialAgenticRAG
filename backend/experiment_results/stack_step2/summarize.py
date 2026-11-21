"""Tables for stacking step 2 (+ answer prompt C2 + question classifier C3) from answers.json (label + rule per question; usage from each row's tokens; run 2026-11-21).
The training set and the test set were run one after the other (4 processes each); run_windows.json holds the two windows.
    python backend/experiment_results/stack_step2/summarize.py   (writes summary.json next to it)"""
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
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"]} for r in rows if r["label"] != "C"]
out["errors"] = sum(bool(r["error"]) for r in rows)
out["per_set"] = {}
for name, key in SETS:
    sel = [r for r in rows if r["set"] == key]
    e = [r["elapsed"] for r in sel]
    p = sum(r["tokens"][0] for r in sel); c = sum(r["tokens"][1] for r in sel)
    out["per_set"][name] = {"calls": len(sel), "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "tokens_per_question": (p + c) // len(sel),
                            "wall_seconds": windows[name]["wall_seconds"], "seconds_per_question": {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": max(e)},
                            "prompt_chars_median": int(st.median(r["prompt_chars"] for r in sel))}
out["total_tokens"] = sum(v["total_tokens"] for v in out["per_set"].values())
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
