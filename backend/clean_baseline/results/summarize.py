"""Tables from single_clean_300.json (labels were read by hand, one answer at a time)
(label = C/S/P/W by EXPERIMENT_PLAN.md section 7; rule = the decision-table row or 7.4 rule it was judged by).
    python backend/clean_baseline/results/summarize.py"""
import collections, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "single_clean_300.json"), encoding="utf-8"))
def tab(sel, key):
    c = collections.Counter(key(r) for r in sel)
    return {k: c[k] for k in "CSPW"}
out = {}
for name, f in (("all", lambda r: True), ("train", lambda r: r["set"] == "original"), ("out", lambda r: r["set"] == "outsample")):
    sel = [r for r in rows if f(r)]
    out[name] = {"n": len(sel), "clean_rag": tab(sel, lambda r: r["label"])}
out["by_type"] = {t: {"n": sum(r["question_type"] == t for r in rows),
                      "clean_rag": tab([r for r in rows if r["question_type"] == t], lambda r: r["label"])}
                  for t in sorted({r["question_type"] for r in rows})}
def wk(sel):
    c = collections.Counter(r.get("w_kind") for r in sel if r["label"] == "W")
    return {"W": c["lack"] + c["wrong"], "lack_or_ask_back": c["lack"], "gave_wrong_answer": c["wrong"]}
out["w_breakdown"] = {"train": wk([r for r in rows if r["set"] == "original"]), "out": wk([r for r in rows if r["set"] == "outsample"]),
                      "by_set_and_type": {f"{s}/{t}": wk([r for r in rows if r["set"] == s and r["question_type"] == t])
                                          for s in ("original", "outsample") for t in sorted({r["question_type"] for r in rows})}}
p = sum(r["tokens"][0] for r in rows); c = sum(r["tokens"][1] for r in rows); e = [r["elapsed"] for r in rows]
out["usage"] = {"calls": len(rows), "errors": sum(bool(r["error"]) for r in rows), "prompt_tokens": p, "completion_tokens": c,
                "total_tokens": p + c, "tokens_per_question": (p + c) // len(rows),
                "seconds_per_question": {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": max(e)}}
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
