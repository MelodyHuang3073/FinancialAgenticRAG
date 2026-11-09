"""Tables for stacking step 4 from answers.json (label + rule per question; usage from each row's tokens).
    python backend/experiment_results/stack_step4/summarize.py   (writes summary.json next to it)"""
import collections, json, os, statistics as st
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
p = sum(r["tokens"][0] for r in rows); c = sum(r["tokens"][1] for r in rows)
# the decomposer is a separate model call per question; its tokens are in each row's meta
dp = sum(r["meta"]["decomposer_tokens"][0] for r in rows); dc = sum(r["meta"]["decomposer_tokens"][1] for r in rows)
dcalls = sum(r["meta"]["decomposer_calls"] for r in rows)
out["usage"] = {"calls": len(rows) + dcalls, "answer_calls": len(rows), "decomposer_calls": dcalls, "prompt_tokens": p + dp, "completion_tokens": c + dc, "total_tokens": p + c + dp + dc,
                "answer_tokens": p + c, "decomposer_tokens": dp + dc,
                "prompt_chars_median": int(st.median(r["prompt_chars"] for r in rows))}
import collections as _c
out["decompose_source"] = {"+".join(k): v for k, v in _c.Counter(tuple(r["meta"]["decompose_source"]) for r in rows).items()}
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
