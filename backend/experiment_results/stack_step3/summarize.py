"""Tables for stacking step 3 (+ decomposer C4 + formula library) from answers.json (label + rule per question; usage from each row's tokens; run 2026-11-21).
The training set and the test set were run one after the other (4 processes each); run_windows.json holds the two windows.
    python backend/experiment_results/stack_step3/summarize.py   (writes summary.json next to it)"""
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
    # the decomposer is a separate model call per question (planner); its calls and tokens are in each row's meta
    dp = sum(r["meta"]["decomposer_tokens"][0] for r in sel); dc = sum(r["meta"]["decomposer_tokens"][1] for r in sel)
    dcalls = sum(r["meta"]["decomposer_calls"] for r in sel)
    n = len(sel)
    out["per_set"][name] = {"calls": n + dcalls, "answer_calls": n, "decomposer_calls": dcalls, "prompt_tokens": p + dp, "completion_tokens": c + dc, "total_tokens": p + c + dp + dc,
                            "answer_tokens": p + c, "decomposer_tokens": dp + dc, "tokens_per_question": (p + c + dp + dc) // n,
                            "wall_seconds": windows[name]["wall_seconds"], "seconds_per_question_wall_x4": round(windows[name]["wall_seconds"] * 4 / n, 1),
                            "answer_call_seconds": {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": max(e)},
                            "prompt_chars_median": int(st.median(r["prompt_chars"] for r in sel)),
                            "decompose_source": {"+".join(k): v for k, v in collections.Counter(tuple(r["meta"]["decompose_source"]) for r in sel).items()}}
out["total_tokens"] = sum(v["total_tokens"] for v in out["per_set"].values())
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
