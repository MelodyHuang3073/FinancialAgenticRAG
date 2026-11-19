"""Tables for the PageIndex baseline (training set, 150 questions) from answers.json, index_summary.json and usage_ledger/.
    python backend/experiment_results/baseline_pageindex/summarize.py   (writes summary.json next to it)

Cost has three parts: what each answered question used (calls / tokens in answers.json), what building each
filing's index used (index_summary.json, one index per filing, shared by every question on that filing), and
the usage ledger (every model call of the run, including retries and the two questions that hit max_turns).
The ledger has no question id, so cost cannot be split further than this."""
import collections, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
index = json.load(open(os.path.join(HERE, "index_summary.json"), encoding="utf-8"))
LAB = "CSPW"


def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {"n": len(sel), **{k: c[k] for k in LAB}}


out = {"train": tab(rows)}
out["by_type"] = {t: tab([r for r in rows if r["question_type"] == t]) for t in sorted({r["question_type"] for r in rows})}
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"]} for r in rows if r["label"] != "C"]
out["max_turns_errors"] = [r["id"] for r in rows if r["error"]]
answered = [r for r in rows if not r["error"]]
e = [r["elapsed"] for r in answered]
out["seconds_per_question"] = {"mean": round(st.mean(e), 1), "median": round(st.median(e), 1), "max": round(max(e), 1)}
tq = [r["tokens"] for r in answered]
out["tokens_per_answered_question"] = {"mean": round(st.mean(tq)), "median": round(st.median(tq)), "max": max(tq)}
out["answered_questions"] = {"n": len(answered), "calls": sum(r["calls"] for r in answered), "tokens": sum(tq)}
out["max_turns_questions_tokens"] = sum(r["tokens"] for r in rows if r["error"])
out["index"] = {"filings": len(index), "calls": sum(v["calls"] for v in index.values()),
                "tokens": sum(v["tokens"] for v in index.values()),
                "seconds_total": round(sum(v["seconds"] for v in index.values())),
                "tokens_per_filing_mean": round(st.mean(v["tokens"] for v in index.values()))}
use = {}
for name in ("train", "trials"):
    c = collections.Counter()
    for line in open(os.path.join(HERE, "usage_ledger", f"{name}.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        c["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            c[k] += d.get(k, 0)
    use[name] = dict(c)
out["usage_ledger"] = use
out["tokens_in_records"] = out["answered_questions"]["tokens"] + out["max_turns_questions_tokens"] + out["index"]["tokens"]
out["ledger_minus_records"] = use["train"]["total_tokens"] - out["tokens_in_records"]
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
