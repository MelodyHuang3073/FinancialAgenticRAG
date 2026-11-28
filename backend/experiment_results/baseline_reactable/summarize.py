"""Tables for the ReAcTable baseline (training set, 150 questions) from answers.json and usage_ledger/.
    python backend/experiment_results/baseline_reactable/summarize.py   (writes summary.json next to it)

Cost has two views. answers.json holds what each recorded question used (calls, tokens, seconds). The usage
ledger holds every model call of the run, which also includes the calls of questions that were cut off when a day's
token cap was reached and were run again from the start the next day."""
import collections, json, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "answers.json"), encoding="utf-8"))
LAB = "CSPW"


def tab(sel):
    c = collections.Counter(r["label"] for r in sel)
    return {"n": len(sel), **{k: c[k] for k in LAB}}


def stat(values):
    return {"mean": round(st.mean(values), 1), "median": round(st.median(values), 1), "max": round(max(values), 1)}


out = {"train": tab(rows)}
out["by_type"] = {t: tab([r for r in rows if r["question_type"] == t]) for t in sorted({r["question_type"] for r in rows})}
out["rules_not_correct"] = {lab: dict(collections.Counter(str(r["rule"]) for r in rows if r["label"] == lab)) for lab in "SPW"}
out["not_correct"] = [{"id": r["id"], "label": r["label"], "rule": r["rule"]} for r in rows if r["label"] != "C"]

# How the released majority vote behaved: it counts identical answer strings and, on a tie, keeps the first one seen.
out["voting"] = {
    "distinct_answer_strings_among_5_runs": dict(sorted(collections.Counter(len(set(r["all_answers"])) for r in rows).items())),
    "questions_with_5_different_strings": sum(len(set(r["all_answers"])) == 5 for r in rows),
    "questions_whose_answer_equals_run_1": sum(r["answer"] == r["all_answers"][0].strip() for r in rows),
}
numeric = []
for r in rows:
    m = re.fullmatch(r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*%?", str(r["gold"]).strip())
    if m and r["label"] in "PW":
        g = m.group(1).replace(",", "")
        g = g.rstrip("0").rstrip(".") if "." in g else g
        if any(re.search(r"(?<![\d.])" + re.escape(g), v.replace(",", "")) for v in r["all_answers"]):
            numeric.append(r["id"])
out["voting"]["p_or_w_numeric_questions_where_a_run_printed_the_gold_number"] = numeric

tokens = [r["input_tokens"] + r["output_tokens"] for r in rows]
out["records"] = {
    "calls": sum(r["calls"] for r in rows), "calls_per_question": round(st.mean([r["calls"] for r in rows]), 1),
    "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
    "tokens": sum(tokens), "tokens_per_question": stat(tokens),
    "seconds_per_question": stat([r["elapsed"] for r in rows]), "seconds_sum": round(sum(r["elapsed"] for r in rows)),
    "forced_answer_calls": sum(len(r["forced_calls"]) for r in rows),
    "questions_with_a_forced_answer_call": sum(bool(r["forced_calls"]) for r in rows),
    "empty_replies": sum(r["empty_replies"] for r in rows),
}
out["usage_ledger"] = {}
for name in ("train", "trials"):
    total, days = collections.Counter(), collections.defaultdict(collections.Counter)
    for line in open(os.path.join(HERE, "usage_ledger", f"{name}.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        for c in (total, days[d["date"]]):
            c["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                c[k] += d.get(k) or 0
    out["usage_ledger"][name] = {**total, "by_utc_day": {k: dict(v) for k, v in sorted(days.items())}}
out["ledger_minus_records_tokens"] = out["usage_ledger"]["train"]["total_tokens"] - out["records"]["tokens"]
out["ledger_minus_records_calls"] = out["usage_ledger"]["train"]["calls"] - out["records"]["calls"]
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "not_correct"}, ensure_ascii=False, indent=1))
