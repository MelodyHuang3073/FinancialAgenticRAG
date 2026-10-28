"""Builds the comparison baseline for ablation experiments from the single-file
full run (single_all_300.json: every question answered with only its own filing
loaded, full system, one run) and the manual labels below.

Labels (my own reading of each answer against the gold answer; the automatic
grader is only a screen):
  C correct, S small error (right conclusion, percentage off by 0.1 point),
  P partially correct, W wrong. A percentage the filing prints as an integer
  counts as C, and so does the same number written with a decimal that rounds to
  it (the filing's 12% written as 12.0%, gold 12.4%). A rate computed from the
  filing's amounts must be the correct rounding of the computed value; one digit
  off (0.1 point) is S (2026-10-28 audit: Amazon service sales growth 15.1%,
  computed 15.045%, should be 15.0%).
Everything not listed in LABELS is C.

Run from backend/:  python ablation_results/build_baseline_standard.py
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "baseline")

LABELS = {
    ("train", 109): ("W", "worst topline region: answered Las Vegas Strip, gold is MGM China (-44%)"),
    ("out", 3): ("P", "net income decline given as 19% from $4,009M; gold 22.1% from $4,173M, same conclusion"),
    ("out", 8): ("S", "ROE written as 0.33; gold 33.28%"),
    ("out", 24): ("S", "service sales growth 15.1%; computed from the filing amounts 15.045%, should be 15.0%"),
    ("out", 12): ("W", "impairment of intangible assets $1.9B vs $714M; gold $1,993M vs $33M"),
    ("out", 33): ("P", "names non-operating/one-time items but not the 'Other (income) expense, net' swing"),
    ("out", 50): ("W", "investment banking fees $9,735M (division table); gold $9,615M"),
    ("out", 70): ("P", "headline restructuring charges $1,092M/$1,987M; gold figures $1,550M/$2,419M appear only as a related line"),
    ("out", 84): ("W", "service revenue $80.9B and wireless equipment $21.8B; gold $112.7B and $25.5B"),
    ("out", 103): ("P", "impairment, D&A and gaming taxes; omits the swing in other non-operating items"),
    ("out", 124): ("P", "profit and amounts right; main driver given as gross margin, gold is other (expense) income"),
}


def numbers(text):
    out = []
    for m in re.finditer(r"-?\(?\$?\d[\d,]*\.?\d*\)?%?", text):
        s = re.sub(r"[^\d.\-]", "", m.group())
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def screen(gold, answer):
    """Mechanical screen only: do the first numbers of the gold answer appear in the answer within 2%?"""
    g, a = numbers(gold)[:3], numbers(answer)
    if not g:
        return None
    return all(any(abs(x - y) <= 0.02 * abs(x) + 1e-9 for y in a) for x in g)


def counts(rows):
    c = Counter(r["label"] for r in rows)
    return {"n": len(rows), **{k: c.get(k, 0) for k in "CSPW"}}


def main():
    rows = json.load(open(os.path.join(HERE, "single_all_300.json"), encoding="utf-8"))
    base = []
    for r in rows:
        label, note = LABELS.get((r["set"], r["n"]), ("C", ""))
        base.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                     "question": r["question"], "gold": r["gold"], "answer": r["model_answer"],
                     "label": label, "note": note, "screen_2pct": screen(r["gold"], r["model_answer"])})
    os.makedirs(OUT, exist_ok=True)
    json.dump(base, open(os.path.join(OUT, "baseline_standard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "baseline_answers.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "n", "question_type", "doc_name", "question", "gold", "answer", "label", "note"])
        for r in base:
            w.writerow([r["set"], r["n"], r["question_type"], r["doc_name"], r["question"], r["gold"], r["answer"], r["label"], r["note"]])
    by_type = defaultdict(list)
    for r in base:
        by_type[r["question_type"]].append(r)
    summary = {
        "all": counts(base),
        "by_set": {s: counts([r for r in base if r["set"] == s]) for s in ("train", "out")},
        "by_type": {t: counts(v) for t, v in sorted(by_type.items())},
        "by_set_and_type": {f"{s}/{t}": counts([r for r in v if r["set"] == s]) for t, v in sorted(by_type.items()) for s in ("train", "out") if any(r["set"] == s for r in v)},
        "screen_2pct": {"numeric_gold": sum(1 for r in base if r["screen_2pct"] is not None),
                        "matched": sum(1 for r in base if r["screen_2pct"])},
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
