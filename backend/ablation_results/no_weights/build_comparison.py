"""Builds the per-question comparison of the no-weights ablation (ABLATE_WEIGHTS=1: every retrieval re-weighting rule
off, ranking by BM25 + keyword overlap only; single-file, all 300 questions, one run) against the baseline
(../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under the written
rules of the fourth-chapter deck 4.1, reviewed by the author). Every one of the 300 answers was read against the gold answer
one by one (2026-10-31); none was graded by program only.

Run from backend/:  python ablation_results/no_weights/build_comparison.py
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("train", 48): ("P", "says no, working capital is negative, but gives -$190M (operating definition: current assets less cash, current liabilities less short-term debt); gold -$1,561M"),
    ("train", 49): ("W", "2.6% from net earnings $918M / $973M / $1,148M and revenue $40,339M / $39,528M / $39,403M; gold 2.8%"),
    ("train", 55): ("P", "Best Buy-format stores 930 to 907; gold is total domestic stores 982 to 969 (-1.32%)"),
    ("train", 69): ("W", "says it cannot determine the COGS margin (no cost of goods sold line in the evidence); gold 39.7%"),
    ("train", 76): ("W", "17.08 from average PP&E (11,349 + 10,292) / 2; gold 17.98"),
    ("train", 77): ("W", "No, not capital intensive (ratio 2.4973 from Health Care Benefits revenue only); gold Yes"),
    ("train", 88): ("W", "2.49 from year-end inventories $12,483M; gold 2.7 (average inventories)"),
    ("train", 103): ("W", "says it cannot calculate the CAGR because the income statement is not in the evidence; gold 0.4%"),
    ("train", 133): ("P", "names Array, Trillium and Arena; gold Trillium, Array and Therachon"),
    ("out", 3): ("P", "net sales growth 2.76% (24,575 to 24,948 is 1.5%); net income -22.1% right"),
    ("out", 8): ("S", "ROE 33.29%; 101,832 / 305,978 = 33.28%"),
    ("out", 9): ("W", "says it cannot determine which grew faster; gold service and other revenue +20.8%, product -1.3%"),
    ("out", 12): ("W", "impairment $1,946M vs $714M; gold $1,993M vs $33M"),
    ("out", 24): ("S", "net service sales growth 15.1%; 420,658 / 365,648 = 15.0%"),
    ("out", 36): ("W", "70.7% (21,218 / 23,769 written as 0.7071); gold 89.3%"),
    ("out", 69): ("W", "acquired IPR&D $1.5B vs $0; gold $1,613M vs $108M"),
    ("out", 70): ("P", "restructuring $1,092M vs $1,987M; gold $1,550M vs $2,419M"),
    ("out", 83): ("W", "23.7% from operating income $32,160M and revenues $135,876M; gold 21.2% (29,259 / 138,191)"),
    ("out", 104): ("P", "casino revenue $9,350M (segment sum); gold about $9.5B; rooms $3,377M is right"),
    ("out", 124): ("P", "profit right; main driver given as gross margin improvement; gold is the Other (expense) income swing"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "single_noweights_300.json"), encoding="utf-8"))
    base = {(b["set"], b["n"]): b for b in json.load(open(os.path.join(HERE, "..", "baseline", "baseline_standard.json"), encoding="utf-8"))}
    out = []
    for r in rows:
        b = base[(r["set"], r["n"])]
        label, note = LABELS.get((r["set"], r["n"]), ("C", ""))
        out.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                    "question": r["question"], "gold": r["gold"],
                    "baseline_label": b["label"], "baseline_answer": b["answer"],
                    "noweights_label": label, "noweights_answer": r["model_answer"], "note": note,
                    "grading_basis": "read-2026-10-31"})
    json.dump(out, open(os.path.join(HERE, "comparison_no_weights.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "noweights": x["noweights_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["noweights_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "no_weights": {"all": cnt("noweights_label"), "train": cnt("noweights_label", "train"), "out": cnt("noweights_label", "out")},
               "read_one_by_one": len(out),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "no_weights")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
