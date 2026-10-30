"""Builds the per-question comparison of the no-router ablation, explanation version (ABLATE_ROUTER=explanation: every
question takes the explanation path, no PoT; single-file, all 300 questions, one run) against the baseline
(../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under the written
rules of the fourth-chapter deck 4.1, reviewed by the author). Every one of the 300 answers was read against the gold answer
one by one (2026-10-30); none was graded by program only.

Run from backend/:  python ablation_results/no_router_explanation/build_comparison.py
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("train", 15): ("S", "free cash flow conversion written as about 155% (7,396 / 4,756 = 155.5%, should be 156%); Yes and 143% right"),
    ("train", 40): ("P", "names the United States versus outside it and a country list; gold names the regions United States, EMEA, APAC and LACC"),
    ("train", 49): ("W", "3.1% from revenues of $36,248M / $36,365M / $36,055M (domestic segment figures); gold 2.8%"),
    ("train", 55): ("P", "Best Buy-format stores 930 to 907; gold is total domestic stores 982 to 969 (-1.32%)"),
    ("train", 66): ("P", "787 and 737 increases named; 777X described as paused with inconsistent disclosures; gold increases for 737, 777X and 787"),
    ("train", 67): ("S", "FY2021 rate 14.7%; the filing prints 14.8%"),
    ("train", 119): ("S", "working capital $1,548M right, but current assets $57,517M less current liabilities $45,101M is $12,416M"),
    ("train", 124): ("W", "main answer EBITDA less capex $14,274M (capex taken as $1.0M); $9,068M appears only as an alternate"),
    ("train", 130): ("W", "says it cannot determine the guidance change; gold 1 percentage point"),
    ("out", 8): ("S", "ROE 33.30%; 101,832 / 305,978 = 33.28%"),
    ("out", 24): ("W", "says it cannot determine; gold net service sales grew faster (15.0% vs 8.8%)"),
    ("out", 68): ("W", "83.3% from gross profit $8,187M / net sales $9,824M; gold 74.3%"),
    ("out", 70): ("P", "restructuring $1,092M vs $1,987M; gold $1,550M vs $2,419M"),
    ("out", 85): ("W", "says diluted EPS is not in the excerpts; gold $4.06 vs $4.14"),
    ("out", 103): ("P", "impairments, write-offs and D&A; omits the operating income decline and the swing in other non-operating items"),
    ("out", 104): ("P", "casino revenue $9,350M (segment sum); gold about $9.5B; rooms $3,377M is right"),
    ("out", 107): ("W", "says net earnings cannot be determined; gold $927M vs $1,241M"),
    ("out", 124): ("P", "profit right; main driver given as gross margin improvement; gold is the Other (expense) income swing"),
    ("out", 142): ("S", "18.0% (from the filing's integer percentage); 4,294 / 23,769 = 18.1%"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "single_norouter_explanation_300.json"), encoding="utf-8"))
    base = {(b["set"], b["n"]): b for b in json.load(open(os.path.join(HERE, "..", "baseline", "baseline_standard.json"), encoding="utf-8"))}
    out = []
    for r in rows:
        b = base[(r["set"], r["n"])]
        label, note = LABELS.get((r["set"], r["n"]), ("C", ""))
        out.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                    "question": r["question"], "gold": r["gold"],
                    "baseline_label": b["label"], "baseline_answer": b["answer"],
                    "norouter_label": label, "norouter_answer": r["model_answer"], "note": note,
                    "grading_basis": "read-2026-10-30"})
    json.dump(out, open(os.path.join(HERE, "comparison_no_router_explanation.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "norouter": x["norouter_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["norouter_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "no_router_explanation": {"all": cnt("norouter_label"), "train": cnt("norouter_label", "train"), "out": cnt("norouter_label", "out")},
               "read_one_by_one": len(out),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "no_router_explanation", "read_one_by_one")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
