"""Builds the per-question comparison of gpt-5-nano (same pipeline, OPENAI_MODEL=gpt-5-nano; single-file, all 300 questions,
one run) against the baseline (gpt-5-mini; ../../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under the written
rules of the fourth-chapter deck 4.1, reviewed by the author). Every one of the 300 answers was read against the gold answer
one by one (2026-10-31); none was graded by program only.

Run from backend/:  python ablation_results/llm_openai/gpt_5_nano/build_comparison.py
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("train", 15): ("W", "says it cannot determine Adobe's free cash flow conversion; gold Yes, 156% vs 143%"),
    ("train", 28): ("W", "swaps the components: $81M called fixed-asset costs and $2M employee costs; gold says 87% is employee liabilities"),
    ("train", 30): ("P", "real change in sales given as +1%; gold flat"),
    ("train", 32): ("S", "quick ratio 1.57 and Yes are right, but the stated inputs (15,019 - 3,771) / 6,369 give 1.77"),
    ("train", 55): ("P", "Best Buy-format stores 930 to 907; gold is total domestic stores 982 to 969"),
    ("train", 65): ("P", "Yes, cyclical, but the reasons are long-term contracts, backlog and government funding; gold is the airline industry's cyclicality"),
    ("train", 66): ("P", "787 and 737 increases named; says 777X production is paused through 2023; gold increases for 737, 777X and 787"),
    ("train", 74): ("P", "Yes, positive working capital, but $2,278M under the standard definition; gold $831M (operating definition)"),
    ("train", 89): ("W", "answers Yes, accelerate (midpoint 4.0%); gold No (3.5% vs 3.6%)"),
    ("train", 90): ("W", "US +4.3% and international +1.0%; gold US +3.0% and international -0.6%"),
    ("train", 95): ("W", "Asset & Wealth Management ($10,003M) as the lowest segment; gold Corporate, -$473M"),
    ("train", 98): ("P", "Corporate & Investment Bank right; net income $5,645M; gold $3,725M"),
    ("train", 119): ("P", "Yes right; working capital $12.4B from the wrong current assets; gold $1.6B"),
    ("train", 121): ("P", "names only Europe, AMESA and APAC; gold also names North America and Latin America"),
    ("train", 122): ("W", "answers Yes (tax audits in Mexico and Canada); gold No material legal battles"),
    ("train", 133): ("P", "Therachon and Array named, then the Upjohn spin-off (not an acquisition); gold Trillium, Array, Therachon"),
    ("train", 135): ("W", "Emerging markets (-53%); gold Developed Rest of the World"),
    ("train", 141): ("W", "says it cannot be determined; gold 36%"),
    ("train", 142): ("W", "says wages expense as a share of sales decreased (23.5% vs 23.9%); gold increased"),
    ("train", 145): ("P", "No right; quick ratio 0.71 from (current assets - inventories) / current liabilities; gold 0.54"),
    ("out", 33): ("P", "net earnings doubling right; driver given as operating earnings and mix; gold is the Other (income) expense swing"),
    ("out", 50): ("W", "investment banking fees $9,735M vs $9,116M; gold $9,615M vs $8,910M"),
    ("out", 84): ("W", "service revenues $80,617M and wireless equipment $21,779M; gold $112,721M and $25,470M"),
    ("out", 103): ("P", "goodwill impairment, write-offs, taxes and amortization; omits the swing in other non-operating items"),
    ("out", 104): ("P", "casino revenue $9,350M (segment sum); gold about $9.5B; rooms $3,377M is right"),
    ("out", 124): ("P", "profit right; main driver given as gross margin improvement; gold is the Other (expense) income swing"),
    ("out", 131): ("P", "FY2025 net income right; prior year given as $1,291,005K (10.7% decline); gold $1,201,118K (4.0%)"),
    ("out", 140): ("S", "total assets right; decrease written as about 5.3%; 2,135 / 39,868 = 5.4%"),
    ("out", 142): ("S", "18.0% (from the filing's integer percentage); 4,294 / 23,769 = 18.1%"),
    ("out", 147): ("W", "says the jump is explained by non-operating currency gains and one-time items rather than operating leverage; gold is operating leverage"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "single_gpt5nano_300.json"), encoding="utf-8"))
    base = {(b["set"], b["n"]): b for b in json.load(open(os.path.join(HERE, "..", "..", "baseline", "baseline_standard.json"), encoding="utf-8"))}
    out = []
    for r in rows:
        b = base[(r["set"], r["n"])]
        label, note = LABELS.get((r["set"], r["n"]), ("C", ""))
        out.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                    "question": r["question"], "gold": r["gold"],
                    "baseline_label": b["label"], "baseline_answer": b["answer"],
                    "nano_label": label, "nano_answer": r["model_answer"], "note": note,
                    "grading_basis": "read-2026-10-31"})
    json.dump(out, open(os.path.join(HERE, "comparison_gpt_5_nano.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "nano": x["nano_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["nano_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "gpt_5_nano": {"all": cnt("nano_label"), "train": cnt("nano_label", "train"), "out": cnt("nano_label", "out")},
               "read_one_by_one": len(out), "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "gpt_5_nano")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
