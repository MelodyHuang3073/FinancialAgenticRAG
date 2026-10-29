"""Builds the per-question comparison of the swap-the-LLM run with gemini-3.1-flash-lite (single-file, all 300 questions,
one run) against the baseline (../../baseline/baseline_standard.json, gpt-5-mini).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under the same fixed
rules as the baseline and the ablations, reviewed by the author). Every one of the 300 answers was read against the gold answer
one by one; none was graded by program only. Answers that were the system's no-LLM fallback (the API call returned nothing)
were rerun and the rerun answer is the one graded (field "rerun" in the comparison file).

Run from backend/:  python ablation_results/llm_gemini/gemini_3_1_flash_lite/build_comparison.py
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    "ou8": ("S", "ROE written as 0.33; the question asks for a percentage (33.28%)"),
    "ou10": ("S", "54.2% (computed 54.148%, should be 54.1%)"),
    "ou12": ("W", "$1,946M vs $714M; gold $1,993M vs $33M"),
    "ou33": ("W", "names the $21.0B Kenvue exchange-offer gain of fiscal 2023 as the driver; the FY2025 driver is the swing in other (income) expense"),
    "ou70": ("P", "main answer $1,092M vs $1,987M (restructuring charges); gold $1,550M vs $2,419M (restructuring charges and certain acquisition-related costs); the gold line is not given"),
    "ou84": ("W", "service revenues $80,617M; gold $112,721M (wireless equipment $21,779M; gold $25,470M)"),
    "ou87": ("W", "net earnings $2,518.6M (FY2025) vs $2,609.6M (FY2024): wrong columns and a decrease reported as the reverse; gold $2,295.2M vs $2,496.6M"),
    "ou101": ("W", "growth stated as 6.66% although the two revenue figures given ($17,240,545K and $17,537,683K) give 1.7%"),
    "ou103": ("P", "explains the operating-income drop with impairments and write-offs; does not name the swing in other non-operating items"),
    "ou105": ("P", "diluted EPS $0.74 computed from net income attributable ($205.86M); the reported diluted EPS is $0.76"),
    "ou116": ("S", "growth 9.95%; (5,140 - 4,684) / 4,684 = 9.7%"),
    "ou118": ("S", "operating margin 36.5% (1,879 / 5,140 = 36.56%, should be 36.6%)"),
    "ou124": ("P", "profit right ($12M vs $(330)M); main driver given as cost optimization and store closures; gold is the Other (expense) income swing"),
    "ou129": ("W", "operating income $1,564,972K vs $1,678,029K: the prior-year columns; gold $1,532,992K vs $1,564,972K"),
    "ou132": ("W", "diluted EPS written as $25.34 vs $26.03 (years shifted); gold $25.64 vs $25.34, up, with the share count falling"),
    "ou133": ("W", "gross margin 38.8% from the prior-year columns; gold 39.1%"),
    "ou147": ("P", "driver given as gross margin expansion and lower restructuring charges; gold is operating leverage (SG&A and R&D growing slower than sales)"),
    "tr6": ("W", "answers that 3M has a reasonably healthy liquidity profile (quick ratio 0.96); gold No"),
    "tr15": ("W", "answers No, free cash flow conversion did not improve; gold Yes (143% to 156%)"),
    "tr25": ("P", "lists two of the three FY2023 acquisitions (Shanghai, New Zealand); the Czech Republic plant is missing"),
    "tr28": ("P", "gives the $93M total and the three categories but not the 87% employee share"),
    "tr51": ("W", "answers No (fluctuations over 2%) although its own figures 22.38/22.49/21.41 move by at most 1.1 points; gold Yes"),
    "tr61": ("P", "says two categories exceed 20% (Commercial Airplanes, Defense); Global Services 26.4% missing; gold names three"),
    "tr67": ("S", "FY2021 rate 14.7%; the filing prints 14.8%"),
    "tr72": ("S", "10.3% right, but the 2019 margin is written as 15.0% ($1,306M / $11,503M is 11.4%)"),
    "tr74": ("P", "working capital $2,278M (unadjusted); gold $831M (operating items only); Yes is right"),
    "tr77": ("W", "answers No, CVS is not capital intensive; gold Yes"),
    "tr119": ("P", "working capital $12,416M (current assets $57,517M); gold about $1.6B; Yes is right"),
    "tr124": ("W", "EBITDA less capex written as $14,274M (it adds operating profit and D&A but the capex deduction is not applied); gold $9,068M"),
    "tr133": ("P", "names Trillium, Array and Hospira; gold Trillium, Array and Therachon"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "single_gemini_3_1_flash_lite_300.json"), encoding="utf-8"))
    base = {(b["set"], b["n"]): b for b in json.load(open(os.path.join(HERE, "..", "..", "baseline", "baseline_standard.json"), encoding="utf-8"))}
    out = []
    for r in rows:
        b = base[(r["set"], r["n"])]
        key = ("tr" if r["set"] == "train" else "ou") + str(r["n"])
        label, note = LABELS.get(key, ("C", ""))
        out.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                    "question": r["question"], "gold": r["gold"],
                    "baseline_label": b["label"], "baseline_answer": b["answer"],
                    "llm_label": label, "llm_answer": r["model_answer"], "note": note,
                    "rerun": r.get("rerun", False), "grading_basis": "read"})
    json.dump(out, open(os.path.join(HERE, "comparison_gemini_3_1_flash_lite.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "llm": x["llm_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["llm_label"]]
    summary = {"model": "gemini-3.1-flash-lite",
               "baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "llm": {"all": cnt("llm_label"), "train": cnt("llm_label", "train"), "out": cnt("llm_label", "out")},
               "reran_after_fallback": sum(1 for x in out if x["rerun"]),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "llm", "reran_after_fallback")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
