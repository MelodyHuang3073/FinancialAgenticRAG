"""Builds the per-question comparison of the swap-the-LLM run with gemini-3.5-flash-lite (single-file, all 300 questions,
one run) against the baseline (../../baseline/baseline_standard.json, gpt-5-mini).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under the same fixed
rules as the baseline and the ablations, reviewed by the author). Every one of the 300 answers was read against the gold answer
one by one; none was graded by program only. Answers that were the system's no-LLM fallback (the API call returned nothing)
were rerun and the rerun answer is the one graded (field "rerun" in the comparison file).

Run from backend/:  python ablation_results/llm_gemini/gemini_3_5_flash_lite/build_comparison.py
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    "ou3": ("P", "net sales growth given as 2.76% (from $24,279M); gold 1.5% (from $24,575M); net income $4,173M to $3,250M and the weaker-profitability conclusion are right"),
    "ou8": ("S", "ROE 0.33 and 33.31%; computed 33.28%"),
    "ou10": ("S", "54.2% (computed 54.148%, should be 54.1%)"),
    "ou12": ("W", "FY2025 $1.9B and FY2024 'no impairment'; gold $1,993M vs $33M"),
    "ou27": ("W", "revenue growth 81.87% from contract revenue $22,726M to $41,332M; gold 34.5%"),
    "ou31": ("S", "67.8% (63,937 / 94,193 = 67.88%, should be 67.9%)"),
    "ou33": ("P", "says net earnings nearly doubled because of prior-year impacts and business growth; does not name the other (income) expense swing"),
    "ou54": ("S", "net income decline written as 44.0%; 3,219 / 5,700 gives 43.5%; the implication (gross margin 44.6% to 42.7%) is right"),
    "ou57": ("S", "operating margin 4.4% (4.35%); computed 4.31%, should be 4.3%"),
    "ou70": ("P", "main answer $1,092M vs $1,987M; gold $1,550M vs $2,419M is the 'restructuring charges and certain acquisition-related costs' line"),
    "ou84": ("W", "says service and wireless equipment revenues are not consolidated line items; gives Consumer-segment $80.6B/$80.9B; gold $112,721M and $25,470M"),
    "ou94": ("S", "gross margin 33.1%; 8,309 / 24,942 = 33.3%"),
    "ou101": ("W", "growth 6.66% from a FY2024 revenue of $16,164,249 thousand; gold 1.7% from $17,240,545 thousand"),
    "ou103": ("P", "explains the operating-income drop (impairments and write-offs); does not name the swing in other non-operating items"),
    "ou116": ("S", "growth 9.94%; (5,140 - 4,684) / 4,684 = 9.7%"),
    "ou118": ("W", "operating margin 39.8% using operating revenues $4,723M (a segment figure); gold 36.6% on $5,140M"),
    "ou124": ("P", "profit right; says lower impairment and other charges (they rose from $80M to $97M) and reduced other expenses; gold main driver is other (expense) income improving"),
    "ou129": ("W", "operating income $1,564,972K vs $1,678,029K: the prior-year columns; gold $1,532,992K vs $1,564,972K"),
    "ou130": ("W", "uses the prior-year columns ($1,564,972K vs $1,678,029K; SG&A $2,808,592K); gold $1,532,992K vs $1,564,972K, SG&A $3,296,411K"),
    "ou132": ("W", "diluted EPS $25.34 vs $26.03 (prior-year columns); gold $25.64 vs $25.34"),
    "ou133": ("W", "gross margin 38.8% from the prior-year columns; gold 39.1%"),
    "ou136": ("P", "leads with 57.3% and adds 29.5% as the company's stated margin; gold 29.5%"),
    "ou147": ("P", "credits gross-margin expansion and lower restructuring; gold main driver is operating leverage (SG&A and R&D growing slower than sales)"),
    "tr5": ("P", "names two segments as drags: Consumer (-0.9%) and Safety and Industrial, which grew +1.0%; gold is Consumer only"),
    "tr6": ("W", "answers Yes (healthy liquidity, quick ratio 0.96); gold No"),
    "tr15": ("P", "Yes, improving, but supported by operating cash flow growth; no FCF conversion (143% to 156%) given"),
    "tr51": ("W", "answers No (says fluctuation exceeds 2%) although its own figures 22.4/22.5/21.4 move by at most 1.1 points; gold Yes"),
    "tr56": ("P", "calls Services (+7.6%) the fastest-growing category although it also lists Entertainment at +9.0%; gold Entertainment +9%"),
    "tr67": ("S", "FY2021 rate written as 14.7%; the filing prints 14.8%"),
    "tr101": ("S", "final ratio 1.33 right, but the average total assets is written as $48,919M (it is $49,119M)"),
    "tr103": ("S", "CAGR 0.5% (computed 0.447%, should be 0.4%)"),
    "tr119": ("S", "working capital $1,548M right, but total current assets written as $57,517M, which does not subtract to $1,548M"),
    "tr124": ("W", "gives EBITDA ($14,274M) as 'EBITDA less capex'; capex not subtracted; gold $9,068M"),
    "tr127": ("W", "says the evidence does not contain the increase; gold $400,000,000"),
    "tr133": ("P", "names Array, Hospira and Trillium; gold Trillium, Array and Therachon"),
    "tr136": ("W", "answers that Pfizer is not spinning off any large business segment; gold Yes, Upjohn"),
    "tr141": ("W", "says the Q4 repurchase share cannot be computed; gold 36%"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "single_gemini_3_5_flash_lite_300.json"), encoding="utf-8"))
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
    json.dump(out, open(os.path.join(HERE, "comparison_gemini_3_5_flash_lite.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "llm": x["llm_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["llm_label"]]
    summary = {"model": "gemini-3.5-flash-lite",
               "baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "llm": {"all": cnt("llm_label"), "train": cnt("llm_label", "train"), "out": cnt("llm_label", "out")},
               "reran_after_fallback": sum(1 for x in out if x["rerun"]),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "llm", "reran_after_fallback")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
