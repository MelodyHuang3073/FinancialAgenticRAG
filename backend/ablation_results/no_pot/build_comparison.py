"""Builds the per-question comparison of the no-PoT ablation (ABLATE_POT=1, single-file, all 300
questions, one run) against the baseline (../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under fixed rules, reviewed by the author).
Grading basis: questions listed in READ were read against the gold answer one by one (117 questions:
baseline not C, or the automatic screen failed, or the gold answer has no number, or yes/no differed).
The other 183 questions had a numeric gold answer that appears in the no-PoT answer within 2%, a
baseline label C and no yes/no conflict; they are labeled C without a word-by-word reading.
2026-10-29: the written rules were applied again to every answer; every answer was read (grading_basis 'read-2026-10-29' for
those that were not in the original reading, 'reread-after-audit' for those whose label the audit changed). No answer was graded by program only.

Run from backend/:  python ablation_results/no_pot/build_comparison.py
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("train", 69): ("S", "39.75% (15,357 / 38,655 = 39.73%); gold 39.7%"),
    ("out", 7): ("S", "45.7% (computed 45.62%, should be 45.6%)"),
    ("out", 54): ("S", "43.6% in the explanation (computed 43.526%, should be 43.5%)"),
    ("out", 24): ("S", "15.1% (computed 15.045%, should be 15.0%)"),
    ("train", 6): ("P", "quick ratio 0.85 (cash+securities+receivables); gold 0.96; conclusion No is right"),
    ("train", 17): ("W", "inventory turnover 12.1 using average inventory; gold 9.5"),
    ("train", 24): ("P", "quick ratio 0.53 to 0.57 with a different definition; gold 0.67 to 0.69; conclusion improved is right"),
    ("train", 55): ("P", "store counts 930 to 907 (-23); gold 982 to 969 (-13); both say decreased"),
    ("train", 61): ("P", "answers with product vs service sales; gold names the three segments over 20%; Yes is right"),
    ("train", 74): ("P", "working capital $2,278M (unadjusted); gold $831M (operating items only); Yes is right"),
    ("train", 82): ("S", "cash conversion cycle -3.62 days; gold -3.7"),
    ("train", 96): ("W", "$82.31 per share (common equity / shares); gold $66.56"),
    ("train", 108): ("W", "interest coverage -3.25; gold 0 (Adjusted EBIT is negative)"),
    ("train", 113): ("W", "EBITDA margin 56.83% (content amortization added back); gold 5.4%"),
    ("train", 119): ("P", "working capital $12,416M; gold about $1.6B; Yes is right"),
    ("train", 145): ("P", "quick ratio 0.71; gold 0.54; conclusion No is right"),
    ("out", 12): ("W", "impairment $1.9B vs $14M; gold $1,993M vs $33M"),
    ("out", 33): ("W", "declines to name a driver (says the excerpts lack the income statement)"),
    ("out", 34): ("W", "total debt $45,938M vs $34,951M (short-term borrowings wrong); gold $47,933M vs $36,634M"),
    ("out", 59): ("P", "margin expansion and lower tax rate; omits the Other (gains) and losses swing"),
    ("out", 70): ("P", "headline restructuring $1,092M/$1,987M; gold figures only as a related line"),
    ("out", 103): ("P", "impairment, D&A, gaming taxes; only a vague mention of other non-operating losses"),
    ("out", 124): ("P", "main driver given as gross margin; gold is other (expense) income"),
}

# 2026-10-28 audit: labels changed after re-checking against the written rules (see ../baseline/README.md)
AUDITED = {('out', 7), ('out', 54), ('train', 69), ('out', 24)}


def numbers(t):
    out = []
    for m in re.finditer(r"(?<![\w.])-?\(?\$?\d[\d,]*\.?\d*\)?%?", t):
        s = re.sub(r"[^\d.\-]", "", m.group())
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def screen(gold, ans):
    g, a = numbers(gold)[:3], numbers(ans)
    if not g:
        return None
    return all(any(abs(x - y) <= 0.02 * abs(x) + 1e-9 for y in a) for x in g)


def yn(t):
    m = re.match(r"\W*(yes|no)\b", t.strip(), re.I)
    return m.group(1).lower() if m else None


def main():
    rows = json.load(open(os.path.join(HERE, "single_nopot_300.json"), encoding="utf-8"))
    base = {(b["set"], b["n"]): b for b in json.load(open(os.path.join(HERE, "..", "baseline", "baseline_standard.json"), encoding="utf-8"))}
    out = []
    for r in rows:
        b = base[(r["set"], r["n"])]
        s = screen(r["gold"], r["model_answer"])
        gy, ay = yn(r["gold"]), yn(r["model_answer"])
        read = b["label"] != "C" or s is None or s is False or (gy and (not ay or gy != ay))
        label, note = LABELS.get((r["set"], r["n"]), ("C", ""))
        out.append({"set": r["set"], "n": r["n"], "doc_name": r["doc_name"], "question_type": r["question_type"],
                    "question": r["question"], "gold": r["gold"],
                    "baseline_label": b["label"], "baseline_answer": b["answer"],
                    "nopot_label": label, "nopot_answer": r["model_answer"], "note": note,
                    "grading_basis": "read" if read else ("reread-after-audit" if (r["set"], r["n"]) in AUDITED else "read-2026-10-29")})
    json.dump(out, open(os.path.join(HERE, "comparison_no_pot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "nopot": x["nopot_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["nopot_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "no_pot": {"all": cnt("nopot_label"), "train": cnt("nopot_label", "train"), "out": cnt("nopot_label", "out")},
               "read_one_by_one": sum(1 for x in out if x["grading_basis"] == "read"),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "no_pot", "read_one_by_one")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
