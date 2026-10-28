"""Builds the per-question comparison of the no-domain-knowledge ablation (ABLATE_DOMAIN=1, single-file, all 300
questions, one run) against the baseline (../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under fixed rules, reviewed by the author).
Grading basis: questions listed in READ were read against the gold answer one by one (118 questions:
baseline not C, or the automatic screen failed, or the gold answer has no number, or yes/no differed).
The other 183 questions had a numeric gold answer that appears in the no-domain answer within 2%, a
baseline label C and no yes/no conflict; they are labeled C without a word-by-word reading.
2026-10-29: the written rules were applied again to every answer; every answer was read (grading_basis 'read-2026-10-29' for
those that were not in the original reading, 'reread-after-audit' for those whose label the audit changed). No answer was graded by program only.

Run from backend/:  python ablation_results/no_domain/build_comparison.py
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("train", 148): ("S", "42.70 days from a cost of sales derived from the gross-margin rate (373,308); the filing's cost of sales 373,396 gives 42.69"),
    ("out", 10): ("S", "54.2% (computed 54.148%, should be 54.1%)"),
    ("out", 24): ("P", "amounts self-derived ($364,290M / $419,141M) differ from the filing lines; growth 15.1% also off by 0.1"),
    ("out", 32): ("S", "6.1% (computed 6.048%, should be 6.0%)"),
    ("out", 52): ("S", "43.6% (computed 43.526%, should be 43.5%)"),
    ("train", 73): ("S", "states -2.63 pp then 'an increase of 2.63 pp': sign contradiction; rates 22.87% / 20.24% are right"),
    ("train", 6): ("P", "quick ratio 0.85; gold 0.96; conclusion No is right"),
    ("train", 17): ("W", "inventory turnover 12.1; gold 9.5"),
    ("train", 18): ("W", "says ROA cannot be computed (no total assets in evidence); gold -0.02"),
    ("train", 24): ("P", "quick ratio 0.531 to 0.573 with a different definition; gold 0.67 to 0.69; improved is right"),
    ("train", 32): ("P", "quick ratio 1.77 ((current assets - inventory)/current liabilities); gold 1.57; Yes is right"),
    ("train", 56): ("W", "names Computing and Mobile Phones (largest by revenue); gold is Entertainment (fastest growth 9%)"),
    ("train", 63): ("W", "says gross margin is not a useful metric for Boeing; gold: Yes, improving 4.8% to 5.3%"),
    ("train", 64): ("P", "airlines and government customers by segment revenue; omits the 40% US government share"),
    ("train", 71): ("W", "DPO 44 days (company's own table); gold 63.86"),
    ("train", 74): ("P", "working capital $2,278M (company's reported figure); gold $831M (operating items only); Yes is right"),
    ("train", 77): ("W", "says CVS is not capital intensive; gold Yes (ROA 1.82%)"),
    ("train", 87): ("P", "gross margin fell 68.2% to 67.3% from cost growth; does not give the vaccine-exit cost, currency and commodity inflation drivers"),
    ("train", 89): ("W", "Yes, accelerate (4.0% vs 3.6%); gold No, decelerates 3.6% to 3.5% on the operational basis"),
    ("train", 96): ("W", "$82.31 per share (common equity / shares); gold $66.56"),
    ("train", 108): ("W", "interest coverage 5.88 using Adjusted EBITDAR; gold 0 (Adjusted EBIT is negative)"),
    ("train", 109): ("W", "names Management and other operations; gold MGM China (-44%)"),
    ("train", 119): ("W", "cannot determine PayPal's working capital; gold positive $1.6B"),
    ("train", 123): ("W", "restructuring costs $0 on the income statement; gold $411M"),
    ("train", 124): ("W", "cannot compute EBITDA less capex (no D&A); gold $9,068M"),
    ("train", 125): ("W", "cannot compute EBITDA margin (no D&A); gold 16.5%"),
    ("train", 133): ("P", "Trillium, Therachon and Warner-Lambert; gold Trillium, Array, Therachon"),
    ("train", 134): ("W", "$700M (the already-incurred amount read as the total); gold $77.78M"),
    ("train", 136): ("W", "No spin-off under way; gold Yes, Upjohn"),
    ("out", 9): ("W", "says it cannot tell which grew faster; gold Service and other revenue +20.8%, Product -1.3%"),
    ("out", 11): ("P", "net income down; lists Rockstar impairment only among several operating causes; gold: intangible impairment jump is the primary driver"),
    ("out", 12): ("W", "impairment $1,946M vs $14M; gold $1,993M vs $33M"),
    ("out", 68): ("W", "gross margin 78.4%; gold 74.3%"),
    ("out", 69): ("W", "says the acquired IPR&D amounts are not in the excerpts; gold $1,613M vs $108M"),
    ("out", 70): ("P", "headline restructuring $1,092M/$1,987M; gold $1,550M/$2,419M not given"),
    ("out", 84): ("W", "Consumer segment service $80,912M and wireless equipment $21,779M; gold $112,721M and $25,470M"),
    ("out", 103): ("P", "impairments, write-offs and D&A; omits the operating income decline and the swing in other non-operating items"),
    ("out", 104): ("P", "casino revenue $9,350M (segment sum); gold about $9.5B; rooms $3,377M is right"),
    ("out", 124): ("P", "profit right; main driver given as gross margin; gold is other (expense) income"),
}

# 2026-10-28 audit: labels changed after re-checking against the written rules (see ../baseline/README.md)
AUDITED = {('out', 32), ('out', 52), ('train', 148), ('out', 10), ('out', 24), ('train', 73)}


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
    rows = json.load(open(os.path.join(HERE, "single_nodomain_300.json"), encoding="utf-8"))
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
                    "nodomain_label": label, "nodomain_answer": r["model_answer"], "note": note,
                    "grading_basis": "read" if read else ("reread-after-audit" if (r["set"], r["n"]) in AUDITED else "read-2026-10-29")})
    json.dump(out, open(os.path.join(HERE, "comparison_no_domain.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "nodomain": x["nodomain_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["nodomain_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "no_domain": {"all": cnt("nodomain_label"), "train": cnt("nodomain_label", "train"), "out": cnt("nodomain_label", "out")},
               "read_one_by_one": sum(1 for x in out if x["grading_basis"] == "read"),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "no_domain", "read_one_by_one")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
