"""Builds the per-question comparison of the no-router ablation (ABLATE_ROUTER=numeric, single-file, all 300
questions, one run) against the baseline (../baseline/baseline_standard.json).

Labels: C correct, S small error, P partially correct, W wrong (the AI assistant's reading of the answer under fixed rules, reviewed by the author).
Grading basis: questions listed in READ were read against the gold answer one by one (106 questions:
baseline not C, or the automatic screen failed, or the gold answer has no number, or yes/no differed).
The other 183 questions had a numeric gold answer that appears in the no-router answer within 2%, a
baseline label C and no yes/no conflict; they are labeled C without a word-by-word reading.
2026-10-29: the written rules were applied again to every answer; every answer was read (grading_basis 'read-2026-10-29' for
those that were not in the original reading, 'reread-after-audit' for those whose label the audit changed). No answer was graded by program only.

Run from backend/:  python ablation_results/no_router/build_comparison.py
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    ("out", 10): ("S", "54.2% (computed 54.148%, should be 54.1%)"),
    ("out", 52): ("S", "43.6% (computed 43.526%, should be 43.5%)"),
    ("out", 102): ("S", "72.5% (computed 72.425%, should be 72.4%)"),
    ("train", 6): ("W", "answers Yes, healthy (quick ratio 0.9578); gold No (0.96 is below 1)"),
    ("train", 25): ("W", "says the extracts do not name the acquired companies; gold lists the FY2023 acquisitions"),
    ("train", 28): ("W", "says the nature of the restructuring liability cannot be determined; gold: 87% employee liabilities"),
    ("train", 44): ("W", "names long-term debt $42.6B as the largest liability; gold customer deposits"),
    ("train", 74): ("P", "working capital $2,278M (unadjusted); gold $831M (operating items only); Yes is right"),
    ("train", 96): ("W", "$82.31 per share (book value per share); gold $66.56"),
    ("out", 8): ("S", "ROE written as 0.33; gold 33.28%"),
    ("out", 12): ("W", "impairment $1.9B vs $22M; gold $1,993M vs $33M"),
    ("out", 32): ("P", "gives operational sales growth 5.3%; gold reported growth about 6.0%"),
    ("out", 33): ("W", "net earnings driver given as stronger Innovative Medicine operating performance; gold is the Other (income) expense swing"),
    ("out", 50): ("W", "investment banking fees $9,735M vs $9,116M (division table); gold $9,615M vs $8,910M"),
    ("out", 59): ("P", "margin expansion and lower tax rate; omits the Other (gains) and losses swing"),
    ("out", 84): ("W", "Consumer segment service $80,912M and wireless equipment $21,779M; gold $112,721M and $25,470M"),
    ("out", 85): ("W", "says the diluted EPS amounts are not in the excerpts; gold $4.06 vs $4.14"),
    ("out", 87): ("W", "net earnings $2,609.6M vs $2,518.6M (increase); gold $2,295.2M vs $2,496.6M (decrease)"),
    ("out", 103): ("P", "impairments, write-offs and D&A; omits the operating income decline and the swing in other non-operating items"),
    ("out", 104): ("P", "casino revenue $9,350M (segment sum); gold about $9.5B; rooms $3,377M is right"),
    ("out", 124): ("P", "profit right; main driver given as gross margin; gold is other (expense) income"),
}

# 2026-10-28 audit: labels changed after re-checking against the written rules (see ../baseline/README.md)
AUDITED = {('out', 10), ('out', 102), ('out', 52)}


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
    rows = json.load(open(os.path.join(HERE, "single_norouter_numeric_300.json"), encoding="utf-8"))
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
                    "norouter_label": label, "norouter_answer": r["model_answer"], "note": note,
                    "grading_basis": "read" if read else ("reread-after-audit" if (r["set"], r["n"]) in AUDITED else "read-2026-10-29")})
    json.dump(out, open(os.path.join(HERE, "comparison_no_router_numeric.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def cnt(k, st=None):
        c = Counter(x[k] for x in out if st is None or x["set"] == st)
        return {"n": sum(c.values()), **{l: c.get(l, 0) for l in "CSPW"}}
    changed = [{"set": x["set"], "n": x["n"], "baseline": x["baseline_label"], "norouter": x["norouter_label"], "question": x["question"][:90]}
               for x in out if x["baseline_label"] != x["norouter_label"]]
    summary = {"baseline": {"all": cnt("baseline_label"), "train": cnt("baseline_label", "train"), "out": cnt("baseline_label", "out")},
               "no_router_numeric": {"all": cnt("norouter_label"), "train": cnt("norouter_label", "train"), "out": cnt("norouter_label", "out")},
               "read_one_by_one": sum(1 for x in out if x["grading_basis"] == "read"),
               "label_changes": changed}
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("baseline", "no_router_numeric", "read_one_by_one")}, ensure_ascii=False))
    print(len(changed), "label changes")


if __name__ == "__main__":
    main()
