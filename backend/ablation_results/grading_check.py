"""Mechanical checks on a group's grading labels, run before any results are committed.

Reading the answers is still done by a person (EXPERIMENT_PLAN.md section 7); this script only
catches inconsistencies that need no reading:

  1. every label file is well formed (300 questions, ids unique, label in C/S/P/W);
  2. every label other than C names the rule it was judged by, and the label is one that rule allows
     (a decision-table row 1-22 of section 7.5, or a section 7.4 rule such as "7.4-6");
  3. across groups: any question whose label differs between groups is listed, with each group's rule,
     so the answers can be re-read together and unified.

Label file: a JSON list of {"id": "train-12" | "out-8", "label": "C|S|P|W", "rule": 3 | "7.4-6" | null}.
Run:  python grading_check.py group_a.json group_b.json ...
"""
import json
import sys

LABELS = ("C", "S", "P", "W")
ANY = set(LABELS)
#: Labels each decision-table row (section 7.5) allows.
ROW_LABELS = {
    1: {"S"}, 2: {"P"}, 3: {"W"}, 4: {"S"}, 5: {"P"}, 6: {"P", "W"}, 7: {"C"}, 8: {"C"},
    9: ANY, 10: {"C", "S"}, 11: {"P"}, 12: {"P"}, 13: {"W"}, 14: {"W"}, 15: {"P"}, 16: {"P"},
    17: {"C"}, 18: {"P", "W"}, 19: {"C", "P", "W"}, 20: {"P"}, 21: ANY, 22: ANY,
}
#: Labels each section 7.4 rule allows ("7.4-<n>").
TEXT_LABELS = {
    "7.4-1": {"C", "P", "W"}, "7.4-2": {"W"}, "7.4-3": {"C", "P", "W"}, "7.4-4": {"P", "W"},
    "7.4-5": {"P"}, "7.4-6": {"C", "W"}, "7.4-7": {"C"}, "7.4-8": {"S", "P", "W"},
}
N_QUESTIONS = 300


def allowed(rule):
    if isinstance(rule, int):
        return ROW_LABELS.get(rule)
    return TEXT_LABELS.get(rule)


def validate(rows, name="group"):
    """Problems found in one label file (an empty list = consistent)."""
    problems = []
    ids = [r.get("id") for r in rows]
    if len(rows) != N_QUESTIONS:
        problems.append(f"{name}: {len(rows)} rows, expected {N_QUESTIONS}")
    if len(set(ids)) != len(ids):
        problems.append(f"{name}: duplicate ids")
    for r in rows:
        label, rule = r.get("label"), r.get("rule")
        if label not in LABELS:
            problems.append(f"{name} {r.get('id')}: label {label!r} is not C/S/P/W")
        elif label != "C" and rule is None:
            problems.append(f"{name} {r.get('id')}: label {label} has no rule")
        elif rule is not None:
            ok = allowed(rule)
            if ok is None:
                problems.append(f"{name} {r.get('id')}: unknown rule {rule!r}")
            elif label not in ok:
                problems.append(f"{name} {r.get('id')}: rule {rule} allows {sorted(ok)}, label is {label}")
    return problems


def differing(groups):
    """Questions whose label differs between groups: {id: {group: (label, rule)}}."""
    out = {}
    names = list(groups)
    for qid in {r["id"] for rows in groups.values() for r in rows}:
        cells = {g: next(((r["label"], r.get("rule")) for r in groups[g] if r["id"] == qid), None) for g in names}
        if len({c[0] for c in cells.values() if c}) > 1:
            out[qid] = cells
    return out


def main(paths):
    groups = {p: json.load(open(p, encoding="utf-8")) for p in paths}
    bad = [x for p, rows in groups.items() for x in validate(rows, p)]
    for x in bad:
        print("PROBLEM", x)
    diff = differing(groups)
    print(f"{len(diff)} questions have different labels across {len(groups)} groups (re-read these together)")
    for qid in sorted(diff):
        print(" ", qid, {g.split('/')[-1]: c for g, c in diff[qid].items()})
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
