"""
Builds the comparison standard for the ablation experiments.

Baseline (full system) per question = the latest all-files regression
(r9 training / r6 outsample, the committed *_results.json), and, for the
questions the automatic grader failed in any earlier all-files round, the
single-file rerun instead (each question sees only its own filing, the
teacher's grading condition).
The first ablation (no PoT, ABLATE_POT=1) is stored the same way beside it.

Labels are manual semantic gradings (C correct, S small error: right
conclusion with a percentage 0.1pp off, P partial, W wrong). Questions not
listed are C. Keys are 1-based positions: outsample in outsample_results.json
order; training = metrics_generated + domain_relevant + novel_generated
results in that order (50 + 50 + 50).
Run from backend/:  python ablation_results/build_baseline_standard.py
"""
import csv
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "tests")
from test_financebench_qa import _numbers_in

OUT_DIR = os.path.join("ablation_results", "baseline")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- manual labels: all-files runs (non-C only) ---------------------------
ALL_POT = {
    "out": {8: "W", 10: "W", 12: "W", 17: "W", 23: "W", 38: "S", 50: "W", 68: "W", 83: "W",
            102: "S", 103: "P", 104: "W", 105: "W", 124: "P"},
    "train": {14: "W", 68: "W", 76: "W", 94: "W", 95: "P", 119: "W", 125: "W", 145: "W", 148: "W"},
}
ALL_NOPOT = {
    "out": {4: "W", 10: "W", 12: "W", 24: "S", 38: "S", 39: "P", 50: "W", 59: "P", 103: "P",
            122: "P", 124: "P", 145: "S"},
    "train": {14: "W", 57: "W", 68: "W", 119: "W", 127: "W", 145: "W", 148: "W",
              53: "P", 59: "P", 62: "P", 79: "P", 81: "P", 92: "P", 95: "P", 113: "P", 131: "P", 132: "P",
              52: "S", 55: "S"},
}
# ---- manual labels: single-file reruns of the historical hard set ----------
SINGLE_POT = {
    "out": {8: "S", 17: "C", 23: "W", 34: "C", 39: "C", 59: "C", 76: "C", 92: "C", 101: "C", 102: "S",
            103: "C", 104: "W", 105: "C", 122: "C", 124: "P", 129: "C", 99: "C", 12: "W", 150: "C"},
    "train": {n: "C" for n in (103, 105, 72, 79, 117, 22, 81, 83, 119, 86, 120, 126, 130, 132, 133, 92,
                               145, 148, 102, 59, 61, 75, 76, 84, 68, 143, 144, 94)} | {14: "W"},
}
SINGLE_NOPOT = {
    "out": {8: "C", 17: "C", 23: "C", 34: "C", 39: "P", 59: "C", 76: "C", 92: "C", 101: "S", 102: "S",
            103: "P", 104: "W", 105: "C", 122: "C", 124: "P", 129: "C", 99: "C", 12: "W", 150: "C"},
    "train": {n: "C" for n in (103, 105, 72, 79, 117, 22, 83, 119, 86, 120, 126, 130, 132, 133,
                               145, 148, 102, 59, 61, 75, 76, 84, 68, 143, 144, 94)}
    | {14: "W", 81: "P", 92: "P"},
}


def head(path):
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:backend/{path}"]).decode("utf-8"))


def load_sets(ablation):
    """-> {set: [records in numbering order]} ; ablation=False reads HEAD (baseline), True reads no_pot copies."""
    r = {}
    for s, files in (("train", ["metrics_generated", "domain_relevant", "novel_generated"]), ("out", ["outsample"])):
        rows = []
        for f in files:
            name = f"{f}_results.json"
            part = json.load(open(os.path.join("ablation_results", "no_pot", name), encoding="utf-8")) if ablation \
                else head(name)
            for x in part:  # the training result files do not carry the type; it is the file's
                x.setdefault("question_type", f.replace("_", "-"))
            rows += part
        r[s] = rows
    return r


def single(prefix):
    d = {}
    for s in ("train", "out"):
        d[s] = {x["question"]: x for x in json.load(open(os.path.join("ablation_results", f"{prefix}_{s}.json"), encoding="utf-8"))}
    return d


def build(allfiles, single_rows, all_labels, single_labels):
    rec = []
    for s in ("train", "out"):
        hard = {}
        for n, a in enumerate(allfiles[s], 1):
            if a["question"] in single_rows[s]:
                hard[n] = single_rows[s][a["question"]]
        assert set(hard) == set(single_labels[s]), (s, set(hard) ^ set(single_labels[s]))
        for n, a in enumerate(allfiles[s], 1):
            src = hard.get(n)
            label = single_labels[s][n] if src else all_labels[s].get(n, "C")
            rec.append({
                "set": s, "n": n, "doc_name": a["doc_name"], "question_type": a["question_type"],
                "question": a["question"], "gold": a["gold"],
                "answer": (src or a)["model_answer"], "source": "single-file" if src else "all-files",
                "label": label, "all_files_label": all_labels[s].get(n, "C"),
            })
    return rec


def decimals(question):
    m = re.search(r"[Rr]ound[^.?]*?(one|two|three|1|2|3) decimal", question)
    return {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}[m.group(1)] if m else None


def mechanical(rec):
    """Auxiliary checks on the merged answers (not the grade): value present at tolerance, exact at requested rounding."""
    tol = {0.001: 0, 0.005: 0, 0.02: 0}
    numeric = exact = exact_ok = 0
    for r in rec:
        g = _numbers_in(r["gold"])
        if not g:
            continue
        numeric += 1
        ms = _numbers_in(r["answer"])
        vals = [m * k for m in ms for k in (1, 100, 0.01, 1e3, 1e-3, 1e6, 1e-6)]
        for t in tol:
            if any((abs(v - g[0]) <= t * abs(g[0])) if g[0] else abs(v) < 1e-9 for v in vals):
                tol[t] += 1
        d = decimals(r["question"])
        if d is not None:
            exact += 1
            tgt = round(g[0], d)
            if any(abs(round(v, d) - tgt) < 10 ** -(d + 3) for v in vals):
                exact_ok += 1
    return {"numeric_gold": numeric, "within": {str(k): v for k, v in tol.items()},
            "round_instructed": exact, "exact_at_requested_decimals": exact_ok}


def tally(rec):
    out = {}
    for s in ("train", "out", "all"):
        sub = [r for r in rec if s == "all" or r["set"] == s]
        c = {k: sum(1 for r in sub if r["label"] == k) for k in "CSPW"}
        by_type = {}
        for t in sorted({r["question_type"] for r in sub}):
            tt = [r for r in sub if r["question_type"] == t]
            by_type[t] = {"n": len(tt), **{k: sum(1 for r in tt if r["label"] == k) for k in "CSPW"}}
        out[s] = {"n": len(sub), **c, "by_type": by_type, "mechanical": mechanical(sub),
                  "mechanical_metrics_only": mechanical([r for r in sub if r["question_type"] == "metrics-generated"])}
    return out


base = build(load_sets(False), single("single_pot"), ALL_POT, SINGLE_POT)
HAS_NOPOT = os.path.isdir(os.path.join("ablation_results", "no_pot"))  # the ablation records live on the experiment branch
summary = {"baseline_full_system": tally(base)}
if HAS_NOPOT:
    nopot = build(load_sets(True), single("single_nopot"), ALL_NOPOT, SINGLE_NOPOT)
    summary["ablation_no_pot"] = tally(nopot)
# all-files only (no single-file replacement), for the "before" column
allonly = {}
for name, sets, labels in (("baseline_full_system", load_sets(False), ALL_POT),) +         ((("ablation_no_pot", load_sets(True), ALL_NOPOT),) if HAS_NOPOT else ()):
    rec = []
    for s in ("train", "out"):
        for n, a in enumerate(sets[s], 1):
            rec.append({"set": s, "question_type": a["question_type"], "label": labels[s].get(n, "C"),
                        "question": a["question"], "gold": a["gold"], "answer": a["model_answer"]})
    allonly[name] = tally(rec)
summary["all_files_only"] = allonly

json.dump(base, open(os.path.join(OUT_DIR, "baseline_standard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if HAS_NOPOT:
    json.dump(nopot, open(os.path.join(OUT_DIR, "ablation_no_pot_standard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in summary.items():
    if k == "all_files_only":
        for kk, vv in v.items():
            print("all-files-only", kk, {s: {x: vv[s][x] for x in ("n", "C", "S", "P", "W")} for s in vv})
    else:
        print(k, {s: {x: v[s][x] for x in ("n", "C", "S", "P", "W")} for s in v})
        print("   mechanical(metrics-generated, all):", v["all"]["mechanical_metrics_only"])
        for t, d in v["all"]["by_type"].items():
            print("   ", t, d)


def write_csv(path, header, rows):
    """Excel-friendly (utf-8 with BOM) copy of the same records."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


write_csv(os.path.join(OUT_DIR, "baseline_answers.csv"),
          ["set", "n", "question_type", "doc_name", "question", "gold", "answer", "answer_source", "label"],
          [[r["set"], r["n"], r["question_type"], r["doc_name"], r["question"], r["gold"], r["answer"],
            r["source"], r["label"]] for r in base])
if HAS_NOPOT:  # side by side, one row per question, for comparing an ablation against the baseline
    write_csv(os.path.join(OUT_DIR, "comparison_no_pot.csv"),
              ["set", "n", "question_type", "doc_name", "question", "gold",
               "baseline_answer", "baseline_source", "baseline_label",
               "no_pot_answer", "no_pot_source", "no_pot_label"],
              [[b["set"], b["n"], b["question_type"], b["doc_name"], b["question"], b["gold"],
                b["answer"], b["source"], b["label"], a["answer"], a["source"], a["label"]]
               for b, a in zip(base, nopot)])
