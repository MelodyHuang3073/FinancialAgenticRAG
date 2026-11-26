"""Turn each filing into ONE long table (CSV) for ReAcTable, and write the training questions in its input format.

ReAcTable answers over a single table: its prompt shows only the first rows of it and the model queries the whole
table with the SQL / Python it writes, so one long table per filing fits and no table-selection step is mixed in.
The table is the same one used for the Google TableRAG baseline, built from the rows our own parser extracted from
the PDF (state this in the thesis). Text passages are not used, because ReAcTable has no text input.

Columns: page, line_item, period, value (text as printed), number (parsed, parentheses = negative), unit (the unit
phrase of that page, else the filing's most common one, else empty).

    python -m baselines.reactable.build_inputs --out <data dir>      (run from backend/)
"""
import argparse
import ast
import csv
import json
import os
import pickle
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tests")
SETS = {
    "train": ("financebench_qa_subset.json", "original_corpus_cache.pkl"),
    "out": ("financebench_outsample_qa.json", "outsample_corpus_cache.pkl"),
}
COLUMNS = ["page", "line_item", "period", "value", "number", "unit"]
_UNIT = re.compile(r"(?i)\b(?:in|dollars in|\$ in|usd in|amounts in)\s+(millions|thousands|billions)\b|\((?:in )?(millions|thousands|billions)\b")


def parse_number(text):
    """'$5,363' -> 5363.0, '(370)' -> -370.0, '0.375%' -> 0.375, anything else -> None."""
    s = str(text).strip().replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if s.startswith("-"):
        neg, s = True, s[1:]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def page_unit(text):
    m = _UNIT.search(text or "")
    return (m.group(1) or m.group(2)).lower() if m else ""


def raw(entry):
    r = entry.get("raw_data") or {}
    return ast.literal_eval(r) if isinstance(r, str) else r


def long_table(corpus, company):
    """One row per (line item, period) cell of the company's `table_row` entries."""
    entries = [e for e in corpus if e.get("company") == company]
    page_units, all_units = {}, Counter()
    for e in entries:
        u = page_unit(e.get("parent_content", "")) or page_unit(e.get("content", ""))
        if u:
            page_units.setdefault(str(e.get("page_number")), Counter())[u] += 1
            all_units[u] += 1
    default = all_units.most_common(1)[0][0] if all_units else ""
    rows = [COLUMNS]
    for e in entries:
        if e.get("type") != "table_row":
            continue
        values = raw(e)
        item = str(values.get("line_item", "")).strip()
        if not item:
            continue
        pu = page_units.get(str(e.get("page_number")))
        unit = pu.most_common(1)[0][0] if pu else default
        for period, value in values.items():
            if period == "line_item" or value is None or str(value).strip() in ("", "-", "—"):
                continue
            number = parse_number(value)
            rows.append([int(e["page_number"]), item, str(period), str(value).strip(),
                         "" if number is None else number, unit])
    return rows


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("dated", ""))


def company_for(doc_name, companies):
    target = _norm(doc_name)
    by_norm = {_norm(c): c for c in companies}
    if target in by_norm:
        return by_norm[target]
    for n, c in by_norm.items():
        if target.startswith(n) or n.startswith(target):
            return c
    raise KeyError(f"no cached filing matches doc_name {doc_name!r}")


def build(out_dir, which="train", tests_dir=TESTS):
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    stats = {}
    for name, (qa_file, cache_file) in SETS.items():
        if which not in (name, "all"):
            continue
        qa = json.load(open(os.path.join(tests_dir, qa_file), encoding="utf-8"))
        with open(os.path.join(tests_dir, cache_file), "rb") as fp:
            cache = pickle.load(fp)                       # our own parsed corpus
        companies = sorted({u["company"] for u in cache["uploaded_files"]})
        with open(os.path.join(out_dir, f"questions_{name}.jsonl"), "w", encoding="utf-8") as out:
            for i, item in enumerate(qa, 1):
                company = company_for(item["doc_name"], companies)
                table = re.sub(r"[^A-Za-z0-9_]", "_", company) + ".csv"
                if company not in stats:
                    rows = long_table(cache["corpus"], company)
                    with open(os.path.join(out_dir, "tables", table), "w", encoding="utf-8", newline="") as fp:
                        csv.writer(fp).writerows(rows)
                    stats[company] = len(rows) - 1
                out.write(json.dumps({"id": f"{name}-{i}", "doc_name": item["doc_name"], "table": table,
                                      "rows": stats[company], "question": item["question"], "label": item["answer"],
                                      "question_type": item["question_type"]}, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="train", choices=["train", "out", "all"])
    args = ap.parse_args()
    stats = build(args.out, args.set)
    sizes = sorted(stats.values())
    print(f"{len(stats)} filings; long rows min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print("filings with an empty table:", [c for c, v in stats.items() if not v])


if __name__ == "__main__":
    main()
