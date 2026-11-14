"""Turn each filing into ONE long table for Google's TableRAG, and write the 300 questions in its input format.

Why one long table per filing: TableRAG answers over a single table (its selling point is very large tables with
schema and cell retrieval), so no table-selection step is needed and no extra method is mixed in. The table
is built from the rows our own parser already extracted from the PDF (generic PDF table finders see almost none of
the borderless statements: PyMuPDF `find_tables` found 5 tables in the 160-page 3M 2018 10-K). Text passages are
not used, because TableRAG has no text input.

Columns: page, line_item, period, value (text as printed), number (parsed, parentheses = negative), unit (the unit
phrase of that page, else the filing's most common one, else empty).

    python -m baselines.tablerag_google.build_tables --out <dir>      (run from backend/)
writes <dir>/tables/<company>.json and <dir>/questions_{train,out}.jsonl
"""
import argparse
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


def page_unit(parent_content):
    m = _UNIT.search(parent_content or "")
    return (m.group(1) or m.group(2)).lower() if m else ""


def long_table(corpus, company):
    """Rows of the company's `table_row` entries, one long row per (line item, period) cell.

    Unit: the unit phrase on the same page (any entry of that page), else the filing's most common unit phrase."""
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
        raw = e.get("raw_data") or {}
        item = str(raw.get("line_item", "")).strip()
        if not item:
            continue
        pu = page_units.get(str(e.get("page_number")))
        unit = pu.most_common(1)[0][0] if pu else default
        for period, value in raw.items():
            if period == "line_item" or value is None or str(value).strip() in ("", "-", "—"):
                continue
            rows.append([int(e["page_number"]), item, str(period), str(value).strip(), parse_number(value), unit])
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


def build(out_dir, tests_dir=TESTS):
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    stats = {}
    for name, (qa_file, cache_file) in SETS.items():
        qa = json.load(open(os.path.join(tests_dir, qa_file), encoding="utf-8"))
        cache = pickle.load(open(os.path.join(tests_dir, cache_file), "rb"))
        companies = sorted({u["company"] for u in cache["uploaded_files"]})
        with open(os.path.join(out_dir, f"questions_{name}.jsonl"), "w", encoding="utf-8") as fp:
            for i, item in enumerate(qa, 1):
                company = company_for(item["doc_name"], companies)
                path = os.path.join(out_dir, "tables", f"{company}.json")
                if company not in stats:
                    table = long_table(cache["corpus"], company)
                    json.dump({"table_caption": f"{company} (all financial tables of the filing)", "table_text": table},
                              open(path, "w", encoding="utf-8"), ensure_ascii=False)
                    stats[company] = len(table) - 1
                fp.write(json.dumps({"id": f"{name}-{i}", "doc_name": item["doc_name"], "table_id": company,
                                     "question": item["question"], "label": item["answer"],
                                     "question_type": item["question_type"]}, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stats = build(args.out)
    sizes = sorted(stats.values())
    print(f"{len(stats)} filings; long rows min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print("smallest:", Counter(stats).most_common()[-5:])


if __name__ == "__main__":
    main()
