"""Build the embedding databases (schema + cell FAISS) once per filing, skipping the ones that already exist.

    python -m baselines.tablerag_google.build_index --data-dir <dir> [--db-dir <dir>] [--set train|out|all]

Run this before the sharded question runs, so that two processes never build the same filing's database.
Embedding calls are billed to the paid balance (the free token program does not cover embeddings).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def missing_tables(db_dir, table_ids, max_cell=10000):
    """Table ids whose schema or cell database is not on disk yet."""
    return [t for t in table_ids
            if not all(os.path.isdir(os.path.join(db_dir, f"{kind}_db_{max_cell}_{t}")) for kind in ("schema", "cell"))]


def main():
    from baselines.tablerag_google import run_questions as rq
    rq.ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--db-dir", default="")
    ap.add_argument("--set", default="all", choices=["train", "out", "all"])
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND, ".env"))
    db_dir = args.db_dir or os.path.join(args.data_dir, "db")
    qs = rq.load_questions(args.data_dir, args.set)
    ids = sorted({q["table_id"] for q in qs})
    todo = missing_tables(db_dir, ids)
    print(f"{len(ids)} filings, {len(ids) - len(todo)} already built, {len(todo)} to build", flush=True)
    if not todo:
        return
    import pandas as pd
    agent = rq.make_agent(db_dir, os.path.join(db_dir, "_log"))
    from utils.utils import infer_dtype
    for i, t in enumerate(todo, 1):
        t0 = time.time()
        tab = json.load(open(os.path.join(args.data_dir, "tables", f"{t}.json"), encoding="utf-8"))["table_text"]
        agent.retriever.init_retriever(t, infer_dtype(pd.DataFrame(tab[1:], columns=tab[0])))
        print(f"[{i}/{len(todo)}] {t} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
