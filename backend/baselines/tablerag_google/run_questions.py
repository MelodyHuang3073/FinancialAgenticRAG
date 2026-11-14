"""Run Google TableRAG on our questions (single-file: each question sees only its own filing's long table).

    python -m baselines.tablerag_google.run_questions --data-dir <dir with tables/ and questions_*.jsonl> \
        --out results.json [--ids train-1,train-2] [--set train|out|all] [--shard 0/4] [--resume]

Needs TABLERAG_SRC (the downloaded google-research/table_rag folder) and OPENAI_API_KEY (backend/.env is read).
The output list uses the field names of the other experiments' answers.json (id, set, n, doc_name, question_type,
question, gold, answer, elapsed, error) plus the reasoning trace and token counts, and is rewritten after every question,
so `--resume` skips the ids already stored (records that failed are retried).
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

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
EMBED_MODEL = "text-embedding-3-large"


def load_questions(data_dir, which):
    qs = []
    for name in ("train", "out"):
        if which not in (name, "all"):
            continue
        for line in open(os.path.join(data_dir, f"questions_{name}.jsonl"), encoding="utf-8"):
            q = json.loads(line)
            q["set"] = "original" if name == "train" else "outsample"
            q["n"] = int(q["id"].split("-")[1])
            qs.append(q)
    return qs


def select(qs, ids="", shard="", limit=0):
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {q["id"]: q for q in qs}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown ids: {missing}")
        qs = [by_id[i] for i in wanted]
    if limit:
        qs = qs[:limit]
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        qs = qs[k::n]
    return qs


def save(path, results):
    tmp = path + ".tmp"
    json.dump(results, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_agent(db_dir, log_dir, src_dir=None):
    from baselines.tablerag_google import compat, fin_prompts
    compat.load(src_dir)
    import prompts
    fin_prompts.register(prompts)
    from agent import TableRAGAgent
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "log"), exist_ok=True)
    return TableRAGAgent(model_name=MODEL, retrieve_mode="embed", embed_model_name=EMBED_MODEL, task="fin",
                         agent_type="TableRAG", top_k=5, sr=0, max_encode_cell=10000, log_dir=log_dir,
                         db_dir=db_dir, load_exist=False, verbose=False)


def run_one(agent, data_dir, q):
    t = json.load(open(os.path.join(data_dir, "tables", f"{q['table_id']}.json"), encoding="utf-8"))
    data = {"id": q["id"], "question": q["question"], "table_id": q["table_id"], "label": q["label"],
            "table_caption": t["table_caption"], "table_text": t["table_text"]}
    # the agent keeps running token totals; report this question's share
    in0, out0 = agent.total_input_token_count, agent.total_output_token_count
    t0 = time.time()
    res = agent.run(data, sc_id=0)
    return {"answer": res["answer"], "elapsed": time.time() - t0, "n_iter": res["n_iter"],
            "solution": res["solution"], "column_queries": res["column_queries"], "cell_queries": res["cell_queries"],
            "n_rows": res["n_rows"], "input_tokens": agent.total_input_token_count - in0,
            "output_tokens": agent.total_output_token_count - out0}


def ensure_utf8():
    """Google's code opens its log files with the platform default encoding (cp950 on this Windows machine), which
    cannot hold characters such as U+2011 that the model writes. Restart once in Python's UTF-8 mode."""
    if not sys.flags.utf8_mode:
        os.execv(sys.executable, [sys.executable, "-X", "utf8", os.path.abspath(__file__)] + sys.argv[1:])


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db-dir", default="", help="FAISS databases (default: <data-dir>/db)")
    ap.add_argument("--log-dir", default="", help="per-question logs (default: next to --out)")
    ap.add_argument("--set", default="all", choices=["train", "out", "all"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BACKEND, ".env"))
    except ImportError:
        pass
    qs = select(load_questions(args.data_dir, args.set), args.ids, args.shard, args.limit)
    results = []
    if args.resume and os.path.exists(args.out):
        results = [r for r in json.load(open(args.out, encoding="utf-8")) if not r.get("error")]
    done = {r["id"] for r in results}
    todo = [q for q in qs if q["id"] not in done]
    print(f"{len(qs)} selected, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    agent = make_agent(args.db_dir or os.path.join(args.data_dir, "db"),
                       args.log_dir or os.path.splitext(args.out)[0] + "_log")
    for i, q in enumerate(todo, 1):
        rec = {"id": q["id"], "set": q["set"], "n": q["n"], "doc_name": q["doc_name"], "question_type": q["question_type"],
               "question": q["question"], "gold": q["label"], "answer": "", "elapsed": 0.0, "error": None}
        t0 = time.time()
        try:
            rec.update(run_one(agent, args.data_dir, q))
        except Exception as e:  # keep going; the failure is part of the record
            rec["error"], rec["elapsed"] = repr(e), time.time() - t0
        results.append(rec)
        save(args.out, results)
        print(f"[{i}/{len(todo)}] {q['id']} {rec['elapsed']:.0f}s {'ERROR ' + rec['error'] if rec['error'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
