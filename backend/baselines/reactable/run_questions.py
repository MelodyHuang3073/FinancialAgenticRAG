"""Run ReAcTable on our training questions (single-file: each question sees only its own filing's long table).
Runs inside the container (see run_docker.ps1):

    python -m baselines.reactable.run_questions --data-dir /data --out /out/results.json
        [--ids train-1,train-2] [--limit 3] [--shard 0/4] [--resume]

As in the authors' notebook, a question whose 5 runs raise is retried up to 3 times; after that the record keeps
the error. The majority answer is the system's answer; all 5 answers are stored.
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

from baselines.reactable import compat   # noqa: E402

TEMPLATE = "prompt_template/original-sql-py-no-intermediate.json"
DEMOS = "few-shot-demo/WikiTQ-sql-py.json"
REPEATS = 5
MAX_ATTEMPTS = 3


def load_questions(data_dir, name="train"):
    qs = []
    for line in open(os.path.join(data_dir, f"questions_{name}.jsonl"), encoding="utf-8"):
        q = json.loads(line)
        q["set"] = "original" if name == "train" else "outsample"
        q["n"] = int(q["id"].split("-")[1])
        qs.append(q)
    return qs


def select(qs, ids="", limit=0, shard=""):
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
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_one(executor_class, data_dir, dataset_dir, q):
    agent = executor_class(TEMPLATE, q["id"], q["question"], os.path.join(data_dir, "tables", q["table"]), "",
                           base_path=dataset_dir, demo_file=DEMOS)
    agent.model, agent.max_demo = compat.MODEL, 5
    agent._gen_gpt_prompt(False)
    agent._get_gpt_prediction_majority_vote(repeat_times=REPEATS)
    return {"answer": str(agent.predicted_result).strip(), "all_answers": [str(a) for a in agent.all_predictions],
            "last_run_steps": [str(s) for s in agent.original_output], "last_execution_error": str(agent.gpt_error or "")[:600]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="train", choices=["train", "out"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    data_dir = os.path.abspath(args.data_dir)
    qs = select(load_questions(data_dir, args.set), args.ids, args.limit, args.shard)
    results = []
    if args.resume and os.path.exists(out):
        results = [r for r in json.load(open(out, encoding="utf-8")) if not r.get("error")]
    done = {r["id"] for r in results}
    todo = [q for q in qs if q["id"] not in done]
    print(f"{len(qs)} selected, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    work = os.path.join(os.path.dirname(out), "work")
    os.makedirs(os.path.join(work, "tmp"), exist_ok=True)
    os.chdir(work)                                   # the released code writes its SQLite files to ./tmp
    executor_class = compat.load()
    dataset_dir = os.path.join(os.environ["REACTABLE_SRC"], "dataset", "WikiTableQuestions")
    for i, q in enumerate(todo, 1):
        rec = {"id": q["id"], "set": q["set"], "n": q["n"], "doc_name": q["doc_name"], "question_type": q["question_type"],
               "question": q["question"], "gold": q["label"], "table_rows": q["rows"], "answer": "", "elapsed": 0.0,
               "error": None}
        compat.reset_counters()
        t0 = time.time()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                rec.update(run_one(executor_class, data_dir, dataset_dir, q))
                rec["error"] = None
                break
            except Exception as e:                   # keep going; the failure is part of the record
                rec["error"] = repr(e)[:600]
            rec["attempts"] = attempt
        rec["elapsed"] = time.time() - t0
        rec.update(compat.counters())
        results.append(rec)
        save(out, results)
        print(f"[{i}/{len(todo)}] {q['id']} {rec['elapsed']:.0f}s calls {rec['calls']} "
              f"tokens {rec['input_tokens'] + rec['output_tokens']:,} "
              f"{'ERROR ' + rec['error'] if rec['error'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
