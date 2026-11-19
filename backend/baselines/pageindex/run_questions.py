"""Run PageIndex (pip package `pageindex`, local mode) on our questions: one filing per question, the SDK's own chat.

    python run_questions.py --out <dir> --set train [--shard 0/4] [--stop-at 8400000]

Documents are processed one at a time: build the filing's index (skipped when already built), then answer every question
on that filing. Results are rewritten after each question and each index, so a rerun resumes. Because PageIndex needs about
1e8 tokens for all 300 questions, the run stops cleanly before each unit of work once today's UTC ledger total reaches
--stop-at (the free allowance is 10M tokens/day); rerun after the next 00:00 UTC to continue.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)
from feasibility import MODEL, Meter, load_env  # noqa: E402


def load_questions(data_dir, which):
    qs = []
    for name, tag in (("train", "original"), ("out", "outsample")):
        if which not in (name, "all"):
            continue
        for line in open(os.path.join(data_dir, f"questions_{name}.jsonl"), encoding="utf-8"):
            q = json.loads(line)
            q["set"], q["n"] = tag, int(q["id"].split("-")[1])
            qs.append(q)
    return qs


def group_by_doc(qs):
    docs = {}
    for q in qs:
        docs.setdefault(q["doc_name"], []).append(q)
    return list(docs.items())


def find_pdf(name, folders):
    """The PDF for a document name; a question file may spell a filing's date differently from the PDF's file name
    ('..._dated-2022-07-01' vs '..._2022-07-01'), so compare names with the word 'dated' and the separators removed."""
    def norm(s):
        return s.lower().replace("dated", "").replace("-", "").replace("_", "")
    for folder in folders:
        exact = os.path.join(folder, name + ".pdf")
        if os.path.exists(exact):
            return exact
    for folder in folders:
        for f in os.listdir(folder):
            if f.lower().endswith(".pdf") and norm(f[:-4]) == norm(name):
                return os.path.join(folder, f)
    return os.path.join(folders[0], name + ".pdf")


def save(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def resume_state(results):
    """Records to keep and the ids that count as done on a rerun. An error is retried, except the agent running out of turns
    ("max_turns"): that is the method's own outcome under our settings, it would cost about 2.5e5 tokens per retry, and it is
    graded as wrong (decision of 2026-11-18), so such a record stays and is not retried."""
    keep = [r for r in results if not r.get("error") or "max_turns" in str(r["error"])]
    return keep, {r["id"] for r in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="train", choices=["train", "out", "all"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stop-at", type=int, default=8_400_000, help="stop when today's ledger total (all callers) reaches this")
    ap.add_argument("--data-dir", default=os.environ.get("TABLERAG_DATA", r"C:\Users\y9205\financial_agentic_rag_runs\tablerag_google\data"))
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--no-index", action="store_true", help="only answer questions on filings that are already indexed")
    args = ap.parse_args()
    k, n = (int(v) for v in args.shard.split("/"))
    load_env()
    os.makedirs(args.out, exist_ok=True)
    from app.agent import usage_tracker
    meter = Meter(10 ** 12)
    meter.install()
    from pageindex import PageIndexLocalClient
    client = PageIndexLocalClient(index_model=MODEL, chat_model=MODEL, storage_path=os.path.join(args.out, f"store_{k}"))
    pdfs = os.path.join(BACKEND, "tests", "financebench_pdfs")
    if args.set != "train":
        pdfs_out = os.path.join(BACKEND, "tests", "financebench_outsample_pdfs")
    res_path, idx_path = os.path.join(args.out, f"results_{k}.json"), os.path.join(args.out, f"index_{k}.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else []
    index = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {}
    results, done = resume_state(results)
    docs = group_by_doc(load_questions(args.data_dir, args.set))[k::n]
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    print(f"shard {k}/{n}: {len(docs)} filings, {len(done)} questions already done", flush=True)

    def over():
        t = usage_tracker.today_total()
        if t >= args.stop_at:
            print(f"STOP: today's total {t:,} >= {args.stop_at:,}; rerun after the next 00:00 UTC", flush=True)
            return True
        return False

    for name, qs in docs:
        todo = [q for q in qs if q["id"] not in done]
        if not todo:
            continue
        if args.no_index and "doc_id" not in index.get(name, {}):
            continue
        if name not in index or "doc_id" not in index[name]:
            if over():
                return
            pdf = find_pdf(name, [pdfs] if args.set == "train" else [pdfs, pdfs_out])
            c0, t0, k0 = meter.calls, time.time(), meter.tokens
            try:
                index[name] = {"doc_id": client.submit_document(pdf)["doc_id"], "seconds": round(time.time() - t0, 1),
                               "calls": meter.calls - c0, "tokens": meter.tokens - k0}
            except Exception as e:
                print("INDEX FAILED", name, repr(e), flush=True)
                index[name] = {"error": repr(e)}
                save(idx_path, index)
                continue
            save(idx_path, index)
            print("indexed", name, index[name], flush=True)
        if "doc_id" not in index[name]:
            continue
        for q in todo:
            if over():
                return
            rec = {"id": q["id"], "set": q["set"], "n": q["n"], "doc_name": q["doc_name"], "question_type": q["question_type"],
                   "question": q["question"], "gold": q["label"], "answer": "", "error": None}
            c0, t0, k0 = meter.calls, time.time(), meter.tokens
            try:
                a = client.chat(q["question"], doc_id=index[name]["doc_id"])
                rec["answer"] = a if isinstance(a, str) else str(a)
            except Exception as e:
                rec["error"] = repr(e)
            rec.update(elapsed=round(time.time() - t0, 1), calls=meter.calls - c0, tokens=meter.tokens - k0)
            results.append(rec)
            save(res_path, results)
            print(q["id"], rec["elapsed"], "s", rec["calls"], "calls", rec["tokens"], "tokens", "ERROR" if rec["error"] else "ok", flush=True)
    print(f"shard {k}/{n}: finished all its filings", flush=True)


if __name__ == "__main__":
    main()
