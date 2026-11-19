"""Feasibility check of PageIndex (VectifyAI/PageIndex, pip package `pageindex`) as a baseline.

Not the baseline run itself: indexes a few filings with PageIndexLocalClient, asks a few questions with the SDK's own
chat, and records time, tokens (through the usage ledger, caller "pageindex") and what the SDK returns.

    python feasibility.py --out <dir> --docs 3M_2018_10K,3M_2022_10K --ids train-1,train-3,train-4 [--budget 1500000]

Needs the pageindex venv (outside the repo) and OPENAI_API_KEY (backend/.env is read). Model: OPENAI_MODEL or gpt-5-mini
(LiteLLM names: a bare name is an OpenAI model).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, BACKEND)
MODEL = os.environ.get("PAGEINDEX_MODEL", "gpt-5-mini")


def load_env():
    for line in open(os.path.join(BACKEND, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "OPENAI_API_KEY":
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Meter:
    """Counts every model call LiteLLM makes and writes it to the usage ledger."""

    def __init__(self, budget):
        self.calls, self.tokens, self.budget = 0, 0, budget

    def install(self):
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from app.agent import usage_tracker
        meter = self

        class Logger(CustomLogger):
            def _rec(self, kwargs, response_obj):
                usage = getattr(response_obj, "usage", None)
                meter.calls += 1
                meter.tokens += getattr(usage, "total_tokens", 0) or 0
                usage_tracker.record_usage(str(kwargs.get("model")), "pageindex", usage, 0)
                if meter.tokens > meter.budget:
                    print(f"BUDGET EXCEEDED: {meter.tokens} tokens", flush=True)
                    os._exit(3)

            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                self._rec(kwargs, response_obj)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self._rec(kwargs, response_obj)

        litellm.callbacks.append(Logger())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--budget", type=int, default=1_500_000)
    ap.add_argument("--data-dir", default=os.environ.get("TABLERAG_DATA", r"C:\Users\y9205\financial_agentic_rag_runs\tablerag_google\data"))
    args = ap.parse_args()
    load_env()
    os.makedirs(args.out, exist_ok=True)
    meter = Meter(args.budget)
    meter.install()
    from pageindex import PageIndexLocalClient
    client = PageIndexLocalClient(index_model=MODEL, chat_model=MODEL, storage_path=os.path.join(args.out, "store"))
    pdfs = os.path.join(BACKEND, "tests", "financebench_pdfs")
    doc_ids, log = {}, {"model": MODEL, "index": [], "questions": []}
    for name in args.docs.split(","):
        c0, t0 = meter.calls, time.time()
        r = client.submit_document(os.path.join(pdfs, name + ".pdf"))
        doc_ids[name] = r["doc_id"]
        rec = {"doc": name, "doc_id": r["doc_id"], "seconds": round(time.time() - t0, 1), "calls": meter.calls - c0, "tokens_total_so_far": meter.tokens}
        log["index"].append(rec)
        print("indexed", rec, flush=True)
        json.dump(log, open(os.path.join(args.out, "feasibility.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    qs = {}
    for fn in ("questions_train.jsonl", "questions_out.jsonl"):
        for line in open(os.path.join(args.data_dir, fn), encoding="utf-8"):
            q = json.loads(line)
            qs[q["id"]] = q
    for qid in args.ids.split(","):
        q = qs[qid]
        c0, t0, k0 = meter.calls, time.time(), meter.tokens
        try:
            ans = client.chat(q["question"], doc_id=doc_ids[q["doc_name"]])
            err = None
        except Exception as e:  # the failure is part of the record
            ans, err = "", repr(e)
        rec = {"id": qid, "question": q["question"], "gold": q["label"], "answer": ans if isinstance(ans, str) else str(ans), "error": err,
               "seconds": round(time.time() - t0, 1), "calls": meter.calls - c0, "tokens": meter.tokens - k0}
        log["questions"].append(rec)
        print(qid, rec["seconds"], "s", rec["calls"], "calls", rec["tokens"], "tokens", flush=True)
        json.dump(log, open(os.path.join(args.out, "feasibility.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("total", meter.calls, "calls", meter.tokens, "tokens")


if __name__ == "__main__":
    main()
