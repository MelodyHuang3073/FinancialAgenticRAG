"""Offline check of the ablation switches: no API call, no cost.

For a fixed sample of questions it runs the real orchestrator (classifier, retrieval, PoT, Tri-Check, retry) with
the two model calls replaced -- the decomposer by a fixed rewrite, the answer call by a recorder -- and writes,
per question, the planned sub-queries, the evidence ids that reached the prompt and the exact messages the
answer call would have sent. Run it once per ABLATE_ROW and compare the files:

    ABLATE_ROW=0 python tests/ablation_offline_check.py --out row0.json     (the full system)
    ABLATE_ROW=3 python tests/ablation_offline_check.py --out row3.json
    python tests/ablation_offline_check.py --compare row0.json row3.json     (what differs, per question)

The same file runs on a tree that has no app/ablation.py (dev before the switches), so `row0.json` of the
switched tree can be compared byte for byte with the output of the unswitched tree.
"""
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

try:
    from app import ablation
    ROW = ablation.ROW
    PLAIN = ablation.off("C1")
except ImportError:          # a tree without the switches: the full system
    ablation, ROW, PLAIN = None, 0, False

from app.rag.corpus_store import FinancialCorpusStore                 # noqa: E402
from app.tools.lexical_retriever import LexicalFinancialRetriever    # noqa: E402
from app.agent import orchestrator as orch_mod                        # noqa: E402
from app.agent.decomposer import QueryDecomposer                      # noqa: E402
from app.agent.llm_client import LLMAnswerGenerator                   # noqa: E402

SETS = {
    "train": ("financebench_qa_subset.json",
              "original_clean_corpus_cache.pkl" if PLAIN else "original_corpus_cache.pkl"),
    "out": ("financebench_outsample_qa.json",
            "outsample_clean_corpus_cache.pkl" if PLAIN else "outsample_corpus_cache.pkl"),
}


def _norm(name):
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("dated", ""))


class _Recorder:
    """Stands in for the model client; keeps the arguments of the single answer call."""

    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="-"))], usage=None)


def _fake_decompose(self, query, target_metrics=None, years=None, entity=None):
    return [{"step": 1, "type": "retrieval", "query": query + " decomposed", "target_metric": "",
             "target_year": "", "source": "llm"}], None


def _fake_topic(self, query, entity=None):
    return "topic terms"


def _store(cached, company):
    store = FinancialCorpusStore()
    if PLAIN:
        store.corpus = ablation.plain_entries(company, cached[company])
        store.uploaded_files = [{"filename": company + ".pdf", "company": company,
                                 "passage_count": len(store.corpus)}]
    else:
        store.corpus = [e for e in cached["corpus"] if e.get("company") == company]
        store.parent_map = cached["parent_map"]
        store.uploaded_files = [u for u in cached["uploaded_files"] if u["company"] == company]
    store.retriever = LexicalFinancialRetriever(store.corpus)
    return store


def run(every, out):
    QueryDecomposer.decompose = _fake_decompose
    QueryDecomposer.suggest_narrative_topic_query = _fake_topic
    rows = []
    for tag, (qa_file, cache_file) in SETS.items():
        qa = json.load(open(os.path.join(HERE, qa_file), encoding="utf-8"))
        cached = pickle.load(open(os.path.join(HERE, cache_file), "rb"))
        companies = sorted(cached) if PLAIN else sorted({u["company"] for u in cached["uploaded_files"]})
        for i, item in enumerate(qa, 1):
            if (i - 1) % every:
                continue
            company = next(c for c in companies if _norm(c) == _norm(item["doc_name"]))
            orch = orch_mod.FinAgentRAGOrchestrator(corpus_store=_store(cached, company))
            recorder = _Recorder()
            orch.llm_generator._client, orch.llm_generator._model = recorder, "recorded"
            res = orch.process_query(item["question"])
            steps = [s for s in res["execution_trace"] if s.get("type") in ("decomposition", "step_retrieval")]
            rows.append({
                "id": f"{tag}-{i}", "mode": res["answer_mode"], "iterations": res["total_iterations"],
                "plan": [s.get("sub_questions") or s.get("detail") for s in steps],
                "evidence": [e["id"] for e in res["evidence_sources"]],
                "messages": (recorder.kwargs or {}).get("messages"),
            })
    json.dump({"ablate_row": ROW, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"row {ROW}: {len(rows)} questions -> {out}")


def compare(a_path, b_path):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (a_path, b_path))
    diff = {"identical": 0, "plan": 0, "evidence": 0, "prompt": 0, "mode": 0, "iterations": 0}
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra["id"] == rb["id"]
        kinds = [k for k in ("mode", "iterations", "plan", "evidence") if ra[k] != rb[k]]
        if ra["messages"] != rb["messages"]:
            kinds.append("prompt")
        for k in kinds:
            diff[k] += 1
        diff["identical"] += not kinds
    print(f"row {a['ablate_row']} vs row {b['ablate_row']} over {len(a['rows'])} questions:", diff)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=5, help="every N-th question of each set")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    compare(*args.compare) if args.compare else run(args.every, args.out)
