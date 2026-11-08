"""Stacking experiment runner: one step per call (EXPERIMENT_PLAN.md section 3, revision 2026-11-08).

Step 0 is the Naive RAG control (backend/clean_baseline/run.py). This file runs the steps that add the
system's components on top of it. Only steps 1 and 2 exist so far; asking for another step stops with an error
instead of running something that is not that step.

Step 1 adds exactly one component, the system's corpus (C1), and nothing else:
  - the filing's passages come from the system's parsed corpus (a page of text, or one table row, per entry,
    with the structured fields) instead of 1,000-character windows of raw PDF text;
  - retrieval is step 0's, unchanged: rank_bm25 (clean_baseline.run.tokenize / retrieve), the question as
    the only query, the 16 best entries. What is indexed is each entry's `content` and nothing else (the
    system's BM25 indexes `content` only; period and section are signals of the system retriever, C5);
  - what the model reads is the system's own evidence display (llm_client): each retrieved entry as
    "- [company / table name] <its whole parent page>", the page cut the way the system cuts it (4,000
    characters, 30 table rows); a page hit by several entries is shown once per entry, as in the system.
    There is NO extra tag line: the full system has none;
  - the prompt is step 0's: "Context:" + the evidence + "Question:" + the question, no system message;
  - one model call, no sampling or reasoning settings.
Not in this step: the answer prompt text and the Answer Mode lines (C2), the question classifier and vocabulary
(C3), the decomposer and formula library (C4), the system retriever (C5), PoT (C6), Tri-Check (C7).

Step 2 adds exactly two components on top of step 1, the answer prompt (C2) and the question classifier with its
vocabulary (C3), and nothing else (see step2_request). Retrieval stays rank_bm25 on `content`.

    python backend/stack_runner/run_step.py --step 1 --set original --dry-run          # no API call
    python backend/stack_runner/run_step.py --step 1 --set original --shard 1/2 --out step1_original_1.json
"""
import argparse
import collections
import importlib.util
import json
import os
import pickle
import re
import sys
import time
from types import SimpleNamespace

from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BACKEND)
TESTS = os.path.join(BACKEND, "tests")

from app.agent import llm_client  # noqa: E402

SETS = {
    "original": ("financebench_qa_subset.json", "original_corpus_cache.pkl"),
    "outsample": ("financebench_outsample_qa.json", "outsample_corpus_cache.pkl"),
}
DISPLAY_MAX_CHARS, DISPLAY_MAX_TABLE_ROWS = 4000, 30    # the arguments llm_client passes


def _load_naive():
    spec = importlib.util.spec_from_file_location("naive_run", os.path.join(BACKEND, "clean_baseline", "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


naive = _load_naive()


def _norm(name: str) -> str:
    # "dated" is only a filename decoration some doc_names carry
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("dated", ""))


def find_filing(companies, doc_name: str) -> str:
    """The cached filing whose name matches a question's doc_name (they can differ in punctuation or an
    appended date). Exactly one filing must match."""
    target = _norm(doc_name)
    exact = [c for c in companies if _norm(c) == target]
    if len(exact) == 1:
        return exact[0]
    loose = [c for c in companies if target.startswith(_norm(c)) or _norm(c).startswith(target)]
    if not exact and len(loose) == 1:
        return loose[0]
    raise SystemExit(f"doc_name {doc_name!r} matches {len(exact) or len(loose)} cached filings")


def filing_entries(entries, company: str):
    """The entries of one filing, in corpus order (single-file mode: a question sees only its own filing)."""
    return [e for e in entries if e.get("company") == company]


def index_text(entry) -> str:
    """What BM25 indexes for an entry: its content, as in the system."""
    return entry["content"]


def evidence_line(entry, parent_map=None) -> str:
    """One retrieved entry as the system's answer prompt shows it (llm_client.generate_answer):
    "- [company / table name] <whole parent page, cut the way the system cuts it>"."""
    item = entry
    if not entry.get("parent_content") and entry.get("parent_id") and parent_map:
        item = dict(entry, parent_content=parent_map.get(entry["parent_id"], ""))
    shown = llm_client._truncate_evidence_content(
        llm_client._evidence_display_text(item), max_chars=DISPLAY_MAX_CHARS, max_table_rows=DISPLAY_MAX_TABLE_ROWS)
    return f"- [{item.get('company', 'Company')} / {item.get('table_name', 'Source')}] {shown}"


def build_prompt(lines, question: str) -> str:
    """Step 0's prompt (clean_baseline.run.build_prompt) around the evidence lines: "Context:", the evidence,
    a blank line, "Question:", the question."""
    return "Context:\n" + "\n".join(lines) + "\n\nQuestion: " + question


def step1_prompt(filing, bm25, question: str, top_k: int, parent_map=None):
    """(prompt, indices of the retrieved entries) for one question in step 1."""
    top = naive.retrieve(bm25, question, top_k)
    return build_prompt([evidence_line(filing[j], parent_map) for j in top], question), top


def step1_request(ctx, question: str, top_k: int, parent_map=None):
    """(messages, retrieved indices) for step 1: one user message, no system message."""
    prompt, top = step1_prompt(ctx["filing"], ctx["bm25"], question, top_k, parent_map)
    return [{"role": "user", "content": prompt}], top


# ---- step 2: the system's answer prompt (C2) and the question classifier with its vocabulary (C3) -----------
class _Capture:
    """Stands in for the model client: generate_answer calls client.chat.completions.create(...) once; the
    arguments are recorded so the runner can send exactly that call itself (with retries and token counts)."""

    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="-"))], usage=None)


def filing_labels_for(filing):
    """The fiscal-year labels of the whole filing, computed the way the orchestrator does it
    (Orchestrator._filing_year_labels: every passage of the filing, joined)."""
    from app.agent.answer_directives import filing_periods
    return filing_periods("\n".join(passage.get("content") or "" for passage in filing))


def search_text_for(question: str, classification) -> str:
    """Step 2's one BM25 query string. A NUMERIC question searches its own text. Any other question searches its
    own text plus the classifier's retrieval queries (each distinct query once, in the classifier's order),
    joined by single spaces -- the classifier already ends its list with the question, which is not repeated."""
    if classification["answer_mode"] == "NUMERIC":
        return question
    queries = [question] + [q for q in classification["retrieval_queries"] if q != question]
    return " ".join(queries)


def step2_request(ctx, question: str, top_k: int, parent_map=None):
    """(messages, retrieved indices) for step 2 = step 1 plus the answer prompt (C2) and the classifier with its
    vocabulary (C3):
      - retrieval is rank_bm25 as in step 1 (here on `search_text_for`: the question, plus the classifier's queries
        for a non-numeric question);
      - the prompt is built by the system's own LLMAnswerGenerator.generate_answer -- system message, question
        first, the "Answer Mode" / "Routing Reason" lines (orchestrator's reason text), evidence, the question-shape
        directives, principles, accounting conventions, output format, checklist; its one model call is captured and
        sent unchanged (model + the two messages). No PoT or verification result is passed, so the "Computed value"
        block (C6) and the "Verification summary" line (C7) are absent, exactly as in the system when those are off;
      - the fiscal-year labels come from the whole filing's entries (as in the orchestrator), not from the 16."""
    from app.agent.question_classifier import FinanceBenchClassifier      # imported only when step 2 runs
    from app.agent import llm_client as client_module
    client_module.record_usage = lambda *args, **kwargs: None         # the runner counts tokens itself (plan section 1)
    if "classifier" not in ctx:
        ctx["classifier"] = FinanceBenchClassifier()
    cls = ctx["classifier"].classify(question)
    reason = f"FinanceBench: {cls.get('question_type', '')} / {cls.get('cognitive_task', '')}"      # as the orchestrator
    ctx["meta"] = {"answer_mode": cls["answer_mode"], "n_queries": len(cls["retrieval_queries"])}
    filing = ctx["filing"]
    top = naive.retrieve(ctx["bm25"], search_text_for(question, cls), top_k)
    items = []
    for rank, j in enumerate(top):
        item = dict(filing[j], relevance_score=len(top) - rank)                       # keeps the BM25 order
        if not item.get("parent_content") and item.get("parent_id") and parent_map:
            item["parent_content"] = parent_map.get(item["parent_id"], "")
        items.append(item)
    if "labels" not in ctx:
        ctx["labels"] = filing_labels_for(filing)
    capture = _Capture()
    generator = llm_client.LLMAnswerGenerator()
    generator._client, generator._model = capture, ctx["model"]
    generator.generate_answer(
        query=question, answer_mode=cls["answer_mode"], evidence=items, route_res={"reason": reason},
        pot_res=None, verification_res=None, filing_labels=ctx["labels"])
    assert capture.kwargs is not None, "generate_answer did not make its model call"
    assert set(capture.kwargs) == {"model", "messages"}, sorted(capture.kwargs)
    return capture.kwargs["messages"], top


STEPS = {1: step1_request, 2: step2_request}


def send(client, model: str, messages):
    """One call with exactly these messages and nothing else (no temperature, no reasoning settings), retried
    like clean_baseline.run.answer. Returns (text, (prompt_tokens, completion_tokens))."""
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=messages)
            u = r.usage
            return (r.choices[0].message.content or ""), ((u.prompt_tokens, u.completion_tokens) if u else None)
        except Exception as e:  # rate limit or transient error: wait and retry
            last = e
            time.sleep(10 * (attempt + 1))
    raise last


def load_done(path):
    """The finished rows of an --out file, by question number. A row with an error or an empty answer is not
    finished: it is left out here, so a restart runs that question again (EXPERIMENT_PLAN.md section 1)."""
    if not path or not os.path.exists(path):
        return {}
    return {r["n"]: r for r in json.load(open(path, encoding="utf-8"))
            if not r.get("error") and (r.get("model_answer") or "").strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--shard", default="1/1", help="i/n: questions whose (number - 1) %% n equals i - 1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--top-k", type=int, default=naive.TOP_K)
    ap.add_argument("--dry-run", action="store_true", help="retrieve and print the first prompt; no API call")
    a = ap.parse_args()
    if a.step not in STEPS:
        raise SystemExit(f"step {a.step} is not implemented in this runner (implemented: {sorted(STEPS)})")
    qa_file, cache_file = SETS[a.set]
    questions = json.load(open(os.path.join(TESTS, qa_file), encoding="utf-8"))
    cached = pickle.load(open(os.path.join(TESTS, cache_file), "rb"))
    entries, parent_map = cached["corpus"], cached.get("parent_map")
    companies = sorted({e["company"] for e in entries if e.get("company")})
    i, n = (int(x) for x in a.shard.split("/"))
    todo = [(k, q) for k, q in enumerate(questions, 1) if (k - 1) % n == i - 1]

    client = None
    if not a.dry_run:
        from openai import OpenAI
        client = OpenAI()
    done = load_done(a.out)
    indexes = {}
    for k, q in todo:
        if k in done:
            continue
        company = find_filing(companies, q["doc_name"])
        if company not in indexes:
            filing = filing_entries(entries, company)
            indexes[company] = {"filing": filing, "model": a.model,
                                "bm25": BM25Okapi([naive.tokenize(index_text(e)) for e in filing])}
        ctx = indexes[company]
        filing = ctx["filing"]
        messages, top = STEPS[a.step](ctx, q["question"], a.top_k, parent_map)
        meta = ctx.pop("meta", None)
        prompt = messages[-1]["content"]
        if a.dry_run:
            for m in messages[:-1]:
                print(f"[{m['role']} message]\n{m['content']}\n")
            print(prompt)
            print("\nentries retrieved:", [filing[j]["id"] for j in top])
            print("entry types:", dict(collections.Counter(filing[j].get("type") for j in top)))
            print("messages:", [m["role"] for m in messages], "| prompt characters:", len(prompt))
            return
        t0 = time.time()
        row = {"set": a.set, "n": k, "doc_name": q["doc_name"], "question_type": q["question_type"],
               "question": q["question"], "gold": q["answer"], "model_answer": "", "error": None,
               "retrieved": [filing[j]["id"] for j in top], "prompt_chars": len(prompt),
               "messages": len(messages), "tokens": None}
        if meta:
            row["meta"] = meta
        try:
            row["model_answer"], row["tokens"] = send(client, a.model, messages)
        except Exception as e:
            row["error"] = str(e)[:300]
        row["elapsed"] = round(time.time() - t0, 1)
        done[k] = row
        print(f"[{a.set} {k}] {row['elapsed']}s tokens={row['tokens']} err={row['error']}", flush=True)
        if a.out:
            json.dump([done[x] for x in sorted(done)], open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
