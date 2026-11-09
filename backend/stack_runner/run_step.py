"""Stacking experiment runner: one step per call (EXPERIMENT_PLAN.md section 3, revision 2026-11-08).

Step 0 is the Naive RAG control (backend/clean_baseline/run.py). This file runs the steps that add the
system's components on top of it. Only steps 1 to 4 exist so far; asking for another step stops with an error
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


def _system_prompt_request(ctx, question: str, search_text: str, cls, top_k: int, parent_map=None):
    """(messages, retrieved indices) shared by steps 2 and 3: rank_bm25 on `search_text` (step 1's retrieval, 16
    entries), the prompt built by the system's own LLMAnswerGenerator.generate_answer with the classifier's
    answer mode and the orchestrator's routing-reason text, its one model call captured and sent unchanged
    (model + the two messages). No PoT or verification result is passed, so the "Computed value" block (C6) and
    the "Verification summary" line (C7) are absent, exactly as in the system when those are off. The fiscal-year
    labels come from the whole filing's entries (as in the orchestrator), not from the 16."""
    filing = ctx["filing"]
    top = naive.retrieve(ctx["bm25"], search_text, top_k)
    items = []
    for rank, j in enumerate(top):
        item = dict(filing[j], relevance_score=len(top) - rank)                       # keeps the BM25 order
        if not item.get("parent_content") and item.get("parent_id") and parent_map:
            item["parent_content"] = parent_map.get(item["parent_id"], "")
        items.append(item)
    return _answer_messages(ctx, question, items, cls), top


def _answer_messages(ctx, question: str, items, cls):
    """The system's own answer prompt (LLMAnswerGenerator.generate_answer) around `items`, its one model call
    captured: returns the messages (model + the two messages)."""
    from app.agent import llm_client as client_module
    client_module.record_usage = lambda *args, **kwargs: None         # the runner counts tokens itself (plan section 1)
    reason = f"FinanceBench: {cls.get('question_type', '')} / {cls.get('cognitive_task', '')}"      # as the orchestrator
    filing = ctx["filing"]
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
    return capture.kwargs["messages"]


def step2_request(ctx, question: str, top_k: int, parent_map=None):
    """(messages, retrieved indices) for step 2 = step 1 plus the answer prompt (C2) and the classifier with its
    vocabulary (C3): retrieval is rank_bm25 on `search_text_for` (the question, plus the classifier's queries for a
    non-numeric question); the prompt is the system's (see _system_prompt_request)."""
    from app.agent.question_classifier import FinanceBenchClassifier      # imported only when step 2 runs
    if "classifier" not in ctx:
        ctx["classifier"] = FinanceBenchClassifier()
    cls = ctx["classifier"].classify(question)
    ctx["meta"] = {"answer_mode": cls["answer_mode"], "n_queries": len(cls["retrieval_queries"])}
    return _system_prompt_request(ctx, question, search_text_for(question, cls), cls, top_k, parent_map)


# ---- step 3: the query decomposer and the formula library (C4) --------------------------------------------
_decomposer_calls = []          # (prompt tokens, completion tokens) of each decomposer model call of the current question


def _count_decomposer_usage(model, caller, usage=None, prompt_chars=0):
    from app.agent.usage_tracker import _usage_numbers
    n = _usage_numbers(usage)
    _decomposer_calls.append((n["prompt"], n["completion"]))


def planned_queries(question: str, answer_mode: str, plan, no_calculation: bool):
    """Every query the system's planner wants searched, each distinct one once, in planning order.
    NUMERIC: the retrieval sub-questions (formula variables, the decomposer's steps, the question itself when
    no formula covers it, the topic query) plus the question when it has no single computed answer.
    Any other mode: the planner's search queries."""
    if answer_mode == "NUMERIC":
        queries = [sq["query"] for sq in plan.sub_questions if sq["type"] == "retrieval"]
        if no_calculation and question not in queries:
            queries.append(question)
    else:
        queries = list(plan.search_queries)
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _orchestrator_for(ctx):
    """One orchestrator per filing, over a stand-in store that holds only that filing (the single-file condition).
    Only its planning and entity-alignment methods are used in step 3; nothing is retrieved through it."""
    if "orch" not in ctx:
        from app.agent import decomposer as decomposer_module
        decomposer_module.record_usage = _count_decomposer_usage
        from app.agent.orchestrator import FinAgentRAGOrchestrator
        store = SimpleNamespace(corpus=ctx["filing"], uploaded_files=ctx.get("uploaded", []))
        ctx["orch"] = FinAgentRAGOrchestrator(corpus_store=store)
    return ctx["orch"]


def plan_for(ctx, question: str):
    """(classification, plan, queries, formula found?) exactly as the orchestrator plans before it retrieves:
    classify, align the company to the filing, look the question up in the formula library, add the implied
    previous year for trend questions, then `_plan_queries`."""
    from app.agent.financial_formula_library import detect_formula
    from app.agent.pot_reasoner import _with_implied_trend_year, is_no_calculation_question
    orch = _orchestrator_for(ctx)
    del _decomposer_calls[:]
    cls = orch.classifier.classify(question)
    clean_entity = cls["entity"]
    cls["entity"] = orch._match_entity_to_corpus(cls["entity"], question)
    clean_entity = orch._readable_company_name(cls["entity"], question) or clean_entity
    library_formula = detect_formula(question)
    retrieval_years = _with_implied_trend_year(cls["years"], question.lower())
    query_entity = clean_entity if clean_entity and clean_entity != "company" else cls["entity"]
    plan = orch._plan_queries(question, cls, cls["answer_mode"], library_formula, query_entity, retrieval_years)
    queries = planned_queries(question, cls["answer_mode"], plan, is_no_calculation_question(question))
    return cls, plan, queries, bool(library_formula)


def step3_request(ctx, question: str, top_k: int, parent_map=None):
    """(messages, retrieved indices) for step 3 = step 2 plus the decomposer and the formula library (C4): the
    system's own planner decides what to search (formula variables, the model's decomposition, the question, the
    topic query, the implied previous year); all distinct queries are joined into ONE rank_bm25 query string (still
    step 1's retrieval, not the system retriever). The prompt is step 2's. The planner's model calls are counted in
    ctx["meta"]."""
    cls, plan, queries, formula = plan_for(ctx, question)
    source = sorted({sq.get("source") for sq in plan.sub_questions if sq.get("source")})      # e.g. ['llm', 'narrative_topic']
    ctx["meta"] = {"answer_mode": cls["answer_mode"], "n_queries": len(queries), "formula": formula,
                   "decompose_source": source, "decomposer_calls": len(_decomposer_calls),
                   "decomposer_tokens": [sum(c[0] for c in _decomposer_calls), sum(c[1] for c in _decomposer_calls)]}
    return _system_prompt_request(ctx, question, " ".join(queries), cls, top_k, parent_map)


# ---- step 4: the system retriever (C5) -------------------------------------------------------------------------
def _system_orchestrator_for(ctx, parent_map):
    """One orchestrator per filing over a real corpus store that holds only that filing (the single-file
    condition, built as tests/run_single_file_subset.py builds it), with PoT, Tri-Check and the answer call
    switched off: the orchestrator's own planning, search loop, de-duplication, quota selection and window of 30
    run unchanged; what it would hand to the answer call is captured instead (see step4_request)."""
    if "orch4" not in ctx:
        from app.agent import decomposer as decomposer_module
        decomposer_module.record_usage = _count_decomposer_usage
        from app.agent.orchestrator import FinAgentRAGOrchestrator
        from app.rag.corpus_store import FinancialCorpusStore
        from app.tools.lexical_retriever import LexicalFinancialRetriever
        store = FinancialCorpusStore()
        store.corpus = list(ctx["filing"])
        store.parent_map = parent_map or {}
        store.uploaded_files = list(ctx.get("uploaded", []))
        store.retriever = LexicalFinancialRetriever(store.corpus)
        orch = FinAgentRAGOrchestrator(corpus_store=store)
        state = ctx["orch4_state"] = {}
        # C6 and C7 off: no computed value, and a verdict that accepts at once (so there is no retry, C7)
        orch.pot_reasoner.generate_and_execute = lambda *a, **k: {
            "code": "", "success": True, "result_value": None, "output_log": "", "extracted_variables": {},
            "used_evidence_ids": []}
        orch.verifier.verify = lambda *a, **k: {"decision": "ACCEPT"}

        def capture(query, evidence, pot_res, verifier_res, classification, sub_questions):
            state.update(final_context=evidence, classification=classification, sub_questions=sub_questions)
            return "-"
        orch._synthesize_final_answer = capture
        plan_queries = orch._plan_queries

        def keep_plan(*a, **k):
            state["plan"] = plan_queries(*a, **k)
            return state["plan"]
        orch._plan_queries = keep_plan
        ctx["orch4"] = orch
    return ctx["orch4"]


def step4_request(ctx, question: str, top_k: int, parent_map=None):
    """(messages, retrieved indices) for step 4 = step 3 plus the system retriever (C5): the orchestrator runs its
    own first attempt -- the planner's queries, one search each with the 8 score weightings, the exclusion of
    already-fetched passages, the (metric, year) skip, a per-query statement-type hint, de-duplication, the
    per-query quota and the window of 30 -- and the system's answer call then keeps its 16 of those 30. The prompt
    is step 3's (no Computed value, no Verification summary). Only the first attempt: PoT is off, so the
    verdict is accepted at once and there is no retry."""
    from app.agent.evidence_selection import select_with_quota
    from app.agent.llm_client import EVIDENCE_PROMPT_CAP
    orch = _system_orchestrator_for(ctx, parent_map)
    state = ctx["orch4_state"]
    state.clear()
    del _decomposer_calls[:]
    orch.process_query(question)
    cls, window = state["classification"], state["final_context"]
    index = {e["id"]: j for j, e in enumerate(ctx["filing"])}
    items = []
    for ev in window:
        item = dict(ev)
        if not item.get("parent_content") and item.get("parent_id") and parent_map:
            item["parent_content"] = parent_map.get(item["parent_id"], "")
        items.append(item)
    kept = select_with_quota(items, EVIDENCE_PROMPT_CAP)             # the 16 the answer call keeps
    plan = state["plan"]
    source = sorted({sq.get("source") for sq in plan.sub_questions if sq.get("source")})
    ctx["meta"] = {"answer_mode": cls["answer_mode"], "window": len(window), "decompose_source": source,
                   "decomposer_calls": len(_decomposer_calls),
                   "decomposer_tokens": [sum(c[0] for c in _decomposer_calls), sum(c[1] for c in _decomposer_calls)]}
    return _answer_messages(ctx, question, items, cls), [index[e["id"]] for e in kept]


STEPS = {1: step1_request, 2: step2_request, 3: step3_request, 4: step4_request}


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
                                "uploaded": [u for u in cached.get("uploaded_files", []) if u.get("company") == company],
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
