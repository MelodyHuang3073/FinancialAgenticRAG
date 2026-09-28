"""FinAgent-RAG Orchestrator

Main flow:
  1. question classification
  2. multi-step decomposition
  3. hybrid retrieval (vector + sparse)
  4. chain-of-thought style reasoning in a sandbox
  5. verification passes and iterative refinement
  6. final answer synthesis via LLM client
"""

import re
from typing import Dict, Any, List, Optional

from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.question_classifier import FinanceBenchClassifier
from app.agent.decomposer import QueryDecomposer
from app.agent.pot_reasoner import ProgramOfThoughtReasoner, _with_implied_trend_year, _get_canonical
from app.agent.verifier import TriCheckSelfVerifier
from app.agent.refiner import QueryRefiner
from app.agent.llm_client import (
    LLMAnswerGenerator, EVIDENCE_PROMPT_CAP, RELIABLE_POT_EVIDENCE_CAP, is_reliable_pot_result,
)
from app.agent.evidence_selection import select_with_quota
from app.agent.financial_formula_library import detect_formula, get_variable_aliases
from app.tools.hybrid_retriever import (
    is_attribution_query, is_geography_query, is_legal_query, is_segment_comparison_query,
)


#: llm_client.py's own "CRITICAL: you MUST quote this exact PoT result
#: number" instruction is worded strongly enough that the model sometimes
#: prints the bare result_value a SECOND time, as its own trailing
#: paragraph, in addition to already using it correctly in prose -- e.g.
#: "...raised guidance by **1 percentage point** (from 8% to 9%).\n\n1" or
#: "...paid a dividend of **$0.55 per share**.\n\nNumeric answer: **0.55**".
#: The number and prose sentence are both already correct; only this
#: redundant echo is wrong, so it is safe to strip mechanically -- but only
#: when it's the LAST paragraph, contains NOTHING but a number (optionally
#: bold-marked, optionally "Numeric answer:"-prefixed), and the answer has
#: other substantive content before it (never strips a genuinely one-line
#: numeric-only answer).
_TRAILING_BARE_NUMBER_RE = re.compile(
    r'^\s*(?:numeric\s+answer:?\s*)?\*{0,2}-?\$?[\d,]*\.?\d+\*{0,2}%?\s*$',
    re.IGNORECASE,
)


def _strip_trailing_bare_number(answer: str) -> str:
    paragraphs = re.split(r'\n\s*\n', answer.strip())
    if len(paragraphs) >= 2 and _TRAILING_BARE_NUMBER_RE.match(paragraphs[-1]):
        return '\n\n'.join(paragraphs[:-1]).strip()
    return answer


class FinAgentRAGOrchestrator:
    # Chunks per sub-question raised from 3 toward 5.
    # A short alias like "net income" can match multiple differently scoped rows on the
    # same statement, so a larger per-subquestion chunk count helps ensure the correctly
    # scoped row appears among candidates.
    RETRIEVAL_TOP_K = 5
    # Use a wider top_k only for narrative/non-numeric retrieval paths.
    # Qualitative questions often require passages ranked lower by topical signals, so
    # expand retrieval window for narrative queries to avoid missing the correct
    # passage.
    RETRIEVAL_TOP_K_NARRATIVE = 15
    # Cap applied just before evidence is passed to the model (keep top N by
    # relevance_score).
    # A too-tight final window can drop a lower-scoring but correct chunk that was
    # retrieved earlier, so ensure final limit preserves diversity across subqueries.
    CONTEXT_CHUNK_LIMIT = 30

    def __init__(self, vector_store: FinancialVectorStoreManager):
        self.vector_store = vector_store
        self.classifier = FinanceBenchClassifier()
        self.decomposer = QueryDecomposer()
        self.pot_reasoner = ProgramOfThoughtReasoner()
        self.verifier = TriCheckSelfVerifier()
        self.refiner = QueryRefiner()
        self.llm_generator = LLMAnswerGenerator()

    def _top_evidence(self, evidence_buffer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the top-scoring CONTEXT_CHUNK_LIMIT items after sorting by relevance,
        not a plain insertion-order slice. evidence_buffer accumulates hits in
        sub-query execution order, so a tail slice can drop earlier high-score items;
        always sort and then cap to avoid silently losing top evidence.
        """
        # Each sub-query keeps its own top passages (scores of different
        # queries are not comparable); the rest is filled by raw score --
        # see evidence_selection.select_with_quota.
        return select_with_quota(evidence_buffer, self.CONTEXT_CHUNK_LIMIT)

    def _build_evidence_info(self, hit: Dict[str, Any], sub_question: str = None) -> Dict[str, Any]:
        """
        Build the evidence-source dict returned to the frontend for one
        retrieved passage. Always includes parent_content (the full page
        text, e.g. the complete Markdown table for a 'table_row' hit) so
        the Source Evidence panel can render the whole table structure
        instead of just the single linearised row that matched the query.
        """
        parent_id = hit.get("parent_id", "")
        parent_content = hit.get("parent_content") or (
            self.vector_store.get_parent_content(parent_id) if parent_id else ""
        )
        info = {
            "id": hit.get("id", ""),
            "table_name": hit.get("table_name", ""),
            "company": hit.get("company", ""),
            "period": hit.get("period", ""),
            "section": hit.get("section", ""),
            "chunk_type": hit.get("type", ""),
            "relevance_score": hit.get("relevance_score", 0.0),
            "snippet": hit.get("content", "")[:120] + "...",
            "content": hit.get("content", ""),
            "parent_id": parent_id,
            "parent_content": parent_content,
        }
        if sub_question:
            info["sub_question"] = sub_question
        return info

    #: Opening words of an answer that says the chosen file lacks the fact.
    _REFUSAL_HEAD_RE = re.compile(
        r"(?:cannot|can't|can not|unable to|not able to)\s+(?:conclusively\s+)?"
        r"(?:determine|confirm|calculate|compute|identify|say|answer|tell|find)"
        r"|(?:do(?:es)?\s+not|don't|doesn't)\s+(?:include|disclose|contain|provide|state|show|say)"
        r"|(?:not|isn't|aren't)\s+(?:disclosed|included|provided|available|stated)"
        r"|insufficient (?:evidence|data|information)|no (?:such )?(?:data|information) (?:in|is)",
        re.IGNORECASE,
    )

    def _looks_like_refusal(self, answer: str) -> bool:
        return bool(answer) and bool(self._REFUSAL_HEAD_RE.search(answer[:220]))

    def _alternate_company_docs(self, resolved: str, query: str, limit: int = 2) -> List[str]:
        """Other files of the SAME company as `resolved`, best guess first
        (a year the question names, then plain annual filings, then more
        recent). Each is tried on its own -- one file per attempt."""
        import re as _re
        def _base(name: str) -> str:
            return _re.split(r"_(?:19|20)\d{2}", name or "", maxsplit=1)[0].lower()
        def _year(name: str) -> str:
            m = _re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name or "")
            return m.group(1) if m else ""
        q_years = set(_re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", query))
        base = _base(resolved)
        cands = []
        for uf in self.vector_store.uploaded_files:
            name = uf.get("company", "")
            if not name or name == resolved or _base(name) != base:
                continue
            has_q = bool(_re.search(r"(?<!\d)(?:19|20)\d{2}q[1-4]", name, _re.IGNORECASE)) or "earnings" in name.lower()
            cands.append((1 if _year(name) in q_years else 0, 0 if has_q else 1, _year(name), name))
        cands.sort(reverse=True)
        return [c[3] for c in cands[:limit]]

    def process_query(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Answer from the best-guess file; if that answer says the file lacks
        the fact, try the same company's other files one at a time (each
        attempt uses a single file) and keep the first non-refusing answer."""
        result = self._process_query_once(query, max_iterations)
        if not self._looks_like_refusal(result.get("final_answer", "")):
            return result
        for alt in self._alternate_company_docs(result.get("resolved_entity", ""), query):
            alt_result = self._process_query_once(query, max_iterations, _entity_override=alt)
            if not self._looks_like_refusal(alt_result.get("final_answer", "")):
                alt_result["retried_from"] = result.get("resolved_entity", "")
                return alt_result
        return result

    def _process_query_once(
        self, query: str, max_iterations: int = 3, _entity_override: Optional[str] = None
    ) -> Dict[str, Any]:
        trace_steps = []
        evidence_buffer: List[Dict[str, Any]] = []  # full evidence objects
        evidence_meta: List[Dict[str, Any]] = []    # per-item sub_question metadata
        retrieved_ids = set()

        # Step 1: classification of financial question type.
        classification = self.classifier.classify(query)
        answer_mode = classification["answer_mode"]
        complexity = classification["complexity"]
        retrieval_strategy = classification["retrieval_strategy"]
        # "Which segment had the highest/lowest X" stays NUMERIC (see
        # hybrid_retriever.is_segment_comparison_query's docstring) so this
        # is computed here, not alongside is_attribution/is_geography/
        # is_legal below (those are only ever used on the non-numeric
        # branch further down).
        is_segment_comparison = is_segment_comparison_query(query)

        # Entity alignment: match extracted entity against corpus company names.
        # Keep a cleaned human-readable entity for query text and use the raw filename-
        # stem form only for the corpus filter; fused filename tokens can lose retrieval
        # signal if used directly in the query.
        clean_entity = classification["entity"]
        classification["entity"] = _entity_override or self._match_entity_to_corpus(
            classification["entity"], query
        )
        if not _entity_override:
            clean_entity = self._readable_company_name(classification["entity"], query) or clean_entity

        # statement_type_hint: from question_classifier (income_statement / balance_sheet /
        # cash_flow / notes).  Used for 1.5x boost in hybrid_retriever.
        statement_type_hint = classification.get("statement_type_hint") or None
        if statement_type_hint == "unknown":
            statement_type_hint = None


        trace_steps.append({
            "step_name": "FinanceBench Classification",
            "type": "classification",
            "detail": (
                f"問題類型: {classification['question_type']} | "
                f"認知任務: {classification['cognitive_task']} | "
                f"檢索策略: {retrieval_strategy} | "
                f"目標指標: {classification['target_metrics']} | "
                f"識別公司: {classification['entity']} | "
                f"回答模式: {answer_mode} | 複雜度: {complexity}"
            )
        })

        # ── Step 2: Query Decomposition (now context-aware) ──
        # When the question matches a KNOWN formula (pot_reasoner will use
        # this exact same formula to compute the answer), retrieval steps
        # are generated directly from that formula's own required_vars
        # alias lists — bypassing LLM-based decomposition, which is
        # non-deterministic and has been confirmed to sometimes retrieve
        # evidence unrelated to what the formula actually needs (real
        # case: a 5-variable "days payable outstanding" formula got
        # LLM-decomposed into sub-queries about cash and marketable
        # securities on one run, and correctly about accounts
        # payable/COGS/inventory on another run of the SAME question —
        # sheer sampling variance). Deriving sub-queries from the
        # formula's own aliases guarantees retrieval searches for
        # exactly what extraction will later look for, deterministically.
        formula_entry = detect_formula(query) if answer_mode == "NUMERIC" else None
        # If a question asks whether a metric is improving/declining as of year Y, treat
        # it as implying a comparison to year Y-1.
        # Without this implied prior-year fetch, retrieval may lack the earlier-year
        # evidence needed for trend calculations, causing fallback answers.
        retrieval_years = _with_implied_trend_year(classification["years"], query.lower())
        query_entity = clean_entity if clean_entity and clean_entity != "company" else classification["entity"]
        if formula_entry:
            sub_questions = self._build_formula_subquestions(
                formula_entry, query_entity, retrieval_years
            )
        elif answer_mode == "NUMERIC":
            sub_questions = self.decomposer.decompose(
                query,
                target_metrics=classification["target_metrics"],
                years=retrieval_years,
                entity=classification["entity"],
            )
        else:
            sub_questions = self._build_non_numeric_subquestions(
                query, answer_mode, classification.get("target_metrics")
            )

        # Both NUMERIC sub-question builders above search only on the target line
        # item's own alias vocabulary, which finds the structured statement row
        # but not a filing's plain-English sentence stating the same fact in
        # different words. For direct-amount/lookup questions (calc_type empty)
        # one extra, additive query built by the LLM suggester covers that
        # prose. Skipped for computed-ratio questions (calc_type set): their
        # alias-matched multi-row retrieval already targets the exact line items,
        # and an extra generic query there displaced correct evidence rows.
        # LLM-decomposed sub-queries vary run to run and can drop the user's own
        # key terms; the original question is always kept as one extra query so
        # retrieval never depends on the decomposition alone (standard
        # multi-query practice; the non-numeric path already does the same).
        if answer_mode == "NUMERIC" and not formula_entry:
            sub_questions.append({
                "step": len(sub_questions) + 1,
                "type": "retrieval",
                "query": query,
                "target_metric": "",
                "target_year": "",
                "source": "original_question",
            })

        if answer_mode == "NUMERIC" and not classification.get("calc_type"):
            topic_terms = self.decomposer.suggest_narrative_topic_query(query, query_entity)
            if topic_terms:
                sub_questions.append({
                    "step": len(sub_questions) + 1,
                    "type": "retrieval",
                    "query": f"{query_entity} {topic_terms}".strip(),
                    "target_metric": "",
                    "target_year": "",
                    "source": "narrative_topic",
                })

        trace_steps.append({
            "step_name": "Query Decomposition",
            "type": "decomposition",
            "sub_questions": sub_questions,
            "detail": f"分解為 {len(sub_questions)} 個子任務。"
        })

        # ── Step 3: Main Execution Loop ──
        iteration_count = 0
        verification_res = None
        pot_res = None

        if answer_mode == "NUMERIC":
            current_query = query
            is_first_iteration = True

            while iteration_count < max_iterations:
                iteration_count += 1
                iter_trace = {
                    "iteration": iteration_count,
                    "query": current_query,
                    "retrieved_passages": [],
                    "pot_code": "",
                    "sandbox_output": "",
                    "verification": {}
                }

                if is_first_iteration:
                    is_first_iteration = False

                    # ── Step-by-step retrieval: one search per decomposed sub-question ──
                    retrieval_steps = [
                        sq for sq in sub_questions if sq["type"] == "retrieval"
                    ]
                    decompose_src = sub_questions[-1].get("source", "rule") if sub_questions else "rule"

                    for sub_q in retrieval_steps:
                        # Every sub-question is ALWAYS retrieved (no early-stop gate):
                        # what finally reaches PoT / the LLM is decided afterwards by
                        # ranking all collected evidence together (CONTEXT_CHUNK_LIMIT,
                        # then EVIDENCE_PROMPT_CAP).
                        step_query = sub_q["query"]
                        target_metric = sub_q.get("target_metric")
                        target_year   = sub_q.get("target_year")

                        # Check if this (metric, year) pair is already in the buffer.
                        # Only structured table_row evidence (a Line Item | year: value
                        # row) counts;
                        # prose chunks that merely contain both words are not reliable
                        # indicators.
                        def _already_has(metric: str, year: str) -> bool:
                            if not metric or not year:
                                return False
                            for ev in evidence_buffer:
                                if ev.get("type") != "table_row":
                                    continue
                                c = ev.get("content", "") or ""
                                if year not in c:
                                    continue
                                # Bare substring co-occurrence of the metric
                                # NAME anywhere in the row's whole serialized
                                # content (company/report prefix included)
                                # false-positives whenever an UNRELATED row
                                # happens to share a word with the target
                                # metric -- e.g. a "Deferred revenue" row
                                # satisfying a "revenue" sub-query, or a
                                # "Total borrowings of long-term debt" row
                                # satisfying a "debt" sub-query for a
                                # different line item entirely. Classify the
                                # row's OWN "Line Item: ..." label through
                                # the same canonical-mapping logic used
                                # everywhere else in this codebase (handles
                                # the "Deferred"/"non-" negation-prefix cases
                                # already) and require it to actually BE the
                                # target canonical, not merely mention it.
                                m = re.search(r"Line Item:\s*([^|]+?)\s*\|", c)
                                if not m:
                                    continue
                                row_canonical = _get_canonical(
                                    m.group(1), ev.get("company", "") or ""
                                )
                                if row_canonical == metric:
                                    return True
                            return False

                        if _already_has(target_metric, target_year):
                            trace_steps.append({
                                "step_name": f"Step {sub_q['step']}: {target_metric} ({target_year})",
                                "type": "step_retrieval",
                                "detail": f"Already retrieved. Skipping duplicate sub-query.",
                            })
                            continue

                        # A single query-level statement_type_hint is wrong for
                        # composite ratios whose sub-questions genuinely need
                        # DIFFERENT statements (e.g. inventory_turnover =
                        # cogs[income statement] / inventory[balance sheet] —
                        # the overall question's hint votes 100% balance_sheet
                        # since "inventory" is the only metric visible in the
                        # WHOLE question text, wrongly applying that hint to
                        # the cogs sub-query too and boosting a balance-sheet
                        # note page over the real income-statement row).
                        # Re-classify each sub-question's OWN text so its hint
                        # reflects what THAT retrieval actually needs; fall
                        # back to the query-level hint when the sub-question's
                        # own text yields nothing.
                        sub_metrics = self.classifier._extract_target_metrics(step_query.lower())
                        sub_hint = self.classifier._infer_statement_type_hint(sub_metrics)
                        effective_hint = sub_hint if sub_hint != "unknown" else statement_type_hint

                        # Narrative-topic sub-questions target prose topics, not
                        # structured line items.
                        # Use a wider retrieval top_k for terse, generically labeled
                        # table rows to avoid missing them.
                        step_top_k = (
                            self.RETRIEVAL_TOP_K_NARRATIVE
                            if sub_q.get("source") == "narrative_topic"
                            else self.RETRIEVAL_TOP_K
                        )
                        hits = self.vector_store.search(
                            step_query, top_k=step_top_k,
                            exclude_ids=list(retrieved_ids),
                            entity=classification.get("entity"),
                            statement_type_hint=effective_hint,
                            is_segment_comparison=is_segment_comparison,
                            query_years=classification.get("years"),
                        )
                        hits = self._tag_subquery(
                            sub_q.get("step", 0),
                            self._deduplicate_hits(hits, entity=classification.get("entity")),
                        )

                        step_hit_infos = []
                        for hit in hits:
                            retrieved_ids.add(hit["id"])
                            evidence_buffer.append(hit)
                            info = self._build_evidence_info(hit, sub_question=step_query)
                            iter_trace["retrieved_passages"].append(info)
                            step_hit_infos.append(info)
                            evidence_meta.append(info)

                        # Record each retrieval step in the trace
                        metric = sub_q.get("target_metric", "")
                        year = sub_q.get("target_year", "")
                        step_label = f"Step {sub_q['step']}"
                        if metric:
                            step_label += f": {metric}"
                        if year:
                            step_label += f" ({year})"
                        trace_steps.append({
                            "step_name": step_label,
                            "type": "step_retrieval",
                            "detail": (
                                f"[{decompose_src}] Query: '{step_query}' → "
                                f"{len(hits)} passage(s) retrieved"
                            ),
                        })

                    # ── Fallback: if zero evidence collected, use classifier retrieval_queries ──
                    if not evidence_buffer:
                        for fb_idx, sq in enumerate(classification["retrieval_queries"]):
                            hits = self.vector_store.search(
                                sq, top_k=self.RETRIEVAL_TOP_K,
                                exclude_ids=list(retrieved_ids),
                                entity=classification.get("entity"),
                                statement_type_hint=statement_type_hint,  # Step 4
                                is_segment_comparison=is_segment_comparison,
                                query_years=classification.get("years"),
                            )
                            for hit in self._tag_subquery(
                                100 + fb_idx,
                                self._deduplicate_hits(hits, entity=classification.get("entity")),
                            ):
                                retrieved_ids.add(hit["id"])
                                evidence_buffer.append(hit)
                                info = self._build_evidence_info(hit)
                                iter_trace["retrieved_passages"].append(info)
                                evidence_meta.append(info)

                else:
                    # ── Subsequent iterations: use refined query ──
                    new_hits = self._deduplicate_hits(
                        self.vector_store.search(
                            current_query, top_k=self.RETRIEVAL_TOP_K,
                            exclude_ids=list(retrieved_ids),
                            entity=classification.get("entity"),
                            statement_type_hint=statement_type_hint,  # Step 4
                            is_segment_comparison=is_segment_comparison,
                            query_years=classification.get("years"),
                        ),
                        entity=classification.get("entity"),
                    )
                    for hit in self._tag_subquery(200 + iteration_count, new_hits):
                        retrieved_ids.add(hit["id"])
                        evidence_buffer.append(hit)
                        info = self._build_evidence_info(hit)
                        iter_trace["retrieved_passages"].append(info)
                        evidence_meta.append(info)

                # ── PoT Execution ──
                # Sort by relevance score so the BEST chunks reach PoT,
                # not just the most recently retrieved ones (RC5 fix)
                raw_window = select_with_quota(evidence_buffer, self.CONTEXT_CHUNK_LIMIT)
                context_window = []
                for ev in raw_window:
                    ev_enriched = dict(ev)
                    parent_id = ev.get("parent_id")
                    if parent_id and not ev_enriched.get("parent_content"):
                        ev_enriched["parent_content"] = self.vector_store.get_parent_content(parent_id)
                    context_window.append(ev_enriched)
                pot_res = self.pot_reasoner.generate_and_execute(
                    query, context_window, entity=classification["entity"]
                )
                iter_trace["pot_code"] = pot_res["code"]
                iter_trace["sandbox_output"] = pot_res["output_log"]
                iter_trace["result_value"] = pot_res["result_value"]

                # ── Tri-Check Verification ──
                verification_res = self.verifier.verify(query, context_window, pot_res)
                iter_trace["verification"] = verification_res

                trace_steps.append({
                    "step_name": f"Iteration {iteration_count}: PoT + Verification",
                    "type": "iteration",
                    "data": iter_trace
                })

                # Accept or simple → done
                if verification_res["decision"] == "ACCEPT" or complexity == "SIMPLE":
                    break

                # Reject → refine and re-search
                current_query = self.refiner.refine(query, verification_res, iteration_count)
                trace_steps.append({
                    "step_name": f"Query Refinement #{iteration_count}",
                    "type": "refinement",
                    "detail": f"REJECT → refined query: '{current_query}'"
                })

        else:
            # ── Non-numeric path (EXPLANATION / ASSESSMENT / EXCLUSION) ──
            iteration_count = 1
            iter_trace = {
                "iteration": 1, "query": query,
                "retrieved_passages": [], "pot_code": "", "sandbox_output": "",
                "verification": {}
            }

            # If a metric matches a registered formula, search for that formula's
            # required variables.
            # The raw question text often omits the derived metric's component line-item
            # names.
            non_numeric_formula = detect_formula(query)
            # Evaluated against the ORIGINAL question, not each individual
            # sub-query below (a keyword-stuffed sub-query like "AMD
            # Revenue Net Revenue" never repeats "what drove" phrasing even
            # when the overall question plainly is an attribution question)
            # -- see hybrid_retriever.is_attribution_query's docstring.
            is_attribution = is_attribution_query(query)
            if non_numeric_formula:
                formula_query_entity = clean_entity if clean_entity and clean_entity != "company" else classification["entity"]
                search_queries = [
                    step["query"] for step in
                    self._build_formula_subquestions(non_numeric_formula, formula_query_entity, classification["years"])
                ]
                # _build_formula_subquestions() emits one query per placeholder only.
                # Attribution questions for a formula-derived metric must also search
                # for causal narrative, not just the numeric inputs.
                if is_attribution:
                    search_queries.append(query)
            else:
                search_queries = classification["retrieval_queries"]
                # Additive: one extra query from the LLM suggester (which filing
                # section/vocabulary would hold this answer). A failed or empty
                # response leaves the classifier's own queries unchanged.
                llm_topic_terms = self.decomposer.suggest_narrative_topic_query(
                    query, clean_entity if clean_entity and clean_entity != "company" else classification["entity"]
                )
                if llm_topic_terms:
                    search_queries = search_queries + [
                        f"{classification['entity']} {llm_topic_terms}".strip()
                    ]
            # This branch runs when no registered formula matches, so it biases ranking
            # toward prose content.
            # Also widen retrieval for attribution questions so narrative boosts can
            # surface causal explanations.
            prefer_narrative = (non_numeric_formula is None or is_attribution)
            is_geography = is_geography_query(query)
            is_legal = is_legal_query(query)
            new_hits = []
            for sq_idx, sq in enumerate(search_queries):
                new_hits.extend(self._tag_subquery(sq_idx, self.vector_store.search(
                    # This retrieval block runs for non-numeric answer modes and always
                    # uses the wider narrative top_k.
                    # prefer_narrative controls scoring boosts but not how many
                    # candidates pass the cutoff.
                    sq, top_k=self.RETRIEVAL_TOP_K_NARRATIVE,
                    exclude_ids=list(retrieved_ids),
                    entity=classification.get("entity"),
                    statement_type_hint=statement_type_hint,  # Step 4
                    prefer_narrative=prefer_narrative,
                    is_attribution=is_attribution,
                    is_geography=is_geography,
                    is_legal=is_legal,
                    query_years=classification.get("years"),
                )))
            new_hits = self._deduplicate_hits(new_hits, entity=classification.get("entity"))
            for hit in new_hits:
                retrieved_ids.add(hit["id"])
                evidence_buffer.append(hit)
                info = self._build_evidence_info(hit)
                iter_trace["retrieved_passages"].append(info)
                evidence_meta.append(info)

            # Explains that some explanation/assessment questions still require a
            # numeric comparison when they match a registered formula; they also need
            # qualitative framing in the final text.
            # Notes that previously the pipeline returned an empty numeric result object
            # so the LLM supplied numbers without any sandbox trace, which can lead to
            # ungrounded outputs; keep sandbox-backed numeric computation to avoid that.
            context_window = []
            for ev in select_with_quota(evidence_buffer, self.CONTEXT_CHUNK_LIMIT):
                ev_enriched = dict(ev)
                parent_id = ev.get("parent_id")
                if parent_id and not ev_enriched.get("parent_content"):
                    ev_enriched["parent_content"] = self.vector_store.get_parent_content(parent_id)
                context_window.append(ev_enriched)

            if non_numeric_formula:
                pot_res = self.pot_reasoner.generate_and_execute(
                    query, context_window, entity=classification["entity"]
                )
            else:
                pot_res = {
                    "code": "", "success": True, "result_value": None,
                    "output_log": "", "extracted_variables": {},
                    "answer_mode": answer_mode,
                }
            pot_res["answer_mode"] = answer_mode
            iter_trace["pot_code"] = pot_res.get("code", "")
            iter_trace["sandbox_output"] = pot_res.get("output_log", "")
            iter_trace["result_value"] = pot_res.get("result_value")

            verification_res = self.verifier.verify(query, self._top_evidence(evidence_buffer), pot_res)
            iter_trace["verification"] = verification_res
            trace_steps.append({
                "step_name": "Evidence Retrieval & Analysis",
                "type": "retrieval",
                "data": iter_trace
            })

        # ── Step 4: Final Answer Synthesis ──
        final_context = self._top_evidence(evidence_buffer)
        final_answer = self._synthesize_final_answer(
            query, final_context, pot_res or {}, verification_res or {},
            classification, sub_questions
        )
        # Mirrors llm_client.generate_answer's own effective_cap exactly --
        # same three conditions (RELIABLE_POT_EVIDENCE_CAP>0, NUMERIC,
        # is_reliable_pot_result) -- so evidence_sources below reports the
        # SAME subset generate_answer actually saw, not the full
        # EVIDENCE_PROMPT_CAP window when a smaller one was really used.
        evidence_prompt_cap = EVIDENCE_PROMPT_CAP
        if (
            RELIABLE_POT_EVIDENCE_CAP > 0
            and answer_mode == "NUMERIC"
            and is_reliable_pot_result(pot_res)
        ):
            evidence_prompt_cap = RELIABLE_POT_EVIDENCE_CAP

        return {
            "query": query,
            "complexity": complexity,
            "answer_mode": answer_mode,
            "question_type": classification["question_type"],
            "cognitive_task": classification["cognitive_task"],
            "retrieval_strategy": retrieval_strategy,
            "total_iterations": iteration_count,
            "final_answer": final_answer,
            "resolved_entity": classification["entity"],
            # The sandbox placeholder for 'result not reliable' is not a computed
            # answer; emitting its raw numeric default can be misinterpreted as a real
            # computed result by downstream UI layers.
            # Log messages still carry the warning and should be surfaced instead of the
            # raw placeholder value.
            "result_value": (
                None
                if pot_res and "result is not reliable" in (pot_res.get("output_log") or "")
                else (pot_res.get("result_value") if pot_res else None)
            ),
            "verification": verification_res,
            "pot_code": pot_res.get("code") if pot_res else "",
            "sandbox_log": pot_res.get("output_log") if pot_res else "",
            "is_degraded_formula": pot_res.get("is_degraded_formula", False) if pot_res else False,
            "degraded_note": pot_res.get("degraded_note", "") if pot_res else "",
            "result_series": pot_res.get("result_series", []) if pot_res else [],
            "result_delta": pot_res.get("result_delta") if pot_res else None,
            "result_direction": pot_res.get("result_direction") if pot_res else None,
            "result_unit": pot_res.get("result_unit", "") if pot_res else "",
            "is_comparison_answer": pot_res.get("is_comparison_answer", False) if pot_res else False,
            "is_qualitative_characterization": pot_res.get("is_qualitative_characterization", False) if pot_res else False,
            # Return ONLY the subset of evidence that actually reached the
            # LLM's prompt (see llm_client.generate_answer's own sort +
            # EVIDENCE_PROMPT_CAP slice, applied here identically to
            # final_context -- the SAME list generate_answer received),
            # not every candidate retrieval ever pulled in. evidence_meta/
            # evidence_buffer can hold every retrieved item
            # across every sub-query; only EVIDENCE_PROMPT_CAP of the
            # highest-scoring ones ever got FORMATTED into the prompt text
            # the model actually read. Returning the full unfiltered list
            # here made the frontend's Source Evidence panel show
            # candidates the LLM never saw, so there was no way to tell
            # from the UI alone whether an answer's citations and its
            # actual grounding evidence agreed. Falls back to rebuilding
            # via _build_evidence_info for any final_context item that
            # (unexpectedly) has no matching evidence_meta entry by id.
            "evidence_sources": [
                next(
                    (m for m in evidence_meta if m.get("id") == h.get("id")),
                    None,
                ) or self._build_evidence_info(h)
                for h in sorted(
                    final_context,
                    key=lambda item: item.get("relevance_score") or 0,
                    reverse=True,
                )[:evidence_prompt_cap]
            ],
            "reasoning_steps": trace_steps,
            "execution_trace": trace_steps,
        }

    # ═══════════════════════════════════════════════════════════════
    # Private Helpers
    # ═══════════════════════════════════════════════════════════════

    #: Wording that points at a company's EARNINGS RELEASE rather than its
    #: 10-K: regional/segment breakdown questions ("region(s)", "segment(s)",
    #: "topline", "US ... international"), non-GAAP "adjusted" measures and
    #: forward guidance -- none of which a 10-K reports in that form.
    _REGIONAL_BREAKDOWN_CUE_RE = re.compile(
        r"\b(?:regions?|segments?|topline)\b|\b(?:us|u\.s\.)\b.*\binternational\b"
        r"|non[- ]?gaap|\badjusted\s+(?:eps|ebitda|ebit|operating|net income|earnings|non)"
        r"|\bguidance\b|\boutlook\b"
        # Maps a descriptive earnings-release phrase to the canonical comparable-
        # constant-currency adjustment concept used for like-for-like comparisons.
        # This is a normalization for aligning different wording to the same analytical
        # bridge.
        r"|constant[- ]currency|pass-?through|one-?off"
    )
    #: A question asking what is EXPECTED for year Y is answered by a filing
    #: dated before Y (it is a forecast), so year Y-1 filings are the match.
    _FORWARD_LOOKING_CUE_RE = re.compile(
        r"\bexpected?\s+to\b|\bexpects?\b|\bguidance\b|\boutlook\b|\bforecast\w*|\banticipat\w+"
    )
    #: If a question about an ongoing multi-period liability names no year, the code
    #: prefers the most recent filing rather than an older annual filing; otherwise
    #: cumulative-in-progress percentages can be stale.
    #: When queries omit a period, ensure selection logic looks for the latest available
    #: filing for that entity.
    _SEPARATION_TOPIC_RE = re.compile(
        r"\bseparat\w+|\bspin[- ]?off\w*|\bdivest\w*", re.IGNORECASE
    )

    @staticmethod
    def _readable_company_name(matched_company: str, query: str) -> str:
        """Derive a human-readable entity name for retrieval queries from the
        corpus filename stem: use the stem's original spacing if it matches,
        otherwise join the stem's underscore-separated parts.
        """
        import re as _re
        base = _re.split(r"(?<!\d)(?:19|20)\d{2}", matched_company or "")[0].strip("_- ")
        if not base:
            return ""
        compact = _re.sub(r"[^a-z0-9]", "", base.lower())
        words = _re.findall(r"[A-Za-z0-9&]+", query)
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                span = words[i:i + n]
                if _re.sub(r"[^a-z0-9]", "", "".join(span).lower()) == compact:
                    return " ".join(span)
        return base.replace("_", " ").title()

    def _match_entity_to_corpus(self, classifier_entity: str, query: str) -> str:
        """
        Always align the classifier's entity name to the actual company string
        stored in the corpus (e.g. '3M' → '3M_2022_10K').
        Also handles the case where the classifier returned generic 'company'.
        """
        import re as _re
        if not self.vector_store.uploaded_files:
            return classifier_entity  # no uploads yet — use classifier result as-is

        q_lower = query.lower()
        # Normalise helper: remove year tokens, underscores, hyphens using digit-
        # boundary checks rather than word-boundary so years inside filenames like
        # X_2022_10K get stripped.
        # This avoids many filings sharing a literal year token and falsely tying query
        # matches across different entities; use (?<!\d)...(?!\d) to detect standalone
        # years.
        def _normalise(s: str) -> str:
            s = _re.sub(r'(?<!\d)(?:20|19)\d{2}(?!\d)', '', s)   # strip years
            s = _re.sub(r'[_\-]+', ' ', s)             # underscores → spaces
            return s.lower().strip()

        norm_classifier = _normalise(classifier_entity)
        abbr_tokens = _re.findall(r"\b[A-Za-z][A-Za-z0-9&]{2,}\b", query)
        compact_query = _re.sub(r"[^a-z0-9]", "", q_lower)
        _DOC_TYPE_WORDS = {"10k", "10q", "8k", "dated", "earnings"}
        q_stop = {"the", "and", "for", "has", "had", "have", "does", "did", "was", "were", "are",
                  "what", "which", "who", "how", "when", "why", "that", "this", "with", "from"}

        def _is_subsequence(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)
        best_company = None
        best_score = 0

        # Use years mentioned in the query only to break ties between multiple filings
        # of the same entity; normalisation still strips year tokens for initial
        # matching so entity names match consistently across filings.
        # Detect years with digit-boundary regex rather than word-boundary so common
        # filename patterns and adjacent characters don't prevent year extraction.
        _YEAR_RE = r'(?<!\d)(?:20|19)\d{2}(?!\d)'
        query_years = set(_re.findall(_YEAR_RE, query))

        # Detect a bare quarter token (q1..q4) in the query text as an independent word
        # token.
        # Used only for tie-breaking between candidate documents; unrelated to extracted
        # query years.
        _query_quarter_m = _re.search(r'\bq([1-4])\b', q_lower)
        query_quarter = f"q{_query_quarter_m.group(1)}" if _query_quarter_m else None
        # For forward-looking questions about year Y, prefer documents dated in year Y-1
        # over those dated in Y.
        # This treats the prior-year filing as the more likely source for forward-
        # looking statements.
        forward_year_prior = None
        if query_years and self._FORWARD_LOOKING_CUE_RE.search(q_lower) and query_quarter is None:
            forward_year_prior = str(int(max(query_years)) - 1)

        # Extract both year and adjacent quarter suffix from filenames (e.g. _2022Q4) to
        # distinguish same-year documents.
        # This prevents wrong-document selection when multiple filings share the same
        # bare year.
        _PERIOD_RE = _re.compile(r'(?<!\d)((?:20|19)\d{2})(q[1-4])?(?!\d)', _re.IGNORECASE)

        def _extract_period(s: str):
            m = _PERIOD_RE.search(s)
            if not m:
                return None, None
            return m.group(1), (m.group(2).lower() if m.group(2) else None)

        # A full calendar date in the question ("...on May 26, 2023") points at the
        # 8-K filed just AFTER that event (files are named "..._dated-YYYY-MM-DD").
        # Used only when a filing's date is within 45 days of the question's date;
        # a filing dated before the event ranks below one dated after it.
        import datetime as _dt
        _MONTHS = {m: i + 1 for i, m in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
        _qdm = _re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:20|19)\d{2})\b",
            q_lower,
        )
        query_date = None
        if _qdm:
            try:
                query_date = _dt.date(int(_qdm.group(3)), _MONTHS[_qdm.group(1)], int(_qdm.group(2)))
            except ValueError:
                query_date = None

        def _date_key(corpus_name: str) -> int:
            if query_date is None:
                return 0
            m = _re.search(r"dated[-_](\d{4})-(\d{2})-(\d{2})", corpus_name)
            if not m:
                return 0
            try:
                delta = (_dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) - query_date).days
            except ValueError:
                return 0
            if abs(delta) > 45:
                return 0
            return 1000 - delta if delta >= 0 else 1000 - (abs(delta) + 100)

        best_tie_key = None
        scored = []

        for uf in self.vector_store.uploaded_files:
            corpus_company = uf.get("company", "")
            if not corpus_company:
                continue
            norm_corpus = _normalise(corpus_company)
            corpus_year, corpus_quarter = _extract_period(corpus_company)

            score = 0
            # Score 1: corpus company words appear in query
            # Filing-type words and leftover date digits are not company names.
            corpus_words = [
                w for w in norm_corpus.split()
                if len(w) >= 2 and not w.isdigit() and w not in _DOC_TYPE_WORDS
            ]
            query_hits = sum(1 for w in corpus_words if w in q_lower)
            score += query_hits * 3

            # Match company names in queries that contain spaces to corpus filenames
            # that omit spaces by normalizing spacing.
            # Ensures consistent name matching regardless of spacing in query vs
            # filename.
            base_name = _re.sub(r"[^a-z0-9]", "", _re.split(r"(?<!\d)(?:19|20)\d{2}", corpus_company)[0].lower())
            if len(base_name) >= 3 and base_name in compact_query:
                score += 6

            # Handle abbreviated company tokens in queries by matching prefixes or
            # acronym-shaped tokens to compact corpus names.
            # Matches tokens that are prefixes or whose letters appear in order in the
            # compact name.
            compact_corpus = norm_corpus.replace(" ", "")
            for tok in abbr_tokens:
                tl = tok.lower()
                if tl in corpus_words or tl in q_stop:
                    continue
                if compact_corpus.startswith(tl):
                    score += 3
                elif (
                    (tok.isupper() or any(ch.isupper() for ch in tok[1:]))
                    and tl[0] == compact_corpus[:1]
                    and _is_subsequence(tl, compact_corpus)
                ):
                    score += 2

            # Score 2: classifier entity words appear in corpus company name
            if norm_classifier and norm_classifier != "company":
                clf_words = [w for w in norm_classifier.split() if len(w) >= 2]
                clf_hits = sum(1 for w in clf_words if w in norm_corpus)
                score += clf_hits * 2

                # Score 3: exact substring match (highest confidence)
                if norm_classifier in norm_corpus or norm_corpus in norm_classifier:
                    score += 5

            # Tie-break among multiple filings for the same company using a four-tier
            # tuple (higher wins):
            # 1) does the filing's quarter match a quarter explicitly named in the
            # query? (only tier that uses corpus quarter)
            # 2) does the filing's bare year appear in the query?
            # 3) does the filing have no quarter suffix (prefer plain annual filings as
            # safer defaults)?
            # 4) the bare year (prefer more recent as final fallback).
            # This favors an explicit quarter signal first, otherwise prefers annual
            # filings when no other signal exists to avoid selecting fragmentary
            # quarterly documents.
            tie_key = (
                _date_key(corpus_company),
                1 if (query_quarter is not None and corpus_quarter == query_quarter) else 0,
                (2 if (forward_year_prior and corpus_year == forward_year_prior)
                 else 1 if corpus_year in query_years else 0),
                # Refers to a specific regex's behavior: it overrides a higher-tier
                # default preference for plain annual reports, but only for a narrowly
                # defined forward-looking separation-cost question shape; it is guarded
                # so it cannot trigger on unrelated no-year questions and will not
                # conflict with the higher-tier rule.
                1 if (
                    not query_years and corpus_quarter
                    and self._FORWARD_LOOKING_CUE_RE.search(q_lower)
                    and self._SEPARATION_TOPIC_RE.search(q_lower)
                ) else 0,
                0 if corpus_quarter else 1,
                corpus_year or "",
            )
            scored.append((score, tie_key, corpus_company))
            if score > best_score or (score > 0 and score == best_score and tie_key > best_tie_key):
                best_score = score
                best_company = corpus_company
                best_tie_key = tie_key

        if best_company and best_score > 0:
            # Tie-breaker when multiple filings from the same filer score identically on
            # all other signals: count, per tied filing, the passages containing at
            # least two of the question's distinctive words and prefer the filing with
            # the higher count; avoids relying solely on metadata ordering.
            tied_docs = [c for (sc, tk, c) in scored if sc == best_score and tk == best_tie_key]
            if len(tied_docs) >= 2:
                _stop = {
                    "the", "and", "for", "has", "had", "have", "does", "did", "was", "were", "are",
                    "what", "which", "who", "whom", "how", "when", "where", "why", "that", "this",
                    "with", "from", "their", "there", "any", "new", "company", "much", "many",
                    "between", "during", "than", "into", "about", "been", "its", "his", "her",
                }
                _name_words = set()
                for _d in tied_docs:
                    _name_words.update(_normalise(_d).split())
                _terms = [
                    w for w in _re.findall(r"[a-z][a-z&-]{2,}", q_lower)
                    if w not in _stop and w not in _name_words
                ]
                if len(_terms) >= 2:
                    _counts = {d: 0 for d in tied_docs}
                    for _p in self.vector_store.corpus:
                        _pc = _p.get("company")
                        if _pc in _counts:
                            _low = (_p.get("content") or "").lower()
                            if sum(1 for _t in _terms if _t in _low) >= 2:
                                _counts[_pc] += 1
                    _ranked = sorted(_counts.items(), key=lambda kv: kv[1], reverse=True)
                    if _ranked[0][1] > _ranked[1][1]:
                        best_company = _ranked[0][0]
            # For regional/segment breakdown questions, prefer a company's simplified
            # regional supplemental table from its earnings release rather than the
            # formal annual report; when same-filer, same-year candidates tie, select
            # the earnings-release file for this question shape because the table format
            # matches the question intent.
            if self._REGIONAL_BREAKDOWN_CUE_RE.search(q_lower):
                tied_earnings = [
                    (tk, c) for (sc, tk, c) in scored
                    if sc == best_score and tk[:3] == best_tie_key[:3]
                    and "earnings" in c.lower()
                ]
                if tied_earnings:
                    return max(tied_earnings)[1]
            return best_company

        # Fallback: if classifier returned a real entity name, keep it
        if classifier_entity and classifier_entity != "company":
            return classifier_entity

        # Last resort: most recently uploaded file
        return self.vector_store.uploaded_files[-1].get("company", "company")

    def _infer_entity_from_corpus(self, query: str) -> str:
        """Legacy method — kept for backward compatibility. Delegates to _match_entity_to_corpus."""
        return self._match_entity_to_corpus("company", query)

    @staticmethod
    def _tag_subquery(idx: int, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remember which sub-query retrieved each passage (for the per-
        sub-query quota in evidence_selection.select_with_quota)."""
        for hit in hits:
            hit["subquery_idx"] = idx
        return hits

    @staticmethod
    def _company_key(name: str) -> str:
        """Extract the entity part of a corpus document name: e.g. STEM_YEAR_TAG
        -> STEMWITHOUTSPACES; return empty when no year/identifier segment exists.
        """
        m = re.match(r"^(.*?)_(?:19|20)\d\d", name or "")
        return re.sub(r"[^A-Za-z0-9]", "", m.group(1)).upper() if m else ""

    def _restrict_to_company(self, hits: List[Dict[str, Any]], entity: Optional[str]) -> List[Dict[str, Any]]:
        """Filter out retrieved passages from other entities when the resolved
        entity maps to a specific corpus document. Retriever down-weighting can
        leave unrelated-entity passages; dropping them avoids extracting numbers
        from the wrong entity. Do not apply this filter if it would leave no evidence.
        """
        key = self._company_key(entity or "")
        if not key:
            return hits
        kept = [h for h in hits if self._company_key(h.get("company") or "") in ("", key)]
        return kept or hits

    def _deduplicate_hits(self, hits: List[Dict[str, Any]], entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Keep the HIGHEST-scoring occurrence of each passage id rather than the first one
        seen.

        Args:
        hits: list of hit records where the same passage id may appear multiple times
        with different scores.

        Returns:
        A deduplicated list preserving the record with the maximum score per passage id.

        Note: This avoids locking in a lower score from an earlier sub-query when a
        later sub-query produces a higher, more relevant score.
        """
        hits = self._restrict_to_company(hits, entity)
        best_by_id: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for hit in hits:
            hit_id = hit.get("id") or hit.get("content")
            existing = best_by_id.get(hit_id)
            if existing is None:
                best_by_id[hit_id] = hit
                order.append(hit_id)
            elif (hit.get("relevance_score") or 0) > (existing.get("relevance_score") or 0):
                best_by_id[hit_id] = hit
        return [best_by_id[hit_id] for hit_id in order]

    # Retrieval-only synonym expansions appended to a placeholder's search query to
    # widen evidence recall for explanation-style questions; these terms are never added
    # to extraction alias lists so numeric extraction logic remains correct. Purpose:
    # retrieve passages that discuss drivers using different wording (e.g., cost-side
    # language) without treating those rows as the numeric target variable.
    _RETRIEVAL_SYNONYM_TERMS: Dict[str, List[str]] = {
        "gross_profit": ["Cost of Products Sold", "Cost of Goods Sold", "Cost of Sales", "COGS"],
        # Add retrieval aliases for specific inventory breakdown terms so the convention
        # check (average vs. ending inventory) can find sub-rows like finished goods;
        # these aliases only affect retrieval ranking and do not change which numeric
        # value extraction treats as the inventory number.
        "inventory": ["Finished Goods", "Merchandise Inventory", "Fuel Inventory", "Raw Materials and Supplies"],
        "inventory_old": ["Finished Goods", "Merchandise Inventory", "Fuel Inventory", "Raw Materials and Supplies"],
        "inventory_new": ["Finished Goods", "Merchandise Inventory", "Fuel Inventory", "Raw Materials and Supplies"],
    }

    def _build_formula_subquestions(
        self, formula_entry: Dict[str, Any], entity: str, years: List[str],
    ) -> List[Dict[str, Any]]:
        """
        One retrieval step per formula placeholder, using that
        placeholder's OWN primary alias as the search text — the same
        alias list _extract_formula_guided() will later score candidate
        rows against, so retrieval and extraction are guaranteed to be
        looking for the same thing, deterministically (see the call site
        for why this replaces LLM decomposition for formula questions).
        Retrieval-only synonym terms (see _RETRIEVAL_SYNONYM_TERMS) are
        appended to the query text for specific placeholders, WITHOUT
        being added to the alias list extraction itself scores against.

        Year targeting mirrors _extract_formula_guided()'s own picking
        rule for each formula shape:
          - period_average: one step per placeholder per year in the
            full requested range (every year is needed to compute the
            average).
          - multi_year (old/new pairs, e.g. fixed_asset_turnover,
            dpo): "_old"-suffixed placeholders target the earliest
            requested year, "_new"-suffixed or bare placeholders target
            the latest.
          - otherwise: every placeholder targets the single latest
            requested year (or no year filter if none was named).
        """
        required_vars = get_variable_aliases(formula_entry)
        is_multi_year = formula_entry.get("multi_year", False)
        is_period_average = formula_entry.get("period_average", False)
        sorted_years = sorted(set(years)) if years else []

        steps: List[Dict[str, Any]] = []
        for placeholder, aliases in required_vars.items():
            # Prefer ASCII/Latin-alphabet aliases first when building queries, falling
            # back to other-language variants only if no ASCII aliases exist. Use up to
            # three distinct ASCII aliases per line item to cover common phrasings
            # across different filers, improving retrieval robustness without changing
            # extraction alias lists.
            ascii_aliases = [a for a in aliases if a.isascii()]
            primary_alias = " ".join(dict.fromkeys(ascii_aliases[:3])) if ascii_aliases else (
                aliases[0] if aliases else placeholder
            )
            extra_terms = self._RETRIEVAL_SYNONYM_TERMS.get(placeholder)
            if extra_terms:
                primary_alias = f"{primary_alias} {' '.join(extra_terms)}"
            if is_period_average and sorted_years:
                target_years = sorted_years
            elif is_multi_year and len(sorted_years) >= 2:
                target_years = [sorted_years[0] if placeholder.endswith("_old") else sorted_years[-1]]
            elif sorted_years:
                target_years = [sorted_years[-1]]
            else:
                target_years = [""]

            for yr in target_years:
                steps.append({
                    "step": len(steps) + 1,
                    "type": "retrieval",
                    "query": f"{entity} {primary_alias} {yr}".strip(),
                    "target_metric": placeholder,
                    "target_year": yr,
                    "source": "formula",
                })
        return steps

    def _build_non_numeric_subquestions(
        self, query: str, answer_mode: str, target_metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        # Earlier logic appended a fixed phrase to every template's retrieval
        # suffix regardless of the question's actual target metrics.
        # Use the classifier-identified target_metrics to build a focused suffix.
        metric_terms = " ".join(m.replace("_", " ") for m in (target_metrics or []))
        fallback_suffix = {
            "ASSESSMENT": "capital expenditure assets depreciation",
            "EXCLUSION": "segment revenue organic growth acquisition",
            "EXPLANATION": "operating margin cost structure segment",
        }.get(answer_mode, "")
        retrieval_query = f"{query} {metric_terms or fallback_suffix}".strip()

        templates = {
            "ASSESSMENT": [
                {"step": 1, "type": "retrieval", "query": retrieval_query},
                {"step": 2, "type": "analysis", "query": "Assess the metric's suitability"},
            ],
            "EXCLUSION": [
                {"step": 1, "type": "retrieval", "query": retrieval_query},
                {"step": 2, "type": "analysis", "query": "Isolate organic vs M&A impact"},
            ],
            "EXPLANATION": [
                {"step": 1, "type": "retrieval", "query": retrieval_query},
                {"step": 2, "type": "analysis", "query": "Identify key drivers"},
            ],
        }
        return templates.get(answer_mode, [
            {"step": 1, "type": "retrieval", "query": query},
            {"step": 2, "type": "analysis", "query": "Synthesize evidence"},
        ])

    def _synthesize_final_answer(
        self, query: str, evidence: List[Dict[str, Any]],
        pot_res: Dict[str, Any], verifier_res: Dict[str, Any],
        classification: Dict[str, Any], sub_questions: List[Dict[str, Any]],
    ) -> str:
        answer_mode = classification.get("answer_mode", "NUMERIC")

        # ── Try LLM first ──
        route_res = {
            "complexity": classification.get("complexity", "SIMPLE"),
            "answer_mode": answer_mode,
            "reason": f"FinanceBench: {classification.get('question_type', '')} / {classification.get('cognitive_task', '')}",
        }
        llm_answer = self.llm_generator.generate_answer(
            query=query, answer_mode=answer_mode, evidence=evidence,
            route_res=route_res, pot_res=pot_res,
            verification_res=verifier_res, sub_questions=sub_questions,
        )
        if llm_answer:
            return _strip_trailing_bare_number(llm_answer)

        # ── Fallback: concise rule-based synthesis ──
        if answer_mode == "NUMERIC":
            return self._synthesize_numeric_concise(query, evidence, pot_res)
        return self._synthesize_qualitative_concise(query, evidence, classification)

    # ─── Concise numeric answer (English) ────────────────────────
    def _synthesize_numeric_concise(
        self, query: str, evidence: List[Dict[str, Any]],
        pot_res: Dict[str, Any]
    ) -> str:
        code_val = pot_res.get("result_value")
        log = pot_res.get("output_log", "")
        extracted_vars = pot_res.get("extracted_variables", {})
        company = evidence[0].get("company", "the company") if evidence else "the company"

        parts = []

        # ── Line 1: Direct answer ──
        if code_val is not None:
            parts.append(f"The calculation result for **{company}** is **`{code_val}`**.")
        else:
            parts.append(f"Based on retrieved financial data for **{company}**, a deterministic calculation result could not be obtained.")

        # ── Line 2: Sandbox output summary ──
        if log:
            parts.append(f"\n> {log}")

        # ── Degraded-formula warning: a different formula was silently
        # substituted (e.g. Current Ratio in place of Quick Ratio) because
        # the exact metric asked for couldn't be computed — this rule-
        # based fallback only fires when the LLM synthesis call is
        # unavailable, so the caveat must be stated here too, not just in
        # the LLM prompt instruction.
        if pot_res.get("is_degraded_formula"):
            parts.append(f"\n⚠️ **Note**: {pot_res.get('degraded_note', '')}")

        # ── Line 3: Key data points used ──
        if extracted_vars:
            data_points = []
            for _, (item, year, val) in list(extracted_vars.items())[:4]:
                data_points.append(f"{item} ({year}): `{val}`")
            if data_points:
                parts.append(f"\nData sources: {' | '.join(data_points)}")

        return "\n".join(parts)

    # ─── Concise qualitative answer (English) ──────────────────────────────
    def _synthesize_qualitative_concise(
        self, query: str, evidence: List[Dict[str, Any]],
        classification: Dict[str, Any]
    ) -> str:
        parts = []
        cog_task = classification.get("cognitive_task", "")

        # Lead sentence
        if cog_task == "LOGICAL_INFERENCE":
            parts.append("Analysis based on retrieved financial evidence:")
        else:
            parts.append("According to the financial report disclosures:")

        # Extract relevant snippets (max 2)
        for ev in evidence[:2]:
            content = ev.get("content", "")
            for marker in ["Content:", "Text:"]:
                if marker in content:
                    content = content.split(marker, 1)[-1].strip()
            snippet = content[:200].strip()
            company = ev.get("company", "")
            table = ev.get("table_name", "")
            if snippet:
                parts.append(f"\n- [{company} / {table}] {snippet}")

        if not evidence:
            parts.append("\n⚠️ Insufficient evidence retrieved. Please upload complete financial report documents.")

        return "\n".join(parts)