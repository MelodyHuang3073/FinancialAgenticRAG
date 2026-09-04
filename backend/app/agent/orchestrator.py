"""
FinAgent-RAG Orchestrator

核心流程：
  1. FinanceBench 問題分類（question_classifier）
  2. 多步拆解（decomposer）
  3. Hybrid RAG 檢索（vector_store + hybrid_retriever）
  4. PoT 推理（pot_reasoner + sandbox）
  5. 三重自我驗證（verifier） + 迭代精練（refiner）
  6. LLM 回答綜合（llm_client）
"""

from typing import Dict, Any, List, Optional

from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.question_classifier import FinanceBenchClassifier
from app.agent.decomposer import QueryDecomposer
from app.agent.pot_reasoner import ProgramOfThoughtReasoner
from app.agent.verifier import TriCheckSelfVerifier
from app.agent.refiner import QueryRefiner
from app.agent.llm_client import LLMAnswerGenerator
from app.agent.financial_formula_library import detect_formula, get_variable_aliases


class FinAgentRAGOrchestrator:
    RETRIEVAL_TOP_K = 3          # chunks per sub-question (was 5)
    # Hard ceiling on total evidence buffer size. Must comfortably fit every
    # sub-query a single formula's own required_vars can generate (top_k=3
    # each) — a composite formula like cash_conversion_cycle needs 8
    # placeholders (cogs, revenue, inv_old/new, ar_old/new, ap_old/new), so
    # 8*3=24 sub-results. The old value of 15 silently cut off mid-formula,
    # dropping ap_old/ap_new before they were ever retrieved (confirmed
    # real case: General Mills FY2019 CCC — accounts payable never entered
    # the evidence buffer at all, and the whole computation fell back to a
    # generic, ungrounded LLM guess). Sized with headroom above today's
    # largest formula rather than pinned to exactly 24, so the next
    # formula with one or two more placeholders doesn't repeat this.
    RETRIEVAL_MAX_TOTAL = 30
    # A SECOND, separate cap applied right before evidence reaches PoT/the
    # LLM (sorted by relevance_score, top N kept) — raising
    # RETRIEVAL_MAX_TOTAL alone isn't enough if this one stays tight,
    # since it can still truncate a lower-but-still-correct-scoring row
    # out of the final window even though it survived the earlier cap.
    # Confirmed real case: General Mills' own real "Net earnings
    # attributable to General Mills" row (score ~43) ranked #3 for its
    # own retrieval query — comfortably inside RETRIEVAL_MAX_TOTAL=30 —
    # but still got squeezed out of the final CONTEXT_CHUNK_LIMIT=8 window
    # by higher-scoring prose chunks from OTHER sub-queries in the same
    # evidence_buffer, leaving retention_ratio's net_income_attributable
    # placeholder unresolved and falling back to an ungrounded guess.
    CONTEXT_CHUNK_LIMIT = 20

    def __init__(self, vector_store: FinancialVectorStoreManager):
        self.vector_store = vector_store
        self.classifier = FinanceBenchClassifier()
        self.decomposer = QueryDecomposer()
        self.pot_reasoner = ProgramOfThoughtReasoner()
        self.verifier = TriCheckSelfVerifier()
        self.refiner = QueryRefiner()
        self.llm_generator = LLMAnswerGenerator()

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

    def process_query(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        trace_steps = []
        evidence_buffer: List[Dict[str, Any]] = []  # full evidence objects
        evidence_meta: List[Dict[str, Any]] = []    # per-item sub_question metadata
        retrieved_ids = set()

        # ── Step 1: FinanceBench Classification ──
        classification = self.classifier.classify(query)
        answer_mode = classification["answer_mode"]
        complexity = classification["complexity"]
        retrieval_strategy = classification["retrieval_strategy"]

        # Entity alignment: match entity against actual corpus company names.
        # The classifier's OWN clean entity ("General Mills") is kept
        # separately as `clean_entity` for embedding in retrieval QUERY
        # TEXT — production's raw filename-stem company field (e.g.
        # "GENERALMILLS_2022_10K") is what classification["entity"] becomes
        # below, and that's the right form for the entity= soft-filter
        # parameter passed to search() (_company_match_score compares it
        # against doc_company), but a poor form to paste into the query
        # string itself: many FinanceBench filenames glue multi-word
        # company names together with no separator ("GENERALMILLS",
        # "BESTBUY", "KRAFTHEINZ", "AMERICANWATERWORKS"...), so the
        # tokenizer produces one fused token that can never match the two
        # separate words ("general", "mills") the filing's own text
        # actually uses — silently losing all of that token's BM25
        # contribution. Confirmed real case: General Mills' FY2022
        # "Net earnings attributable to General Mills" row scored far
        # lower against a query built from "GENERALMILLS_2022_10K" than
        # the identical query built from "General Mills", pushing an
        # unrelated row into the retrieval results the formula extraction
        # then had to guess from.
        clean_entity = classification["entity"]
        classification["entity"] = self._match_entity_to_corpus(
            classification["entity"], query
        )

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
        if formula_entry:
            query_entity = clean_entity if clean_entity and clean_entity != "company" else classification["entity"]
            sub_questions = self._build_formula_subquestions(
                formula_entry, query_entity, classification["years"]
            )
        elif answer_mode == "NUMERIC":
            sub_questions = self.decomposer.decompose(
                query,
                target_metrics=classification["target_metrics"],
                years=classification["years"],
                entity=classification["entity"],
            )
        else:
            sub_questions = self._build_non_numeric_subquestions(
                query, answer_mode, classification.get("target_metrics")
            )

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
                        # ── Early-stop: skip if total evidence is already large enough ──
                        if len(evidence_buffer) >= self.RETRIEVAL_MAX_TOTAL:
                            trace_steps.append({
                                "step_name": f"Step {sub_q['step']}: Early-Stop",
                                "type": "step_retrieval",
                                "detail": f"Evidence buffer full ({len(evidence_buffer)} chunks). Skipping remaining sub-queries.",
                            })
                            break

                        step_query = sub_q["query"]
                        target_metric = sub_q.get("target_metric")
                        target_year   = sub_q.get("target_year")

                        # ── Check if this (metric, year) is already in the buffer ──
                        def _already_has(metric: str, year: str) -> bool:
                            if not metric or not year:
                                return False
                            for ev in evidence_buffer:
                                c = ev.get("content", "").lower()
                                if year in c and metric.replace("_", " ") in c:
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

                        hits = self.vector_store.search(
                            step_query, top_k=self.RETRIEVAL_TOP_K,
                            exclude_ids=list(retrieved_ids),
                            entity=classification.get("entity"),
                            statement_type_hint=effective_hint,
                        )
                        hits = self._deduplicate_hits(hits)

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
                        for sq in classification["retrieval_queries"]:
                            hits = self.vector_store.search(
                                sq, top_k=self.RETRIEVAL_TOP_K,
                                exclude_ids=list(retrieved_ids),
                                entity=classification.get("entity"),
                                statement_type_hint=statement_type_hint,  # Step 4
                            )
                            for hit in self._deduplicate_hits(hits):
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
                        )
                    )
                    for hit in new_hits:
                        retrieved_ids.add(hit["id"])
                        evidence_buffer.append(hit)
                        info = self._build_evidence_info(hit)
                        iter_trace["retrieved_passages"].append(info)
                        evidence_meta.append(info)

                # ── PoT Execution ──
                # Sort by relevance score so the BEST chunks reach PoT,
                # not just the most recently retrieved ones (RC5 fix)
                raw_window = sorted(
                    evidence_buffer,
                    key=lambda x: x.get("relevance_score", 0.0),
                    reverse=True
                )[:self.CONTEXT_CHUNK_LIMIT]
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

            # If the target metric happens to match a registered formula
            # (e.g. "working_capital" = current_assets - current_liabilities),
            # search for its OWN required_vars instead of the classifier's
            # generic retrieval_queries — those are often just the raw
            # question text plus a fixed boilerplate suffix ("operating
            # margin cost structure segment" for every EXPLANATION
            # question), which can score near zero against a filing that
            # never literally prints the derived metric's own name.
            # Confirmed real case: "Does American Water Works have
            # positive working capital..." searched for "Working Capital"
            # itself, which appears nowhere as a real line item, and
            # retrieved unrelated debt-exhibit boilerplate instead of
            # "Total current assets"/"Total current liabilities" (both of
            # which retrieve cleanly on their own).
            non_numeric_formula = detect_formula(query)
            if non_numeric_formula:
                formula_query_entity = clean_entity if clean_entity and clean_entity != "company" else classification["entity"]
                search_queries = [
                    step["query"] for step in
                    self._build_formula_subquestions(non_numeric_formula, formula_query_entity, classification["years"])
                ]
            else:
                search_queries = classification["retrieval_queries"]
            new_hits = []
            for sq in search_queries:
                new_hits.extend(self.vector_store.search(
                    sq, top_k=self.RETRIEVAL_TOP_K,
                    exclude_ids=list(retrieved_ids),
                    entity=classification.get("entity"),
                    statement_type_hint=statement_type_hint,  # Step 4
                ))
            new_hits = self._deduplicate_hits(new_hits)
            for hit in new_hits:
                retrieved_ids.add(hit["id"])
                evidence_buffer.append(hit)
                info = self._build_evidence_info(hit)
                iter_trace["retrieved_passages"].append(info)
                evidence_meta.append(info)

            pot_res = {
                "code": "", "success": True, "result_value": None,
                "output_log": "", "extracted_variables": {},
                "answer_mode": answer_mode,
            }
            verification_res = self.verifier.verify(query, evidence_buffer[-self.CONTEXT_CHUNK_LIMIT:], pot_res)
            iter_trace["verification"] = verification_res
            trace_steps.append({
                "step_name": "Evidence Retrieval & Analysis",
                "type": "retrieval",
                "data": iter_trace
            })

        # ── Step 4: Final Answer Synthesis ──
        final_context = evidence_buffer[-self.CONTEXT_CHUNK_LIMIT:]
        final_answer = self._synthesize_final_answer(
            query, final_context, pot_res or {}, verification_res or {},
            classification, sub_questions
        )

        return {
            "query": query,
            "complexity": complexity,
            "answer_mode": answer_mode,
            "question_type": classification["question_type"],
            "cognitive_task": classification["cognitive_task"],
            "retrieval_strategy": retrieval_strategy,
            "total_iterations": iteration_count,
            "final_answer": final_answer,
            "result_value": pot_res.get("result_value") if pot_res else None,
            "verification": verification_res,
            "pot_code": pot_res.get("code") if pot_res else "",
            "sandbox_log": pot_res.get("output_log") if pot_res else "",
            "is_degraded_formula": pot_res.get("is_degraded_formula", False) if pot_res else False,
            "degraded_note": pot_res.get("degraded_note", "") if pot_res else "",
            "result_series": pot_res.get("result_series", []) if pot_res else [],
            "result_delta": pot_res.get("result_delta") if pot_res else None,
            "result_direction": pot_res.get("result_direction") if pot_res else None,
            # Return ALL evidence items (with sub_question tag) so the frontend
            # can display every data point that contributed to the calculation
            "evidence_sources": evidence_meta if evidence_meta else [
                self._build_evidence_info(h) for h in evidence_buffer
            ],
            "reasoning_steps": trace_steps,
            "execution_trace": trace_steps,
        }

    # ═══════════════════════════════════════════════════════════════
    # Private Helpers
    # ═══════════════════════════════════════════════════════════════

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
        # Normalise helper: strip year tokens, underscores, hyphens
        def _normalise(s: str) -> str:
            s = _re.sub(r'\b(20|19)\d{2}\b', '', s)   # strip years
            s = _re.sub(r'[_\-]+', ' ', s)             # underscores → spaces
            return s.lower().strip()

        norm_classifier = _normalise(classifier_entity)
        best_company = None
        best_score = 0
        best_year: Optional[str] = None

        # Years the query itself mentions — used only to break ties between
        # multiple filings of the SAME company (see below), since
        # _normalise() deliberately strips year tokens before scoring so
        # "Corning" can match either "CORNING_2021_10K" or
        # "CORNING_2022_10K" equally well in the first place.
        #
        # Uses (?<!\d)...(?!\d) rather than \b: \b only fires at a
        # word/non-word transition, and both "_" and digits count as word
        # characters to regex — so \b2021\b never matches inside
        # "CORNING_2021_10K" (underscore before) or "FY2021" (letter
        # before, no separator) at all, silently defeating year detection
        # in exactly the two places years actually show up here.
        _YEAR_RE = r'(?<!\d)(?:20|19)\d{2}(?!\d)'
        query_years = set(_re.findall(_YEAR_RE, query))

        for uf in self.vector_store.uploaded_files:
            corpus_company = uf.get("company", "")
            if not corpus_company:
                continue
            norm_corpus = _normalise(corpus_company)
            corpus_year_match = _re.search(_YEAR_RE, corpus_company)
            corpus_year = corpus_year_match.group(0) if corpus_year_match else None

            score = 0
            # Score 1: corpus company words appear in query
            corpus_words = [w for w in norm_corpus.split() if len(w) >= 2]
            query_hits = sum(1 for w in corpus_words if w in q_lower)
            score += query_hits * 3

            # Score 2: classifier entity words appear in corpus company name
            if norm_classifier and norm_classifier != "company":
                clf_words = [w for w in norm_classifier.split() if len(w) >= 2]
                clf_hits = sum(1 for w in clf_words if w in norm_corpus)
                score += clf_hits * 2

                # Score 3: exact substring match (highest confidence)
                if norm_classifier in norm_corpus or norm_corpus in norm_classifier:
                    score += 5

            if score > best_score or (
                # Tie-break between multiple filings of the SAME company
                # (identical score, since the company-name portion is
                # identical once years are stripped): prefer whichever
                # filing's OWN year is the one the query actually asks
                # about, falling back to the most recent filing — never an
                # arbitrary "whichever was uploaded first". Confirmed real
                # case: a "how did Corning's tax rate change between
                # FY2021 and FY2022" question, with both CORNING_2021_10K
                # and CORNING_2022_10K loaded, tied at the same score and
                # picked CORNING_2021_10K purely by upload order — a
                # filing that structurally CANNOT contain FY2022 figures
                # at all, since it predates that fiscal year.
                score > 0 and score == best_score and corpus_year and (
                    (corpus_year in query_years and best_year not in query_years)
                    or (corpus_year in query_years and best_year in query_years and corpus_year > best_year)
                    or (not query_years and (best_year is None or corpus_year > best_year))
                )
            ):
                best_score = score
                best_company = corpus_company
                best_year = corpus_year

        if best_company and best_score > 0:
            return best_company

        # Fallback: if classifier returned a real entity name, keep it
        if classifier_entity and classifier_entity != "company":
            return classifier_entity

        # Last resort: most recently uploaded file
        return self.vector_store.uploaded_files[-1].get("company", "company")

    def _infer_entity_from_corpus(self, query: str) -> str:
        """Legacy method — kept for backward compatibility. Delegates to _match_entity_to_corpus."""
        return self._match_entity_to_corpus("company", query)

    def _deduplicate_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for hit in hits:
            hit_id = hit.get("id") or hit.get("content")
            if hit_id in seen:
                continue
            seen.add(hit_id)
            unique.append(hit)
        return unique

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
            # This codebase's alias lists consistently put the Chinese
            # term first (e.g. required_vars["ap_old"] ==
            # ["應付帳款", "accounts payable"]) — aliases[0] would search
            # an all-English 10-K for Chinese text, retrieving nothing
            # relevant (confirmed real case: DPO's own retrieval queries
            # came out as "Amazon 應付帳款 2016" etc., matching zero real
            # content in the English filing). Prefer ASCII/Latin-alphabet
            # aliases — every formula in the library also lists an English
            # variant — falling back to aliases[0] only if none exists.
            #
            # Uses up to the first THREE distinct ASCII aliases, not just
            # one: different companies genuinely use different phrasings
            # for the same line item (e.g. "net income attributable to
            # shareowners" vs. "net earnings attributable to <company>"),
            # and picking only the single first alias means the query only
            # ever matches ONE company's convention. Confirmed real case:
            # General Mills' "Net earnings attributable to General Mills"
            # row scored below an unrelated NCI row when the query only
            # contained "net income attributable to shareowners" (Coca-
            # Cola's own phrasing) — combining alias variants into one
            # query correctly ranks the right row #1 for EITHER company's
            # wording, without needing a second retrieval round-trip.
            # Bumped from 2 to 3: cogs alone has FOUR genuinely common
            # phrasings across real 10-Ks ("cost of goods sold", "cost of
            # sales", "cost of revenue", "cost of products sold"), and
            # with only 2 covered, a company using the 3rd/4th variant
            # (Kraft Heinz: "Cost of products sold") got literally zero
            # _line_item_match_score credit for its own real row while an
            # unrelated OTHER company's row using one of the covered
            # phrasings scored an exact match and outranked it even after
            # the entity-mismatch penalty.
            ascii_aliases = [a for a in aliases if a.isascii()]
            primary_alias = " ".join(dict.fromkeys(ascii_aliases[:3])) if ascii_aliases else (
                aliases[0] if aliases else placeholder
            )
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
        # The retrieval suffix for each template used to be a fixed phrase
        # ("operating margin cost structure segment" for EVERY EXPLANATION
        # question, regardless of what the question actually asked about),
        # which only coincidentally overlaps with what a given question
        # needs. When the classifier already identified specific
        # target_metrics, search for THOSE instead — a general improvement
        # for any qualitative question, not just this one. Confirmed real
        # case: "Does American Water Works have positive working capital"
        # (target_metrics=['working_capital']) retrieved evidence about
        # operating margin and cost structure instead of current assets/
        # liabilities, so the model's answer never stated the actual
        # -$1,561M figure at all — just a generic non-answer.
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
            return llm_answer

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