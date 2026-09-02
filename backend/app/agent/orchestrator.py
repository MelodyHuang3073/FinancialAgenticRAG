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

from typing import Dict, Any, List

from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.question_classifier import FinanceBenchClassifier
from app.agent.decomposer import QueryDecomposer
from app.agent.pot_reasoner import ProgramOfThoughtReasoner
from app.agent.verifier import TriCheckSelfVerifier
from app.agent.refiner import QueryRefiner
from app.agent.llm_client import LLMAnswerGenerator


class FinAgentRAGOrchestrator:
    RETRIEVAL_TOP_K = 3          # chunks per sub-question (was 5)
    RETRIEVAL_MAX_TOTAL = 15     # hard ceiling on total evidence buffer size
    CONTEXT_CHUNK_LIMIT = 8

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

        # Entity alignment: match entity against actual corpus company names
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
        if answer_mode == "NUMERIC":
            sub_questions = self.decomposer.decompose(
                query,
                target_metrics=classification["target_metrics"],
                years=classification["years"],
                entity=classification["entity"],
            )
        else:
            sub_questions = self._build_non_numeric_subquestions(query, answer_mode)

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
                pot_res = self.pot_reasoner.generate_and_execute(query, context_window)
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

        for uf in self.vector_store.uploaded_files:
            corpus_company = uf.get("company", "")
            if not corpus_company:
                continue
            norm_corpus = _normalise(corpus_company)

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

            if score > best_score:
                best_score = score
                best_company = corpus_company

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

    def _build_non_numeric_subquestions(self, query: str, answer_mode: str) -> List[Dict[str, str]]:
        templates = {
            "ASSESSMENT": [
                {"step": 1, "type": "retrieval", "query": f"{query} capital expenditure assets depreciation"},
                {"step": 2, "type": "analysis", "query": "Assess the metric's suitability"},
            ],
            "EXCLUSION": [
                {"step": 1, "type": "retrieval", "query": f"{query} segment revenue organic growth acquisition"},
                {"step": 2, "type": "analysis", "query": "Isolate organic vs M&A impact"},
            ],
            "EXPLANATION": [
                {"step": 1, "type": "retrieval", "query": f"{query} operating margin cost structure segment"},
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