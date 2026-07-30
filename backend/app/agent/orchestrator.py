"""
FinAgent-RAG Orchestrator (重構版)

修復清單:
- Bug #2: _build_search_queries 現在在後續迭代中正確使用 refined query
- Bug #7: 回答精簡為 2-3 行核心結論 + 可展開細節
- FinanceBench: 整合 question_classifier 取代舊 router + financial_knowledge
"""

from typing import Dict, Any, List

from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.question_classifier import FinanceBenchClassifier
from app.agent.decomposer import QueryDecomposer
from app.agent.pot_reasoner import ProgramOfThoughtReasoner
from app.agent.verifier import TriCheckSelfVerifier
from app.agent.refiner import QueryRefiner
from app.agent.llm_client import LLMAnswerGenerator
from app.agent.external_retriever import ExternalFinanceRetriever


class FinAgentRAGOrchestrator:
    RETRIEVAL_TOP_K = 5
    CONTEXT_CHUNK_LIMIT = 8

    def __init__(self, vector_store: FinancialVectorStoreManager):
        self.vector_store = vector_store
        self.classifier = FinanceBenchClassifier()
        self.decomposer = QueryDecomposer()
        self.pot_reasoner = ProgramOfThoughtReasoner()
        self.verifier = TriCheckSelfVerifier()
        self.refiner = QueryRefiner()
        self.llm_generator = LLMAnswerGenerator()
        self.external_retriever = ExternalFinanceRetriever()

    def process_query(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        trace_steps = []
        evidence_buffer: List[Dict[str, Any]] = []
        retrieved_ids = set()

        # ── Step 1: FinanceBench Classification ──
        classification = self.classifier.classify(query)
        answer_mode = classification["answer_mode"]
        complexity = classification["complexity"]
        retrieval_strategy = classification["retrieval_strategy"]

        # ── Entity injection: if classifier returned generic 'company',
        #    try to infer the real company from uploaded files in the corpus ──
        if classification["entity"] == "company":
            inferred = self._infer_entity_from_corpus(query)
            classification["entity"] = inferred

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

                    for sub_q in retrieval_steps:
                        step_query = sub_q["query"]
                        hits = self.vector_store.search(
                            step_query, top_k=self.RETRIEVAL_TOP_K,
                            exclude_ids=list(retrieved_ids)
                        )
                        hits = self._deduplicate_hits(hits)

                        step_hit_infos = []
                        for hit in hits:
                            retrieved_ids.add(hit["id"])
                            evidence_buffer.append(hit)
                            info = {
                                "id": hit["id"],
                                "table_name": hit.get("table_name", ""),
                                "company": hit.get("company", ""),
                                "period": hit.get("period", ""),
                                "relevance_score": hit.get("relevance_score", 0.0),
                                "snippet": hit.get("content", "")[:120] + "...",
                                "sub_question": step_query,
                            }
                            iter_trace["retrieved_passages"].append(info)
                            step_hit_infos.append(info)

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
                                f"Query: '{step_query}' → "
                                f"{len(hits)} passage(s) retrieved"
                            ),
                        })

                    # ── Fallback: if zero evidence collected, use classifier retrieval_queries ──
                    if not evidence_buffer:
                        for sq in classification["retrieval_queries"]:
                            hits = self.vector_store.search(
                                sq, top_k=self.RETRIEVAL_TOP_K,
                                exclude_ids=list(retrieved_ids)
                            )
                            for hit in self._deduplicate_hits(hits):
                                retrieved_ids.add(hit["id"])
                                evidence_buffer.append(hit)
                                iter_trace["retrieved_passages"].append({
                                    "id": hit["id"],
                                    "table_name": hit.get("table_name", ""),
                                    "company": hit.get("company", ""),
                                    "period": hit.get("period", ""),
                                    "relevance_score": hit.get("relevance_score", 0.0),
                                    "snippet": hit.get("content", "")[:120] + "...",
                                })

                else:
                    # ── Subsequent iterations: use refined query ──
                    new_hits = self._deduplicate_hits(
                        self.vector_store.search(
                            current_query, top_k=self.RETRIEVAL_TOP_K,
                            exclude_ids=list(retrieved_ids)
                        )
                    )
                    for hit in new_hits:
                        retrieved_ids.add(hit["id"])
                        evidence_buffer.append(hit)
                        iter_trace["retrieved_passages"].append({
                            "id": hit["id"],
                            "table_name": hit.get("table_name", ""),
                            "company": hit.get("company", ""),
                            "period": hit.get("period", ""),
                            "relevance_score": hit.get("relevance_score", 0.0),
                            "snippet": hit.get("content", "")[:120] + "...",
                        })

                # ── PoT Execution ──
                context_window = evidence_buffer[-self.CONTEXT_CHUNK_LIMIT:]
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
                    exclude_ids=list(retrieved_ids)
                ))
            new_hits = self._deduplicate_hits(new_hits)
            for hit in new_hits:
                retrieved_ids.add(hit["id"])
                evidence_buffer.append(hit)
                iter_trace["retrieved_passages"].append({
                    "id": hit["id"],
                    "table_name": hit.get("table_name", ""),
                    "company": hit.get("company", ""),
                    "period": hit.get("period", ""),
                    "relevance_score": hit.get("relevance_score", 0.0),
                    "snippet": hit.get("content", "")[:120] + "..."
                })

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
            "evidence_sources": evidence_buffer[-5:],
            "reasoning_steps": trace_steps,
            "execution_trace": trace_steps,
        }

    # ═══════════════════════════════════════════════════════════════
    # Private Helpers
    # ═══════════════════════════════════════════════════════════════

    def _infer_entity_from_corpus(self, query: str) -> str:
        """
        When the classifier can't identify the company (returns 'company'),
        try to match uploaded file company names against the query text.
        """
        import re as _re
        if not self.vector_store.uploaded_files:
            return "company"

        q_lower = query.lower()

        for uf in self.vector_store.uploaded_files:
            company = uf.get("company", "")
            if not company:
                continue
            # Split on common separators and check if any meaningful part appears in query
            parts = _re.split(r'[_\-\s]+', company.lower())
            significant = [p for p in parts if len(p) >= 2 and not _re.match(r'^\d{4}$', p)]
            if any(part in q_lower for part in significant):
                return company

        # Default to the most recently uploaded file's company
        return self.vector_store.uploaded_files[-1].get("company", "company")

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