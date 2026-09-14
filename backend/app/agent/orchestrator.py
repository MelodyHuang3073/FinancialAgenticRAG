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

import re
from typing import Dict, Any, List, Optional

from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.question_classifier import FinanceBenchClassifier, _detect_narrative_topic_query
from app.agent.decomposer import QueryDecomposer
from app.agent.pot_reasoner import ProgramOfThoughtReasoner, _with_implied_trend_year, _get_canonical
from app.agent.verifier import TriCheckSelfVerifier
from app.agent.refiner import QueryRefiner
from app.agent.llm_client import LLMAnswerGenerator, EVIDENCE_PROMPT_CAP
from app.agent.financial_formula_library import detect_formula, get_variable_aliases
from app.tools.hybrid_retriever import is_attribution_query, is_geography_query, is_legal_query


class FinAgentRAGOrchestrator:
    # Chunks per sub-question. Raised from 3 back toward the original 5:
    # a bare alias like "net income" can legitimately match several
    # differently-scoped rows on the SAME income statement ("Net income
    # from continuing operations", "Consolidated net income", "Net income
    # attributable to shareowners of ..."), and the one GAAP convention
    # actually wants can rank #4-#5 for a generic query even though it's
    # sitting cleanly in a real evidence chunk — confirmed real case:
    # Coca-Cola FY2017 net income for ROA, where top_k=3 never retrieved
    # any of the pages carrying the correctly-labeled "attributable to
    # shareowners" row at all, leaving only mis-scoped rows to choose
    # from no matter how good the downstream tie-break logic is.
    RETRIEVAL_TOP_K = 5
    # A wider top_k used ONLY for the non-numeric/narrative retrieval path
    # (prefer_narrative=True) below -- a genuinely qualitative question
    # ("who are Boeing's primary customers", "what drove JnJ's gross
    # margin change", "is 3M capital-intensive") only ever issues 1-2
    # search queries total (a topic query plus the bare question text),
    # nowhere near RETRIEVAL_MAX_TOTAL/CONTEXT_CHUNK_LIMIT's headroom, so
    # there's no accumulation-cap risk in widening just this path the way
    # there would be for a composite NUMERIC formula's 8-placeholder fan-
    # out. Confirmed real, repeated pattern across three separate
    # questions: the genuinely correct passage consistently ranked
    # #8-#24 -- comfortably inside a wider window, but always just
    # outside the narrower RETRIEVAL_TOP_K=5 this shared constant used to
    # apply everywhere -- while a topically-adjacent but wrong passage
    # (a same-company statistic about a DIFFERENT metric, an unrelated
    # accounting-policy note, a different business segment's own
    # sub-table) narrowly won the 5 available slots instead. Confirmed
    # cases: Boeing's "primary customers" question retrieved a real
    # "non-U.S. customers = 41% of revenue" sentence (rank ~1, a true
    # statistic about a DIFFERENT metric) while the actual gold-relevant
    # "U.S. government = 40% of revenue" sentence (rank ~8-9) never made
    # the cut; Johnson & Johnson's "what drove gross margin change"
    # question needed a passage with zero direct "gross margin" wording
    # at all (rank ~2-24 depending on phrasing) that always lost to
    # shorter, more topically-generic prose.
    RETRIEVAL_TOP_K_NARRATIVE = 15
    # Hard ceiling on total evidence buffer size. Must comfortably fit every
    # sub-query a single formula's own required_vars can generate (top_k=5
    # each) — a composite formula like cash_conversion_cycle needs 8
    # placeholders (cogs, revenue, inv_old/new, ar_old/new, ap_old/new), so
    # 8*5=40 sub-results. The old value of 15 silently cut off mid-formula,
    # dropping ap_old/ap_new before they were ever retrieved (confirmed
    # real case: General Mills FY2019 CCC — accounts payable never entered
    # the evidence buffer at all, and the whole computation fell back to a
    # generic, ungrounded LLM guess). Sized with headroom above today's
    # largest formula rather than pinned to exactly 40, so the next
    # formula with one or two more placeholders doesn't repeat this.
    RETRIEVAL_MAX_TOTAL = 45
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
        """
        The CONTEXT_CHUNK_LIMIT-item slice of evidence_buffer that actually
        reaches verification/answer synthesis -- sorted by relevance_score
        (highest first) and THEN capped, never a plain `[-N:]` insertion-
        order slice. evidence_buffer accumulates hits from every retrieval
        sub-query in the order those sub-queries happened to run, not in
        score order -- a plain `[-CONTEXT_CHUNK_LIMIT:]` slice keeps
        whichever sub-queries ran LAST, silently dropping a genuinely
        top-scoring item from an EARLIER sub-query the instant later
        sub-queries together contribute CONTEXT_CHUNK_LIMIT-or-more items
        of their own -- regardless of how low those later items scored.
        Confirmed real case: Amcor's "what are major acquisitions" question
        (HYBRID strategy, 3 sub-queries) retrieved its own "Note 5 -
        Acquisitions and Divestitures" note as the #1-scoring passage
        overall (from the FIRST sub-query), but the 2nd and 3rd sub-queries
        together still contributed CONTEXT_CHUNK_LIMIT-or-more MORE items
        afterward, so the plain insertion-order slice dropped it entirely
        -- the model then denied the note was ever supplied, when it had
        simply never been shown it despite retrieval finding it perfectly.
        """
        return sorted(
            evidence_buffer, key=lambda item: item.get("relevance_score") or 0, reverse=True
        )[:self.CONTEXT_CHUNK_LIMIT]

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
        # A question asking whether a metric is "improving"/"declining" as
        # of year Y implies a comparison against year Y-1, even when the
        # classifier's own year extraction names only Y -- without this,
        # RETRIEVAL never fetches the prior year at all, so by the time
        # pot_reasoner.py's own _with_implied_trend_year (used at
        # CALCULATION time) tries to compare two years, there is no prior-
        # year evidence in the buffer to find, and the formula silently
        # falls back to "0.0 -- not a useful metric" instead of the real
        # trend answer. Confirmed real case: "Does Boeing have an
        # improving gross margin profile as of FY2022?" -- classification
        # extracted only ["2022"], retrieval fetched gross_profit(2022)
        # alone, and the answer came back 0.0 even though Boeing's real
        # FY2021->FY2022 gross margin trend (4.8% -> 5.3%) is a clean,
        # directly answerable "Yes".
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

        # Both NUMERIC sub-question builders above (formula-guided and
        # LLM-decomposed) search purely on the target line item's OWN
        # alias vocabulary (e.g. "dividends paid to common shareholders"),
        # which reliably finds the STRUCTURED statement row (a dollar
        # total) but never a filing's plain-English narrative sentence
        # stating the same fact in different words (e.g. Item 5's "we
        # maintained an annual dividend of $0.01 per share throughout
        # 2022") -- a prose detail FinanceBench gold answers often want
        # alongside the total, that alias-matching alone will never
        # surface within RETRIEVAL_TOP_K. Reuses the same narrative-topic
        # vocabulary bridge the non-numeric path already relies on
        # (_detect_narrative_topic_query) as one extra retrieval step --
        # additive only, appended after whichever NUMERIC path already
        # ran, never replacing its steps. A no-op for the non-numeric
        # `else` branch above, which already gets topic-aware queries via
        # classification["retrieval_queries"]. Confirmed real case: "Has
        # MGM Resorts paid dividends to common shareholders in FY2022?"
        # retrieved the correct $4,048K cash-flow total but never the
        # $0.01/share sentence, because "dividends paid to common
        # shareholders" has almost no vocabulary overlap with "annual
        # dividend...per share".
        if answer_mode == "NUMERIC":
            topic_query = _detect_narrative_topic_query(query.lower())
            if topic_query:
                sub_questions.append({
                    "step": len(sub_questions) + 1,
                    "type": "retrieval",
                    "query": f"{query_entity} {topic_query}".strip(),
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
                        # Restricted to table_row evidence: only a structured
                        # "Line Item: X | year: value" row's bare co-occurrence
                        # of the metric name and year genuinely means a usable
                        # value was already captured. A text_note chunk merely
                        # CONTAINING both words somewhere is no such guarantee
                        # -- prose routinely mentions a year and a metric name
                        # in unrelated sentences, and this got dramatically
                        # more likely once chunk_size grew from 800 to 3000
                        # chars (a single chunk covers much more of a page's
                        # MD&A prose). Confirmed real case: Activision
                        # Blizzard's capex query at 3000-char chunks pulled in
                        # a text_note chunk that happened to also say "2017"
                        # and "revenue" incidentally, so ALL THREE of the
                        # question's own separate revenue(2017/2018/2019)
                        # sub-queries got silently skipped as "already
                        # retrieved" -- no revenue evidence was ever actually
                        # fetched, and the calculation fell back to 0.0.
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

                        hits = self.vector_store.search(
                            step_query, top_k=self.RETRIEVAL_TOP_K,
                            exclude_ids=list(retrieved_ids),
                            entity=classification.get("entity"),
                            statement_type_hint=effective_hint,
                            query_years=classification.get("years"),
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
                                query_years=classification.get("years"),
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
                            query_years=classification.get("years"),
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
            # No registered formula matched at all — this is reached ONLY
            # by genuinely qualitative/narrative questions (every formula-
            # backed non-numeric question, e.g. working_capital/inventory_
            # turnover/effective_tax_rate, took the `if` branch above
            # instead), so it's safe to bias ranking toward prose content
            # here without touching anything a numeric/formula answer
            # depends on — see hybrid_retriever.search()'s prefer_narrative
            # docstring for the confirmed real case this fixes.
            prefer_narrative = non_numeric_formula is None
            # Evaluated against the ORIGINAL question, not each individual
            # sub-query below (a keyword-stuffed sub-query like "AMD
            # Revenue Net Revenue" never repeats "what drove" phrasing even
            # when the overall question plainly is an attribution question)
            # -- see hybrid_retriever.is_attribution_query's docstring.
            is_attribution = is_attribution_query(query)
            is_geography = is_geography_query(query)
            is_legal = is_legal_query(query)
            new_hits = []
            for sq in search_queries:
                new_hits.extend(self.vector_store.search(
                    # This whole retrieval block only ever runs for the
                    # non-numeric answer_mode branch (ASSESSMENT/
                    # EXPLANATION/EXCLUSION) -- always uses the wider
                    # narrative top_k here regardless of prefer_narrative
                    # (which only controls search()'s internal SCORING
                    # boost, not how many candidates get through at all).
                    # Confirmed real case beyond the pure-narrative ones
                    # RETRIEVAL_TOP_K_NARRATIVE was first added for: 3M's
                    # FY2022 "is 3M capital-intensive" question (a
                    # formula-backed non-numeric question, so
                    # prefer_narrative=False) needed its real "Net sales"
                    # row, which ranked #6 for its own dedicated revenue
                    # sub-query -- just one past the base
                    # RETRIEVAL_TOP_K=5 cutoff, with several unrelated
                    # accounting-policy notes from the SAME filing
                    # occupying the 5 available slots instead.
                    sq, top_k=self.RETRIEVAL_TOP_K_NARRATIVE,
                    exclude_ids=list(retrieved_ids),
                    entity=classification.get("entity"),
                    statement_type_hint=statement_type_hint,  # Step 4
                    prefer_narrative=prefer_narrative,
                    is_attribution=is_attribution,
                    is_geography=is_geography,
                    is_legal=is_legal,
                    query_years=classification.get("years"),
                ))
            new_hits = self._deduplicate_hits(new_hits)
            for hit in new_hits:
                retrieved_ids.add(hit["id"])
                evidence_buffer.append(hit)
                info = self._build_evidence_info(hit)
                iter_trace["retrieved_passages"].append(info)
                evidence_meta.append(info)

            # An EXPLANATION/ASSESSMENT question ("does X have positive
            # working capital", "did Y's margin improve") still turns on a
            # real number comparison whenever it matches a registered
            # formula — it just ALSO needs qualitative framing in the
            # final text. This used to always return a stub pot_res with
            # no code and result_value=None, meaning the LLM derived
            # every number itself straight from raw evidence text with
            # zero sandbox grounding — exactly the failure mode the
            # "trust the sandbox" instruction in llm_client.py exists to
            # prevent elsewhere, just never reached here at all. Confirmed
            # real case: American Water Works' FY2022 working-capital
            # question got the right numbers this time purely by LLM
            # luck, with no Python trace to show for it or to have caught
            # it if the LLM had been wrong.
            context_window = []
            for ev in sorted(evidence_buffer, key=lambda x: x.get("relevance_score", 0.0), reverse=True)[:self.CONTEXT_CHUNK_LIMIT]:
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
            "result_unit": pot_res.get("result_unit", "") if pot_res else "",
            # Return ONLY the subset of evidence that actually reached the
            # LLM's prompt (see llm_client.generate_answer's own sort +
            # EVIDENCE_PROMPT_CAP slice, applied here identically to
            # final_context -- the SAME list generate_answer received),
            # not every candidate retrieval ever pulled in. evidence_meta/
            # evidence_buffer can hold up to RETRIEVAL_MAX_TOTAL=45 items
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
                )[:EVIDENCE_PROMPT_CAP]
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
        #
        # \b only fires at a word/non-word transition, and both "_" and
        # digits count as word characters to regex -- so \b2022\b never
        # matches inside "3M_2022_10K" (underscore on both sides) at
        # all, silently leaving the year token in norm_corpus. Every
        # OTHER same-year company's filing then ALSO keeps its own
        # literal year token, and a query merely mentioning that year
        # (as a two-year comparison like "between 2022 and 2021"
        # routinely does) scores a spurious match against EVERY one of
        # them equally via the "corpus words appear in query" signal
        # below -- ties broken by nothing but iteration order once the
        # real classifier-entity-based scores (which correctly stay at
        # 0 for a company genuinely not yet in uploaded_files) can't
        # break them. (?<!\d)...(?!\d) checks for a DIGIT boundary
        # instead, matching the same fix already applied to this
        # method's own query_years extraction just below (_YEAR_RE) --
        # underscore isn't a digit, so this correctly strips the year
        # out of "3M_2022_10K" too. Confirmed real case: "Has Verizon
        # increased its debt...between 2022 and the 2021 fiscal
        # period?" resolved matched-entity to "3M_2022_10K" -- the
        # first 2022-year filing in upload order -- because "2022"
        # survived normalisation on every 2022 filing's own company
        # string, tying all of them at the same score.
        def _normalise(s: str) -> str:
            s = _re.sub(r'(?<!\d)(?:20|19)\d{2}(?!\d)', '', s)   # strip years
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

    # Retrieval-only synonym terms, appended to a placeholder's own search
    # query TEXT but deliberately NEVER fed into required_vars/extraction
    # (pot_reasoner._extract_formula_guided() still scores candidate rows
    # against the formula library's own unmodified alias lists). A real
    # filing routinely discusses what moved a DERIVED metric entirely in
    # terms of its own COST-side line ("cost of products sold increased
    # as a percent to sales driven by...") without ever literally saying
    # "gross profit"/"gross margin" in that passage -- a direct accounting
    # equivalence (gross margin moves inversely to cost-of-sales-as-%-of-
    # revenue), but with zero real word overlap against a query built
    # purely from gross_profit's own aliases. Widening the retrieval query
    # alone (not the extraction-time alias list) is deliberate: adding
    # "cost of products sold" etc. AS an extraction alias for gross_profit
    # would make _extract_formula_guided() treat a COGS row's own VALUE as
    # if it WERE gross profit for a NUMERIC gross-margin calculation --
    # wrong by definition (COGS is revenue MINUS gross profit, not gross
    # profit itself) and a correctness regression risk for every
    # already-passing NUMERIC gross_margin question. This dict only ever
    # widens what evidence gets RETRIEVED for a qualitative/EXPLANATION-
    # mode "what drove X change" question to read from; it changes
    # nothing about what value gets treated as gross_profit. Confirmed
    # real case: Johnson & Johnson's FY2022 "what drove gross margin
    # change" question -- the filing's own driver bullets ("One-time
    # COVID-19 vaccine manufacturing exit related costs...") sit entirely
    # under a "Cost of products sold... driven by:" heading with the
    # words "gross"/"margin"/"profit" nowhere in it.
    _RETRIEVAL_SYNONYM_TERMS: Dict[str, List[str]] = {
        "gross_profit": ["Cost of Products Sold", "Cost of Goods Sold", "Cost of Sales", "COGS"],
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
        #
        # NOTE: this function's output is ONLY used for the "Query
        # Decomposition" trace display and the (currently unused by the
        # LLM prompt) sub_questions parameter — NOT for actual retrieval.
        # The real non-numeric retrieval query, whenever no formula
        # matches the question, comes from
        # classification["retrieval_queries"] (built by
        # FinanceBenchClassifier._build_retrieval_queries()), a
        # completely separate code path. A general query-vocabulary fix
        # for narrative questions (legal proceedings, dividends,
        # restructuring, etc.) belongs there, not here — confirmed by
        # tracing an actual failing case (Boeing legal-battles question)
        # end to end: this function's suffix showed up correctly in the
        # trace, but the retrieved evidence was completely unaffected by
        # it.
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