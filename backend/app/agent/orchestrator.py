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
    # nowhere near CONTEXT_CHUNK_LIMIT's headroom, so
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
    # Cap applied right before evidence reaches PoT/the
    # LLM (sorted by relevance_score, top N kept) — raising
    # A tight window here can truncate a lower-but-still-correct-scoring row
    # out of the final window even though it was retrieved.
    # Confirmed real case: General Mills' own real "Net earnings
    # attributable to General Mills" row (score ~43) ranked #3 for its
    # own retrieval query — comfortably inside the retrieved set —
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

        # ── Step 1: FinanceBench Classification ──
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
        classification["entity"] = _entity_override or self._match_entity_to_corpus(
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
                        # Every sub-question is ALWAYS retrieved (no early-stop gate):
                        # what finally reaches PoT / the LLM is decided afterwards by
                        # ranking all collected evidence together (CONTEXT_CHUNK_LIMIT,
                        # then EVIDENCE_PROMPT_CAP).
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

                        # A narrative_topic-sourced sub-question (see
                        # _detect_narrative_topic_query) targets a PROSE/
                        # narrative topic bridge, not a structured line
                        # item -- the alias-matched NUMERIC sub-queries
                        # around it already cover the structured value.
                        # The narrower RETRIEVAL_TOP_K=5 used for every
                        # OTHER sub-question here is tuned for a specific,
                        # well-aliased line item that reliably ranks near
                        # the top; a bridge query's TARGET is often a
                        # terse, generic-labeled row (e.g. a filer's own
                        # "Results of Operations" table literally printing
                        # just "Total | 2022: 1.3%") with very little
                        # distinctive vocabulary for BM25 to rank highly on
                        # -- confirmed real case: JnJ's own stated revenue
                        # %-change row ranked outside the top 8 for both
                        # this bridge query AND the plain "Total revenue
                        # 2022" query, so top_k=5 missed it entirely even
                        # though a wider net would have caught it.
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
                # _build_formula_subquestions() only ever emits ONE query
                # PER PLACEHOLDER (e.g. "3M op income 2022", "3M revenue
                # 2022") -- unlike classification["retrieval_queries"]
                # below (which always appends the bare question text as a
                # fallback), it has no equivalent, so a "what DROVE X
                # change" attribution question whose metric X happens to
                # match a registered ratio formula NEVER actually searches
                # for the causal narrative itself -- only the bare numbers
                # that go into computing X. Confirmed real case: 3M's own
                # "what drove operating margin change" question matched
                # the operating_margin formula and only ever searched "3M
                # op income 2022"/"3M revenue 2022", never surfacing the
                # MD&A page naming the real drivers (Combat Arms Earplugs
                # litigation, PFAS manufacturing exit costs) at all.
                if is_attribution:
                    search_queries.append(query)
            else:
                search_queries = classification["retrieval_queries"]
            # No registered formula matched at all — this is reached ONLY
            # by genuinely qualitative/narrative questions (every formula-
            # backed non-numeric question, e.g. working_capital/inventory_
            # turnover/effective_tax_rate, took the `if` branch above
            # instead), so it's safe to bias ranking toward prose content
            # here without touching anything a numeric/formula answer
            # depends on — see hybrid_retriever.search()'s prefer_narrative
            # docstring for the confirmed real case this fixes. Also
            # widened to attribution questions even when a formula DID
            # match, for the same reason the bare query got appended just
            # above -- the narrative-content and causal-language boosts
            # (see hybrid_retriever.search()'s own prefer_narrative/
            # attribution_active handling) only ever activate together
            # under prefer_narrative=True, so without this the bare query
            # just appended would compete on equal footing with dense
            # table rows and rarely win anyway.
            # "What was the LARGEST liability in the Balance Sheet?" is answered by
            # comparing the statement's own rows, so the narrative-prose boost must
            # not demote those rows (AmEx: the balance-sheet rows ranked 4th/6th
            # per query at ~90 but fell below the top-16 cut behind prose chunks
            # scoring 100-200; the right answer came from a table on another page).
            is_statement_item_question = bool(
                re.search(r"\b(largest|biggest|highest|smallest|lowest)\b[^?]{0,60}\b(liabilit\w*|asset\w*|expense\w*|equity)\b", query, re.IGNORECASE)
                and re.search(r"balance sheet|income statement|cash flow statement", query, re.IGNORECASE)
            )
            prefer_narrative = (non_numeric_formula is None or is_attribution) and not is_statement_item_question
            is_geography = is_geography_query(query)
            is_legal = is_legal_query(query)
            new_hits = []
            for sq_idx, sq in enumerate(search_queries):
                new_hits.extend(self._tag_subquery(sq_idx, self.vector_store.search(
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
                )))
            new_hits = self._deduplicate_hits(new_hits, entity=classification.get("entity"))
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
            # The sandbox's "result is not reliable" placeholder (result = 0.0
            # printed with that warning when no retrieved data matched the
            # question) is not a computed answer; sending its bare 0.0 made
            # the frontend headline "0" as the final calculation result
            # (11+ real cases: 3M dividend trend, Amcor adjusted EBITDA, Best
            # Buy cash drop / store count, Boeing production rates and tax
            # rate, MGM EBITDAR region, ...). sandbox_log below still carries
            # the warning text.
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
        # "excluding the impact of FX, passthrough costs and one-off items" is the
        # earnings release's "comparable constant currency" bridge (Amcor QA 29
        # routed to the 10-K, which has no such table)
        r"|constant[- ]currency|pass-?through|one-?off"
    )
    #: A question asking what is EXPECTED for year Y is answered by a filing
    #: dated before Y (it is a forecast), so year Y-1 filings are the match.
    _FORWARD_LOOKING_CUE_RE = re.compile(
        r"\bexpected?\s+to\b|\bexpects?\b|\bguidance\b|\boutlook\b|\bforecast\w*|\banticipat\w+"
    )
    #: A question about an ongoing separation/spin-off/divestiture cost, with
    #: no year at all named (so the plain-annual-10-K default tier below has
    #: nothing to override it), needs the MOST RECENT filing -- the running
    #: cumulative "percent incurred so far" figure is stale in an older
    #: annual 10-K. Confirmed real case: "How much does Pfizer expect to pay
    #: to spin off Upjohn in the future?" resolved to PFIZER_2021_10K (its
    #: own "~75% incurred through December 31, 2021" sentence) instead of
    #: Pfizer_2023Q2_10Q (the more complete "~90% incurred through Q2 2023"),
    #: purely because the tier-3 "prefer the plain annual 10-K" default below
    #: has no year-based signal to lose to when the query names none at all.
    _SEPARATION_TOPIC_RE = re.compile(
        r"\bseparat\w+|\bspin[- ]?off\w*|\bdivest\w*", re.IGNORECASE
    )

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

        # Quarter the QUERY itself names (e.g. "In 2022 Q2, which of JPM's
        # segments...", "...net revenue in 2021 Q1?") -- a bare word-
        # boundary "q1"-"q4" token, independent of adjacency to a year.
        # Used only for the tie-break below; unrelated to query_years.
        _query_quarter_m = _re.search(r'\bq([1-4])\b', q_lower)
        query_quarter = f"q{_query_quarter_m.group(1)}" if _query_quarter_m else None
        # Forward-looking question about year Y ("is X expected to ... in FY2023?"):
        # the source is a filing from year Y-1, so that year outranks Y itself.
        forward_year_prior = None
        if query_years and self._FORWARD_LOOKING_CUE_RE.search(q_lower) and query_quarter is None:
            forward_year_prior = str(int(max(query_years)) - 1)

        # Quarter-aware period extraction for the SAME purpose the bare
        # _YEAR_RE above already served (tie-breaking between multiple
        # filings of the same company) — now also captures an adjacent
        # "Q1"-"Q4" suffix (e.g. "MGMRESORTS_2022Q4_EARNINGS" -> year
        # "2022", quarter "q4"), which the bare year-only regex collapsed
        # to plain "2022", indistinguishable from "MGMRESORTS_2022_10K".
        # Confirmed real case: "What was MGM's interest coverage ratio
        # using FY2022 Adjusted EBIT...?" — no quarter word anywhere in
        # the question itself, so the old tie-break's ONLY signal (bare
        # year, identical for both filings) couldn't distinguish them and
        # silently fell back to whichever was inserted first into
        # DOC_TO_FILE, resolving to the wrong document (MGMRESORTS_2022_
        # 10K instead of the intended MGMRESORTS_2022Q4_EARNINGS) and
        # extracting nonsense values from unrelated line items.
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

            # Tie-break key between multiple filings of the SAME company
            # (identical score, since the company-name portion is identical
            # once years are stripped) -- compared as a tuple, higher wins:
            #   1) does this filing's quarter match one the query itself
            #      names (e.g. "Q2 2023")? Highest-confidence signal when
            #      present -- this is the ONLY tier that uses corpus_quarter
            #      at all. Fixes "What was MGM's interest coverage ratio
            #      using FY2022 Adjusted EBIT...?" IF the question had named
            #      a quarter, and robustly (not by accident) fixes JPM's own
            #      "In 2022 Q2, which of JPM's segments..." shape.
            #   2) is this filing's bare year one the query mentions at all?
            #   3) does this filing have NO quarter suffix at all (i.e. is
            #      it a plain annual 10-K rather than a 10-Q/earnings-
            #      release/8-K)? Preferred as the safer default source when
            #      nothing else disambiguates, since it's far more likely to
            #      be a complete, self-contained annual filing than a
            #      quarterly document is.
            #   4) the bare year itself, as the final "prefer more recent"
            #      fallback among same-specificity candidates.
            #
            # Tier 3 exists because bare-year recency ALONE (the entire
            # fallback prior to today) is no longer a safe proxy for "most
            # complete/appropriate filing" now that the corpus can contain
            # quarterly documents dated LATER in calendar terms than an
            # older but more complete annual 10-K. Confirmed real
            # regression this fixes: "Are Best Buy's gross margins
            # historically consistent...?" (no year or quarter named at
            # all) resolved to BESTBUY_2024Q2_10Q -- purely because "2024"
            # sorts after "2023" -- instead of BESTBUY_2023_10K, which is
            # what actually carries the multi-year income-statement trend
            # this question needs; a 10-Q fragment doesn't.
            #
            # Tier 1 remains the ONLY tier that uses corpus_quarter to
            # PREFER a quarter-suffixed filing (when the query itself names
            # that exact quarter, e.g. JPM's "In 2022 Q2, which of JPM's
            # segments..."). An earlier version of this fix used
            # corpus_quarter more broadly, as a blanket "prefer the more
            # specific filing" default tiebreaker -- that caused a separate
            # real regression (JnJ's "Roughly how many times has JnJ sold
            # its inventory in FY2022?", resolved to JOHNSON_JOHNSON_2022Q4_
            # EARNINGS, a press release with no balance sheet at all,
            # instead of the 10-K that actually has inventory data) and was
            # removed for the same reason tier 3 now exists: whether the
            # MORE or LESS specific filing is correct depends on what kind
            # of data the question needs, which isn't something this
            # function can infer -- so the safe default is the plain annual
            # filing, not the quarterly one, absent an explicit signal.
            tie_key = (
                _date_key(corpus_company),
                1 if (query_quarter is not None and corpus_quarter == query_quarter) else 0,
                (2 if (forward_year_prior and corpus_year == forward_year_prior)
                 else 1 if corpus_year in query_years else 0),
                # See _SEPARATION_TOPIC_RE's docstring: overrides tier 3's
                # "prefer the plain annual 10-K" default, but ONLY for this
                # narrow forward-looking-separation-cost shape, so it cannot
                # fire on an unrelated no-year question (e.g. Best Buy's
                # gross-margin-consistency question, which tier 3 exists
                # for) and never collides with it.
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
            # Several filings of the same company that tie on EVERY signal above
            # (e.g. two 8-Ks of one year) are told apart by which one actually
            # talks about what the question asks: count, per tied filing, the
            # passages containing at least two of the question's distinctive
            # words. Confirmed real case: "Does Foot Locker's new CEO have
            # previous CEO experience...?" chose the May 8-K (shareholder vote
            # results) instead of the August 8-K that announces the CEO change.
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
            # Regional/segment BREAKDOWN questions ("which region had the
            # worst topline...", "how did US sales growth compare to
            # international...") are answered from the simplified regional
            # supplemental tables a company's own EARNINGS RELEASE carries;
            # the formal 10-K reports a different (reportable-segment)
            # cut. Each question needs exactly ONE file, so among
            # same-company, same-named-year candidates that TIE, an
            # earnings-release file wins for this question shape only.
            # BM25 content mass could not make this call (the larger 10-K
            # always scores higher), so it is keyed on the question shape.
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
        """Company part of a corpus doc name: "JOHNSON_JOHNSON_2022Q4_EARNINGS"
        -> "JOHNSONJOHNSON"; "" when the name has no year segment (e.g. the
        unresolved placeholder "company")."""
        m = re.match(r"^(.*?)_(?:19|20)\d\d", name or "")
        return re.sub(r"[^A-Za-z0-9]", "", m.group(1)).upper() if m else ""

    def _restrict_to_company(self, hits: List[Dict[str, Any]], entity: Optional[str]) -> List[Dict[str, Any]]:
        """Drop retrieved passages that belong to a DIFFERENT company than the
        resolved entity's document. Every question in the benchmark is about
        one named company, but the retriever only down-weights (0.4x) other
        companies' passages, so they still surface when the resolved
        company has few strong matches -- and the PoT sandbox then extracts
        numbers from them. Confirmed real case: "How did JnJ's US sales
        growth compare to international sales growth" had MGM Resorts table
        rows in its evidence and the sandbox computed 77.29% from MGM's
        "Las Vegas Strip Resorts net revenues". Only applied when the entity
        resolved to a real corpus document, and never when it would leave
        nothing."""
        key = self._company_key(entity or "")
        if not key:
            return hits
        kept = [h for h in hits if self._company_key(h.get("company") or "") in ("", key)]
        return kept or hits

    def _deduplicate_hits(self, hits: List[Dict[str, Any]], entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Keeps the HIGHEST-scoring occurrence of a passage id, not just the
        first one encountered. `hits` here is the concatenation of every
        sub-query's own results in whichever order those sub-queries ran
        (see the `for sq in search_queries: new_hits.extend(...)` call
        site) -- the SAME passage routinely gets found by more than one
        sub-query with a DIFFERENT score each time (BM25 scores depend on
        the exact query text), and a plain first-seen-wins dedup locks in
        whichever score its EARLIEST matching sub-query happened to give
        it, discarding a later, more-targeted sub-query's much higher
        score for the exact same passage. Confirmed real case: AMD's
        FY2022 "what drove revenue change" question retrieves its own
        real driver passage ("...driven by a 64% increase in Data Center
        segment revenue... EPYC...") via TWO sub-queries -- a generic
        "AMD Revenue Net Revenue" alias query (which only weakly matches
        it, score ~97) that happens to run FIRST, and the bare question
        text itself (which matches it strongly via the causal-language
        boost, score ~195) that runs second. First-seen-wins kept the
        weak 97 score, which then ranked the passage outside the top-12
        evidence cap the LLM actually sees -- even though its own
        genuinely-best score would have ranked it comfortably inside.
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
        # pot_reasoner._has_finished_goods_inventory() decides the
        # inventory_turnover convention (average vs. ending) from whatever
        # evidence happens to reach PoT -- but a plain "inventory"/
        # "inventories" alias query alone retrieves only the TOTAL
        # inventory line (top ~5), never the breakdown sub-rows the
        # convention check actually looks for. Confirmed real case: JnJ's
        # own "Finished goods" row (page 58, needed to trigger the average
        # convention) never made the top-5 for the plain "inventory"
        # query -- only "Total inventories"/"Inventories (Notes 1 and 3)"
        # did, both generic totals with neither "finished goods" nor
        # "fuel"/"spare parts" wording, so the convention check silently
        # saw no signal and fell back to the wrong (ending) default. These
        # terms widen the SAME retrieval query (not the extraction alias
        # list _extract_formula_guided() scores against, so the inventory
        # VALUE used in the calculation is unaffected) to give the
        # breakdown row a real chance at the top-5, for either convention.
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