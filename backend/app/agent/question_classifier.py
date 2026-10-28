"""Question classifier for financial QA taxonomy

Classifies questions along three axes:
1. Question Type:
   - GENERIC: Broad questions applicable to any filer
   - ENTITY_SPECIFIC: Company-specific, complex analytic prompts
   - METRICS_FOCUSED: Questions that require precise numeric calculations

2. Cognitive Task:
   - LOOKUP: Direct extraction of values from statements
   - NUMERICAL_REASONING: Calculations using extracted data (growth, ratios, margins)
   - LOGICAL_INFERENCE: Interpretation, trend assessment, causal reasoning

3. Retrieval Strategy (drives evidence retrieval):
   - TARGETED_ROW: Retrieve specific line-item rows from structured tables
   - MULTI_ROW: Retrieve multiple related rows for cross-computation
   - NARRATIVE: Retrieve prose sections such as management discussion or notes
   - HYBRID: Combine table rows and narrative passages
"""

import os
import re
from typing import Dict, Any, List, Optional

from app.agent.financial_vocabulary import metric_view


# ─── Financial term dictionaries ────────────────────────────────────
# How a question phrases each metric -- see financial_vocabulary.QUESTION_METRICS.
_METRIC_KEYWORDS: Dict[str, List[str]] = metric_view("keywords")

# Narrative (non-line-item) questions get their topic-specific retrieval query
# from decomposer.QueryDecomposer.suggest_narrative_topic_query() (LLM-based,
# called from orchestrator.py), not from a keyword table in this module.

# ─── Metric → statement_type routing table ──────────────────────────────────
# Maps canonical metric key → which financial statement to target
_STATEMENT_TYPE_MAP: Dict[str, str] = metric_view("statement")

# ─── Answer-mode classification keywords (NUMERIC / EXPLANATION / ASSESSMENT /
# EXCLUSION triggers, keyword matching, year extraction) ─────────────────────
_CALC_KEYWORDS = {
    "yoy":   ["yoy", "year over year", "成長率", "年增率", "同比", "growth rate", "growth"],
    "cagr":  ["cagr", "複合成長率", "compound annual growth"],
    "margin":["毛利率", "利潤率", "margin", "利益率"],
    "ratio": ["比率", "ratio", "倍", "百分比", "percent"],
    "change":["變動", "change", "difference", "差異", "增加多少", "減少多少"],
    "compare":["比較", "compare", "versus", "vs", "對比"],
}

_CHANGE_VERBS = ["grow", "drop", "decline", "decrease", "increase", "rise", "fell"]

_EXPLANATION_KEYWORDS = [
    "why", "what drove", "what caused", "driver", "drove", "explain",
    "原因", "驅動", "帶動", "造成", "解釋", "說明", "變化原因", "原因分析",
]

_ASSESSMENT_KEYWORDS = [
    "capital-intensive", "capital intensive", "useful metric",
    "not a useful metric", "should", "whether", "是否", "適用",
    "有沒有意義", "是否適合", "判斷",
]

_EXCLUSION_KEYWORDS = [
    "exclude", "excluding", "without", "adjusted for", "m&a",
    "organic", "merger",
    "剔除", "排除", "不含", "併購", "併入",
]
# The word "acquisition" alone should not trigger exclusion intent.
# Treat it as exclusion only when paired with explicit isolation language (e.g.,
# "excluding", "adjusted for").
_ACQUISITION_EXCLUSION_PAIR_KEYWORDS = [
    "exclude", "excluding", "without", "adjusted for", "organic",
    "剔除", "排除", "不含",
]


def _kw_match(triggers, q_lower: str) -> bool:
    """Return True if q_lower contains any trigger with a genuine word boundary
    immediately before it. Short trigger keywords can occur as substrings
    inside ordinary words, causing false matches; enforcing a left boundary
    reduces those false positives. Only a left boundary is required so
    multi-word phrases still match regardless of what follows.
    """
    for t in triggers:
        if re.search(r'\b' + re.escape(t), q_lower):
            return True
    return False


def _extract_years(query: str) -> List[str]:
    """Return every distinct year mentioned in the query. Year ranges written
    with a dash or 'to' are expanded to include all years in between
    inclusive, avoiding omission of interior years when building retrieval
    steps.
    """
    years: List[str] = []
    seen: set = set()
    for m in re.finditer(r'(?:FY)?(20\d{2})\s*(?:[-–—]|to)\s*(?:FY)?(20\d{2})', query, re.IGNORECASE):
        start, end = int(m.group(1)), int(m.group(2))
        if 0 < end - start <= 10:
            for y in range(start, end + 1):
                ys = str(y)
                if ys not in seen:
                    years.append(ys)
                    seen.add(ys)
    for y in re.findall(r'(?:FY)?(20\d{2})', query):
        if y not in seen:
            years.append(y)
            seen.add(y)
    return sorted(years)


class FinanceBenchClassifier:
    """Classify an input question into the system taxonomy and choose the best
    retrieval strategy for that class.
    """

    def classify(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()

        # ── Step 1: Extract structured signals ──
        entity = self._extract_entity(query)
        target_metrics = self._extract_target_metrics(q_lower)
        calc_type = self._detect_calc_type(q_lower)
        years = _extract_years(query)
        quarters = sorted(set(re.findall(r'[Qq][1-4]', query)))
        has_explanation = any(kw in q_lower for kw in _EXPLANATION_KEYWORDS)
        has_assessment = any(kw in q_lower for kw in _ASSESSMENT_KEYWORDS)
        has_exclusion = any(kw in q_lower for kw in _EXCLUSION_KEYWORDS) or (
            "acquisition" in q_lower
            and any(kw in q_lower for kw in _ACQUISITION_EXCLUSION_PAIR_KEYWORDS)
        )
        # Terms like "what drove", "what caused", and "driver" always indicate a request
        # for a causal explanation, not a numeric value.
        # Track this separately from broader explanation flags so explanatory intent
        # overrides numeric calc-type triggers.
        has_causal_driver = any(
            kw in q_lower for kw in ("what drove", "what caused", "driver", "drove",
                                      "帶動", "造成", "驅動")
        )

        # Step 2: classify high-level question type for finance QA routing.
        question_type = self._classify_question_type(
            target_metrics, calc_type, years, has_explanation, has_assessment, has_exclusion
        )

        # ── Step 3: Classify cognitive_task ──
        cognitive_task = self._classify_cognitive_task(
            calc_type, has_explanation, has_assessment, has_exclusion, len(years)
        )

        # ── Step 4: Determine retrieval_strategy ──
        retrieval_strategy = self._determine_retrieval_strategy(
            question_type, cognitive_task, target_metrics, calc_type
        )

        # ── Step 5: Build optimized retrieval queries ──
        retrieval_queries = self._build_retrieval_queries(
            query, entity, target_metrics, years, retrieval_strategy
        )

        # ── Step 6: Determine statement_type_hint (Step 3/4 Section Anchoring) ──
        # Prefer the question's OWN explicit statement restriction (e.g. "...by
        # primarily using information that is shown in the balance sheet") over
        # the metric-inferred guess below -- it's a direct statement of intent,
        # not a heuristic. Only overrides when exactly one statement is named;
        # a question naming two (a composite ratio needing both, e.g. DPO) is
        # already handled per-sub-question by the formula-decomposition path
        # further down, which re-derives its own hint from each sub-question's
        # own text rather than this single query-level value.
        literal_hint = self._extract_literal_statement_hint(query)
        statement_type_hint = literal_hint or self._infer_statement_type_hint(target_metrics)

        # ── Step 7: Determine answer_mode ─────────────────────────────────────
        # Priority: EXCLUSION > NUMERIC(with data) > EXPLANATION/ASSESSMENT
        # Key insight: if the question compares years OR involves calc_type (ratio, yoy, etc.),
        # we ALWAYS go NUMERIC even if the question also asks for explanation.
        # The LLM synthesizer will handle the assessment part of the answer.
        if has_exclusion:
            answer_mode = "EXCLUSION"
        elif has_causal_driver:
            # Phrases asking "what drove/caused X's change" request causal reasons
            # rather than computed percentages.
            # Ensure causal-intent detection overrides calc-type signals that would
            # force numeric answers.
            answer_mode = "EXPLANATION"
        elif calc_type in ("yoy", "cagr", "change", "ratio", "compare", "margin") or (
            len(years) >= 2 and target_metrics
        ) or (
            len(years) >= 2 and calc_type
        ):
            # Has numerical signals → always route through NUMERIC path first.
            # The synthesizer/LLM will still produce the assessment narrative.
            answer_mode = "NUMERIC"
        elif (
            re.search(r"\bhow\s+much\b", query, re.IGNORECASE)
            and re.search(r"\b(?:usd|dollars?|millions?|billions?)\b|\$", query, re.IGNORECASE)
            and not has_assessment
        ):
            # Explicit currency/units in the question (e.g., "in USD million") request a
            # single numeric amount even if no standard metric keyword is found.
            # If the question is explanatory rather than quantitative, do not produce a
            # numeric result card.
            answer_mode = "NUMERIC"
        elif has_assessment:
            answer_mode = "ASSESSMENT"
        elif has_explanation:
            answer_mode = "EXPLANATION"
        elif target_metrics and not has_explanation:
            answer_mode = "NUMERIC"
        else:
            answer_mode = "EXPLANATION"

        # ── Step 7: Determine complexity ──
        complexity = "COMPLEX" if any([
            calc_type in ("yoy", "cagr", "change"),
            len(years) >= 2,
            has_exclusion,
            len(target_metrics) >= 2,
        ]) else "SIMPLE"

        # Ablation switch (default off): ABLATE_ROUTER=numeric|explanation sends every
        # question down one path -- the answer mode, retrieval strategy and statement
        # hint the classifier would have chosen are replaced by fixed values.
        forced = os.environ.get("ABLATE_ROUTER", "").lower()
        if forced in ("numeric", "explanation"):
            answer_mode = "NUMERIC" if forced == "numeric" else "EXPLANATION"
            retrieval_strategy = "HYBRID"
            retrieval_queries = self._build_retrieval_queries(
                query, entity, target_metrics, years, retrieval_strategy
            )
            statement_type_hint = None

        return {
            "entity": entity,
            "target_metrics": target_metrics,
            "calc_type": calc_type,
            "years": years,
            "quarters": quarters,
            "question_type": question_type,
            "cognitive_task": cognitive_task,
            "retrieval_strategy": retrieval_strategy,
            "retrieval_queries": retrieval_queries,
            "answer_mode": answer_mode,
            "complexity": complexity,
            "statement_type_hint": statement_type_hint,   # Step 3/4: section anchoring
        }

    # ─── Private helpers ────────────────────────────────────────────

    def _extract_entity(self, query: str) -> str:
        q = query.lower()
        q_orig = query

        # Matches capitalized single-token entities in queries (e.g., X, Y, Z).
        # Only single tokens are matched to avoid capturing sentence-start words.
        skip_words = {
            # Question starters & auxiliary verbs
            'The', 'What', 'How', 'Why', 'Which', 'When', 'Where', 'Is', 'Are',
            'Was', 'Were', 'Has', 'Have', 'Had', 'Did', 'Does', 'Do',
            'Can', 'Could', 'Would', 'Should', 'May', 'Might', 'Will', 'Shall',
            # Common non-company words that start with uppercase
            'In', 'Of', 'For', 'And', 'Or', 'To', 'From', 'At', 'On', 'By',
            'If', 'Then', 'That', 'This', 'These', 'Those', 'It',
            'Net', 'Total', 'Gross', 'Operating', 'Non', 'Other',
            'FY', 'Q1', 'Q2', 'Q3', 'Q4', 'H1', 'H2',
            'Report', 'Annual', 'Quarter', 'Revenue', 'Income', 'Profit',
            'Loss', 'Asset', 'Equity', 'Cash', 'Flow', 'Growth', 'Rate',
            'Margin', 'Cost', 'Expense', 'Sales', 'Segment', 'Business',
            'Company', 'Fiscal', 'Year', 'Period', 'Please', 'Tell',
            'Give', 'Calculate', 'Compute', 'Find', 'Show', 'State', 'Explain',
            'CAGR', 'YOY', 'EPS', 'ROE', 'ROA', 'CapEx', 'EBITDA', 'MD',
            'Quick', 'Current', 'Ratio', 'Debt', 'Working', 'Capital',
            'Inventory', 'Receivable', 'Liabilities', 'Assets', 'Interest',
            # Captures uppercase metric abbreviations (e.g., SG&A, EBITDA) that look
            # like entities.
            # Avoid treating such abbreviations as companies in multi-entity corpora to
            # prevent retrieval mismatch.
            'SG&A', 'SGA', 'G&A',
        }
        # Only match single-token company-like words (no multi-word groups)
        cap_tokens = re.findall(
            r'\b[A-Z][A-Za-z0-9&\-]+\b',
            q_orig
        )
        candidates = [
            w for w in cap_tokens
            if w not in skip_words
            and len(w) >= 2
            and not re.match(r'^FY?20\d\d$', w)  # skip FY2022, 2023 etc.
            and not re.match(r'^Q[1-4]$', w)      # skip Q1-Q4
            and not re.match(r'^\d+$', w)          # skip pure numbers
        ]
        if candidates:
            return candidates[0]

        return "company"


    def _extract_target_metrics(self, q_lower: str) -> List[str]:
        """Return a list of canonical metric keys found in the query."""
        found = []
        for metric_key, keywords in _METRIC_KEYWORDS.items():
            if _kw_match(keywords, q_lower):
                found.append(metric_key)
        # Detects narrative questions asking for named debt instruments and avoids
        # triggering numeric-only lookup mode.
        # If a query also contains explicit debt-securities phrasing, ignore the generic
        # "debt" numeric trigger.
        if "debt" in found and _kw_match(
            ["debt securit", "registered to trade", "national securities exchange"], q_lower
        ):
            found.remove("debt")
        return found

    def _detect_calc_type(self, q_lower: str) -> str:
        """Detect the primary calculation type requested."""
        for calc_key, keywords in _CALC_KEYWORDS.items():
            if _kw_match(keywords, q_lower):
                return calc_key
        # Handles plain increase/decrease verbs by assigning a year-over-year change
        # intent when calc_type is empty.
        # Run after other keyword matches and skip for driver/explanatory or superlative
        # questions.
        if (_kw_match(_CHANGE_VERBS, q_lower)
                and not _kw_match(_EXPLANATION_KEYWORDS, q_lower)
                and not re.search(r"\bwhich\b", q_lower)):
            return "change"
        return ""

    def _classify_question_type(self, target_metrics, calc_type, years, has_expl, has_assess, has_excl) -> str:
        """Taxonomy categories:
        - METRICS_GENERATED: requires calculation on extracted data
        - NOVEL_GENERATED: requires company-specific complex analysis
        - DOMAIN_RELEVANT: generic, applicable across companies
        """
        if calc_type in ("yoy", "cagr", "margin", "ratio", "change"):
            return "METRICS_GENERATED"
        if has_excl or has_assess or (has_expl and target_metrics):
            return "NOVEL_GENERATED"
        if target_metrics and len(years) >= 1:
            return "METRICS_GENERATED"
        if target_metrics:
            return "DOMAIN_RELEVANT"
        return "DOMAIN_RELEVANT"

    def _classify_cognitive_task(self, calc_type, has_expl, has_assess, has_excl, n_years) -> str:
        """
        Cognitive task type:
        - LOOKUP: direct value retrieval
        - NUMERICAL_REASONING: needs computation
        - LOGICAL_INFERENCE: needs interpretation / judgment
        """
        if calc_type:
            return "NUMERICAL_REASONING"
        if has_expl or has_assess or has_excl:
            return "LOGICAL_INFERENCE"
        if n_years >= 2:
            return "NUMERICAL_REASONING"
        return "LOOKUP"

    #: Matches the handful of ways FinanceBench-style questions name a
    #: specific financial statement. Grouped so the match itself tells us
    #: which canonical statement type was named (see
    #: _extract_literal_statement_hint).
    _LITERAL_STATEMENT_RE = re.compile(
        r'(balance sheet|statement of financial position)|'
        r'(income statement|statement of income|p&l statement|profit and loss statement)|'
        r'(cash flow statement|statement of cash flows)',
        re.IGNORECASE,
    )
    _LITERAL_STATEMENT_GROUPS = {1: "balance_sheet", 2: "income_statement", 3: "cash_flow"}

    def _extract_literal_statement_hint(self, query: str) -> Optional[str]:
        """
        The question's own explicit statement restriction, when it names
        EXACTLY ONE financial statement (e.g. "...by primarily using
        information that is shown in the balance sheet", "...who can only
        use the details shown within the balance sheet"). Returns None when
        no statement is named, or when more than one is named -- a
        composite-ratio question needing two statements (e.g. DPO needs both
        the balance sheet and the income statement) is not something a
        single hint value can represent correctly; that case is already
        handled per-sub-question elsewhere.
        """
        found = set()
        for m in self._LITERAL_STATEMENT_RE.finditer(query):
            for group_idx, stmt in self._LITERAL_STATEMENT_GROUPS.items():
                if m.group(group_idx):
                    found.add(stmt)
        if len(found) == 1:
            return next(iter(found))
        return None

    def _infer_statement_type_hint(self, target_metrics: List[str]) -> str:
        """
        Given the list of canonical target metrics, infer which financial
        statement is the primary source.  Returns the most-voted statement type,
        or "unknown" if target_metrics is empty.

        Voting: each metric casts one vote; the plurality wins.
        Tie-break order: income_statement > balance_sheet > cash_flow.
        """
        if not target_metrics:
            return "unknown"

        votes: Dict[str, int] = {}
        for metric in target_metrics:
            stmt = _STATEMENT_TYPE_MAP.get(metric)
            if stmt:
                votes[stmt] = votes.get(stmt, 0) + 1

        if not votes:
            return "unknown"

        # Tie-break rule: cash_flow-specific metrics beat balance_sheet ties.
        # e.g. if both 'cash' (balance_sheet) and 'operating_cf' (cash_flow) appear,
        # the question is about cash flows, not balance sheet.
        _CASH_FLOW_METRICS = {"capex", "operating_cf", "investing_cf", "financing_cf", "fcf", "depreciation"}
        has_cf_specific = any(m in _CASH_FLOW_METRICS for m in target_metrics)
        if has_cf_specific and votes.get("cash_flow", 0) > 0:
            # Promote cash_flow: remove any balance_sheet vote that came from 'cash' only
            only_cash_bs = all(
                _STATEMENT_TYPE_MAP.get(m) != "balance_sheet"
                or m == "cash"
                for m in target_metrics
                if _STATEMENT_TYPE_MAP.get(m) == "balance_sheet"
            )
            if only_cash_bs:
                votes.pop("balance_sheet", None)  # clear the ambiguous 'cash' vote

        if not votes:
            return "unknown"

        # Return the statement type with the most votes (tie-break: priority order)
        priority = ["cash_flow", "income_statement", "balance_sheet", "notes"]
        best_count = max(votes.values())
        for p in priority:
            if votes.get(p, 0) == best_count:
                return p
        return next(iter(votes))  # fallback: any with best count

    def _determine_retrieval_strategy(self, q_type, cog_task, metrics, calc_type) -> str:
        """
        Retrieval strategy determines HOW we search:
        - TARGETED_ROW: need a single specific line item from table
        - MULTI_ROW: need multiple related rows (e.g., revenue + COGS for margin)
        - NARRATIVE: need MD&A text passages
        - HYBRID: need both table data and narrative evidence
        """
        if q_type == "METRICS_GENERATED":
            if calc_type in ("margin", "ratio"):
                return "MULTI_ROW"
            if calc_type in ("yoy", "cagr", "change", "compare"):
                return "TARGETED_ROW"
            return "MULTI_ROW"
        if cog_task == "LOGICAL_INFERENCE":
            if metrics:
                return "HYBRID"
            return "NARRATIVE"
        if cog_task == "LOOKUP":
            return "TARGETED_ROW"
        return "HYBRID"

    def _build_retrieval_queries(self, query, entity, metrics, years, strategy) -> List[str]:
        """
        Build optimized retrieval queries based on classification.
        Returns a list of search queries that target the right evidence.
        """
        queries = []

        # A narrative-topic question (legal proceedings, dividends, customers,
        # ...) has no _METRIC_KEYWORDS entry, so `metrics` is empty and the
        # strategy branches below fall back to the bare question text. The
        # topic-specific query for those is added afterwards by
        # orchestrator.py via QueryDecomposer.suggest_narrative_topic_query().

        # Map canonical metric keys to actual financial line item names
        # How filings word each metric -- see financial_vocabulary.QUESTION_METRICS.
        metric_names = metric_view("search_terms")

        if strategy == "TARGETED_ROW":
            # Build precise queries for each target metric + year combination
            for m in metrics:
                name = metric_names.get(m, m)
                if years:
                    for y in years:
                        queries.append(f"{entity} {name} {y}")
                else:
                    queries.append(f"{entity} {name}")
            if not queries:
                queries.append(query)

        elif strategy == "MULTI_ROW":
            # Need multiple line items (e.g., gross profit AND revenue for margin)
            for m in metrics:
                name = metric_names.get(m, m)
                year_str = " ".join(years) if years else ""
                queries.append(f"{entity} {name} {year_str}".strip())
            if not queries:
                queries.append(query)

        elif strategy == "NARRATIVE":
            # Search for text notes and MD&A passages
            queries.append(f"{entity} MD&A 經營討論 附註 {' '.join(years)}".strip())
            queries.append(query)

        elif strategy == "HYBRID":
            # Combine metric rows + narrative
            for m in metrics[:2]:
                name = metric_names.get(m, m)
                queries.append(f"{entity} {name} {' '.join(years)}".strip())
            queries.append(f"{entity} 經營討論 MD&A 附註 {' '.join(years)}".strip())

        # Always include the original query as fallback
        if query not in queries:
            queries.append(query)

        return queries[:5]  # Cap at 5 queries to avoid over-retrieval
