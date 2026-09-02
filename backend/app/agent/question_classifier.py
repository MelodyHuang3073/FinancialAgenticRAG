"""
FinanceBench Question Classifier

Classifies financial questions using the FinanceBench taxonomy:
1. Question Type (FinanceBench):
   - DOMAIN_RELEVANT: Generic questions applicable to any public company
   - NOVEL_GENERATED: Company-specific, complex analytical questions
   - METRICS_GENERATED: Questions requiring precise financial calculations

2. Cognitive Task:
   - LOOKUP: Direct value extraction from financial statements
   - NUMERICAL_REASONING: Calculations using extracted data (YoY, CAGR, margins, ratios)
   - LOGICAL_INFERENCE: Interpretation, trend analysis, driver assessment

3. Retrieval Strategy (driven by classification):
   - TARGETED_ROW: Retrieve specific line item rows from linearized tables
   - MULTI_ROW: Retrieve multiple related rows for cross-computation
   - NARRATIVE: Retrieve MD&A text / note passages
   - HYBRID: Combine table rows + narrative evidence
"""

import re
from typing import Dict, Any, List


# ─── Financial term dictionaries ────────────────────────────────────
_METRIC_KEYWORDS = {
    "revenue":       ["營業收入", "revenue", "sales", "net sales", "total revenue", "營收"],
    "gross_profit":  ["營業毛利", "gross profit", "毛利"],
    "gross_margin":  ["毛利率", "gross margin"],
    "op_income":     ["營業利益", "operating income", "operating profit", "營業淨利"],
    "op_expense":    ["營業費用", "operating expense", "opex"],
    "op_margin":     ["營業利益率", "operating margin", "op margin"],
    "net_income":    ["本期淨利", "net income", "net profit", "淨利", "淨利潤"],
    "eps":           ["每股盈餘", "eps", "earnings per share"],
    "rd_expense":    ["研發費用", "r&d", "research", "研發"],
    "sga":           ["推銷與管理費用", "sg&a", "selling general"],
    "cost_revenue":  ["營業成本", "cost of revenue", "cogs", "cost of goods sold", "cost of sales"],
    "total_assets":  ["總資產", "total assets"],
    "total_liab":    ["總負債", "total liabilities"],
    "equity":        ["股東權益", "shareholders equity", "stockholders equity", "equity"],
    "cash":          ["現金及約當現金", "cash and cash equivalents", "cash & equivalents",
                      "cash equivalents"],  # bare 'cash' removed to avoid collision with cash flow
    "capex":         ["資本支出", "capital expenditure", "capex"],
    # Property, Plant & Equipment is a BALANCE SHEET asset (the net book
    # value of fixed assets), distinct from capex (the CASH FLOW
    # STATEMENT spend to acquire/build those assets during the period).
    # Conflating the two (as "pp&e" used to be a trigger for "capex")
    # routed PP&E questions to the wrong statement and searched for the
    # wrong term — confirmed root cause of fixed_asset_turnover's PP&E
    # never being retrieved for a real Activision Blizzard question.
    "ppe":           ["pp&e", "property, plant and equipment", "property and equipment",
                       "fixed assets", "不動產、廠房及設備", "固定資產"],
    "depreciation":  ["折舊", "depreciation", "amortization"],
    "ebitda":        ["ebitda"],
    "roe":           ["roe", "return on equity"],
    "roa":           ["roa", "return on assets"],
    "dividend":      ["股利", "dividend", "dividends"],
    "data_center":   ["data center", "資料中心"],
    "fcf":           ["自由現金流", "free cash flow", "fcf"],
    # ── Balance sheet liquidity metrics ──────────────────────────────
    "quick_ratio":      ["quick ratio", "acid-test ratio", "acid test"],
    "current_ratio":    ["current ratio", "流動比率"],
    "current_assets":   ["current assets", "流動資產"],
    "current_liab":     ["current liabilities", "流動負債"],
    "inventory":        ["inventory", "存貨", "inventories"],
    "accounts_rec":     ["accounts receivable", "應收帳款", "receivables"],
    "accounts_payable": ["accounts payable", "應付帳款", "payables"],
    "dpo":              ["days payable outstanding", "dpo"],
    "dividends_paid":   ["dividends paid", "cash dividends paid", "dividends", "股利", "現金股利"],
    "net_income_attributable": ["net income attributable to shareowners",
                                 "net income attributable to shareholders",
                                 "net income attributable to"],
    "dividend_payout_ratio": ["dividend payout ratio", "payout ratio", "股利發放率"],
    "net_debt":         ["net debt", "淨負債"],
    "debt":             ["debt", "long-term debt", "short-term debt", "借款", "負債"],
    "working_capital":  ["working capital", "營運資金"],
    "fixed_asset_turnover": ["fixed asset turnover", "fixed-asset turnover", "net ppe turnover"],
    "operating_cash_flow_ratio": ["operating cash flow ratio", "cash flow ratio", "ocf ratio"],
    # ── Leverage / solvency metrics ───────────────────────────────────
    "debt_equity":      ["debt-to-equity", "d/e ratio", "leverage", "financial leverage"],
    "interest_coverage":["interest coverage", "times interest earned"],
    # ── Profitability metrics ─────────────────────────────────────────
    "roce":             ["roce", "return on capital employed"],
    "roic":             ["roic", "return on invested capital"],
    "net_margin":       ["net margin", "net profit margin", "profit margin"],
    # ── Cash flow metrics ─────────────────────────────────────────────
    "operating_cf":     ["operating cash flow", "cash from operations", "cfo",
                          "cash from operating", "cash provided by operating",
                          "operating activities", "generate from operations",
                          "generated by operating", "operating cash"],
    "investing_cf":     ["investing activities", "investing cash flow"],
    "financing_cf":     ["financing activities", "financing cash flow"],
}

# ─── Metric → statement_type routing table ──────────────────────────────────
# Maps canonical metric key → which financial statement to target
_STATEMENT_TYPE_MAP: Dict[str, str] = {
    # Income Statement metrics
    "revenue":          "income_statement",
    "gross_profit":     "income_statement",
    "gross_margin":     "income_statement",
    "op_income":        "income_statement",
    "op_expense":       "income_statement",
    "op_margin":        "income_statement",
    "net_income":       "income_statement",
    "eps":              "income_statement",
    "rd_expense":       "income_statement",
    "sga":              "income_statement",
    "cost_revenue":     "income_statement",
    "ebitda":           "income_statement",
    "net_margin":       "income_statement",
    "roce":             "income_statement",
    "roic":             "income_statement",
    # Balance Sheet metrics
    "total_assets":     "balance_sheet",
    "total_liab":       "balance_sheet",
    "equity":           "balance_sheet",
    "cash":             "balance_sheet",
    "current_assets":   "balance_sheet",
    "current_liab":     "balance_sheet",
    "inventory":        "balance_sheet",
    "accounts_rec":     "balance_sheet",
    "accounts_payable": "balance_sheet",
    "dpo":              "balance_sheet",
    "dividends_paid":   "cash_flow",
    "net_income_attributable": "income_statement",
    "dividend_payout_ratio":   "income_statement",
    "net_debt":         "balance_sheet",
    "debt":             "balance_sheet",
    "working_capital":  "balance_sheet",
    "quick_ratio":      "balance_sheet",
    "current_ratio":    "balance_sheet",
    "debt_equity":      "balance_sheet",
    "interest_coverage":"balance_sheet",
    "fixed_asset_turnover": "balance_sheet",
    "operating_cash_flow_ratio": "cash_flow",
    "ppe":              "balance_sheet",
    # Cash Flow metrics
    "capex":            "cash_flow",
    "depreciation":     "cash_flow",
    "fcf":              "cash_flow",
    "operating_cf":     "cash_flow",
    "investing_cf":     "cash_flow",
    "financing_cf":     "cash_flow",
    # Computed from income + balance (hint to income as primary)
    "roe":              "income_statement",
    "roa":              "income_statement",
    "dividend":         "income_statement",
}

_CALC_KEYWORDS = {
    "yoy":   ["yoy", "year over year", "成長率", "年增率", "同比", "growth rate", "growth"],
    "cagr":  ["cagr", "複合成長率", "compound annual growth"],
    "margin":["毛利率", "利潤率", "margin", "利益率"],
    "ratio": ["比率", "ratio", "倍", "百分比", "percent"],
    "change":["變動", "change", "difference", "差異", "增加多少", "減少多少"],
    "compare":["比較", "compare", "versus", "vs", "對比"],
}

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
    "acquisition", "organic", "merger",
    "剔除", "排除", "不含", "併購", "併入",
]


def _kw_match(triggers, q_lower: str) -> bool:
    """
    True if q_lower contains ANY trigger with a genuine word boundary
    immediately BEFORE it — not merely as a substring anywhere. Short
    trigger keywords ("roa", "roe") are prone to appearing INSIDE
    ordinary English words with no boundary at all (confirmed real case:
    "roa" — the Return on Assets keyword — matched inside "Approach the
    question..." boilerplate instruction text with zero relation to ROA,
    silently misclassifying the whole question). Only a LEFT boundary is
    required (not a trailing one) so multi-word phrases still match
    normally regardless of what follows them. Duplicated from
    pot_reasoner._kw_match (same rationale) since this module builds its
    own classification independently.
    """
    for t in triggers:
        if re.search(r'\b' + re.escape(t), q_lower):
            return True
    return False


def _extract_years(query: str) -> List[str]:
    """
    Every distinct fiscal year mentioned in the query. A dash/'to'-joined
    range ("FY2017-FY2019", "2017 to 2019") is expanded to every year in
    between (inclusive) — a plain year-token regex would otherwise return
    only the two range endpoints, silently dropping the years in between.
    Confirmed real impact: a "FY2017 - FY2019 3 year average" question
    only generated retrieval steps for 2017 and 2019, never 2018, because
    this used to be a bare findall with no range awareness (the same bug
    already fixed in pot_reasoner._extract_query_years — duplicated here
    since this module builds its own `years` independently).
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
    """
    Classifies a user question into FinanceBench taxonomy categories
    and determines the optimal retrieval strategy.
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
        has_exclusion = any(kw in q_lower for kw in _EXCLUSION_KEYWORDS)

        # ── Step 2: Classify FinanceBench question_type ──
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
        statement_type_hint = self._infer_statement_type_hint(target_metrics)

        # ── Step 7: Determine answer_mode ─────────────────────────────────────
        # Priority: EXCLUSION > NUMERIC(with data) > EXPLANATION/ASSESSMENT
        # Key insight: if the question compares years OR involves calc_type (ratio, yoy, etc.),
        # we ALWAYS go NUMERIC even if the question also asks for explanation.
        # The LLM synthesizer will handle the assessment part of the answer.
        if has_exclusion:
            answer_mode = "EXCLUSION"
        elif calc_type in ("yoy", "cagr", "change", "ratio", "compare", "margin") or (
            len(years) >= 2 and target_metrics
        ) or (
            len(years) >= 2 and calc_type
        ):
            # Has numerical signals → always route through NUMERIC path first.
            # The synthesizer/LLM will still produce the assessment narrative.
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

        # ── 1. Hardcoded known companies (fast path) ──────────────────────
        known = {
            "activision blizzard": "Activision Blizzard", "activision": "Activision Blizzard",
            "blizzard": "Activision Blizzard", "atvi": "Activision Blizzard",
            "台積電": "台積電 (TSMC 2330)", "tsmc": "台積電 (TSMC 2330)",
            "nvidia": "NVIDIA", "輝達": "NVIDIA",
            "apple": "Apple", "蘋果": "Apple",
            "microsoft": "Microsoft", "微軟": "Microsoft",
            "amazon": "Amazon", "亞馬遜": "Amazon",
            "google": "Google", "alphabet": "Google",
            "meta": "Meta",
            "jpmorgan": "JPMorgan", "jp morgan": "JPMorgan",
            "3m": "3M",
            "pfizer": "Pfizer",
            "costco": "Costco",
            "walmart": "Walmart",
            "tesla": "Tesla",
            "ford": "Ford",
            "boeing": "Boeing",
            "johnson & johnson": "Johnson & Johnson", "j&j": "Johnson & Johnson",
            "coca-cola": "Coca-Cola", "cocacola": "Coca-Cola",
            "pepsi": "PepsiCo", "pepsico": "PepsiCo",
            "disney": "Walt Disney",
            "netflix": "Netflix",
            "uber": "Uber",
            "salesforce": "Salesforce",
            "adobe": "Adobe",
            "intel": "Intel",
            "amd": "AMD",
            "qualcomm": "Qualcomm",
            "visa": "Visa",
            "mastercard": "Mastercard",
            "goldman sachs": "Goldman Sachs", "goldman": "Goldman Sachs",
            "morgan stanley": "Morgan Stanley",
            "berkshire": "Berkshire Hathaway",
            "chevron": "Chevron",
            "exxon": "ExxonMobil",
            "unitedhealth": "UnitedHealth",
            "abbvie": "AbbVie",
            "eli lilly": "Eli Lilly", "lilly": "Eli Lilly",
            "bristol": "Bristol-Myers Squibb",
            "merck": "Merck",
            "abbott": "Abbott",
            "medtronic": "Medtronic",
            "starbucks": "Starbucks",
            "nike": "Nike",
            "mcdonald": "McDonald's",
            "lockheed": "Lockheed Martin",
            "raytheon": "Raytheon",
            "caterpillar": "Caterpillar",
            "deere": "John Deere",
            "honeywell": "Honeywell",
            "ge": "General Electric",
            "general electric": "General Electric",
            "general motors": "General Motors",
            "ibm": "IBM",
            "oracle": "Oracle",
            "sap": "SAP",
            "accenture": "Accenture",
            "paypal": "PayPal",
            "block": "Block (Square)",
            "shopify": "Shopify",
            "airbnb": "Airbnb",
            "lyft": "Lyft",
            "snap": "Snap",
            "twitter": "Twitter/X",
            "spotify": "Spotify",
            "pinterest": "Pinterest",
        }
        for keyword, name in known.items():
            if keyword in q:
                return name

        # ── 2. Dynamic: extract individual capitalized tokens from original query ────
        # Matches single tokens like: 3M, JPMorgan, AMCOR, Pfizer, Costco, S&P500
        # NOTE: We match SINGLE tokens only (no multi-word groups) to avoid
        #       grabbing sentence starters like "Has AMCOR" as one entity.
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
        return found

    def _detect_calc_type(self, q_lower: str) -> str:
        """Detect the primary calculation type requested."""
        for calc_key, keywords in _CALC_KEYWORDS.items():
            if _kw_match(keywords, q_lower):
                return calc_key
        return ""

    def _classify_question_type(self, target_metrics, calc_type, years, has_expl, has_assess, has_excl) -> str:
        """
        FinanceBench taxonomy:
        - METRICS_GENERATED: requires calculation on extracted data
        - NOVEL_GENERATED: company-specific complex analysis
        - DOMAIN_RELEVANT: generic applicable to any company
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

        # Map canonical metric keys to actual financial line item names
        metric_names = {
            "revenue":          "Revenue Net Revenue 營業收入 營收",
            "gross_profit":     "Gross Profit 營業毛利 毛利",
            "gross_margin":     "Gross Margin 毛利率 Gross Profit Revenue",
            "op_income":        "Operating Income Operating Profit 營業利益",
            "op_expense":       "Operating Expense 營業費用",
            "op_margin":        "Operating Margin 營業利益率 Operating Income Revenue",
            "net_income":       "Net Income Net Profit 本期淨利 淨利",
            "eps":              "EPS Earnings Per Share 每股盈餘",
            "rd_expense":       "R&D Research Development 研發費用",
            "sga":              "SG&A Selling General Administrative 推銷管理費用",
            "cost_revenue":     "Cost of Revenue COGS Cost of Goods Sold 營業成本",
            "total_assets":     "Total Assets 總資產",
            "total_liab":       "Total Liabilities 總負債",
            "equity":           "Shareholders Equity 股東權益",
            "cash":             "Cash Equivalents 現金及約當現金",
            "capex":            "Capital Expenditure CapEx 資本支出",
            "ppe":              "Property Plant and Equipment PP&E Fixed Assets 不動產廠房及設備 固定資產",
            "depreciation":     "Depreciation Amortization 折舊",
            "ebitda":           "EBITDA",
            "roe":              "Return on Equity ROE Net Income Equity",
            "roa":              "Return on Assets ROA Net Income Total Assets",
            "dividend":         "Dividend 股利",
            "data_center":      "Data Center Revenue",
            "fcf":              "Free Cash Flow FCF 自由現金流",
            # Balance sheet liquidity
            "quick_ratio":      "Quick Ratio Acid Test Current Assets Inventory Current Liabilities",
            "current_ratio":    "Current Ratio Current Assets Current Liabilities 流動比率",
            "current_assets":   "Current Assets 流動資產",
            "current_liab":     "Current Liabilities 流動負債",
            "inventory":        "Inventory Inventories 存貨",
            "accounts_rec":     "Accounts Receivable Receivables 應收帳款",
            "net_debt":         "Net Debt 淨負債",
            "debt":             "Debt Long-term Debt Short-term Debt Borrowings 負債",
            "working_capital":  "Working Capital 營運資金",
            "fixed_asset_turnover": "Property Plant and Equipment PP&E Fixed Assets Revenue",
            "operating_cash_flow_ratio": "Cash from Operations Operating Cash Flow Current Liabilities",
            # Leverage
            "debt_equity":      "Debt-to-Equity Leverage Financial Leverage",
            "interest_coverage":"Interest Coverage Times Interest Earned",
            # Profitability
            "roce":             "ROCE Return on Capital Employed",
            "roic":             "ROIC Return on Invested Capital",
            "net_margin":       "Net Margin Net Profit Margin Profit Margin",
            # Cash flow
            "operating_cf":     "Operating Cash Flow Cash from Operations CFO",
            "investing_cf":     "Investing Activities Investing Cash Flow",
            "financing_cf":     "Financing Activities Financing Cash Flow",
        }

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
