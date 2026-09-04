import re
from typing import List, Dict, Any, Optional


class HybridFinancialRetriever:
    # Used for a 1.5x _line_item_match_score boost when a query and a
    # candidate passage both mention the same term — this is what lets a
    # short, clean balance-sheet/cash-flow table ROW outrank a long prose
    # page that happens to repeat the company's own name many times (e.g.
    # a subsidiary list or legal exhibit index) purely on raw BM25 term
    # frequency. Several formula-library primary aliases (the literal text
    # used to build retrieval queries — see
    # orchestrator._build_formula_retrieval_steps) were missing here
    # entirely, leaving those specific lookups with no such protection.
    # Confirmed real case: Kraft Heinz's real "Inventories" row (page 52)
    # ranked #12, behind 11 boilerplate/legal pages that just happened to
    # repeat "Kraft Heinz" and a stray "2018" many times — "inventory"/
    # "inventories" wasn't in this list at all.
    FINANCIAL_TERMS = [
        "營業收入", "營收", "毛利", "毛利率", "營業利益", "營業利益率", "營業費用",
        "研發費用", "本期淨利", "淨利", "每股盈餘", "資本支出", "銷貨成本", "營業成本",
        "存貨", "應付帳款", "應收帳款",
        "revenue", "gross profit", "gross margin", "operating income", "operating margin",
        "net income", "eps", "capital expenditure", "capex", "r&d", "net sales",
        "total revenue", "cost of revenue", "ebitda", "free cash flow",
        "inventory", "inventories", "accounts payable", "accounts receivable", "receivables",
        "cost of goods sold", "cost of sales", "cost of products sold",
        "total assets", "total liabilities", "current liabilities", "current assets",
        "provision for income taxes", "income tax", "dividends paid",
        "net cash provided by operating activities", "property and equipment",
        "property, plant and equipment",
    ]

    #: A row whose own Line Item label starts with "Total" (e.g. "Total
    #: cost of sales", "Total current assets") is a genuine consolidated
    #: total, not a sub-item/segment/note breakdown row — the SAME
    #: total-row-priority principle already applied at the extraction
    #: stage (_score_row_match() in pot_reasoner.py), needed here too
    #: because a wrong sub-item row can outrank the real total on pure
    #: BM25 score before extraction ever gets a chance to choose between
    #: them (confirmed real case: AES Corporation's "Cost of Sales—Non-
    #: Regulated" note row, and even a different page's own "Cost of
    #: Sales" sub-row, both outscored the real "Total cost of sales" row
    #: on the same page — the total row was retrieved, ranked 7th, and
    #: never made the top-3 cutoff actually used).
    _TOTAL_ROW_RE = re.compile(r'Line Item:\s*Total\b', re.IGNORECASE)

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus

    # ──────────────────────────────────────────────────────────────
    # Tokenisation (handles Chinese characters + English/numbers)
    # ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        text_lower = text.lower()
        # Split Chinese chars individually; keep alphanumeric words and decimal numbers.
        # Comma-grouped numbers (e.g. "7,772") MUST be matched as one token before the
        # plain [a-z0-9]+ alternative, which stops at the comma \u2014 otherwise "7,772"
        # becomes two tokens ("7","772") while an unrelated same-row value like "13"
        # stays one token, artificially inflating doc_len (and thus penalising BM25
        # score) for every row that happens to have 4-digit accounting figures.
        # Confirmed real case: Corning's real "Cost of sales" row (7,772/7,468/6,829)
        # scored BELOW an unrelated footnote row with tiny 2-digit values (13/11/13)
        # purely because of this length-normalisation artifact, not any real relevance
        # difference.
        tokens = re.findall(
            r'[\u4e00-\u9fff]'
            r'|\d{1,3}(?:,\d{3})+(?:\.\d+)?'
            r'|[a-z0-9]+(?:\.\d+)?'
            r'|\d+[,.]?\d*',
            text_lower,
        )
        # Also add multi-char financial terms as atomic tokens for better matching
        for term in self.FINANCIAL_TERMS:
            if term in text_lower:
                tokens.append(term)
        return [t for t in tokens if len(t) > 0]

    # ──────────────────────────────────────────────────────────────
    # BM25-style scoring
    # ──────────────────────────────────────────────────────────────

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        score = 0.0
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        doc_set = set(doc_tokens)
        for token in query_tokens:
            if token in doc_set:
                tf = doc_tokens.count(token)
                score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / 50.0)))
        return score

    # ──────────────────────────────────────────────────────────────
    # Financial line-item keyword boost
    # ──────────────────────────────────────────────────────────────

    def _line_item_match_score(self, query: str, doc_content: str) -> float:
        q_lower = query.lower()
        content_lower = doc_content.lower()
        for item in self.FINANCIAL_TERMS:
            if item in q_lower and item in content_lower:
                return 1.5
        return 1.0

    # ──────────────────────────────────────────────────────────────
    # Company entity filter (RC3 fix)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_company(name: str) -> str:
        """Strip years, underscores, extensions; lowercase."""
        n = re.sub(r'\b(?:20|19)\d{2}\b', '', name)  # remove years
        n = re.sub(r'\.(pdf|csv|txt|xlsx?|json)$', '', n, flags=re.IGNORECASE)
        n = re.sub(r'[_\-]+', ' ', n)
        return n.lower().strip()

    def _company_match_score(self, doc_company: str, entity: str) -> float:
        """
        Returns a multiplier based on how well doc_company matches entity.
          2.0  → strong match  (boost)
          1.0  → neutral
          0.15 → mismatch      (heavy penalty, not hard exclusion)
        """
        if not entity or entity.lower() in ("company", "unknown", ""):
            return 1.0  # no filter if entity is generic

        norm_doc = self._normalise_company(doc_company)
        norm_ent = self._normalise_company(entity)

        if not norm_doc or not norm_ent:
            return 1.0

        # Exact normalised match
        if norm_doc == norm_ent:
            return 2.0

        # One is a substring of the other
        if norm_ent in norm_doc or norm_doc in norm_ent:
            return 1.8

        # Word-level overlap
        ent_words = [w for w in norm_ent.split() if len(w) >= 2]
        doc_words = set(norm_doc.split())
        if not ent_words:
            return 1.0

        matches = sum(1 for w in ent_words if w in doc_words)
        if matches == len(ent_words):
            return 1.8   # all entity words found in doc company name
        if matches > 0:
            return 1.2   # partial match — mild boost
        return 0.15      # no word overlap → very likely a different company

    # ──────────────────────────────────────────────────────────────
    # Main search
    # ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_ids: Optional[List[str]] = None,
        entity: Optional[str] = None,
        section: Optional[str] = None,
        statement_type_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the corpus with BM25 + overlap scoring.

        Args:
            query              : user query
            top_k              : max results to return
            exclude_ids        : passage IDs to skip (already retrieved)
            entity             : company filter (soft, via _company_match_score)
            section            : legacy Step-3 section label (0.05x penalty on mismatch)
            statement_type_hint: Step-4 report type hint — income_statement | balance_sheet |
                                 cash_flow | notes | unknown.  Matching docs get a 1.5x boost;
                                 non-matching docs are unaffected (no penalty).
        """
        exclude_ids = set(exclude_ids or [])
        query_tokens = self._tokenize(query)
        query_years = set(re.findall(r'(?:FY)?(20\d\d)', query))
        query_quarters = set(re.findall(r'q[1-4]', query.lower()))

        scored_results = []
        for doc in self.corpus:
            if doc['id'] in exclude_ids:
                continue

            content = doc['content']
            doc_tokens = self._tokenize(content)
            bm25 = self._bm25_score(query_tokens, doc_tokens)
            overlap_count = sum(1 for q in query_tokens if q in content.lower())

            multiplier = 1.0

            # ── RC3: company entity filter ────────────────────────────────
            doc_company = doc.get('company', '')
            multiplier *= self._company_match_score(doc_company, entity)

            # ── Financial line item boost ──────────────────────────────────
            multiplier *= self._line_item_match_score(query, content)

            # ── Total-row boost ──────────────────────────────────────────────
            if self._TOTAL_ROW_RE.search(content):
                multiplier *= 1.3

            # ── Year / quarter boost ───────────────────────────────────────
            doc_period = str(doc.get('period', '')) + " " + content
            if query_years:
                # Normalise FY prefix for comparison
                doc_years_found = set(re.findall(r'(?:FY)?(20\d\d)', doc_period))
                if query_years & doc_years_found:
                    multiplier *= 1.4
                else:
                    # Penalise docs with completely different recent years
                    recent_years = {'2021', '2022', '2023', '2024', '2025'}
                    if doc_years_found & recent_years:
                        multiplier *= 0.6

            if query_quarters:
                if any(q in doc_period.lower() for q in query_quarters):
                    multiplier *= 1.3

            # ── Step 3: section anchoring (soft filter / penalty) ─────────────
            if section:
                doc_section = doc.get("section", "")  # empty = untagged old passage
                if doc_section and doc_section != section:
                    # Wrong section: heavy penalty but not hard exclusion
                    multiplier *= 0.05

            # ── Step 4: statement_type_hint boost (soft preference) ──────────
            # Use 'statement_type' field (set during ingestion by parser/table_parser).
            # A matching document gets a 1.5x boost; non-matching docs are unchanged.
            if statement_type_hint and statement_type_hint != "unknown":
                doc_stmt_type = doc.get("statement_type", "") or doc.get("section", "")
                if doc_stmt_type == statement_type_hint:
                    multiplier *= 1.5

            final_score = (bm25 * 0.7 + overlap_count * 0.3) * multiplier
            if final_score > 0.01:
                scored_results.append((final_score, doc))


        scored_results.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_results[:top_k]:
            doc_copy = dict(doc)
            doc_copy['relevance_score'] = round(float(score), 4)
            results.append(doc_copy)
        return results
