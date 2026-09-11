import re
from typing import List, Dict, Any, Optional, Tuple


_TERM_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _term_present(term: str, text: str) -> bool:
    """
    Word-boundary-aware presence check for a financial term/alias inside a
    (already-lowercased) query or document string -- NOT a plain `term in
    text` substring check, which lets a short alias accidentally match
    inside an unrelated word. Confirmed real case: the "eps" alias
    (Earnings Per Share) is a literal substring of "pepsico" -- so a plain
    substring check gave EVERY PepsiCo-related query/row pair a spurious
    1.5x _line_item_match_score boost regardless of whether the row had
    anything to do with EPS, silently neutralising the boost's ability to
    discriminate for any PepsiCo question at all (it ends up applying
    uniformly to every candidate, right and wrong alike).

    An optional trailing "s" is allowed before the closing boundary --
    plain word-boundary matching alone regresses a term like "total
    revenue" (singular, as many aliases are written) against a filing's
    own "Total revenues" (plural): "revenue" immediately followed by "s"
    is NOT a word boundary, so a naive `\\bterm\\b` no longer matches what
    the OLD plain-substring check used to catch by pure accident (a
    singular alias is always also a literal PREFIX of its regular
    plural). Confirmed real case: CVS Health's real "Total revenues" row
    dropped out of a formula's own retrieval top-5 once word-boundary
    matching alone was added, because "total revenue" (the alias) could
    no longer match "total revenues" (the row's real label) at all.
    """
    pattern = _TERM_PATTERN_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(r'\b' + re.escape(term) + r's?\b')
        _TERM_PATTERN_CACHE[term] = pattern
    return bool(pattern.search(text))


_FULL_YEAR_RE = re.compile(r'(?:FY\s*)?(20\d\d)\b', re.IGNORECASE)
_FY_SHORT_YEAR_RE = re.compile(r'\bFY\s*(\d{2})\b', re.IGNORECASE)


def _extract_years(text: str) -> "set[str]":
    """
    Every 4-digit year (optionally "FY"-prefixed, e.g. "FY2022" or bare
    "2022") PLUS the common 2-digit fiscal-year shorthand ("FY22") that
    the plain 4-digit pattern alone cannot see at all -- \\d{2} after "FY"
    never matches inside "FY2022" (the boundary check after the first two
    captured digits fails, since "20" is immediately followed by more
    digits, "22"), so it only ever fires on a genuine 2-digit shorthand.
    Returns each match normalised to its 4-digit form ("22" -> "2022").
    Confirmed real case: "What drove revenue change as of the FY22 for
    AMD?" -- the old 4-digit-only pattern found no year in this query at
    all, so the entire year-relevance boost/penalty in search() silently
    never activated, letting AMD's unrelated FY2015 filing content compete
    on equal footing with the real FY2022 content for a question that
    explicitly names its year.
    """
    years = set(_FULL_YEAR_RE.findall(text))
    years.update('20' + m for m in _FY_SHORT_YEAR_RE.findall(text))
    return years


#: A question asking WHAT DROVE/CAUSED a change ("What drove revenue
#: change...", "What caused the increase in...", "Why did X change...")
#: has one specific, narrow correct answer shape: a sentence using real
#: causal language ("driven by", "primarily due to", "as a result of") to
#: name the actual driver(s) -- not just any prose that happens to repeat
#: the metric name and a similar-sounding number. Module-level (not a
#: HybridFinancialRetriever class attribute) so orchestrator.py can also
#: import and evaluate it against the ORIGINAL full question -- a single
#: attribution-shaped question typically gets decomposed into SEVERAL
#: retrieval sub-queries (e.g. a plain keyword-stuffed "AMD Revenue Net
#: Revenue" alongside the full original question text), and only checking
#: each SUB-query's own text for "what drove" phrasing misses every
#: sub-query that doesn't happen to repeat it -- confirmed real case: AMD's
#: FY2022 "What drove revenue change" question's OWN keyword sub-query
#: ("AMD Revenue Net Revenue...") never triggered the causal-language
#: boost below at all, letting an unrelated ASC 606 boilerplate footnote
#: win that sub-query's own top-5 and dilute the combined evidence pool
#: even though the SAME question's other sub-query (the full original
#: text) correctly boosted the real MD&A driver sentence.
_ATTRIBUTION_QUERY_RE = re.compile(
    r'\bwhat\s+(?:drove|caused|led\s+to)\b|\bwhy\s+(?:did|has|have)\b|\bdrivers?\s+of\b',
    re.IGNORECASE,
)


def is_attribution_query(text: str) -> bool:
    return bool(_ATTRIBUTION_QUERY_RE.search(text))


#: A question asking WHAT GEOGRAPHIES/REGIONS a company operates in has
#: one specific correct source: a 10-K's own "Geographic Operations" /
#: "Geographic Information" footnote (Note 24 or similar), which is
#: usually the SECOND of two closely-related, heavily-overlapping
#: sub-sections in the SAME segment-reporting note -- the first
#: sub-section ("Reportable Operating Segments") describes BUSINESS
#: segments (e.g. American Express's USCS/CS/ICS/GMNS), not geographies,
#: and shares almost all the same vocabulary ("revenue", "segment",
#: "operations", the fiscal year). Both sub-sections score close enough
#: on plain BM25 that the wrong (business-segment) one can edge out the
#: real geographic table even after every other ranking fix. Module-level
#: for the same reason as is_attribution_query -- orchestrator.py needs to
#: evaluate it against the ORIGINAL question, not each individual
#: sub-query (a keyword-stuffed sub-query built from this topic still
#: usually keeps "geographic" in it, but the guard is here for the same
#: reason and consistency as the attribution case).
_GEOGRAPHY_QUERY_RE = re.compile(r'\bgeograph(?:y|ies|ic(?:al)?)\b', re.IGNORECASE)
#: The literal section heading text a 10-K prints above its real
#: geographic breakdown -- confirmed real case: American Express's own
#: Note 24 prints "GEOGRAPHIC OPERATIONS" as a bold sub-heading
#: immediately followed by "The following table presents our total
#: revenues ... in different geographic regions", naming United States /
#: EMEA / APAC / LACC. A plain "geographic" substring match alone isn't
#: selective enough (the sibling business-segment sub-section's own body
#: text also uses the word "geographic" in passing), so this requires the
#: heading-style phrasing specifically -- deliberately narrow to
#: "operations"/"information" only, NOT "areas" or "segments": both of
#: those are common enough as ordinary prose (not a section title) that
#: they produce false positives of their own. Confirmed real case:
#: PepsiCo's "Our Customers" page discusses bottler distribution
#: contracts "for specified geographic areas" -- completely unrelated to
#: the company's own geographic revenue breakdown -- and "geographic
#: areas" alone was enough to trigger this boost and outrank the real
#: geographic-revenue table.
_GEOGRAPHIC_SECTION_RE = re.compile(
    r'geographic\s+(?:operations|information)\b', re.IGNORECASE
)

#: Not every filing labels its geography breakdown with the word
#: "geographic" at all -- PepsiCo's own "geographies primarily operated
#: in" answer lives entirely inside Item 1 Business's "Our Divisions"
#: segment-by-segment description ("...Africa, the Middle East and South
#: Asia...", "...Asia Pacific, Australia and New Zealand, and China
#: region...") without the word "geographic" appearing anywhere in it, so
#: _GEOGRAPHIC_SECTION_RE above (which requires that literal phrase)
#: never fires for it -- confirmed real case: that passage ranked ~155th
#: out of 200 candidates for the geography query, nowhere near the top-6
#: that reaches the LLM. Region NAMES themselves are the signal instead:
#: a passage mentioning several of them close together is very likely to
#: be an actual geography enumeration regardless of company or filing
#: structure, whereas a single incidental mention (e.g. "...Europe also
#: manufactures...") isn't -- so this only counts as a match at 3+
#: DISTINCT region names in the same passage, not a bare single hit.
_GEOGRAPHIC_REGION_NAME_RE = re.compile(
    r'\b(?:north america|latin america|south america|asia pacific|'
    r'middle east|south asia|africa|europe|australia|new zealand)\b',
    re.IGNORECASE,
)


def _has_dense_geographic_region_names(content: str, min_distinct: int = 3) -> bool:
    matches = {m.group(0).lower() for m in _GEOGRAPHIC_REGION_NAME_RE.finditer(content)}
    return len(matches) >= min_distinct


def is_geography_query(text: str) -> bool:
    return bool(_GEOGRAPHY_QUERY_RE.search(text))


#: A question asking about ongoing legal battles/litigation has one
#: specific, SEC-mandated source every 10-K carries under the exact same
#: heading: "Item 3. Legal Proceedings." -- a far more reliable structural
#: signal than any keyword overlap, since the heading text itself is
#: standardized across every filer regardless of company or industry.
#: Existing topic-query text alone ("Item 3 Legal Proceedings litigation
#: lawsuit claims") isn't enough of a ranking signal on its own when a
#: filing's OTHER pages also score reasonably on the same bag-of-words
#: query -- confirmed real case: PepsiCo's real Item 3 page ranked 23rd
#: among its own 2022 filing's pages (score 30.2) for that exact topic
#: query, well outside the top-5 per-sub-query cutoff, even after
#: eliminating the wrong-fiscal-year competition from PEPSICO_2021_10K.
_LEGAL_QUERY_RE = re.compile(
    r'\blegal\s+(?:battle|proceeding|matter)s?\b|\blitigation\b|\blawsuit\b',
    re.IGNORECASE,
)
_LEGAL_PROCEEDINGS_SECTION_RE = re.compile(
    r'item\s*3\.?\s*legal\s+proceedings', re.IGNORECASE
)


def is_legal_query(text: str) -> bool:
    return bool(_LEGAL_QUERY_RE.search(text))


def _combined_pattern(terms: List[str]) -> "re.Pattern":
    """
    One alternation regex matching ANY of `terms` at a word boundary
    (each with an optional trailing "s" -- see _term_present) -- lets a
    document be probed with a single .search() call instead of one per
    term/alias. Longest-first ordering so a multi-word alias like
    "property, plant and equipment, net" isn't pre-empted by a shorter
    alias that's also a prefix of it (regex alternation takes the first
    branch that matches, not the longest).
    """
    ordered = sorted({t for t in terms}, key=len, reverse=True)
    return re.compile(r'\b(?:' + '|'.join(re.escape(t) + 's?' for t in ordered) + r')\b')


def _load_alias_groups() -> List[List[str]]:
    """
    Lazily loads pot_reasoner.py's canonical-metric alias table
    (_CANONICAL_TO_ALIASES) as a list of synonym groups, each group being
    every alias (plus the canonical key itself) for one underlying metric
    -- e.g. capex's group includes "capital expenditure", "capital
    spending", "purchases of ppe", "資本支出", "capex". Reused by
    _line_item_match_score so a query built around one alias ("capital
    expenditure") still recognises a row that uses a DIFFERENT alias for
    the same metric ("Capital spending", PepsiCo's own cash-flow-statement
    label) as a genuine line-item match, instead of requiring the literal
    same string on both sides -- which silently fails whenever the query's
    wording and a filing's own wording for the same line item differ.
    Imported lazily (not at module load) since app.agent.pot_reasoner is a
    much larger module with its own heavier import chain; failures here
    degrade gracefully to the plain FINANCIAL_TERMS check rather than
    breaking retrieval.
    """
    try:
        from app.agent.pot_reasoner import _CANONICAL_TO_ALIASES
        return [
            [canonical] + list(aliases)
            for canonical, aliases in _CANONICAL_TO_ALIASES.items()
        ]
    except Exception:
        return []


#: Computed once per process (not per search() call, let alone per
#: candidate document) -- see _load_alias_groups's docstring.
_ALIAS_GROUPS: Optional[List[List[str]]] = None


def _get_alias_groups() -> List[List[str]]:
    global _ALIAS_GROUPS
    if _ALIAS_GROUPS is None:
        _ALIAS_GROUPS = _load_alias_groups()
    return _ALIAS_GROUPS


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
        "net income", "eps", "capital expenditure", "capex", "capital spending", "r&d", "net sales",
        "total revenue", "cost of revenue", "ebitda", "free cash flow",
        "inventory", "inventories", "accounts payable", "accounts receivable", "receivables",
        "cost of goods sold", "cost of sales", "cost of products sold",
        "total assets", "total liabilities", "current liabilities", "current assets",
        "provision for income taxes", "income tax", "dividends paid",
        "net cash provided by operating activities", "property and equipment",
        "property, plant and equipment",
        # The cash-flow statement's own line for acquisition spend --
        # present in every filing whether or not any acquisition
        # actually happened that year, unlike a narrative "Acquisitions
        # and Divestitures" note (which only exists for a filer that HAD
        # one to write about). Without this, a table_row candidate like
        # this one gets NO boost at all under prefer_narrative (only
        # text_note passages do, via the narrative-content boost below),
        # so a company with no acquisitions to narrate about — whose
        # only real evidence IS this bare table row — loses to unrelated
        # prose that merely shares generic vocabulary. Confirmed real
        # case: Ulta Beauty's own "Acquisitions, net of cash acquired"
        # cash-flow row (proving zero acquisitions in FY2023/FY2022)
        # ranked outside the top 5 under an acquisitions-topic query,
        # losing to MD&A/results-of-operations prose that never
        # mentions acquisitions at all.
        "acquisitions, net of cash acquired", "acquisitions net of cash acquired",
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

    #: A table_row whose own linearized content contains a generic "ColN:"
    #: placeholder (parser.py's own fallback whenever it couldn't resolve
    #: a real year/period header for one of a row's value columns -- see
    #: e.g. parser._inject_missing_year_header, _reconstruct_table_from_
    #: word_positions) is a low-confidence extraction: the row's real
    #: label may be right, but at least one of its numbers has an
    #: unreliable or entirely wrong year attached to it. A CORRECTLY
    #: parsed row never contains this literal text (a real year header
    #: always reads "2022:"/"2021:" etc, never "Col4:"), so this only
    #: ever demotes genuinely uncertain rows -- soft penalty, not
    #: exclusion, since the row may still be the only evidence available
    #: for its line item. Confirmed real case: MGM Resorts' "Consolidated
    #: Statements of Stockholders' Equity" is a rollforward/waterfall
    #: statement (one narrative block per year, not year-column pairs),
    #: which the standard table parser mis-splits into "Col4"/"Col6"/
    #: "Col8" placeholders -- its own confusing "(77,606)" value (really
    #: 2020's dividend total) outranked the cash-flow-statement's cleanly
    #: 2022-labeled "(4,048)" row for a FY2022 dividends question, and
    #: even survived an explicit "trust the PoT sandbox result" prompt
    #: instruction because the raw evidence looked more detailed/complete.
    _LOW_CONFIDENCE_COLUMN_RE = re.compile(r'\bCol\d+:')

    _CAUSAL_LANGUAGE_RE = re.compile(
        r'\bdriven\s+by\b|\bprimarily\s+due\s+to\b|\bmainly\s+due\s+to\b|'
        r'\bas\s+a\s+result\s+of\b|\battributable\s+to\b|\bresulted\s+from\b',
        re.IGNORECASE,
    )

    #: A table_row's own Line Item label is often a far more reliable
    #: signal for which financial statement it belongs to than the
    #: page-level `section`/`statement_type` tag the parser assigned at
    #: ingestion time — that tag is a coarse, page-range heuristic that
    #: can mis-fire for a filing's own unusual layout (e.g. a "Financial
    #: Highlights"/MD&A summary page placed right before the real
    #: statements), silently excluding the correct row from the
    #: statement_type_hint boost below while an unrelated page elsewhere
    #: (mis-tagged with the matching label) still gets it purely by
    #: document-level metadata. Confirmed real case: MGM Resorts' own
    #: "Net revenues" row (the real consolidated FY2018-2020 total) was
    #: tagged section="general_mda" instead of "income_statement", so an
    #: "income_statement" hint gave it no boost at all — while a
    #: completely unrelated revenue-RECOGNITION accounting-policy
    #: footnote (prose explaining casino/hotel revenue recognition
    #: rules, no dollar figures) was tagged section="income_statement"
    #: and got the full 1.5x boost instead, burying the real total below
    #: narrative noise and a segment note's own smaller "Reportable
    #: segment net revenues" sub-total, which then won retrieval by
    #: default with no real competition.
    _CORE_STATEMENT_LINE_ITEMS: Dict[str, List[str]] = {
        "income_statement": [
            "revenue", "net revenue", "net sales", "total revenue",
            "cost of revenue", "cost of goods sold", "cost of sales",
            "gross profit", "operating income", "net income", "net earnings",
        ],
        "balance_sheet": [
            "total assets", "total liabilities", "total current assets",
            "total current liabilities", "total stockholders equity",
            "total shareholders equity",
        ],
        "cash_flow_statement": [
            "capital expenditures", "net cash provided by operating activities",
            "net cash used in operating activities", "net increase in cash",
            "net decrease in cash",
        ],
    }
    _LINE_ITEM_LABEL_RE = re.compile(r'Line Item:\s*([^|]+)', re.IGNORECASE)

    def _matches_core_statement_line_item(self, content: str, hint: str) -> bool:
        terms = self._CORE_STATEMENT_LINE_ITEMS.get(hint)
        if not terms:
            return False
        m = self._LINE_ITEM_LABEL_RE.search(content)
        if not m:
            return False
        label = m.group(1).lower().strip()
        return any(_term_present(t, label) for t in terms)

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        # BM25's length-normalization term needs THIS corpus's own average
        # document length, not an arbitrary guess -- _bm25_score used to
        # hardcode avgdl=50, which systematically penalizes every
        # text_note passage here (this corpus's real text_note average is
        # ~125 tokens, ~80 overall across text_note + table_row) as if it
        # were 2-3x longer than "normal", silently burying long,
        # information-dense narrative passages under short ones that only
        # superficially match. Confirmed real case: Amcor's own "Note 5 -
        # Acquisitions and Divestitures" passage (the ACTUAL list of the
        # three FY2023 acquisitions the gold answer names) scored BELOW a
        # much shorter, less informative passage that merely says "refer
        # to Note 5" without any of the real content, purely because it's
        # longer -- not because it's less relevant.
        doc_lens = [len(self._tokenize(d.get('content', ''))) for d in corpus]
        self._avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 50.0

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
        avgdl = self._avg_doc_len or 50.0
        for token in query_tokens:
            if token in doc_set:
                tf = doc_tokens.count(token)
                score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / avgdl)))
        return score

    # ──────────────────────────────────────────────────────────────
    # Financial line-item keyword boost
    # ──────────────────────────────────────────────────────────────

    def _query_line_item_candidates(self, query: str) -> List["re.Pattern"]:
        """
        Precomputes, as a small list of COMBINED alternation regexes, which
        FINANCIAL_TERMS and alias groups (see _load_alias_groups) the QUERY
        itself mentions -- call ONCE per search(), not per candidate
        document.

        Two compounding costs had to be fixed here, both only visible once
        measured against the real 74k-passage corpus (a single search()
        call regressed from ~1-2s to 30s+):
          1. Re-scanning all ~28 alias groups (~150 aliases total) against
             the query on every one of ~74,000 per-document calls, when the
             query string never changes within one search() -- fixed by
             computing the query-side match ONCE here instead of inside
             _line_item_match_score.
          2. Even after (1), a document with genuinely relevant terms was
             still probed with one separate regex .search() per matched
             term/alias (profiling showed 1M+ individual .search() calls
             for a single query) -- fixed by compiling every matched
             term/group into ONE alternation pattern each, so a document's
             own check is a small, fixed number of .search() calls (one
             per matched group, plus one for terms) instead of one per
             alias.
        Narrowing each document's own check down to only the groups the
        query ACTUALLY matched is also a precision improvement on top of
        the speed fix: a document can only earn the 1.5x boost via a term/
        group the query is genuinely about, not via incidentally sharing
        an unrelated group with some other candidate.
        """
        q_lower = query.lower()
        matched_terms = [t for t in self.FINANCIAL_TERMS if _term_present(t, q_lower)]
        matched_groups = [g for g in _get_alias_groups() if any(_term_present(a, q_lower) for a in g)]
        patterns = []
        if matched_terms:
            patterns.append(_combined_pattern(matched_terms))
        patterns.extend(_combined_pattern(g) for g in matched_groups)
        return patterns

    @staticmethod
    def _line_item_match_score(doc_content: str, query_patterns: List["re.Pattern"]) -> float:
        content_lower = doc_content.lower()
        for pattern in query_patterns:
            if pattern.search(content_lower):
                return 1.5
        return 1.0

    # ──────────────────────────────────────────────────────────────
    # Company entity filter (RC3 fix)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_company(name: str) -> str:
        """Strip years, underscores/extensions, and 10-K/10-Q filing-type
        boilerplate; lowercase.

        Years and the "10K"/"10Q" filing-type suffix used to be stripped
        with `\\b(?:20|19)\\d{2}\\b` applied BEFORE underscores were turned
        into spaces. `\\b` requires a transition between a word char and a
        non-word char, and `_` counts as a word char to regex -- so in a
        real corpus company id like "BESTBUY_2017_10K" there is no `\\b`
        on either side of "2017" (underscore before, underscore after),
        and the year survived normalisation entirely, along with the "10k"
        boilerplate. Confirmed real case: this let "BESTBUY_2017_10K" and
        "COCACOLA_2017_10K" share the surviving "2017"/"10k" tokens, so
        _company_match_score's word-overlap check scored them as a
        PARTIAL MATCH (1.2x mild boost) instead of a mismatch (0.05x
        penalty) -- silently turning the entity filter into a same-year
        same-filing-type free-for-all and letting Coca-Cola's/Microsoft's
        own "Net income" table rows outrank Best Buy's on raw BM25 alone
        for a Best Buy net-profit-margin question. Fixed by tokenising
        AFTER the underscore/hyphen-to-space conversion and dropping any
        token that IS a bare year, a fused year+quarter ("2023q2"), or the
        "10k"/"10q" boilerplate -- rather than trying to regex them out of
        an underscore-joined string first.
        """
        n = re.sub(r'\.(pdf|csv|txt|xlsx?|json)$', '', name, flags=re.IGNORECASE)
        n = re.sub(r'[_\-]+', ' ', n)
        words = []
        for w in n.split():
            wl = w.lower()
            if re.fullmatch(r'(?:19|20)\d{2}(?:q[1-4])?', wl):
                continue
            if wl in ('10k', '10q'):
                continue
            words.append(wl)
        return ' '.join(words).strip()

    @staticmethod
    def _extract_company_filing_year(name: str) -> Optional[str]:
        """First bare 4-digit year (optionally with a Q1-4 suffix, e.g.
        "2023q2") found in a raw company/doc id string, or None if it has
        none. Used only to DEMOTE `_company_match_score`'s top tier when
        two DIFFERENT fiscal years of the SAME company are both in the
        corpus -- see that method's docstring for the confirmed real
        case this fixes.
        """
        m = re.search(r'(?:19|20)\d{2}(?:q[1-4])?', name, flags=re.IGNORECASE)
        return m.group(0).lower() if m else None

    def _company_match_score(self, doc_company: str, entity: str) -> float:
        """
        Returns a multiplier based on how well doc_company matches entity.
          2.0  → strong match  (boost)
          1.0  → neutral
          0.05 → mismatch      (heavy penalty, not hard exclusion)

        Deliberately still a SOFT penalty, not a hard filter — several
        confirmed fixes this session (Activision Blizzard's capex,
        General Mills' CCC placeholders) depend on a right-company
        PARTIAL/sub-item match being able to outrank a wrong-company
        EXACT match, which requires the wrong-company candidate to still
        be scoreable at all rather than excluded outright; the actual
        correctness guarantee against a wrong-company row winning lives
        in pot_reasoner.py's entity-identity-aware reduction, not here.
        Lowered from 0.15 (still 3x stricter) purely to cut down how
        often an obviously-wrong-company row is visible at all in the
        Source Evidence panel for an already-correct answer — a cosmetic/
        noise concern, verified via the full calc-question suite to
        confirm no previously-correct answer actually depended on a
        wrong-company candidate surviving at the old, looser penalty.
        """
        if not entity or entity.lower() in ("company", "unknown", ""):
            return 1.0  # no filter if entity is generic

        norm_doc = self._normalise_company(doc_company)
        norm_ent = self._normalise_company(entity)

        if not norm_doc or not norm_ent:
            return 1.0

        # Exact normalised match. _normalise_company deliberately STRIPS
        # the fiscal year (see its own docstring — needed to fix Best
        # Buy's word-boundary bug), which means it can no longer tell
        # apart two DIFFERENT fiscal years of the SAME company once both
        # are in the corpus — "MGMRESORTS_2022_10K" and
        # "MGMRESORTS_2018_10K" both normalise to "mgmresorts" and would
        # otherwise tie at this same top tier. Demote to 1.5 (still well
        # above the 0.05 different-company penalty and the 1.0 neutral
        # tier, but below the exact-right-year match below it AND below
        # the 1.8 substring/word-overlap tiers) whenever BOTH sides carry
        # a detectable year that DIFFERS — a real disambiguating signal
        # this project's own doc-id convention (COMPANY_YEAR_10K) always
        # carries. Left at the full 2.0 boost whenever either side has no
        # detectable year (a generic entity like "company", or a doc-id
        # convention without one) — unchanged from before, since there is
        # then no year signal to disambiguate with at all. Confirmed real
        # case: "Has MGM Resorts paid dividends to common shareholders in
        # FY2022?" — the real answer's own "Dividend Policy" paragraph
        # (MGMRESORTS_2022_10K) ranked #8, just outside the top-6 sent to
        # the LLM, while two malformed, blank-period rows from
        # MGMRESORTS_2018_10K (a completely different filing year) rode
        # this same 2.0 tier into #2/#3 purely because "mgmresorts" ==
        # "mgmresorts" post year-stripping.
        if norm_doc == norm_ent:
            doc_year = self._extract_company_filing_year(doc_company)
            ent_year = self._extract_company_filing_year(entity)
            if doc_year and ent_year and doc_year != ent_year:
                return 1.5
            return 2.0

        # One is a substring of the other
        if norm_ent in norm_doc or norm_doc in norm_ent:
            return 1.8

        # Collapsed (no-space) comparison: catches a human-readable name
        # like "Best Buy"/"General Mills"/"Coca Cola" against this
        # project's doc_name convention, which concatenates multi-word
        # company names WITHOUT a space ("BESTBUY_2023_10K",
        # "GENERALMILLS_2020_10K", "COCACOLA_2021_10K"). Neither the
        # exact-match nor the word-overlap check below can ever catch
        # this -- there's no word boundary inside "bestbuy" to compare
        # against the separate word "best" -- so every candidate
        # document, same-company or not, fell straight through to the
        # 0.05 "different company" penalty, silently turning the entity
        # filter into a no-op for any multi-word company whose doc_name
        # strips the space. Confirmed real case: a "Best Buy" entity
        # query got the SAME 0.05x penalty on Best Buy's OWN
        # BESTBUY_2023_10K passages as on every other company's, so an
        # unrelated company's page could freely outrank Best Buy's own
        # actual answer on raw term overlap alone.
        collapsed_doc = norm_doc.replace(' ', '')
        collapsed_ent = norm_ent.replace(' ', '')
        if collapsed_ent and (collapsed_ent in collapsed_doc or collapsed_doc in collapsed_ent):
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
        return 0.05      # no word overlap → very likely a different company

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
        prefer_narrative: bool = False,
        is_attribution: bool = False,
        is_geography: bool = False,
        is_legal: bool = False,
        query_years: Optional[List[str]] = None,
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
            prefer_narrative   : True for genuinely qualitative/narrative questions with
                                 no matching formula and no recognized financial-metric
                                 keyword (legal proceedings, dividend disclosures,
                                 business-combination lists, geographies, customers,
                                 industry/product overviews — see
                                 question_classifier._detect_narrative_topic_query).
                                 Defaults to False and is only ever passed True from
                                 orchestrator.py's non-numeric, no-formula retrieval
                                 branch, so every numeric/formula-driven retrieval call
                                 (everything the calc-question suite depends on) is
                                 bit-for-bit unaffected by the two behaviors below.
                                 Confirmed real case (Boeing FY2022 legal-proceedings
                                 question): with this off, the one passage actually
                                 describing the Lion Air/Ethiopian Airlines litigation
                                 ranked ~100th out of 200 candidates — table ROWS
                                 (a handful of tokens each) get a much friendlier BM25
                                 length-normalization than a full prose paragraph, and
                                 the unconditional "Total row" boost below adds a 1.3x
                                 bonus to any "Total X" balance-sheet row regardless of
                                 whether the query has anything to do with financial
                                 totals at all — both systematically bury narrative
                                 content under unrelated financial tables.
            is_attribution      : True when the ORIGINAL user question (not
                                 necessarily THIS specific sub-query's own
                                 text) asks what drove/caused a change --
                                 see is_attribution_query's module-level
                                 docstring for why the caller must pass
                                 this explicitly rather than relying on
                                 this call's own `query` text alone.
            is_geography        : True when the ORIGINAL user question asks
                                 what geographies/regions a company
                                 operates in -- see is_geography_query's
                                 module-level docstring.
            query_years         : years the classifier extracted from the
                                 ORIGINAL question (classification["years"]),
                                 used to disambiguate between multiple
                                 fiscal years of the SAME company's filings
                                 when `entity` itself is a bare, year-less
                                 company name (e.g. "PepsiCo", from
                                 question_classifier's hardcoded fast-path
                                 lookup) -- _company_match_score's own
                                 same-company-different-year demotion only
                                 fires when BOTH sides carry a detectable
                                 year, so it's a no-op here. See the
                                 "Preferred-year boost" below for the
                                 confirmed real case this fixes.
        """
        exclude_ids = set(exclude_ids or [])
        query_tokens = self._tokenize(query)
        query_years = _extract_years(query)
        query_quarters = set(re.findall(r'q[1-4]', query.lower()))
        # Computed ONCE per search() call, not per candidate document --
        # see _query_line_item_candidates's docstring.
        query_patterns = self._query_line_item_candidates(query)
        # OR'd with this call's own query text so a direct, single-query
        # caller (a diagnostic script, a future call site) still gets the
        # boost without needing to pass is_attribution/is_geography explicitly.
        attribution_active = is_attribution or is_attribution_query(query)
        geography_active = is_geography or is_geography_query(query)
        legal_active = is_legal or is_legal_query(query)
        # The LATEST year mentioned, not just any of them -- matches this
        # project's established convention of sourcing a multi-year
        # question from that single filing's own comparative columns
        # (e.g. Boeing's gross-margin-trend question is answered entirely
        # from BOEING_2022_10K alone, never a separate BOEING_2021_10K),
        # so when a question spans several years, only the most recent
        # filing gets the boost -- not every year mentioned, which would
        # boost multiple competing fiscal years equally and never actually
        # break the tie it exists to break.
        preferred_filing_year = max(query_years) if query_years else None

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
            multiplier *= self._line_item_match_score(content, query_patterns)

            # ── Total-row boost ──────────────────────────────────────────────
            # Skipped under prefer_narrative: this boost exists so a real
            # "Total X" statement row can outrank a sub-item/note row for
            # a NUMERIC lookup — meaningless (and actively harmful, since
            # it applies to ANY "Total ..." row regardless of topic) for a
            # question that isn't about a financial total at all.
            if not prefer_narrative and self._TOTAL_ROW_RE.search(content):
                multiplier *= 1.3

            # ── Narrative-content boost (only when prefer_narrative) ──────────
            # Counteracts BM25's inherent length bias: a table row chunk
            # is typically a handful of tokens, so even one matching term
            # dominates its score, while a full prose paragraph needs
            # several matches to reach the same magnitude purely because
            # of the doc_len normalization in _bm25_score. For a question
            # this module has already determined has no financial-metric
            # or formula match at all, the answer is almost certainly in
            # prose, not a table row, so this compensates rather than
            # relying on raw BM25 alone to surface it.
            if prefer_narrative and doc.get("type") == "text_note":
                multiplier *= 1.5

            # ── Causal-language boost (attribution questions only) ────────────
            # Only ever active alongside prefer_narrative, so this is
            # bit-for-bit unreachable from any numeric/formula-driven call
            # too — see _ATTRIBUTION_QUERY_RE's docstring for the confirmed
            # AMD real case this fixes.
            if prefer_narrative and attribution_active and self._CAUSAL_LANGUAGE_RE.search(content):
                multiplier *= 1.4

            # ── Geographic-section boost (geography questions only) ───────────
            # Only ever active alongside prefer_narrative -- see
            # _GEOGRAPHIC_SECTION_RE's docstring for the confirmed AmEx real
            # case (the sibling "Reportable Operating Segments" sub-section
            # scores close enough on plain BM25 to edge out the real
            # "Geographic Operations" table otherwise).
            if prefer_narrative and geography_active and _GEOGRAPHIC_SECTION_RE.search(content):
                multiplier *= 1.4
            # ── Dense region-name boost (geography questions only) ─────────────
            # Catches the case above's own literal-heading requirement can't:
            # a passage enumerating a company's own operating geographies
            # without ever using the word "geographic" at all -- see
            # _has_dense_geographic_region_names's docstring for the
            # confirmed PepsiCo real case.
            elif prefer_narrative and geography_active and _has_dense_geographic_region_names(content):
                multiplier *= 1.4

            # ── Legal-proceedings-section boost (legal questions only) ──────────
            # See _LEGAL_PROCEEDINGS_SECTION_RE's docstring for the
            # confirmed PepsiCo real case -- a stronger, more targeted
            # boost than the geography one above (1.8x, not 1.4x) since
            # the section heading itself is SEC-mandated and identical
            # across every filer, a much more reliable signal than any
            # bag-of-words topic query alone.
            if prefer_narrative and legal_active and _LEGAL_PROCEEDINGS_SECTION_RE.search(content):
                multiplier *= 1.5

            # ── Preferred-year boost (bare, year-less entity only) ──────────────
            # Applies regardless of prefer_narrative. entity is frequently a
            # bare human-readable company name with no fiscal year at all
            # (question_classifier's hardcoded "pepsico" -> "PepsiCo"
            # fast-path lookup, not "PEPSICO_2022_10K"), so
            # _company_match_score's own same-company-different-year
            # demotion never fires (it requires a detectable year on BOTH
            # sides) -- two fiscal years of the same company's filings tie
            # at the same top entity-match tier, competing for the same
            # top-k slots on nothing but raw content similarity, which is
            # often nearly identical boilerplate year over year. This
            # breaks that tie using the year(s) the classifier already
            # extracted from the question itself. Confirmed real case:
            # "Has Pepsico reported any materially important ongoing legal
            # battles from FY2022 and FY2021?" needed PEPSICO_2022_10K's
            # own Item 3 Legal Proceedings page, but PEPSICO_2021_10K's
            # near-identical boilerplate page narrowly outscored it under
            # a bare "PepsiCo" entity match with top_k=5, pushing the
            # right filing's own page out of the retrieved evidence
            # entirely.
            if preferred_filing_year and not self._extract_company_filing_year(entity or ""):
                doc_filing_year = self._extract_company_filing_year(doc.get("company", ""))
                if doc_filing_year == preferred_filing_year:
                    multiplier *= 1.3

            # ── Low-confidence column penalty ─────────────────────────────────
            # Applies regardless of prefer_narrative -- a mis-parsed year
            # header is exactly as misleading for a NUMERIC lookup (the
            # confirmed MGM dividends real case is answer_mode=NUMERIC) as
            # for a narrative one. See _LOW_CONFIDENCE_COLUMN_RE's docstring.
            if self._LOW_CONFIDENCE_COLUMN_RE.search(content):
                multiplier *= 0.5

            # ── Year / quarter boost ───────────────────────────────────────
            doc_period = str(doc.get('period', '')) + " " + content
            if query_years:
                # Normalise FY prefix for comparison
                doc_years_found = _extract_years(doc_period)
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
                if doc_stmt_type == statement_type_hint or self._matches_core_statement_line_item(
                    content, statement_type_hint
                ):
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
