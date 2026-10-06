import math
import re
from typing import List, Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Query/text matching utilities (word-boundary term presence, year extraction)
# ─────────────────────────────────────────────────────────────────────────────

_TERM_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _term_present(term: str, text: str) -> bool:
    """Word-boundary-aware presence check for a term/alias in a lowercased text.
    Matches whole words and allows an optional trailing 's' so singular aliases match
    plural labels, avoiding accidental substring matches inside unrelated words.
    """
    pattern = _TERM_PATTERN_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(r'\b' + re.escape(term) + r's?\b')
        _TERM_PATTERN_CACHE[term] = pattern
    return bool(pattern.search(text))


_FULL_YEAR_RE = re.compile(r'(?:FY\s*)?(20\d\d)\b', re.IGNORECASE)
_FY_SHORT_YEAR_RE = re.compile(r'\bFY\s*(\d{2})\b', re.IGNORECASE)


def _extract_years(text: str) -> "set[str]":
    """Match either a 4-digit year (optionally prefixed with FY) or the 2-digit fiscal
    shorthand (FYXX).
    Normalize any 2-digit shorthand to its 4-digit form before returning.
    Needed because a 2-digit FY token cannot be captured by the plain 4-digit pattern.
    """
    years = set(_FULL_YEAR_RE.findall(text))
    years.update('20' + m for m in _FY_SHORT_YEAR_RE.findall(text))
    return years


#: Detects questions asking what caused a metric change and flags them
#: as attribution-style queries.
#: Used so retriever/orchestrator can boost passages containing explicit causal phrasing;
#: check the original full question text when deciding boosts.

# ─────────────────────────────────────────────────────────────────────────────
# Query-topic detectors (attribution / geography / legal) -- decide whether
# to apply that topic's dedicated evidence-scoring boost below
# ─────────────────────────────────────────────────────────────────────────────

_ATTRIBUTION_QUERY_RE = re.compile(
    r'\bwhat\s+(?:drove|caused|led\s+to)\b|\bwhy\s+(?:did|has|have)\b|\bdrivers?\s+of\b',
    re.IGNORECASE,
)


def is_attribution_query(text: str) -> bool:
    return bool(_ATTRIBUTION_QUERY_RE.search(text))


#: Detects questions asking which geographies a filer operates in.
#: Used to prefer the filing's geographic-disclosure note over similar segment text;
#: evaluate this guard against the original question text.
_GEOGRAPHY_QUERY_RE = re.compile(r'\bgeograph(?:y|ies|ic(?:al)?)\b', re.IGNORECASE)
#: Matches literal section headings that indicate a geographic breakdown
#: (e.g., headings containing "Geographic Operations" or "Geographic Information").
#: Narrow phrasing reduces false positives from generic words like "areas" or "segments."
_GEOGRAPHIC_SECTION_RE = re.compile(
    r'geographic\s+(?:operations|information)\b', re.IGNORECASE
)

#: Catches geographic answers that do not use the word "geographic" by
#: looking for multiple distinct region names in the same passage.
#: Require at least three different region names close together to qualify.
_GEOGRAPHIC_REGION_NAME_RE = re.compile(
    r'\b(?:north america|latin america|south america|asia pacific|'
    r'middle east|south asia|africa|europe|australia|new zealand)\b',
    re.IGNORECASE,
)


def _geographic_region_name_count(content: str) -> int:
    return len({m.group(0).lower() for m in _GEOGRAPHIC_REGION_NAME_RE.finditer(content)})


def _has_dense_geographic_region_names(content: str, min_distinct: int = 3) -> bool:
    return _geographic_region_name_count(content) >= min_distinct


def is_geography_query(text: str) -> bool:
    return bool(_GEOGRAPHY_QUERY_RE.search(text))


#: Identifies litigation questions that should be answered from the
#: standard "Legal Proceedings" section present in filings.
#: Use the standardized section heading as a strong structural signal for ranking.
_LEGAL_QUERY_RE = re.compile(
    r'\blegal\s+(?:battle|proceeding|matter)s?\b|\blitigation\b|\blawsuit\b',
    re.IGNORECASE,
)
_LEGAL_PROCEEDINGS_SECTION_RE = re.compile(
    r'item\s*3\.?\s*legal\s+proceedings|'
    # Handles multi-part legal proceedings sections where relevant content
    # appears under later sub-headings within the same Item 3 block.
    # Boost standard category sub-headings too, since useful content can reside beyond
    # the first page.
    r'usual\s+and\s+customary\s+pricing\s+litigation|pbm\s+litigation|'
    r'controlled\s+substances\s+litigation|opioid\s+litigation',
    re.IGNORECASE,
)


def is_legal_query(text: str) -> bool:
    return bool(_LEGAL_QUERY_RE.search(text))


#: Identifies questions asking which companies a filer acquired. Used to boost a
#: filing's own acquisitions-note company sub-headings over other passages that
#: merely mention "acquisition" in passing -- see _ACQUISITION_SUBHEADING_RE.
_ACQUISITION_QUERY_RE = re.compile(
    r'\bacqui(?:sition|red)\b|\bcompanies\s+acquired\b|\bbusiness\s+combinations?\b',
    re.IGNORECASE,
)
#: A filing's own acquisitions note names each acquired company as a short
#: sub-heading immediately followed by a dated "we acquired / completed the
#: acquisition of ..." sentence -- the same structural shape
#: decomposer.suggest_narrative_topic_query's answer-synthesis counterpart
#: (llm_client._ACQUISITION_SUBHEADING_RE) already extracts a company list
#: from, duplicated here as a RETRIEVAL-time ranking signal: a chunk matching
#: this shape is close to certainly part of the acquisitions note itself, not
#: a passing mention, and multi-chunk pages (an acquisitions note spanning
#: several thousand characters splits into several overlapping chunks) can
#: otherwise leave one acquired company's own chunk just outside the
#: retrieval top-k while a neighboring chunk from the same note makes it in --
#: confirmed real case: a filing's note split so that only ONE of two
#: adjacent chunks (each naming a different acquired company) ranked inside
#: the top 15 for every query actually used, so the company named in the
#: other chunk was never shown to the model at all, even though both chunks
#: are equally part of the same note.
_ACQUISITION_SUBHEADING_RE = re.compile(
    r'\n[A-Z][A-Za-z0-9&.,\' \-]{1,40}\n\s*On\s+\w+\s+\d{1,2},\s*\d{4},\s*we\s+'
    r'(?:acquired|completed\s+the\s+acquisition\s+of)',
)


def is_acquisition_query(text: str) -> bool:
    return bool(_ACQUISITION_QUERY_RE.search(text))


#: "Which segment/division/business unit had the highest/lowest ..." is a
#: NUMERIC-mode question (the classifier still routes it as such --
#: pot_reasoner.py's own _SELECTION_QUERY_RE recognizes the identical
#: phrasing to skip PoT computation entirely), so it never gets the
#: prefer_narrative-gated boosts above; this is threaded independently
#: into the NUMERIC retrieval path instead. Deliberately scoped to
#: segment/division/business-unit wording only (not the more general
#: "region/category/type" a plain ranking question might also use) since
#: the boost below targets a specific 10-Q/10-K structural feature -- a
#: consolidated segment-results table, not any ranking table.
_SEGMENT_COMPARISON_QUERY_RE = re.compile(
    r'\bwhich\b[^.?]{0,120}\b(?:segment|division|business\s+unit)s?\b[^.?]{0,80}'
    r'\b(?:highest|lowest|largest|biggest|smallest|greatest|fewest|most|least|best|worst)\b'
    r'|\b(?:highest|lowest|largest|biggest|smallest)\b[^.?]{0,80}\b(?:segment|division|business\s+unit)s?\b',
    re.IGNORECASE,
)

#: Notes that a filer may print a consolidated, multi-segment results table with a
#: section heading directly above it; such pages can answer cross-segment comparison
#: queries but may be ranked lower by retrieval when many single-segment narrative pages
#: match query terms more tightly.
#: This is a retrieval-ranking issue rather than a table-parsing error; expect rank gaps
#: when short, focused prose pages dominate term-overlap signals.
_SEGMENT_RESULTS_SECTION_RE = re.compile(
    r'segment\s+results\b|results?\s+by\s+segment\b', re.IGNORECASE
)


def is_segment_comparison_query(text: str) -> bool:
    return bool(_SEGMENT_COMPARISON_QUERY_RE.search(text))


#: A config expands a formula query for inventory-related ratios with composition-
#: category terms (e.g., Finished Goods) so composition-specific rows get considered;
#: presence of that marker indicates the query was broadened.
#: Query-term widening alone may still leave important breakdown rows unretrieved due to
#: other ranking boosts.
_INVENTORY_COMPOSITION_QUERY_RE = re.compile(r'\bfinished\s+goods\b', re.IGNORECASE)

#: A filing's own inventory-note breakdown category labels -- the row
#: pot_reasoner._has_finished_goods_inventory needs to actually see, not
#: just the TOTAL inventory line already well-retrieved by the plain
#: "inventory"/"inventories" alias.
_INVENTORY_COMPOSITION_LABEL_RE = re.compile(
    r'\bfinished\s+goods\b|\bmerchandise\s+inventor(?:y|ies)\b|\bfuel\s+inventory\b|'
    r'\braw\s+materials\s+and\s+supplies\b|\bgoods\s+in\s+process\b|'
    r'\bspare\s+parts\s+and\s+supplies\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Alias-group pattern loading (financial_formula_library.py's variable
# aliases -> combined regex patterns, used by the line-item match boost)
# ─────────────────────────────────────────────────────────────────────────────

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
    """Lazily load the canonical-metric alias groups from the metric module.
    Each group lists aliases for one metric so queries using one alias match rows using
    another; imported lazily so failures fall back to basic term checks.
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


# ─────────────────────────────────────────────────────────────────────────────
# Main retriever: BM25 scoring + financial-domain relevance boosts (line-item
# match, company match, the topic detectors above)
# ─────────────────────────────────────────────────────────────────────────────

class HybridFinancialRetriever:
    # Apply a 1.5x boost when a query and candidate passage share the same line-item
    # term so concise table rows can outrank verbose non-relevant prose that repeats
    # company tokens.
    # Ensure primary formula aliases include expected terms (e.g.,
    # inventory/inventories) or those lookups get no protection.
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
        # Ensure the cash-flow statement row for acquisition-related cash flows is
        # eligible for ranking even when a filer has no narrative note; this row is the
        # only direct evidence of zero acquisition activity for some filers.
        # Do not rely solely on narrative boosts when the factual signal resides only in
        # a table row.
        "acquisitions, net of cash acquired", "acquisitions net of cash acquired",
    ]

    #: Treat rows whose label begins with Total as consolidated totals rather than sub-
    #: item or breakdown rows; apply total-row priority during retrieval to prevent
    #: specific sub-item rows from outranking the true total on raw term overlap.
    #: This is a ranking safeguard prior to any extraction-stage tie-breaking.
    _TOTAL_ROW_RE = re.compile(r'Line Item:\s*Total\b', re.IGNORECASE)

    #: Rows containing a generic ColN: placeholder in their linearized content indicate
    #: low-confidence extraction for one or more period/value columns; apply a soft
    #: demotion since year alignment may be unreliable.
    #: Do not exclude these rows outright — they can still be sole evidence for a line
    #: item but should be treated cautiously.
    _LOW_CONFIDENCE_COLUMN_RE = re.compile(r'\bCol\d+:')

    _CAUSAL_LANGUAGE_RE = re.compile(
        r'\bdriven\s+by\b|\bprimarily\s+due\s+to\b|\bmainly\s+due\s+to\b|'
        r'\bas\s+a\s+result\s+of\b|\battributable\s+to\b|\bresulted\s+from\b',
        re.IGNORECASE,
    )

    #: Table-row label is often a stronger indicator of its statement than a coarse page-
    #: level section tag assigned at ingestion.
    #: Rationale: page-level tags are page-range heuristics and can misclassify isolated
    #: summary or policy pages, causing incorrect boosts.
    #: Caveat: rely on row labels for statement hints; treat page-level tags as noisy.
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
        # BM25 length-normalization must use this corpus's actual average document
        # length, not an arbitrary constant; otherwise long informative passages are
        # unfairly penalized.
        # Also compute document-frequency (IDF) from the corpus so rare topical terms
        # score more than ubiquitous stopwords; both doc_lens and DF are built from a
        # single tokenization pass per document.
        # Caveat: keep tokenization/DF consistent across components to avoid ranking
        # artifacts.
        doc_freq: Dict[str, int] = {}
        doc_lens: List[int] = []
        for d in corpus:
            toks = self._tokenize(d.get('content', ''))
            doc_lens.append(len(toks))
            for t in set(toks):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        self._avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 50.0
        self._doc_freq = doc_freq
        self._n_docs = len(corpus) or 1

    def _idf(self, token: str) -> float:
        """
        Standard Okapi-BM25 IDF, the "+1 inside the log" (BM25+-style)
        variant specifically so it can never go negative for an
        ultra-common token (plain BM25's classic IDF formula goes negative
        once a term appears in over half the corpus, which would ACTIVELY
        penalize a document for containing a stopword rather than just
        failing to reward it -- not the intended fix here, and a much
        bigger behavior change than closing the stopword-noise gap this
        was added for).
        """
        df = self._doc_freq.get(token, 0)
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

    # ──────────────────────────────────────────────────────────────
    # Tokenisation (handles Chinese characters + English/numbers)
    # ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        text_lower = text.lower()
        # Tokenize Chinese characters individually; keep alphanumeric words and decimal
        # numbers as units.
        # Ensure comma-grouped numbers (e.g. 7,772) are tokenized as one token before
        # the plain alphanumeric rule, to avoid inflating document length and skewing
        # BM25.
        # Caveat: improper ordering creates length-normalization bias against rows with
        # grouped numerals.
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

    def _bm25_score(
        self, query_tokens: List[str], doc_tokens: List[str],
        query_idf: Optional[Dict[str, float]] = None,
    ) -> float:
        score = 0.0
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        doc_set = set(doc_tokens)
        avgdl = self._avg_doc_len or 50.0
        for token in query_tokens:
            if token in doc_set:
                tf = doc_tokens.count(token)
                tf_component = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / avgdl)))
                # IDF weighting -- see _idf()'s docstring. Without this, a
                # stopword hit ("a", "of", "is") scored identically to a
                # rare, meaningful term hit ("margin", "jnj"), which let
                # incidental stopword overlap dominate ranking whenever the
                # query's real content words weren't literally present in
                # the one genuinely relevant passage.
                # query_idf, when given, is precomputed ONCE per search()
                # call for that query's own (small) token set -- calling
                # self._idf() fresh here instead would repeat the same
                # dict-lookup+log() for every one of the ~30 query tokens
                # on EVERY one of the corpus's ~60,000 documents each
                # search() call, a 10-20x per-call slowdown confirmed via
                # direct profiling (one question that used to take under a
                # minute never finished in 10+ minutes). Falls back to a
                # fresh per-token lookup only for a caller that doesn't
                # have a query_idf table handy (e.g. a unit test calling
                # this directly).
                idf = query_idf[token] if query_idf is not None else self._idf(token)
                score += tf_component * idf
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
        """Normalize a raw company/doc id by lowercasing and removing year and filing-type
        boilerplate after converting underscores/hyphens to spaces.
        Tokenize first, then drop tokens that are bare years, fused year+quarter (e.g.
        2023q2), or filing-type boilerplate (e.g. 10k/10q) to avoid false matches.
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
        """Return the first bare 4-digit year (optionally with a Q1-4 suffix like 2023q2)
        from a raw company/doc id, or None.
        Used to detect and demote same-company candidates from different filing periods.
        """
        m = re.search(r'(?:19|20)\d{2}(?:q[1-4])?', name, flags=re.IGNORECASE)
        return m.group(0).lower() if m else None

    def _company_match_score(self, doc_company: str, entity: str) -> float:
        """Return a multiplier for how well a document's company metadata matches the
        target entity:
        2.0 → strong match (boost)
        1.0 → neutral
        0.4 → same company but wrong filing period (demotion)
        0.05 → mismatch (heavy penalty)

        Keep this a soft penalty so partially-matching candidates remain scoreable; the
        stricter demotion for wrong-period same-company docs prevents wrong-period
        documents from unfairly outranking correct-period or neutral documents on raw
        relevance.
        """
        if not entity or entity.lower() in ("company", "unknown", ""):
            return 1.0  # no filter if entity is generic

        norm_doc = self._normalise_company(doc_company)
        norm_ent = self._normalise_company(entity)

        if not norm_doc or not norm_ent:
            return 1.0

        # _normalise_company strips fiscal-period markers, so identical base company
        # strings can represent different filing periods.
        # Apply a demotion factor when both items have detectable but differing periods;
        # leave unset if either side lacks a period signal.
        # Caveat: period stripping prevents match tiers from distinguishing filing years
        # unless an explicit period signal is present.
        doc_period = self._extract_company_filing_year(doc_company)
        ent_period = self._extract_company_filing_year(entity)
        period_mismatch = bool(doc_period and ent_period and doc_period != ent_period)

        # Exact normalised match.
        if norm_doc == norm_ent:
            return 0.4 if period_mismatch else 2.0

        # One is a substring of the other
        if norm_ent in norm_doc or norm_doc in norm_ent:
            return 0.4 if period_mismatch else 1.8

        # Handle collapsed multiword company names (no-space doc_name convention)
        # specially: exact-match and word-overlap checks miss internal word boundaries.
        # Detect and treat collapsed forms so same-company documents do not receive a
        # generic different-company penalty.
        # Caveat: without this, multiword companies may be incorrectly penalized,
        # weakening the entity filter.
        collapsed_doc = norm_doc.replace(' ', '')
        collapsed_ent = norm_ent.replace(' ', '')
        if collapsed_ent and (collapsed_ent in collapsed_doc or collapsed_doc in collapsed_ent):
            return 0.4 if period_mismatch else 1.8

        # Word-level overlap
        ent_words = [w for w in norm_ent.split() if len(w) >= 2]
        doc_words = set(norm_doc.split())
        if not ent_words:
            return 1.0

        matches = sum(1 for w in ent_words if w in doc_words)
        if matches == len(ent_words):
            return 0.4 if period_mismatch else 1.8   # all entity words found in doc company name
        if matches > 0:
            return 0.4 if period_mismatch else 1.2   # partial match — mild boost
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
        is_acquisition: bool = False,
        is_segment_comparison: bool = False,
        query_years: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the corpus with BM25 plus overlap scoring.

        Args:
            query: user query
            top_k: max results to return
            exclude_ids: passage IDs to skip
            entity: company filter (soft via _company_match_score)
            section: legacy section label (mismatch gives a small penalty)
            statement_type_hint: hint for statement type (income_statement | balance_sheet | cash_flow | notes | unknown); matching docs get a boost
            prefer_narrative: True for qualitative/narrative questions (defaults False); when True, apply behaviors that favor paragraph narrative over short table rows
            is_attribution: True when the original question asks what caused a change; caller must set this explicitly
            is_geography: True when the original question asks about geographies/regions
            is_acquisition: True when the original question asks which companies were acquired
            query_years: years extracted from the original question; used to disambiguate among multiple filings for the same company

        Behavior notes:
            Matching statement types receive a positive boost; non-matching documents are not penalized.
            prefer_narrative changes ranking to avoid short table rows systematically outranking prose for narrative questions.
            query_years help prefer documents covering the requested period when entity names lack year information.
        """
        exclude_ids = set(exclude_ids or [])
        query_tokens = self._tokenize(query)
        # Computed ONCE per search() call (this query's own token set is
        # small, ~10-30 tokens) and reused for every one of the corpus's
        # ~60,000 documents below -- see _bm25_score's query_idf param
        # docstring for the confirmed 10-20x per-call slowdown this avoids
        # versus calling self._idf() fresh per document per token.
        query_idf = {t: self._idf(t) for t in set(query_tokens)}
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
        acquisition_active = is_acquisition or is_acquisition_query(query)
        segment_comparison_active = is_segment_comparison or is_segment_comparison_query(query)
        # No caller flag for this one -- "Finished Goods" appearing in the
        # query text IS the signal (see _INVENTORY_COMPOSITION_QUERY_RE's
        # docstring), always auto-detected.
        inventory_composition_active = bool(_INVENTORY_COMPOSITION_QUERY_RE.search(query))
        # When questions span multiple years, boost only the latest year mentioned per
        # the project's convention of sourcing multi-year queries from one filing.
        # Rationale: boosting every year mentioned would amplify competing periods
        # equally and fail to disambiguate.
        # Caveat: use the latest-year signal to break ties between filings.
        preferred_filing_year = max(query_years) if query_years else None

        scored_results = []
        for doc in self.corpus:
            if doc['id'] in exclude_ids:
                continue

            content = doc['content']
            doc_tokens = self._tokenize(content)
            bm25 = self._bm25_score(query_tokens, doc_tokens, query_idf)
            # IDF-weighted, not a flat count -- same reasoning as
            # _bm25_score's own IDF fix just above: a plain `1 for q in
            # query_tokens if q in content.lower()` counted a "not"/"is"/
            # "this" substring hit exactly the same as a "margin"/"jnj"
            # hit, so this secondary signal was just as vulnerable to
            # stopword noise dominating the final ranking as bm25 was
            # before its own fix. Reuses the precomputed query_idf table
            # (see just above) rather than calling self._idf() fresh here.
            content_lower = content.lower()
            overlap_count = sum(query_idf[q] for q in query_tokens if q in content_lower)

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
                # Handle cases where a generic "Total ..." table row in a document can
                # outrank a query-specific total row after more rows are added to the
                # corpus.
                # This prevents unrelated total rows from being selected when the query
                # expects a specific total row.
                _lab = re.search(r'Line Item:\s*([^|]+)', content)
                _lab_stems = {
                    t[:5] for t in self._tokenize(_lab.group(1)) if len(t) > 2
                } - {"total", "other", "and", "the", "of"} if _lab else set()
                _q_stems = {t[:5] for t in query_tokens}
                if not _lab_stems or (_lab_stems & _q_stems):
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

            # Apply a causal-language relevance boost only for attribution-style
            # questions and only when narrative-preference is enabled.
            # This boost is never used for purely numeric or formula-driven retrieval
            # paths.
            if prefer_narrative and attribution_active and self._CAUSAL_LANGUAGE_RE.search(content):
                multiplier *= 1.4

            # ── Geographic-section boost (geography questions only) ───────────
            # Only ever active alongside prefer_narrative -- see
            # _GEOGRAPHIC_SECTION_RE's docstring: a sibling "Reportable
            # Operating Segments" sub-section can score close enough on
            # plain BM25 to edge out the real "Geographic Operations"
            # table otherwise, since both share most of their vocabulary.
            if prefer_narrative and geography_active and _GEOGRAPHIC_SECTION_RE.search(content):
                multiplier *= 1.4
            # Boost passages that enumerate distinct geographic regions when the passage
            # does not use explicit geography headings.
            # Scale the boost by the number of distinct regions found so stronger multi-
            # region lists receive higher weight than borderline lists.
            elif prefer_narrative and geography_active and _has_dense_geographic_region_names(content):
                region_count = _geographic_region_name_count(content)
                multiplier *= min(1.4 + 0.3 * (region_count - 3), 2.9)

            # Apply a stronger boost for passages that appear under legal-proceedings
            # section headings, since those headings are standardized and reliable
            # signals.
            # Use this boost for legal questions to prefer section-heading matches over
            # bag-of-words topic matches.
            if prefer_narrative and legal_active and _LEGAL_PROCEEDINGS_SECTION_RE.search(content):
                multiplier *= 1.5

            # Boost chunks matching a filing's own acquisitions-note sub-heading shape
            # (see _ACQUISITION_SUBHEADING_RE) for acquisition questions. An acquisitions
            # note spanning several thousand characters splits into multiple overlapping
            # chunks, and without this boost one acquired company's own chunk can rank
            # just outside the retrieval cutoff while a neighboring chunk from the same
            # note (naming a different acquired company) ranks comfortably inside it.
            if prefer_narrative and acquisition_active and _ACQUISITION_SUBHEADING_RE.search(content):
                multiplier *= 1.5

            # Boost documents that contain side-by-side segment results for segment-
            # comparison questions; this applies even when narrative preference is off.
            # This ensures multi-segment summary pages rank above multiple single-
            # segment pages that individually have high overlap.
            if segment_comparison_active and _SEGMENT_RESULTS_SECTION_RE.search(content):
                multiplier *= 2.0

            # Boost label matches that indicate inventory composition when retrieving
            # inputs for inventory-turnover formulas; applies in numeric retrieval mode.
            # This helps surface the composition row over shorter, generic inventory
            # total rows that would otherwise rank higher.
            if inventory_composition_active and _INVENTORY_COMPOSITION_LABEL_RE.search(content):
                multiplier *= 3.0

            # Preferred-year boost for bare entity matches.
            # When a query names an entity without a year, boost documents whose year
            # matches the year(s) extracted from the query to break ties among same-
            # entity filings with similar boilerplate.
            if preferred_filing_year and not self._extract_company_filing_year(entity or ""):
                doc_filing_year = self._extract_company_filing_year(doc.get("company", ""))
                if doc_filing_year == preferred_filing_year:
                    multiplier *= 1.3

            # Low-confidence column penalty applies to numeric and narrative lookups.
            # A mis-parsed year header can mislead any lookup mode; see the column-
            # confidence pattern's docstring for details.
            if self._LOW_CONFIDENCE_COLUMN_RE.search(content):
                multiplier *= 0.5

            # Avoid answering "X and equivalents" from a combined "Total X, X
            # equivalents and restricted X" row.
            # This prevents returning an aggregated total when the question requests the
            # plain row.
            if (
                doc.get("type") == "table_row"
                and re.search(r'Line Item:[^|]*\brestricted\b', content, re.IGNORECASE)
                and "restricted" not in query.lower()
            ):
                multiplier *= 0.6

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
