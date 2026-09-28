"""ProgramOfThoughtReasoner (PoT) — v3

Variable extraction hierarchy:
1. Linearized-table (pipe-delimited) — preferred when structured JSON-like data is
available
2. Formula-guided (alias search) — used for questions requiring composed calculations
3. Raw-number fallback — last resort when other methods fail

Calculation hierarchy:
1. Formula template (financial_formula_library.py)
2. Ratio/margin: match numerator and denominator from the SAME year
3. YoY/CAGR: locate the same item across different years
4. Direct lookup: return the first matching item

Notes: fixes include correct same-year matching for margins, improved canonical labels
to avoid false matches, prioritizing year mentioned in the query, and stripping common
prefixes before year matching.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from app.tools.sandbox import execute_pot_code
from app.agent.financial_formula_library import detect_formula, get_variable_aliases, FORMULA_LIBRARY
from app.tools.table_parser import is_markdown_separator_row

# ─────────────────────────────────────────────────────────────────────────────
# Canonical item taxonomy  (order matters — first match wins)
# ─────────────────────────────────────────────────────────────────────────────
_ITEM_TAXONOMY: List[Tuple[str, List[str]]] = [
    # Revenue / top line
    ("revenue",        ["total revenue", "net revenue", "net sales", "revenue",
                         "sales to customers", "total revenues", "total net revenues",
                         "net revenues", "total net revenue", "total net sales",
                         "total sales", "營業收入", "營收"]),
    # Gross
    ("gross_profit",   ["gross profit", "gross margin amount", "營業毛利", "毛利"]),
    # Operating
    ("cost_of_revenue",["cost of revenue", "cost of goods sold", "cogs",
                         "cost of sales", "cost of products", "cost of services",
                         "cost of products sold", "營業成本"]),
    ("op_expense",     ["operating expenses", "operating expense", "opex", "營業費用"]),
    ("op_income",      ["operating income", "operating profit", "operating earnings",
                         "income from operations", "earnings from operations",
                         "ebit", "營業利益", "營業淨利"]),
    ("rd_expense",     ["research & development", "r&d", "research and development",
                         "research and development expense", "research and development expenses",
                         "研發費用"]),
    ("sga",            ["sg&a", "selling general", "selling and marketing",
                         "推銷管理費用", "推銷與管理費用"]),
    ("restructuring_costs", ["restructuring and impairment charges",
                              "restructuring charges", "restructuring costs",
                              "重組費用", "重組成本"]),
    # Net income / EPS
    ("net_income_btax",["net income before tax", "income before tax", "pretax income",
                         "稅前淨利"]),
    ("income_tax",     ["income tax", "tax expense", "所得稅費用"]),
    ("net_income",     ["net income", "net profit", "net earnings",
                         "本期淨利", "淨利"]),
    ("eps",            ["eps", "earnings per share", "diluted eps", "基本每股盈餘",
                         "稀釋每股盈餘", "每股盈餘"]),
    # Already an eps/book_value_per_share formula placeholder -- adding the
    # canonical itself lets that formula's "try to fill from linearized
    # table" fallback find a filing's own "Weighted average shares
    # outstanding" row when the primary extraction path doesn't.
    ("shares_outstanding", ["shares outstanding", "weighted average shares",
                             "diluted shares", "common shares outstanding"]),
    # Balance sheet
    ("cash",           ["cash and cash equivalents", "cash & equivalents",
                         "現金及約當現金"]),
    # Was previously untagged (no canonical at all) despite already being
    # used as a formula placeholder (cash_ratio's own "short_term_investments"
    # required_vars alias list) -- this canonical is purely a fallback
    # source for that formula's "try to fill from linearized table" step
    # when the primary alias-matched extraction path comes up empty; a
    # no-op for every filing without a distinct short-term-investments row.
    ("short_term_investments", ["short-term investments", "short term investments",
                                 "marketable securities"]),
    ("total_assets",   ["total assets", "總資產"]),
    ("current_assets", ["total current assets", "current assets", "流動資產"]),
    ("inventory",      ["inventory", "inventories", "存貨"]),
    ("accounts_rec",   ["accounts receivable", "trade receivables", "receivables", "receivable", "應收帳款"]),
    ("accounts_payable", ["accounts payable", "trade payables", "payables", "payable", "應付帳款"]),
    ("current_liab",   ["total current liabilities", "current liabilities", "流動負債"]),
    ("total_liab",     ["total liabilities", "總負債"]),
    # Same rationale as short_term_investments above -- already a formula
    # placeholder name (debt_change_yoy's composite fallback,
    # long_term_debt_to_capitalization) but previously had no direct
    # taxonomy canonical of its own for a plain "how much long-term debt
    # does X have" style question to route to.
    ("long_term_debt", ["long-term debt", "long term debt"]),
    # Standalone P&L line, distinct from ebit/operating_income -- a plain
    # "what is X's interest expense" question had no canonical to route to
    # even though interest_coverage's formula already extracts it under a
    # differently-scoped alias list of its own.
    ("interest_expense", ["interest expense"]),
    ("goodwill",       ["goodwill"]),
    ("intangible_assets", ["intangible assets", "other intangible assets"]),
    ("retained_earnings", ["retained earnings", "accumulated deficit"]),
    # Ensure suffix variants like ", net" are checked early so exact balance-sheet rows
    # get higher match scores than shorter, incidental footnote rows.
    # This prevents wrong selection when multiple rows contain the same core phrase but
    # differ by short suffixes.
    ("ppe",            ["property, plant and equipment, net", "property, plant, and equipment, net",
                         "property and equipment, net",
                         "property, plant and equipment", "property, plant, and equipment",
                         "property and equipment", "net ppe", "pp&e", "fixed assets",
                         "不動產、廠房及設備", "固定資產"]),
    ("equity",         ["total equity", "total shareholders equity",
                         "stockholders equity", "股東權益總額", "股東權益"]),
    # Include the specific phrasing variant "provided (used) by" for operating cash
    # flows so it matches contiguous aliases that other cash-flow groups already catch.
    # Avoid adding a bare "operations"/"operating activities" alias because those tokens
    # are too generic and would misclassify unrelated lines.
    ("operating_cf",   ["operating cash flow", "cash from operations",
                         "cash provided by operating",
                         "cash provided (used) by operations",
                         "cash provided (used) by operating activities",
                         "營業活動現金"]),
    # Siblings of operating_cf above -- the OTHER two SEC-standard cash-
    # flow-statement summary lines, needed for a "which of operations/
    # investing/financing activities brought in the most cash?" question
    # (see the dedicated comparison branch in _build_calculation_code).
    ("investing_cf",   ["cash used in investing", "cash provided by investing",
                         "investing activities", "投資活動現金"]),
    ("financing_cf",   ["cash used in financing", "cash provided by financing",
                         "financing activities", "籌資活動現金"]),
    ("capex",          ["capital expenditure", "purchases of ppe",
                         "purchases of property and equipment",
                         "purchases of property, plant and equipment",
                         "additions to property and equipment",
                         "capital spending", "資本支出"]),
    ("depreciation",   ["depreciation and amortization", "depreciation & amortization",
                         "depreciation", "amortization", "d&a", "折舊"]),
    ("fcf",            ["free cash flow", "fcf", "自由現金流"]),
    # Add a canonical tag for dividend cash outflows so direct-lookup answers can route
    # to the extracted "Dividends paid" row instead of falling back to generic output.
    # This fills a missing mapping that prevents returning zero or no direct result.
    ("dividends_paid", ["dividends paid", "cash dividends paid", "dividends", "股利"]),
    # Treat dividend-per-share queries as requests for the per-share rate, not the
    # aggregate cash outflow; provide aliases that explicitly include "per share".
    # This lets the per-share exclusion filter allow matching per-share rows only for
    # this canonical, avoiding misrouting to aggregate cash-flow lines.
    ("dividends_per_share", [
        "dividends declared per share", "dividends declared per common share",
        "cash dividends declared per share", "dividends per common share",
        "dividends per share", "dividend per share",
        "每股股利", "每股現金股利",
    ]),
    # Misc
    ("data_center_rev",["data center revenue", "data center"]),
]

# Pre-build a flat lookup: lowercase alias → canonical key
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in _ITEM_TAXONOMY:
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a.lower()] = _canonical


#: Exclude narrow "Reportable segment"/"Segment" subtotals from matching consolidated
#: metrics by adding variants that distinguish segment-level labels from full-company
#: totals.
#: This avoids mixing different scopes (segment vs consolidated) when selecting the
#: canonical row.

# ─────────────────────────────────────────────────────────────────────────────
# Attribution / negation / query-intent matching (does this evidence item
# actually answer what THIS question is asking, vs. a superficially similar
# but wrong-context row?)
# ─────────────────────────────────────────────────────────────────────────────

_NEGATION_PREFIX_RE = re.compile(
    r'\b(non[- ]?|not\s+|deferred\s+|unearned\s+|change(?:s|d)?\s+in\s+|'
    r'(?:reportable\s+)?segment\s+)$'
)

# Disambiguate "X attributable to" matches by checking the text after the match to
# ensure the beneficiary is the parent company, not a redeemable or noncontrolling
# interest.
# This prevents attributability aliases from matching minority/carve-out rows that are
# not the parent company's figure.
_ATTRIBUTABLE_TO_CARVEOUT_RE = re.compile(
    r'^\s*(the\s+)?(redeemable|noncontrolling|non-controlling|minority)\b'
)


def _is_carveout_attribution_match(item_lower: str, match_start: int, match_len: int) -> bool:
    """True if an alias match ending in "...attributable to" is
    immediately followed by a redeemable/noncontrolling/minority-interest
    carve-out rather than the reporting company's own figure."""
    if not item_lower[:match_start + match_len].rstrip().endswith("attributable to"):
        return False
    return bool(_ATTRIBUTABLE_TO_CARVEOUT_RE.match(item_lower[match_start + match_len:]))


# Generic terms for "the reporting company's own equity holders" — used
# alongside the reporting entity's own name (see
# _is_attributable_to_reporting_entity below) to recognize a row like
# "Net income (loss) attributable to The AES Corporation" as the real
# parent-level headline figure, not just another carve-out variant.
_GENERIC_PARENT_ATTRIBUTION_RE = re.compile(
    r'^\s*(the\s+)?(shareowners|shareholders|stockholders|'
    r'common stockholders|common shareholders|the company)\b'
)


def _is_attributable_to_reporting_entity(text_after: str, ev_company: str) -> bool:
    """Return True if the phrase following "attributable to" refers to a generic parent-
    equity term (shareholders/stockholders/the company) or the reporting entity's
    normalized name.

    Uses the module's entity-name normalization to match compact identifiers against
    naturally spaced/punctuated names. Avoids false negatives caused by differing
    whitespace/punctuation formats in filenames versus prose.
    """
    if _GENERIC_PARENT_ATTRIBUTION_RE.match(text_after):
        return True
    text_core = re.sub(r'[^a-z0-9]', '', text_after.lower())
    for w in _entity_words(ev_company):
        if len(w) >= 3 and w in text_core:
            return True
    return False


# Some labels contain a trailing carve-out phrase (e.g., "attributable to") far after a
# short alias match like "net income".
# This check scans the entire label for those disqualifying phrases so such rows are
# treated as slice/attributable items, not the reporting entity's consolidated figure.
_CARVEOUT_ANYWHERE_RE = re.compile(
    r'(?:attributable|paid|allocated|distributed)\s+to\s+(?:the\s+)?'
    r'(?:redeemable|noncontrolling|non-controlling|minority)\b'
)

# Rows expressing per-share metrics (EPS, dividends per share) are ratios, not aggregate
# amounts.
# Avoid matching short aliases that can substring-match per-share labels; also handle
# footnote-suffixed words (e.g., "PER SHARE1") so regex boundary checks don't
# misclassify per-share rows as aggregates.
_PER_SHARE_ANYWHERE_RE = re.compile(r'per\s+(?:common\s+|diluted\s+|basic\s+)?share')

#: A "Has X paid dividends to common shareholders?" query expects a Yes/No about the per-
#: share rate, not the aggregate cash outflow.
#: Keep this mapping narrow so explicit aggregate-amount questions still return the
#: company-level total.
_YESNO_DIVIDEND_QUERY_RE = re.compile(r'\bhas\b[^.?]{0,60}\bpaid\s+dividends?\b', re.IGNORECASE)

#: "Are there any product/service categories/segments that represent
#: more than N% of X's revenue?" -- see generate_and_execute's own use
#: of this for why PoT is skipped entirely for this question shape
#: rather than mechanically returning a misleading single number.
_CATEGORY_THRESHOLD_QUERY_RE = re.compile(
    r'\b(?:categor(?:y|ies)|segments?)\b[^.?]{0,80}'
    r'(?:represent|account(?:s|ed)?\s+for|exceed|more\s+than|greater\s+than)[^.?]{0,40}%',
    re.IGNORECASE,
)

#: Some question forms ask for a selection, guidance text, or a comparison that the
#: numeric-calculation pipeline can't produce.
#: For those shapes, do not attempt to compute a numeric result; return None for the
#: computed value so the evidence text selection is used as the answer.
_SELECTION_QUERY_RE = re.compile(
    r'\bwhich\b[^.?]{0,120}\b(?:highest|lowest|largest|biggest|smallest|greatest|fewest|most|least|best|worst)\b'
    r'|\b(?:highest|lowest|largest|biggest|smallest)\b[^.?]{0,60}\b(?:segments?|regions?|categor(?:y|ies)|types?|divisions?|geograph\w*)\b',
    re.IGNORECASE,
)
_GUIDANCE_QUERY_RE = re.compile(
    r'\b(?:guidance|outlook|forecast(?:s|ed|ing)?)\b'
    r'|\bexpected\s+to\s+(?:accelerate|decelerate|grow|increase|decrease|decline|slow)\b',
    re.IGNORECASE,
)
_GROUP_COMPARE_QUERY_RE = re.compile(
    r'\b(?:u\.?s\.?|domestic|international|foreign)\b[^.?]{0,60}\bcompare[sd]?\b'
    r'|\bcompare[sd]?\b[^.?]{0,60}\b(?:u\.?s\.?|domestic|international|foreign)\b',
    re.IGNORECASE,
)


_EXISTENCE_QUERY_RE = re.compile(
    r'\b(?:were|are|is|was|has|have)\s+there\s+any\s+(?:[a-z-]+\s+){0,3}?(?:events?|items?|factors?|matters?|transactions?|nominees?)\b',
    re.IGNORECASE,
)


#: Direction (increase/decrease) questions about an item expressed as a percent of sales
#: should be treated as Yes/No direction lookups on that ratio, not raw amounts.
_RATIO_DIRECTION_QUERY_RE = re.compile(
    r'^\s*(?:did|does|do|was|were|is|are|has|have)\b[^?]{0,140}'
    r'\bas\s+a\s+(?:percent(?:age)?|%|share|proportion)\s+of\b[^?]{0,80}'
    r'\b(?:increase[sd]?|decrease[sd]?|rise[n]?|fall|fell|grow|improve[sd]?|change[sd]?|decline[sd]?)\b',
    re.IGNORECASE,
)


def _no_calculation_path(q_lower: str) -> bool:
    """True for selection / guidance / group-comparison question shapes."""
    return bool(
        _SELECTION_QUERY_RE.search(q_lower)
        or _GUIDANCE_QUERY_RE.search(q_lower)
        or _GROUP_COMPARE_QUERY_RE.search(q_lower)
        or _EXISTENCE_QUERY_RE.search(q_lower)
    )


#: Not every filer states its dividend rate as a clean standalone
#: "Dividends declared per share" table row the way CVS does — many
#: only ever state it in a narrative sentence (e.g. "we paid dividends
#: of $0.0025 per share [in each of several months], totaling $X
#: million for <year>"), which the dividends_per_share canonical above
#: (row-label matching only) can never see. A dollar amount ending in
#: "per share" within one SENTENCE that also names the target year is
#: a reliable enough anchor — a filing routinely restates a PRIOR
#: year's now-superseded rate elsewhere for context (e.g. "we reduced
#: our dividend to $X per share in <earlier year>"), so requiring the
#: target year inside the SAME sentence, not just the same page/chunk,
#: is what keeps this from grabbing a stale rate.
#:
#: The "any char but a period" sentence-boundary idiom ([^.]*) is
#: WRONG for financial text specifically: a dollar amount's own decimal
#: point ("$0.55") is also a literal ".", so [^.]* stops dead at the
#: FIRST dollar amount's decimal point and can never reach a LATER one
#: in the same sentence — e.g. "...dividend was $0.55, $0.50 and $0.50
#: per share, respectively" would never match at all, since crossing
#: from "dividend" to "per share" requires passing three separate
#: decimal points. _NOT_SENTENCE_END matches any character (including
#: newlines) that ISN'T a period immediately followed by whitespace —
#: a decimal point is always followed by a digit, never whitespace, so
#: it's transparently crossed, while a genuine sentence-ending period
#: (always followed by a space in PDF-extracted text) still stops the
#: scan.

# ─────────────────────────────────────────────────────────────────────────────
# Dividend narrative extraction (per-share sentences, "respectively" lists)
# ─────────────────────────────────────────────────────────────────────────────

_NOT_SENTENCE_END = r'(?:(?!\.\s)[\s\S])'
_DIVIDEND_PER_SHARE_SENTENCE_RE = re.compile(
    rf'({_NOT_SENTENCE_END}*?\$\s*(\d+\.\d+)\s*per\s+share{_NOT_SENTENCE_END}*\.)', re.IGNORECASE
)

#: A standard, generic SEC-filing convention for stating several years'
#: dividend rate in ONE sentence: a list of years and a list of dollar
#: amounts, tied together only by a trailing "respectively" and
#: matching LIST ORDER -- e.g. "During 2022, 2021 and 2020, the
#: quarterly cash dividend was $0.55, $0.50 and $0.50 per share,
#: respectively" (years-then-amounts) or "...was $0.55 and $0.50 per
#: share in 2022 and 2021, respectively" (amounts-then-years). Neither
#: shape is reachable by _DIVIDEND_PER_SHARE_SENTENCE_RE above, which
#: requires a dollar amount immediately adjacent to "per share" --
#: here only the LAST amount in the list sits next to that phrase, so
#: that regex alone would silently grab an EARLIER year's now-
#: superseded rate instead of the target year's own. Not specific to
#: CVS -- stating N years' rates in one sentence via "respectively" is
#: a routine, generic SEC drafting convention for any recurring metric,
#: not just dividends.
_DIVIDEND_RESPECTIVELY_SENTENCE_RE = re.compile(
    rf'({_NOT_SENTENCE_END}*?\bdividends?\b{_NOT_SENTENCE_END}*?\brespectively\b{_NOT_SENTENCE_END}*\.)', re.IGNORECASE
)
_YEAR_TOKEN_RE = re.compile(r'\b(?:19|20)\d{2}\b')
_DOLLAR_AMOUNT_TOKEN_RE = re.compile(r'\$\s*(\d+\.\d+)')


def _parse_respectively_dividend_sentence(sentence: str) -> List[Tuple[str, float]]:
    """
    Extract (year, value) pairs from one "...$A, $B and $C per share
    [...] <year1>, <year2> and <year3>, respectively"-shaped sentence
    (or the amounts-after-years variant) by zipping the years and
    dollar amounts found IN THEIR OWN LEFT-TO-RIGHT ORDER — matching
    list length is what confirms this sentence really is the "N years,
    N amounts, respectively" shape rather than some other unrelated
    construction that happens to contain both a year and a dollar
    amount. See _DIVIDEND_RESPECTIVELY_SENTENCE_RE's docstring.
    """
    years = _YEAR_TOKEN_RE.findall(sentence)
    amounts = _DOLLAR_AMOUNT_TOKEN_RE.findall(sentence)
    if not years or len(years) != len(amounts):
        return []
    pairs = []
    for yr, amt in zip(years, amounts):
        val = _to_float(amt)
        if val is not None and val > 0:
            pairs.append((yr, val))
    return pairs

#: A rate INCREASE the Board authorizes near a fiscal year's end
#: routinely doesn't take effect until the FOLLOWING year (e.g. "In
#: December 2022, the Board authorized a 10% increase in the quarterly
#: cash dividend to $0.605 per share effective in 2023") -- the target
#: year appears in the sentence (the authorization date), but the rate
#: itself was never actually paid during that year. A trailing
#: "effective (in) <year>" naming a DIFFERENT year than the one being
#: asked about is a reliable, generic signal this candidate describes
#: a future rate, not the target year's own.
_DIVIDEND_EFFECTIVE_YEAR_RE = re.compile(
    r'effective\s+(?:in\s+|as\s+of\s+)?(?:[A-Za-z]+\s+\d{1,2},?\s+)?(\d{4})', re.IGNORECASE
)

#: A question that names a specific quarter requests that quarter's per-share rate, not
#: the annual total.
#: Route quarter-specific per-share queries to the narrative per-share row rather than
#: annual aggregate rows.
_QUARTER_QUERY_RE = re.compile(
    r'\bq[1-4]\b|\b(?:first|second|third|fourth)\s+quarter\b', re.IGNORECASE
)


def _extract_narrative_dividend_per_share(
    evidence_list: List[Dict[str, Any]], target_year: Optional[str],
    prefer_quarterly: bool = False,
) -> Optional[Tuple[float, str]]:
    """
    Last-resort scan of narrative evidence for a dividend-per-share rate
    tied to `target_year`, for filers with no clean structured
    "Dividends declared per share" row at all (or, when
    `prefer_quarterly` is set, for a per-QUARTER rate no structured row
    ever states at all — see _QUARTER_QUERY_RE's docstring). See
    _DIVIDEND_PER_SHARE_SENTENCE_RE's docstring for why the year must
    be inside the SAME sentence as the dollar amount.

    A filing routinely states BOTH a per-QUARTER rate ("we paid
    dividends of $0.0025 per share [in each of four months]") and the
    already-annualized total ("we maintained an annual dividend of
    $0.01 per share throughout <year>") for the SAME year — the two
    aren't interchangeable, and which one a "Has X paid dividends...?"
    question wants depends on whether it named a specific quarter.
    Collects every matching sentence across all evidence first and
    prefers whichever granularity was asked for over the other, rather
    than returning on the first match found — retrieval order is non-
    deterministic, so "first found" would otherwise flip between the
    two per run for the exact same underlying filing.
    """
    if not target_year:
        return None
    # (matches_granularity, is_positionally_verified, val, detail)
    # is_positionally_verified is True only when a candidate value is paired to a year
    # by matching list index positions.
    # This is more reliable than heuristics that pick the nearest numeric token before a
    # keyword, which can misassign values when multiple amounts appear.
    candidates: List[Tuple[bool, bool, float, str]] = []
    for ev in evidence_list:
        content = ev.get("parent_content") or ev.get("content", "")
        if not content or "dividend" not in content.lower():
            continue
        for sentence, amount in _DIVIDEND_PER_SHARE_SENTENCE_RE.findall(content):
            # Require the word indicating a dividend to appear in the same sentence as
            # the per-share amount.
            # Page-wide checks can confuse unrelated per-share mentions (e.g., treasury
            # stock) with dividend rates.
            if "dividend" not in sentence.lower():
                continue
            if target_year not in sentence:
                continue
            eff_match = _DIVIDEND_EFFECTIVE_YEAR_RE.search(sentence)
            if eff_match and eff_match.group(1) != target_year:
                continue
            val = _to_float(amount)
            if val is None or val <= 0:
                continue
            is_annual = "annual" in sentence.lower()
            is_quarterly = bool(re.search(r'\bquarter(?:ly)?\b', sentence, re.IGNORECASE))
            matches_granularity = is_quarterly if prefer_quarterly else is_annual
            candidates.append((matches_granularity, False, val, " ".join(sentence.split())[:200]))

        # "$A, $B and $C per share ... <year1>, <year2> and <year3>,
        # respectively"-shaped sentences (see
        # _DIVIDEND_RESPECTIVELY_SENTENCE_RE's docstring) aren't
        # reachable by the single-value regex above at all — handled as
        # a separate pass rather than folded into it, since this one
        # genuinely needs the WHOLE sentence to positionally pair each
        # year with its own amount, not just the text immediately
        # around a single "$X.XX per share" match.
        for sentence in _DIVIDEND_RESPECTIVELY_SENTENCE_RE.findall(content):
            for yr, val in _parse_respectively_dividend_sentence(sentence):
                if yr != target_year:
                    continue
                is_annual = "annual" in sentence.lower()
                is_quarterly = bool(re.search(r'\bquarter(?:ly)?\b', sentence, re.IGNORECASE))
                matches_granularity = is_quarterly if prefer_quarterly else is_annual
                candidates.append((matches_granularity, True, val, " ".join(sentence.split())[:200]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, val, detail = candidates[0]
    return val, detail


def _is_negated_match(item_lower: str, match_start: int) -> bool:
    """Return True when an alias match is immediately preceded by a prefix that changes its
    accounting meaning: negations (non-, not), recognition-timing modifiers (deferred,
    unearned), or period-delta qualifiers (change in, changes in).

    This prevents mapping stock-balance labels to flow/delta or unrelated liability
    concepts.
    """
    return bool(_NEGATION_PREFIX_RE.search(item_lower[:match_start]))


# ─────────────────────────────────────────────────────────────────────────────
# Canonical item-label resolution
# ─────────────────────────────────────────────────────────────────────────────

def _get_canonical(item_name: str, company_name: str = "") -> str:
    """Map a raw line item label to a canonical key.

    Lookup priority:
    1. Global taxonomy exact match
    2. Global taxonomy longest substring match

    Reject substring matches if immediately preceded by a negation prefix.

    Args:
        item_name    : raw line item string
        company_name : entity name from the evidence

    Returns:
        canonical metric key or "unknown"

    Matching prefers longest match length; ties favor exact matches.
    """
    item_lower = item_name.lower().strip()

    # (match_len, canonical, source_rank) — source_rank breaks ties:
    # global exact (1) > global substring (2).
    candidates: List[Tuple[int, str, int]] = []

    # Some supplemental disclosures use a fixed phrasing distinct from the main
    # statement rows (e.g., "Operating cash flows from finance leases").
    # Match these footnote sub-lines separately to avoid confusion with the main
    # operating cash flow line.
    _lease_cf_disclosure = bool(re.search(
        r'cash flows? from (?:operating|finance) leases?', item_lower
    ))

    # ── Global taxonomy exact match ──────────────────────────────────────────
    if item_lower in _ALIAS_TO_CANONICAL:
        candidates.append((len(item_lower), _ALIAS_TO_CANONICAL[item_lower], 1))

    # ── Global taxonomy substring match ───────────────────────────────────────
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        idx = item_lower.find(alias)
        if idx == -1:
            continue
        if _is_negated_match(item_lower, idx):
            continue
        if _lease_cf_disclosure and canonical in ("operating_cf", "investing_cf", "financing_cf"):
            continue
        candidates.append((len(alias), canonical, 2))

    if not candidates:
        return "unknown"
    candidates.sort(key=lambda c: (-c[0], c[2]))
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(name: str, maxlen: int = 24) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()[:maxlen] or "val"


def _to_float(s: str) -> Optional[float]:
    """Parse a table-cell numeric string, handling accounting conventions such as
    parentheses for negatives and embedded currency symbols.

    Standard float() cannot parse values like (7,616) or strings with $ symbols, so this
    parser strips currency markers and converts parenthesized numbers to negatives to
    avoid silently dropping valid numeric data.

    Caveat: ensure input is a single cell string; ambiguous multi-value cells may need
    prior splitting.
    """
    cleaned = str(s).strip().replace(",", "").replace("%", "").replace("$", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    try:
        val = float(cleaned)
    except (ValueError, AttributeError):
        return None
    return -val if negative else val


def _normalize_year(y: str) -> str:
    """Convert a string fiscal-year prefix to its numeric year.

    Args:
    input_str: string possibly starting with a fiscal-year prefix like 'FYYYY' or 'FY-'.

    Returns:
    Normalized string or number with the fiscal-year prefix removed.
    """
    return re.sub(r"^FY", "", y.strip())


def _extract_query_years(query: str) -> List[str]:
    """List every distinct fiscal year mentioned in the query, in first-seen order.
    Expand dashed or "to" ranges (e.g. "Y1-Y3", "Y1 to Y3") to include every year in
    between so multi-year formulas receive all years in the range.
    """
    years: List[str] = []
    seen: set = set()
    for m in re.finditer(
        r"(?:FY)?(20\d{2})\s*(?:[-–—]|to)\s*(?:FY)?(20\d{2})", query, re.IGNORECASE
    ):
        start, end = int(m.group(1)), int(m.group(2))
        if 0 < end - start <= 10:
            for y in range(start, end + 1):
                ys = str(y)
                if ys not in seen:
                    years.append(ys)
                    seen.add(ys)
    for y in re.findall(r"(?:FY)?(20\d{2})", query):
        ny = _normalize_year(y)
        if ny not in seen:
            years.append(ny)
            seen.add(ny)
    return years


def _kw_match(triggers, q_lower: str) -> bool:
    """Return true if a trigger token in the lowercased query has a genuine word-boundary
    immediately to its left.
    Allows prefix matches (e.g. "improv" matching "improve") but avoids matching short
    triggers that appear inside ordinary words.
    """
    for t in triggers:
        if re.search(r'\b' + re.escape(t), q_lower):
            return True
    return False


#: Phrases implying a comparison (e.g., "improving" or "declining") should be treated as
#: two-point trend questions even if only the latest year is named.
#: This avoids answering with a single-period snapshot when the intent is a comparison
#: against a prior period.
_TREND_KEYWORDS = (
    "improv", "declin", "trend", "increased or decreased",
    "increase or decrease", "compared to", "year over year", "yoy",
    # Questions that ask about a "change" to a metric should be interpreted as
    # requesting a two-point comparison, even when only one year is stated.
    # If only one year is given, infer the prior period to compute and report the
    # magnitude of the change.
    "change",
)
# Remove the lone keyword "profile" as a trend trigger; it causes false positives when
# used as a static noun.
# Keep explicit trend words (improv/declin) as the signals for two-period comparisons;
# do not append a prior year based solely on "profile".


def _with_implied_trend_year(query_years: Optional[List[str]], q_lower: str) -> List[str]:
    """
    If trend language is present but query_years names only one year,
    append year-1 so multi-year comparison logic has a second year to
    compare against.
    """
    years = list(query_years or [])
    if len(set(years)) == 1 and _kw_match(_TREND_KEYWORDS, q_lower):
        try:
            years = years + [str(int(years[0]) - 1)]
        except ValueError:
            pass
    return years


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: standard Markdown table (header row + |---|---| separator + data rows)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_markdown_table_block(content: str, ev_company: str = "") -> Dict[str, Dict]:
    """Parse every standard Markdown pipe table block found in content.
    Each block's header row is the line immediately above the separator row; rows below
    the separator until the first blank/non-pipe line are that block's data rows.
    Return entries in the same shape as the linearized-table extractor so downstream
    code can consume additional blocks without changes.
    """
    extracted: Dict[str, Dict] = {}
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        if i == 0 or not is_markdown_separator_row(lines[i]):
            i += 1
            continue

        sep_idx = i
        header_cells = [c.strip() for c in lines[sep_idx - 1].strip().strip("|").split("|")]
        if len(header_cells) < 2:
            i = sep_idx + 1
            continue
        period_headers = header_cells[1:]

        j = sep_idx + 1
        # Cash-flow statements include a "changes in operating assets and liabilities"
        # section whose rows are period deltas labeled with bare balance-sheet nouns
        # (e.g., "Inventories", "Accounts payable"). Those bare labels are deltas, not
        # point-in-time balances, but they can exactly match a balance-sheet label and
        # be mistaken for it. Detect this by the section header and stop at the section
        # terminator (e.g., "Net cash provided by/used in operating activities").
        in_wc_changes_section = False
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or "|" not in stripped:
                break  # this block ended — outer loop resumes scanning from here

            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 2 or not cells[0]:
                j += 1
                continue
            line_item_name = cells[0]

            _label_lower = line_item_name.lower()
            if "change" in _label_lower and "working capital" in _label_lower:
                in_wc_changes_section = True
            elif "change" in _label_lower and "operating assets" in _label_lower:
                in_wc_changes_section = True
            elif "net cash" in _label_lower:
                in_wc_changes_section = False
            elif in_wc_changes_section:
                line_item_name = f"Change in {line_item_name}"

            canonical = _get_canonical(line_item_name, company_name=ev_company)
            sanitized = _sanitize(line_item_name)

            for k, val_str in enumerate(cells[1:]):
                if k >= len(period_headers):
                    break
                ym = re.search(r"(20\d{2}|FY\d{4})", period_headers[k])
                if not ym:
                    continue
                year = _normalize_year(ym.group(1))
                val = _to_float(val_str)
                if val is None:
                    continue

                base_key = f"val_{year}_{sanitized}"
                key = base_key
                idx = 1
                while key in extracted and abs(extracted[key]["val"] - val) > 0.001:
                    key = f"{base_key}_{idx}"
                    idx += 1
                if key not in extracted:
                    extracted[key] = {
                        "item": line_item_name,
                        "canonical": canonical,
                        "year": year,
                        "val": val,
                        "code_key": key,
                    }
            j += 1

        i = j  # resume scanning for the NEXT '|---|' block from here

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: linearized-table (pipe-delimited)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_linearized_table(
    evidence_list: List[Dict[str, Any]], entity: str = "",
) -> Dict[str, Dict]:
    """
    Works on pipe-delimited linearized table rows:
      ... | Line Item: 營業收入 (Revenue) | 2023 年 (全年度): 2,161.7 | 2024 年: 2,894.3
    AND on standard Markdown tables (header + |---|---| separator + data
    rows) — see _extract_from_markdown_table_block() above, dispatched to
    per evidence item whenever a separator row is detected. Both formats
    can appear across the same evidence_list; entries from either path are
    merged into the same returned dict using the same key-collision rule.
    Returns:
      {code_key: {item, canonical, year, val, code_key}}
    """
    extracted: Dict[str, Dict] = {}
    year_col_re = re.compile(
        r"(.*?(?:20\d{2}|FY\d{4})[^:]*?)\s*:\s*([\d,]+\.?\d*)",
        re.IGNORECASE,
    )
    # The retriever's company filter is a soft penalty, so chunks from the wrong company
    # can still appear in evidence_list; some selection paths lacked an entity-identity
    # check and thus could pick rows from another company. Ensure evidence selection
    # enforces entity-aware filtering or reduction before choosing the best-scoring row.
    entity_target_words = _entity_words(entity) if entity and entity.lower() not in ("company", "unknown", "") else None

    def _entity_ok(ev_company: str) -> bool:
        if entity_target_words is None:
            return True
        if _entity_period_conflicts(entity, ev_company):
            return False
        doc_words = _entity_words(ev_company)
        if not doc_words:
            return True  # no company tag at all — nothing to contradict the target
        if (entity_target_words <= doc_words or doc_words <= entity_target_words
                or (entity_target_words & doc_words)):
            return True
        # Raw word-set comparison fails when document naming collapses multiword company
        # names (e.g., "X Y" -> "XY"), since the token sets don't intersect. Use a
        # collapsed/no-space match in addition to word-set checks to catch concatenated
        # doc_name conventions.
        collapsed_ent = _entity_collapsed(entity)
        collapsed_doc = _entity_collapsed(ev_company)
        return bool(collapsed_ent and (
            collapsed_ent in collapsed_doc or collapsed_doc in collapsed_ent
        ))

    for ev in evidence_list:
        # Prefer parent_content for richer context
        content = ev.get("parent_content") or ev.get("content", "")
        if not content:
            continue
        if _is_supplementary_schedule(content):
            # Same guard as _extract_from_free_text() / the formula-guided
            # path: a guarantor/parent-only/combining schedule reuses the
            # real statements' line-item labels for a smaller reporting
            # entity, so its numbers must never stand in for the actual
            # company's consolidated figures here either.
            continue
        if _is_quarterly_breakdown_table(content):
            # A quarter's own value (e.g. Q1 revenue) must never stand in
            # for the annual total it shares a label with — see
            # _is_quarterly_breakdown_table()'s docstring.
            continue

        # Company name of this evidence row
        ev_company = ev.get("company", "")
        if not ev_company:
            import re as _re
            m_co = _re.search(r"Company:\s*([^|]+)", content)
            ev_company = m_co.group(1).strip() if m_co else ""

        if not _entity_ok(ev_company):
            continue

        # ── Standard Markdown table (header + |---|---| separator) ──────────
        if any(is_markdown_separator_row(l) for l in content.split("\n")):
            for entry in _extract_from_markdown_table_block(content, ev_company).values():
                base_key = entry["code_key"]
                key = base_key
                idx = 1
                while key in extracted and abs(extracted[key]["val"] - entry["val"]) > 0.001:
                    key = f"{base_key}_{idx}"
                    idx += 1
                if key not in extracted:
                    extracted[key] = dict(entry, code_key=key)
            continue  # this evidence item is fully handled by the Markdown parser

        # ── Legacy single-line "Line Item: X | Year: Val" format ────────────
        fields = [f.strip() for f in content.split("|")]
        line_item_name: Optional[str] = None
        for f in fields:
            if f.lower().startswith("line item:"):
                line_item_name = f.split(":", 1)[1].strip()
                break

        if line_item_name is None:
            # Might be a free-text chunk (PDF); skip the table parser
            continue

        canonical = _get_canonical(line_item_name, company_name=ev_company)

        for f in fields:
            m = year_col_re.search(f)
            if not m:
                continue
            year_header = m.group(1).strip()
            val = _to_float(m.group(2))
            if val is None:
                continue
            ym = re.search(r"(20\d{2}|FY\d{4})", year_header)
            if not ym:
                continue
            year = _normalize_year(ym.group(1))
            sanitized = _sanitize(line_item_name)
            base_key = f"val_{year}_{sanitized}"
            key = base_key
            idx = 1
            while key in extracted and abs(extracted[key]["val"] - val) > 0.001:
                key = f"{base_key}_{idx}"
                idx += 1
            if key not in extracted:
                extracted[key] = {
                    "item": line_item_name,
                    "canonical": canonical,
                    "year": year,
                    "val": val,
                    "code_key": key,
                }

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: formula-guided (alias-based)
# ─────────────────────────────────────────────────────────────────────────────

#: Standard SEC-filing terminology for a supplementary/condensed
#: financial schedule covering a SUBSET of the reporting entity —
#: guarantor subsidiaries (required by SEC Rule 3-10 for guaranteed
#: debt), parent-company-only statements, segment combining schedules.
#: Not specific to any one company: these are generic terms used across
#: many real 10-Ks. "deed of cross guarantee" is the equivalent term used
#: by companies with Australian subsidiaries (e.g. under ASIC Class
#: Order relief) — same underlying concept, different jurisdiction's
#: wording, still generic rather than tied to one filer.
_SUPPLEMENTARY_SCHEDULE_MARKERS = (
    "guarantor", "obligor group", "obligor", "parent company only",
    "parent-company-only", "condensed consolidating", "combining schedule",
    "deed of cross guarantee",
    # Business-combination purchase-price-allocation tables reuse generic balance-sheet
    # labels for the acquired entity's fair-valued assets, which can match consolidated
    # labels and be misattributed. Treat PPA tables as distinct from the reporting
    # company's consolidated statements when matching labels.
    "previously held equity interest", "previously held equity investment",
    "purchase price allocation", "assets acquired and liabilities assumed",
    # Equity-method notes can include summarized balance sheets for investees that reuse
    # the same labels as the filer, leading to using investee figures instead of the
    # filer’s consolidated numbers. Use contextual headers/dates to distinguish investee
    # summaries from the reporting entity's statements.
    "on a 100 percent basis", "summary combined financial information",
    "recognized amounts of identified assets",
    # Pro forma disclosures for business combinations present hypothetical combined
    # figures using the same line-item labels as real consolidated statements, which can
    # be mistaken for reported GAAP figures. Detect and exclude pro forma/hypothetical
    # tables or mark them so they are not treated as actual reported values.
    "pro forma results", "pro forma revenue", "pro forma information",
    "unaudited pro forma", "supplemental pro forma",
)

#: Detect when an attached exhibit contains a complete set of another entity's financial
#: statements using a structural pattern: exhibit number immediately followed by an
#: entity name and a consolidated-statement heading. This avoids misattributing tables
#: that merely reference an exhibit elsewhere in the same filing.
_EXHIBIT_FINANCIALS_RE = re.compile(
    r'exhibit\s+\d+\.\d+\s*\n[^\n]{0,80}\n\s*consolidated\s+(balance\s+sheets?|statements?)',
    re.IGNORECASE,
)


def _is_supplementary_schedule(content: str) -> bool:
    """Return true if an evidence chunk appears to be a supplementary schedule for a subset
    of the reporting entity or a one-time acquisition snapshot rather than consolidated
    statements.
    This detects cases where identical line-item labels appear but the data corresponds
    to a different sub-entity or point-in-time, preventing misattribution of values.
    """
    lower = content.lower()
    if any(m in lower for m in _SUPPLEMENTARY_SCHEDULE_MARKERS):
        return True
    return bool(_EXHIBIT_FINANCIALS_RE.search(content))


#: Recognize a filer’s multi-year summary/selected-financial-data table as the
#: authoritative source for unqualified, headline metric queries when multiple same-word
#: labels compete across statements. Prefer the filer’s summary table label over
#: incidental matches elsewhere.
_SUMMARY_TABLE_MARKERS = (
    "selected financial data", "summary of operations",
    "selected consolidated financial data", "five-year selected",
    "five-year financial summary", "five year summary",
)


def _is_summary_reference_table(content: str) -> bool:
    """True if this evidence chunk is (part of) the filer's own multi-year
    "Selected Financial Data" / "Summary of Operations" reference table."""
    lower = content.lower()
    return any(m in lower for m in _SUMMARY_TABLE_MARKERS)


#: Markers indicating a "Selected Quarterly Financial Data" disclosure —
#: a standard SEC 10-K schedule that breaks a fiscal year's figures down
#: into Q1-Q4 (plus a full-year total column), reusing the SAME line-item
#: labels ("Revenue", "Net earnings...") as the real annual income
#: statement. Not specific to any one filer — this is a routine Item 8
#: disclosure most 10-Ks include.
_QUARTERLY_DATA_MARKERS = (
    "quarterly financial data", "quarterly results", "selected quarterly",
    "quarterly financial information",
)

#: A table header cell naming a real calendar/fiscal year (e.g. "2017").
_YEAR_HEADER_CELL_RE = re.compile(r"^(?:FY)?(?:19|20)\d{2}$")
#: table_parser.py's fallback name for a column it found no real period
#: header for (e.g. an unlabeled quarter column).
_COL_PLACEHOLDER_CELL_RE = re.compile(r"^Col\d+$")


def _is_quarterly_breakdown_table(content: str) -> bool:
    """Return true if content contains a quarter-by-quarter breakdown table rather than a
    multi-year annual comparison.
    Detect either by quarter-specific title markers or by a header with one year
    followed by four generic ColN columns plus a data row where the first four numeric
    values sum to the fifth (within rounding), avoiding using quarterly slices as annual
    totals.
    """
    lower = content.lower()
    if any(m in lower for m in _QUARTERLY_DATA_MARKERS):
        return True
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|") or is_markdown_separator_row(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        data_cells = cells[1:]  # skip the row-label cell
        year_cells = [c for c in data_cells if _YEAR_HEADER_CELL_RE.match(c)]
        col_placeholder_cells = [c for c in data_cells if _COL_PLACEHOLDER_CELL_RE.match(c)]
        if len(year_cells) != 1 or len(col_placeholder_cells) != 4:
            continue
        # Header shape matches — confirm with the arithmetic invariant on
        # a nearby data row before concluding this is genuinely quarterly.
        for j in range(i + 1, min(i + 6, len(lines))):
            data_line = lines[j].strip()
            if not data_line.startswith("|"):
                break
            if is_markdown_separator_row(data_line):
                continue
            data = [c.strip() for c in data_line.strip("|").split("|")][1:]
            if len(data) != 5:
                continue
            nums = [_to_float(c) for c in data]
            if any(n is None for n in nums):
                continue
            quarters_sum = sum(nums[:4])
            annual_total = nums[4]
            if annual_total != 0 and abs(quarters_sum - annual_total) / abs(annual_total) < 0.02:
                return True
    return False


def _score_row_match(label: str, aliases: List[str]) -> float:
    """Score how well a table row label matches any alias for a canonical variable.
    2 = exact normalized alias match or "total {alias}"; 1.x = alias appears as a
    substring (fractional part = length of longest matching alias/1000); 0 = no valid
    match (including negated forms like "non-...").
    When multiple rows compete for the same (placeholder, year), the highest score wins
    to prefer precise totals over partial or generic matches.
    """
    # Remove stray currency-symbol tokens from extracted label text before comparing to
    # aliases to avoid false non-matches caused by symbol leakage from adjacent columns.
    label_norm = re.sub(r'\s+', ' ', label.lower().strip().rstrip(':'))
    label_norm = re.sub(r'(?:(?<=\s)|^)\$(?=\s|$)', '', label_norm)
    # Treat spaced dashes (dash/en-dash/em-dash) used as clause separators as equivalent
    # to commas when matching labels so punctuation differences across filers do not
    # block exact matches.
    label_norm = re.sub(r'\s+[\-–—]\s+', ', ', label_norm)
    label_norm = re.sub(r'\s+', ' ', label_norm).strip()
    if _CARVEOUT_ANYWHERE_RE.search(label_norm):
        return 0
    # Strip end-of-label parenthetical sign qualifiers like (loss)/(deficit)/(benefit)
    # when matching aliases, treating them as boilerplate indicating sign rather than
    # distinct line items; do this only as an end-of-string variant and only if the
    # original label is not already an exact match.
    label_norm_unqualified = re.sub(
        r'\s*\((?:loss|deficit|expense|benefit)\)\s*$', '', label_norm
    ).strip()
    # Handle the narrow caption suffix "interest expense, net of amounts capitalized" as
    # a recognized variant for capitalized-interest accounting; do not apply a broad
    # "strip any net of X" rule because bare "net" suffixes can indicate genuinely
    # different figures.
    label_norm_unqualified = re.sub(
        r',?\s*net of amounts capitalized\s*$', '', label_norm_unqualified
    ).strip()
    # Per-share rows must only match aliases that themselves request a per-share figure
    # (the per-share placeholder's alias list).
    # Any match to a non-per-share placeholder is a disqualifying mismatch.
    row_is_per_share = bool(_PER_SHARE_ANYWHERE_RE.search(label_norm))
    best_substring_len = 0
    for alias in aliases:
        a = alias.lower().strip()
        if not a:
            continue
        if row_is_per_share and 'per share' not in a and a != 'eps':
            continue
        idx = label_norm.find(a)
        if idx == -1 or _is_negated_match(label_norm, idx):
            continue
        if _is_carveout_attribution_match(label_norm, idx, len(a)):
            continue
        if (
            label_norm == a or label_norm == f"total {a}"
            or label_norm_unqualified == a or label_norm_unqualified == f"total {a}"
        ):
            return 2
        best_substring_len = max(best_substring_len, len(a))
    if best_substring_len:
        return 1 + min(best_substring_len, 999) / 1000.0
    return 0


#: Some variables are reported as multiple sub-items rather than a single line.
#: Keyed by the placeholder name, each value lists sub-item alias fragments to sum for
#: the composite concept.
_COMPOSITE_ITEM_ALIASES: Dict[str, List[str]] = {
    "inventory": [
        "raw materials", "raw materials and supplies",
        "work in process", "work in process and finished goods",
        "finished goods", "merchandise inventory",
        "存貨", "原料", "在製品", "製成品",
    ],
    # Questions about a filer’s borrowings require that filer’s own borrowings, not a
    # broader total-liabilities placeholder.
    # Borrowings may be split across long-term and current portions, so both must be
    # summed when resolving the borrowing placeholder in multi-year formulas.
    "total_borrowings_old": [
        "long-term debt", "current portion of long-term debt",
        "short-term borrowings", "current maturities of long-term debt",
    ],
    "total_borrowings_new": [
        "long-term debt", "current portion of long-term debt",
        "short-term borrowings", "current maturities of long-term debt",
    ],
}


def _resolve_composite_item(
    evidence_list: List[Dict[str, Any]],
    sub_aliases: List[str],
    entity: str = "",
) -> Dict[str, Tuple[float, List[str]]]:
    """Approximate a composite line item by summing its known sub-items across a filer’s
    own statement structure when no single row matches the composite.
    Each sub-alias is resolved independently to its single best-scoring row per year to
    avoid double-counting; distinct sub-alias values are then summed. Supplementary
    schedules are excluded.
    Returns {year: (summed_value, [line_item_label, ...])} for provenance.
    """
    entity_target_words = _entity_words(entity) if entity and entity.lower() not in ("company", "unknown", "") else None

    # sub_candidates[sub_alias] = {year: (value, score, is_primary, line_item_label)}
    sub_best: Dict[str, Dict[str, Tuple[float, int, bool, str]]] = {a: {} for a in sub_aliases}

    for ev in evidence_list:
        if entity_target_words is not None:
            doc_words = _entity_words(ev.get("company", "") or "")
            if doc_words and not (entity_target_words <= doc_words or doc_words <= entity_target_words or (entity_target_words & doc_words)):
                continue
        content = ev.get("parent_content") or ev.get("content", "")
        if not content or _is_quarterly_breakdown_table(content):
            continue
        is_primary = not _is_supplementary_schedule(content)

        rows: List[Tuple[str, str, float]] = []  # (label, year, val)
        if any(is_markdown_separator_row(l) for l in content.split("\n")):
            ev_company = ev.get("company", "")
            for row in _extract_from_markdown_table_block(content, ev_company).values():
                rows.append((row["item"], row["year"], row["val"]))
        else:
            year_col_re = re.compile(
                r"(.*?(?:20\d{2}|FY\d{4})[^:]*?)\s*:\s*([\d,]+\.?\d*)", re.IGNORECASE
            )
            for seg in re.split(r'(?=Line Item:)', content):
                label_m = re.search(r'Line Item:\s*([^|]+)', seg)
                if not label_m:
                    continue
                seg_label = label_m.group(1).strip()
                for f in [x.strip() for x in seg.split("|")]:
                    m_f = year_col_re.search(f)
                    if not m_f:
                        continue
                    ym = re.search(r"(20\d{2}|FY\d{4})", m_f.group(1))
                    if not ym:
                        continue
                    val = _to_float(m_f.group(2))
                    if val is not None and val != 0:
                        rows.append((seg_label, _normalize_year(ym.group(1)), val))

        for label, year, val in rows:
            for sub_alias in sub_aliases:
                score = _score_row_match(label, [sub_alias])
                if score == 0:
                    continue
                existing = sub_best[sub_alias].get(year)
                priority = (is_primary, score)
                if existing is None or priority > (existing[2], existing[1]):
                    sub_best[sub_alias][year] = (val, score, is_primary, label)

    # Sum distinct sub-aliases' best value per year. A sub-alias with no
    # match anywhere just doesn't contribute (a partial sum from whatever
    # sub-items ARE found is still better signal than nothing — the
    # caller marks the result is_approximate regardless).
    by_year: Dict[str, Tuple[float, List[str]]] = {}
    for sub_alias, year_map in sub_best.items():
        for year, (val, score, is_primary, label) in year_map.items():
            total, labels = by_year.get(year, (0.0, []))
            if label in labels:
                continue  # same row already counted under a different matching sub-alias
            by_year[year] = (total + val, labels + [label])
    return by_year


def _entity_words(name: str) -> set:
    r"""Produce a lowercase, year/underscore/extension-stripped word set for a company
    identifier or raw corpus company field (e.g. FILESTEM_YYYY_TYPE -> {"filestem"},
    "Human Name" -> {"human","name"}).
    Used only for a cheap entity-match check; not a full company-match scoring routine.
    Excludes common filing-type tokens so generic suffixes do not create false overlaps.
    Uses boundary-aware year-stripping that works when underscores neighbor digits.
    """
    n = re.sub(r'(?<!\d)(?:20|19)\d{2}(?!\d)', '', name)
    n = re.sub(r'[_\-]+', ' ', n)
    return {w for w in n.lower().split() if len(w) >= 2 and w != "10k"}


#: Year, optionally with an adjacent "Q1"-"Q4" suffix (e.g. "2022Q4" in
#: "MGMRESORTS_2022Q4_EARNINGS"). Same pattern as orchestrator.py's own
#: _match_entity_to_corpus fix for the identical underlying issue in a
#: different function.
_ENTITY_PERIOD_RE = re.compile(r'(?<!\d)((?:20|19)\d{2})(q[1-4])?(?!\d)', re.IGNORECASE)


def _entity_period_conflicts(entity: str, doc_company: str) -> bool:
    """Return True when both entity and document-company have extractable period tokens
    (year or year+quarter) and those periods differ — this vetoes matches that mere
    word-overlap cannot distinguish.
    Returns False if either side lacks an extractable period, so it only narrows
    existing matches; companies with one filing per year are unaffected.
    """
    em = _ENTITY_PERIOD_RE.search(entity or "")
    dm = _ENTITY_PERIOD_RE.search(doc_company or "")
    if not em or not dm:
        return False
    e_period = (em.group(1), (em.group(2) or "").lower())
    d_period = (dm.group(1), (dm.group(2) or "").lower())
    return e_period != d_period


def _entity_collapsed(name: str) -> str:
    """Same normalization as the word-set version but return a space-stripped single string
    (e.g. "mgmresorts") so multi-word human names can be matched via substring against
    the mashed doc_name convention.
    """
    n = re.sub(r'(?<!\d)(?:20|19)\d{2}(?!\d)', '', name)
    n = re.sub(r'[_\-]+', ' ', n)
    n = re.sub(r'\b10k\b', '', n.lower())
    return re.sub(r'\s+', '', n)


def _extract_formula_guided(
    evidence_list: List[Dict[str, Any]],
    formula_entry: Dict[str, Any],
    query_years: List[str],
    entity: str = "",
    q_lower: str = "",
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, str]]], Dict[str, Dict[str, Any]]]:
    """
    For each required variable in formula_entry, search evidence for a chunk whose
    Line Item matches a known alias. Returns (final, resolved, meta):
      - final:    {placeholder -> best single float} — what every existing
                  caller expects (codegen for non-period_average formulas,
                  the extracted-variables summary shown to the user).
      - resolved: {placeholder -> [(value, year), ...]} — every WINNING
                  match (see below), needed by period_average formulas,
                  which average a ratio across EVERY year the query asks
                  about rather than picking one old/new pair (see
                  _gen_formula_code()).
      - meta:     {placeholder -> {"source": str, "is_approximate": bool}}
                  source is one of "table-total" (row label IS the alias
                  or "Total {alias}"), "table-partial" (matched a sub-
                  item/compound label instead — is_approximate=True), a
                  "-supplementary" suffixed variant of either (matched in
                  a guarantor/parent-only/combining schedule rather than
                  the primary statement — always is_approximate=True,
                  even for an otherwise-exact "-total" match, since it's
                  real data for the wrong reporting entity), "legacy",
                  "free-text", "period-average" (mixed years), or
                  "unresolved". Lets callers show exactly how each
                  variable was actually obtained instead of a blanket
                  "(formula)" label, so a suspicious value is visible at
                  a glance rather than indistinguishable from a solid one.

    Extraction priority per evidence chunk — Priority 1 is the fix for
    the operating_income/revenue mix-up bug: matching alias against each
    ROW's own label — not the whole chunk — makes it structurally
    impossible for one row's alias hit to resolve to a DIFFERENT row's
    value. When several rows in the same table match the same alias
    (e.g. both "Total current liabilities" and "Other current
    liabilities" contain "current liabilities"), _score_row_match() picks
    the real total over the sub-item instead of whichever happened to be
    scanned first.
      1. Standard Markdown table (the format real PDF tables are stored
         in) — matched ROW BY ROW via _extract_from_markdown_table_block(),
         scored via _score_row_match(), highest score per year wins.
      2. Legacy single-line "Line Item: X | Year: Val" format (kept for
         backward compatibility with content still in that shape).
      3. Free-text keyword-proximity fallback (_extract_from_free_text),
         for genuinely unstructured narrative evidence only — and ONLY
         when a canonical whose NAME actually matches this placeholder's
         aliases is found. There is deliberately no "grab whichever
         canonical happens to have data for our query years" fallback
         beyond that: that exact mechanism was root-caused to
         fixed_asset_turnover's ppe_new silently taking on revenue's
         value (no real PP&E match existed anywhere, so the old fallback
         grabbed the first unrelated canonical that had matching-year
         data instead of leaving ppe_new unresolved).
    """
    var_aliases = get_variable_aliases(formula_entry)
    is_multi_year = formula_entry.get("multi_year", False)
    # A formula able to compute an N-year average does not imply every matched question
    # wants an average.
    # Only questions explicitly requesting an average should trigger the period_average
    # behavior.
    is_period_average = formula_entry.get("period_average", False) and any(
        kw in q_lower for kw in ("average", "avg", "平均")
    )

    # candidates[placeholder] = [(value, year, score, source, is_primary,
    # evidence_index, line_item_label), ...] -- every match found, BEFORE
    # reducing to one winner per year. Always scans the FULL evidence
    # list (no "first chunk match wins" early exit, even for simple
    # formulas) -- a match found early is no longer trusted just for
    # being early, since it could be a supplementary schedule's row
    # scanned before the real primary statement's; the reduction step
    # below picks the best candidate regardless of discovery order.
    # evidence_index/line_item_label exist purely for provenance (see
    # meta["source_detail"] below) -- so a wrong extraction can be traced
    # straight back to which evidence item and which line item produced
    # it, without re-diagnosing from the raw PDF each time.
    candidates: Dict[str, List[Tuple[float, str, int, str, bool, int, str]]] = {
        k: [] for k in var_aliases
    }

    year_col_re = re.compile(
        r"(.*?(?:20\d{2}|FY\d{4})[^:]*?)\s*:\s*([\d,]+\.?\d*)", re.IGNORECASE
    )

    for placeholder, aliases in var_aliases.items():
        for ev_idx, ev in enumerate(evidence_list):
            content = ev.get("parent_content") or ev.get("content", "")
            if not content or _is_quarterly_breakdown_table(content):
                continue
            ev_company = ev.get("company", "")
            is_primary = not _is_supplementary_schedule(content)

            # ── Priority 1: standard Markdown table, matched per ROW, scored ──
            if any(is_markdown_separator_row(l) for l in content.split("\n")):
                for row in _extract_from_markdown_table_block(content, ev_company).values():
                    score = _score_row_match(row["item"], aliases)
                    if score == 0:
                        continue
                    source = "table-total" if score == 2 else "table-partial"
                    if not is_primary:
                        source += "-supplementary"
                    candidates[placeholder].append(
                        (row["val"], row["year"], score, source, is_primary, ev_idx, row["item"])
                    )
                continue  # this chunk is fully handled by the table parser

            # Legacy content blobs use a single-segment-per-line-item format with a
            # fixed marker that starts each line item segment.
            # Alias matching must be scoped to an individual line-item segment so a
            # label match cannot pull a number from a different segment.
            for seg in re.split(r'(?=Line Item:)', content):
                label_m = re.search(r'Line Item:\s*([^|]+)', seg)
                if not label_m:
                    continue
                seg_label = label_m.group(1).strip()
                score = _score_row_match(seg_label, aliases)
                if score == 0:
                    continue
                source = "legacy-total" if score == 2 else "legacy-partial"
                if not is_primary:
                    source += "-supplementary"
                for f in [x.strip() for x in seg.split("|")]:
                    m_f = year_col_re.search(f)
                    if not m_f:
                        continue
                    ym = re.search(r"(20\d{2}|FY\d{4})", m_f.group(1))
                    if not ym:
                        continue
                    year = _normalize_year(ym.group(1))
                    val = _to_float(m_f.group(2))
                    if val is not None and val != 0:
                        candidates[placeholder].append(
                            (val, year, score, source, is_primary, ev_idx, seg_label)
                        )

    # ── Reduce to the single best-scoring candidate per year ────────────────
    # Priority is (is_primary, score) compared as a tuple: a match from a
    # supplementary/guarantor schedule NEVER outranks one from the real
    # primary statement, even a lower-scoring sub-item match — using the
    # WRONG entity's "clean" total is worse than the RIGHT entity's
    # partial data for computing a ratio about the actual company. Only
    # within the same primary-ness tier does score (total vs. partial)
    # break the tie.
    resolved: Dict[str, List[Tuple[float, str]]] = {}
    # winner_meta[placeholder][year] = (score, source, evidence_index,
    # line_item_label) of the entry that won that year, kept alongside
    # `resolved` (whose tuples must stay plain (value, year) — every
    # existing consumer of the return value unpacks exactly two fields).
    winner_meta: Dict[str, Dict[str, Tuple[int, str, int, str]]] = {}
    # Cache mapping evidence index -> the filing's reported year string, parsed once per
    # evidence item.
    filing_year_cache: Dict[int, Optional[str]] = {}

    def _filing_year(ev_idx: int) -> Optional[str]:
        if ev_idx not in filing_year_cache:
            ev = evidence_list[ev_idx]
            text = ev.get("parent_content") or ev.get("content", "") or ev.get("table_name", "")
            m = re.search(r"_(\d{4})_10K", text)
            filing_year_cache[ev_idx] = m.group(1) if m else None
        return filing_year_cache[ev_idx]

    # Cache mapping evidence index -> whether that evidence item's company field matches
    # the target entity.
    # Retrieval's company-match score is a soft multiplier (a mismatch penalizes but
    # does not exclude), so candidates from the wrong company can still appear; this
    # cache lets later logic know which rows originated from the target entity.
    entity_target_words = _entity_words(entity) if entity and entity.lower() not in ("company", "unknown", "") else None
    entity_match_cache: Dict[int, bool] = {}

    def _entity_matches(ev_idx: int) -> bool:
        if entity_target_words is None:
            return True
        if ev_idx not in entity_match_cache:
            doc_company = evidence_list[ev_idx].get("company", "") or ""
            if _entity_period_conflicts(entity, doc_company):
                entity_match_cache[ev_idx] = False
            else:
                doc_words = _entity_words(doc_company)
                entity_match_cache[ev_idx] = bool(entity_target_words) and (
                    entity_target_words <= doc_words or doc_words <= entity_target_words
                    or bool(entity_target_words & doc_words)
                )
        return entity_match_cache[ev_idx]

    summary_table_cache: Dict[int, bool] = {}

    def _is_summary_table(ev_idx: int) -> bool:
        if ev_idx not in summary_table_cache:
            content = evidence_list[ev_idx].get("parent_content") or evidence_list[ev_idx].get("content", "")
            summary_table_cache[ev_idx] = _is_summary_reference_table(content)
        return summary_table_cache[ev_idx]

    def _is_attributable_row(label: str, ev_idx: int) -> bool:
        ll = label.lower()
        idx = ll.find("attributable to")
        if idx == -1:
            return False
        after = ll[idx + len("attributable to"):]
        if _ATTRIBUTABLE_TO_CARVEOUT_RE.match(after):
            return False
        return _is_attributable_to_reporting_entity(after, evidence_list[ev_idx].get("company", "") or "")

    for placeholder in var_aliases:
        query_year_set = set(query_years or [])

        # Handle cases where both a pre-split subtotal and a parent-level headline
        # figure exist with similar labels.
        # If a same-year sibling candidate clearly attributable to the reporting entity
        # exists, cap a bare exact-match score so subtotal and headline compete on
        # downstream heuristics rather than label exactness alone.
        attributable_sibling_year: set = {
            _yr for _val, _yr, _score, _source, _is_primary, _ev_idx, _label in candidates[placeholder]
            if _is_attributable_row(_label, _ev_idx)
        }

        # Two-pass reduction. Pass 1: compute each candidate's priority
        # tuple and bucket by year. Pass 2 (below): among a year's
        # MAX-priority candidates, pick the winner by frequency first —
        # see the block comment there for why a straight "biggest wins"
        # tie-break isn't safe on its own.
        by_year: Dict[str, List[Tuple[Tuple[bool, bool, bool, int, bool], float, int, str, int, str]]] = {}
        for val, yr, score, source, is_primary, ev_idx, line_item_label in candidates[placeholder]:
            # Prefer a candidate whose own filing's reporting year matches the data year
            # over an exact-label match that appears as a historical column in a
            # different filing.
            # This tie-break ranks same-filing matches above label-only exactness when
            # years conflict, but it never overrides is_primary.
            year_matches_filing = _filing_year(ev_idx) == yr
            # entity_matches leads the tuple: identity correctness beats
            # every other signal — a right-company partial/sub-item match
            # must always outrank a wrong-company "Total X" exact match.
            # is_summary_table: when a bare alias ties in score against
            # several differently-scoped rows scattered across the full
            # financial statements, the row the filer itself chose to
            # headline in its own "Selected Financial Data"/"Summary of
            # Operations" reference table is the strongest available
            # signal for which one a plain, unqualified question means —
            # see _is_summary_reference_table's docstring. Label length
            # is deliberately NOT part of this tuple — see the length
            # tie-break inside the "still tied" block below for why it
            # must run AFTER the magnitude-outlier filter rather than
            # before it (a short label like "Basic net income" is an EPS
            # figure, not automatically the better match).
            effective_score = score
            if (
                score == 2 and yr in attributable_sibling_year
                and not _is_attributable_row(line_item_label, ev_idx)
            ):
                effective_score = 1
            priority = (
                _entity_matches(ev_idx), is_primary, year_matches_filing, effective_score,
                _is_summary_table(ev_idx),
            )
            by_year.setdefault(yr, []).append((priority, val, score, source, ev_idx, line_item_label))

        # When the same evidence item is the top candidate for every year in a multi-
        # year query, force all those years to use that single source for internal
        # consistency.
        # This preserves per-year source winners while avoiding mixing independently
        # selected sources across years.
        forced_ev_by_year: Dict[str, int] = {}
        if len(query_year_set) >= 2 and query_year_set <= set(by_year.keys()):
            ceiling_evs: Optional[set] = None
            for yr in query_year_set:
                cands = by_year[yr]
                max_p = max(c[0] for c in cands)
                evs_at_ceiling = {c[4] for c in cands if c[0] == max_p}
                ceiling_evs = evs_at_ceiling if ceiling_evs is None else (ceiling_evs & evs_at_ceiling)
            if ceiling_evs:
                chosen_ev = min(ceiling_evs)
                for yr in query_year_set:
                    forced_ev_by_year[yr] = chosen_ev

        best_by_year: Dict[str, Tuple[float, int, str, Tuple[bool, bool, bool, int, bool], int, str]] = {}
        for yr, cands in by_year.items():
            # After forcing a consistent source across years, still run
            # magnitude/frequency/length reductions on the rows from that source.
            # A single evidence item can contain multiple same-scored rows; keep the
            # reduction to distinguish subtotals from sub-line items rather than picking
            # arbitrarily.
            if yr in forced_ev_by_year:
                cands = [c for c in cands if c[4] == forced_ev_by_year[yr]]
            max_priority = max(c[0] for c in cands)
            tied = [c for c in cands if c[0] == max_priority]
            if len(tied) == 1:
                priority, val, score, source, ev_idx, line_item_label = tied[0]
            else:
                # Apply a magnitude-outlier filter before counting frequency.
                # Exclude candidates below a fraction of the largest tied magnitude so
                # small-scale figures (e.g., per-share EPS) do not outvote dollar-scale
                # totals when alias matching is broad.
                max_abs = max(abs(c[1]) for c in tied)
                dominant = [c for c in tied if max_abs == 0 or abs(c[1]) >= max_abs * 0.05]
                if not dominant:
                    dominant = tied

                # Frequency next: prefer the candidate that appears with the same
                # canonical label across multiple independent evidence items.
                # A one-off table that shares a label with the target concept should not
                # outweigh corroboration from multiple independent sources.
                counts: Dict[float, int] = {}
                for c in dominant:
                    counts[c[1]] = counts.get(c[1], 0) + 1
                max_count = max(counts.values())
                by_freq = [c for c in dominant if counts[c[1]] == max_count]
                if len(by_freq) == 1:
                    priority, val, score, source, ev_idx, line_item_label = by_freq[0]
                else:
                    # Still tied on frequency — fall back to magnitude,
                    # but ONLY when the gap is dramatic (>5x) even after
                    # the outlier filter above (e.g. two legitimately
                    # different-but-plausible dollar-scale candidates
                    # remain). A modest gap like 91,601 vs 87,896 (~4%) is
                    # NOT that signal — two candidates that close are
                    # plausibly two genuine but different totals, and
                    # picking the bigger one by default has no basis.
                    by_freq.sort(key=lambda c: abs(c[1]), reverse=True)
                    biggest, smallest = by_freq[0], by_freq[-1]
                    if abs(smallest[1]) > 0 and abs(biggest[1]) / abs(smallest[1]) > 5:
                        priority, val, score, source, ev_idx, line_item_label = biggest
                    else:
                        # If frequency and magnitude still tie, prefer a candidate
                        # explicitly marked as attributable to the reporting entity over
                        # an unqualified subtotal.
                        # Only if attributable-ness also ties should shorter label
                        # length serve as a final tie-breaker; longer labels often
                        # indicate further qualification.
                        by_freq.sort(key=lambda c: (
                            not _is_attributable_row(c[5], c[4]), len(c[5]),
                        ))
                        priority, val, score, source, ev_idx, line_item_label = by_freq[0]
            best_by_year[yr] = (val, score, source, priority, ev_idx, line_item_label)
        # A supplementary-schedule match is NEVER actually used, even as
        # a last resort with nothing else available for that year — real
        # data for the WRONG reporting entity is worse than no data at
        # all for computing a ratio about the actual company (same
        # philosophy as root cause 2: unresolved beats confidently
        # wrong). The tuple-priority ordering above already lets a
        # primary match win whenever both exist; this prunes the
        # remaining case where supplementary was the ONLY candidate for
        # a year, so that year falls through to the free-text fallback
        # (or stays unresolved) instead of silently using it.
        best_by_year = {y: v for y, v in best_by_year.items() if v[3][0]}
        resolved[placeholder] = [(v, y) for y, (v, s, src, p, ei, lbl) in best_by_year.items()]
        winner_meta[placeholder] = {
            y: (s, src, ei, lbl) for y, (v, s, src, p, ei, lbl) in best_by_year.items()
        }

    # ── Free-text fallback: fill any placeholders still incomplete ──────────
    # Triggers when:
    #   (a) a placeholder has no matches at all, OR
    #   (b) formula is multi_year but a placeholder only has ONE distinct year
    #       (the pipe parser grabbed '2022: 34,229' but missed '2021: 35,355'), OR
    #   (c) formula is period_average but not every query year is covered yet
    def _needs_ft_fallback(ph: str, matches_list: List[Tuple[float, str]]) -> bool:
        if not matches_list:
            return True
        distinct_years = {y for _, y in matches_list}
        if is_period_average and query_years:
            return not set(query_years).issubset(distinct_years)
        if is_multi_year and len(distinct_years) < 2:
            return True
        return False

    # Composite-item fallback: try summing structured sub-item rows when a concept is
    # normally reported only as components.
    # Run this before free-text fallback and allow it even when a weak direct match
    # exists; do not let a weak match permanently block attempting a composite sum.
    for placeholder in var_aliases:
        if placeholder not in _COMPOSITE_ITEM_ALIASES:
            continue
        composite_by_year = _resolve_composite_item(
            evidence_list, _COMPOSITE_ITEM_ALIASES[placeholder], entity
        )
        if not composite_by_year:
            continue
        ph_meta = winner_meta.setdefault(placeholder, {})
        existing_val_by_year = {y: v for v, y in resolved[placeholder]}
        for yr, (total_val, labels) in composite_by_year.items():
            # Exact label matches are not always more reliable: identical short labels
            # can appear for distinct tables.
            # When a concept can be reconstructed from sub-items, prefer the larger
            # magnitude candidate over a bare score-based label match.
            existing_val = existing_val_by_year.get(yr)
            if existing_val is not None and abs(existing_val) >= abs(total_val):
                continue
            resolved[placeholder] = [(v, y) for v, y in resolved[placeholder] if y != yr]
            resolved[placeholder].append((total_val, yr))
            ph_meta[yr] = (1, "composite", -1, " + ".join(labels))

    any_needs_fallback = any(_needs_ft_fallback(ph, v) for ph, v in resolved.items())
    if any_needs_fallback and query_years:
        ft = _extract_from_free_text(evidence_list, query_years)
        if ft:
            # Map canonical -> list of (val, year) from free_text
            ft_by_canonical: Dict[str, List[Tuple[float, str]]] = {}
            for item in ft.values():
                c = item["canonical"]
                ft_by_canonical.setdefault(c, []).append((item["val"], item["year"]))

            for placeholder, aliases in var_aliases.items():
                if not _needs_ft_fallback(placeholder, resolved[placeholder]):
                    continue  # already has complete multi-year data
                # Try to match placeholder to a canonical BY NAME.
                base = placeholder.replace("_new", "").replace("_old", "")
                matched_canonical = None
                for c in ft_by_canonical:
                    if c == base or any(c in a.lower() or a.lower() in c for a in aliases):
                        matched_canonical = c
                        break
                # Root cause 2 fix: no fallback beyond this. Previously,
                # finding no name match here fell back to "whichever
                # canonical has data for one of our query years" — which
                # is how ppe_new silently took on revenue's value with no
                # real PP&E match anywhere. If nothing genuinely matches
                # this placeholder's own aliases, it stays unresolved.
                if matched_canonical is None:
                    continue
                existing_years = {y for _, y in resolved[placeholder]}
                for val, yr in ft_by_canonical[matched_canonical]:
                    if yr in existing_years:
                        continue  # keep the stronger table-sourced value for a year we already have
                    resolved[placeholder].append((val, yr))
                    winner_meta.setdefault(placeholder, {})[yr] = (1, "free-text", -1, matched_canonical)
                    existing_years.add(yr)

    # ── Choose the best value for each placeholder ────────────────────────────
    final: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for placeholder, matches in resolved.items():
        if not matches:
            meta[placeholder] = {"source": "unresolved", "is_approximate": False}
            continue

        picked_year: Optional[str] = None
        if is_period_average:
            # No single "best" value — the real per-year computation is
            # done by _gen_formula_code() from `resolved` directly. This
            # mean is only for the truthiness gate and the variable
            # summary shown to the user.
            final[placeholder] = sum(v for v, _ in matches) / len(matches)
            year_meta = [
                winner_meta.get(placeholder, {}).get(y, (1, "unknown", -1, "?")) for _, y in matches
            ]
            meta[placeholder] = {
                "source": "period-average",
                "is_approximate": any(s < 2 or "supplementary" in src for s, src, ei, lbl in year_meta),
                "source_detail": "; ".join(
                    f"{y}={v} <- sum of sub-items: {lbl}" if src == "composite"
                    else (f"{y}={v} <- evidence[{ei}] Line Item \"{lbl}\"" if ei >= 0 else f"{y}={v} <- free-text")
                    for (v, y), (s, src, ei, lbl) in zip(matches, year_meta)
                ),
            }
            continue
        elif is_multi_year:
            # _new / _old suffix: pick based on query years
            base = placeholder.replace("_new", "").replace("_old", "")
            sorted_years = sorted(set(y for _, y in matches))
            if query_years and len(query_years) >= 2:
                old_yr = sorted(query_years)[0]
                new_yr = sorted(query_years)[-1]
            elif len(sorted_years) >= 2:
                # If a query names fewer than two years, interpret it as asking for the
                # change into that year from the immediately preceding year among the
                # most recent consecutive years found.
                # This avoids comparing the query year to an older period when filings
                # present more historical columns.
                old_yr, new_yr = sorted_years[-2], sorted_years[-1]
            else:
                old_yr = new_yr = sorted_years[0] if sorted_years else "N/A"
            if "_old" in placeholder:
                best = next((v for v, y in matches if y == old_yr), matches[0][0])
                picked_year = next((y for v, y in matches if y == old_yr), matches[0][1])
            else:
                best = next((v for v, y in matches if y == new_yr), matches[-1][0])
                picked_year = next((y for v, y in matches if y == new_yr), matches[-1][1])
            final[placeholder] = best
        else:
            # Prefer value from query year
            best = None
            for val, yr in matches:
                if yr in query_years:
                    best = val
                    picked_year = yr
                    break
            if best is None:
                best, picked_year = matches[0]
            final[placeholder] = best

        score, source, ev_idx, line_item_label = winner_meta.get(placeholder, {}).get(
            picked_year, (1, "unknown", -1, "?")
        )
        if source == "composite":
            detail = f"sum of sub-items: {line_item_label}"
        elif ev_idx >= 0:
            detail = f"evidence[{ev_idx}] Line Item \"{line_item_label}\""
        else:
            detail = f"free-text match on \"{line_item_label}\""
        meta[placeholder] = {
            "source": source,
            "is_approximate": score < 2 or "supplementary" in source,
            "source_detail": detail,
        }

    return final, resolved, meta


#: Values at or below this magnitude are exempt from the duplicate-value
#: check below — 0, 1, and -1 can legitimately coincide between two
#: genuinely unrelated metrics (e.g. both being exactly zero), so
#: treating every such coincidence as suspected reuse would be too noisy
#: to be useful.
_DUPLICATE_VALUE_MIN_MAGNITUDE = 1.0


def _detect_and_strip_duplicate_values(
    resolved_formula: Dict[str, float],
    resolved_formula_series: Dict[str, List[Tuple[float, str]]],
    resolved_formula_meta: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Detect when two different variables resolve to the exact same numeric value — a
    strong signal one variable may have borrowed the other's number instead of finding
    its own match.
    When detected, drop the conflicting pairs back to unresolved in place so downstream
    code treats them as missing, and emit visible warning lines so the collision appears
    in generated code/logs.
    """
    warnings: List[str] = []
    placeholders = list(resolved_formula.keys())
    to_drop: set = set()
    for i in range(len(placeholders)):
        for j in range(i + 1, len(placeholders)):
            ph_a, ph_b = placeholders[i], placeholders[j]
            base_a = ph_a.replace("_new", "").replace("_old", "")
            base_b = ph_b.replace("_new", "").replace("_old", "")
            if base_a == base_b:
                continue
            val_a, val_b = resolved_formula[ph_a], resolved_formula[ph_b]
            if abs(val_a) <= _DUPLICATE_VALUE_MIN_MAGNITUDE or abs(val_b) <= _DUPLICATE_VALUE_MIN_MAGNITUDE:
                continue
            if val_a == val_b:
                to_drop.add(ph_a)
                to_drop.add(ph_b)
                warnings.append(
                    f"# WARNING: '{ph_a}' and '{ph_b}' both resolved to {val_a} — "
                    f"suspected value reuse between unrelated variables. Both dropped; "
                    f"falling back to a different extraction path."
                )
    for ph in to_drop:
        resolved_formula.pop(ph, None)
        resolved_formula_series.pop(ph, None)
        resolved_formula_meta[ph] = {"source": "unresolved", "is_approximate": False}
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: raw-number fallback
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raw_numbers(evidence_list: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Last-resort extraction from unstructured text.
    Filters out year-like numbers (1900-2099) and tiny values to avoid
    returning years (e.g. 2017, 2024) as financial results.
    """
    combined = "\n".join(ev.get("content", "") for ev in evidence_list)
    numbers = re.findall(r"(?<!\d)([\d,]{1,12}(?:\.\d+)?)(?!\d)", combined)
    extracted: Dict[str, Dict] = {}
    idx = 1
    for n_str in numbers:
        if idx > 8:
            break
        val = _to_float(n_str)
        if val is None:
            continue
        # Skip year-like values and near-zero noise
        if 1900 <= val <= 2099:  # year numbers — NOT financial data
            continue
        if abs(val) < 0.1:      # tiny values — likely noise
            continue
        key = f"num_{idx}"
        extracted[key] = {"item": f"Value_{idx}", "canonical": "unknown",
                          "year": "N/A", "val": val, "code_key": key}
        idx += 1
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: free-text keyword matching (PDF / MD&A narrative chunks)
# ─────────────────────────────────────────────────────────────────────────────

# (keyword_triggers, canonical_label, display_name)
# Triggers are checked in order; put longer/more-specific phrases first to
# avoid short tokens (e.g. "sales") matching before "net sales".
# Bug 2 fix: added "net sales", "net revenues", "cost of goods sold", etc.
_TEXT_PATTERNS: List[Tuple[List[str], str, str]] = [
    # Revenue extraction uses common report wordings and excludes bare terms that
    # overmatch (e.g., a short verb like "sales") to avoid capturing unrelated lines
    # such as proceeds labeled similarly in other statements.
    # This reduces false positives where similarly worded rows are not operating
    # revenue.
    (["net sales", "net revenues", "total net revenue",
      "revenue", "net revenue", "total revenue",
      "\u71df\u696d\u6536\u5165", "\u71df\u6536"],
     "revenue", "Revenue"),
    (["gross profit", "\u71df\u696d\u7e6a\u5229", "\u6bdb\u5229"],
     "gross_profit", "Gross Profit"),
    (["gross margin", "\u6bdb\u5229\u7387"],
     "gross_margin_pct", "Gross Margin"),
    (["operating income", "operating profit", "income from operations",
      "\u71df\u696d\u5229\u76ca"],
     "op_income", "Operating Income"),
    (["operating margin", "\u71df\u696d\u5229\u76ca\u7387"],
     "op_margin_pct", "Operating Margin"),
    (["net income", "net earnings", "net profit",
      "\u672c\u671f\u6de8\u5229", "\u6de8\u5229"],
     "net_income", "Net Income"),
    (["net margin", "\u6de8\u5229\u7387"],
     "net_margin_pct", "Net Margin"),
    (["earnings per share", "diluted earnings per share", "diluted eps",
      "eps", "\u6bcf\u80a1\u76c8\u9918"],
     "eps", "EPS"),
    (["research and development", "r&d expense", "r&d", "\u7814\u767c\u8cbb\u7528"],
     "rd_expense", "R&D"),
    (["capital expenditures", "capital expenditure", "capex",
      "\u8cc7\u672c\u652f\u51fa"],
     "capex", "CapEx"),
    (["total assets", "\u7e3d\u8cc7\u7522"],
     "total_assets", "Total Assets"),
    (["shareholders equity", "stockholders equity", "total equity",
      "\u80a1\u6771\u6b0a\u76ca"],
     "equity", "Equity"),
    (["free cash flow", "fcf", "\u81ea\u7531\u73fe\u91d1\u6d41"],
     "fcf", "Free Cash Flow"),
    # Cost / expense items
    (["cost of sales", "cost of goods sold", "cost of revenue",
      "cost of products", "cogs", "\u9500\u8ca8\u6210\u672c"],
     "cost_of_revenue", "Cost of Revenue"),
    (["selling, general and administrative", "selling, general",
      "sg&a", "sga", "\u63a8\u9500\u8cbb\u7528"],
     "sga", "SG&A"),
    (["ebitda"],
     "ebitda", "EBITDA"),
    # Balance sheet
    (["long-term debt", "long term debt"],
     "lt_debt", "Long-Term Debt"),
    (["total current assets", "current assets"],
     "current_assets", "Current Assets"),
    (["total current liabilities", "current liabilities"],
     "current_liab", "Current Liabilities"),
]

# Regex: number optionally followed by % or B/M/T suffix
_NUM_PATTERN = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(%|percent|billion|million|trillion|\u5104|B|M)?",
    re.IGNORECASE,
)

#: Allow canonical values that are legitimately percentages.
#: Other canonicals (revenue, net_income, etc.) must be absolute amounts;
#: nearby % figures usually indicate a rate of change, not the metric itself.
_PCT_CANONICALS = {"gross_margin_pct", "op_margin_pct", "net_margin_pct"}
# Year pattern
_YEAR_NEAR = re.compile(r"(?:FY)?(20\d{2})")


def _extract_from_free_text(
    evidence_list: List[Dict[str, Any]],
    query_years: List[str],
) -> Dict[str, Dict]:
    """
    Extract financial values from narrative/PDF text by keyword proximity.

    Multi-year fix (Bug 1 root cause fix):
    - Deduplication keyed by (canonical, year), not just canonical, so the
      same line item is captured for BOTH year-A AND year-B (needed for YoY).
    - After finding a trigger keyword, scans for ALL query years nearby and
      records each year's value by looking strictly AFTER the year token
      (avoids grabbing the adjacent year's value by accident).
    """
    extracted: Dict[str, Dict] = {}
    seen_canonical_year: set = set()  # key = (canonical, year)

    for ev in evidence_list:
        content = ev.get("parent_content") or ev.get("content", "")
        if not content or "Line Item:" in content:
            # legacy "Line Item: X | Year: Val" chunk — handled by
            # _extract_from_linearized_table
            continue
        if any(is_markdown_separator_row(l) for l in content.split("\n")):
            # Detect standard Markdown tables to avoid reprocessing by a flat-text
            # keyword-proximity heuristic.
            # Otherwise a flattened table can match the wrong cell's number.
            continue
        if _is_supplementary_schedule(content):
            # Avoid using evidence from guarantor/parent-only or combining schedules
            # that reuse consolidated labels for a different, smaller entity.
            # Row-scoped matching is required to prevent attributing the wrong entity's
            # numbers.
            continue
        if _is_quarterly_breakdown_table(content):
            continue
        content_lower = content.lower()

        for triggers, canonical, display_name in _TEXT_PATTERNS:
            # Skip only if ALL requested query years already found for this canonical
            found_years = {yr for (c, yr) in seen_canonical_year if c == canonical}
            if query_years and set(query_years).issubset(found_years):
                continue

            for trigger in triggers:
                pos = content_lower.find(trigger.lower())
                if pos == -1:
                    continue

                # Wide context around the trigger keyword
                ctx_start = max(0, pos - 20)
                ctx_end   = min(len(content), pos + 400)
                wide_ctx  = content[ctx_start:ctx_end]

                # ── Strategy A: match each query year explicitly ──────────
                # For each query year found in the context, look for the FIRST
                # non-year number that appears AFTER the year token.
                year_found_any = False
                for qy in (query_years or []):
                    if qy not in wide_ctx:
                        continue
                    if (canonical, qy) in seen_canonical_year:
                        continue

                    yr_pos_in_ctx = wide_ctx.find(qy)
                    # Look strictly AFTER the year token (skip up to 80 chars after)
                    after_year = wide_ctx[yr_pos_in_ctx + len(qy): yr_pos_in_ctx + len(qy) + 80]

                    val = None
                    for m in _NUM_PATTERN.finditer(after_year):
                        candidate = _to_float(m.group(1))
                        if candidate is None:
                            continue
                        if 1900 <= candidate <= 2099:   # skip year tokens
                            continue
                        if abs(candidate) < 0.01:       # skip near-zero
                            continue
                        suffix = (m.group(2) or "").lower()
                        if suffix in ("%", "percent") and canonical not in _PCT_CANONICALS:
                            continue  # a rate-of-change % is not this dollar-value metric's own value
                        val = candidate
                        break

                    if val is None:
                        continue

                    key = f"txt_{canonical}_{qy}"
                    if key not in extracted:
                        extracted[key] = {
                            "item": display_name,
                            "canonical": canonical,
                            "year": qy,
                            "val": val,
                            "code_key": key,
                        }
                        seen_canonical_year.add((canonical, qy))
                        year_found_any = True

                # ── Strategy B: fallback — first non-year number near trigger ──
                if not year_found_any:
                    window = content[pos: pos + 150]
                    val = None
                    for m in _NUM_PATTERN.finditer(window):
                        candidate = _to_float(m.group(1))
                        if candidate is None:
                            continue
                        if 1900 <= candidate <= 2099:
                            continue
                        if abs(candidate) < 0.01:
                            continue
                        suffix = (m.group(2) or "").lower()
                        if suffix in ("%", "percent") and canonical not in _PCT_CANONICALS:
                            continue
                        val = candidate
                        break

                    if val is None:
                        break  # try next trigger in group

                    yr_m = _YEAR_NEAR.search(window)
                    year = yr_m.group(1) if yr_m else (query_years[-1] if query_years else "N/A")
                    if query_years and year not in query_years:
                        wider = content[max(0, pos - 40): pos + 150]
                        for qy in query_years:
                            if qy in wider:
                                year = qy
                                break

                    if (canonical, year) not in seen_canonical_year:
                        key = f"txt_{canonical}_{year}"
                        if key not in extracted:
                            extracted[key] = {
                                "item": display_name,
                                "canonical": canonical,
                                "year": year,
                                "val": val,
                                "code_key": key,
                            }
                            seen_canonical_year.add((canonical, year))

                break  # trigger matched; move to next pattern group

    return extracted




# ─────────────────────────────────────────────────────────────────────────────
# Code generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gen_period_average_code(
    fk: str,
    label: str,
    unit: str,
    expr: str,
    resolved_series: Dict[str, List[Tuple[float, str]]],
    query_years: Optional[List[str]] = None,
) -> List[str]:
    """Generate code that computes an expression separately for every year common to all
    placeholders, then averages those per-year results (useful for N-year average
    ratios).
    When specific query years are given, restrict the average to exactly those years; do
    not expand the year set based on additional retrieved evidence.
    """
    if not resolved_series or not expr:
        return []
    for series in resolved_series.values():
        if not series:
            return []

    lines: List[str] = []
    by_year_names = []
    for ph, series in resolved_series.items():
        by_year = {y: v for v, y in series}  # last value per year wins on duplicates
        by_year_name = f"{ph}_by_year"
        lines.append(f"{by_year_name} = {by_year!r}")
        by_year_names.append(by_year_name)

    # list(), not sorted(list()) — the sandbox's builtin whitelist doesn't
    # include sorted(), and order doesn't matter here since the years are
    # only ever averaged, never displayed positionally. Likewise no
    # explicit "raise ValueError(...)" guard for an empty intersection —
    # ValueError isn't in the whitelist either; an empty _years naturally
    # leaves _ratios empty and sum(_ratios)/len(_ratios) raises
    # ZeroDivisionError (a real, silent Python operator behaviour, not a
    # name lookup), which the caller's repair loop already handles.
    intersection_expr = " & ".join(f"set({n})" for n in by_year_names)
    distinct_query_years = sorted(set(query_years)) if query_years else []
    if len(distinct_query_years) >= 2:
        intersection_expr += f" & {set(distinct_query_years)!r}"
    lines.append(f"_years = list({intersection_expr})")
    lines.append("_ratios = []")
    lines.append("for _y in _years:")
    for ph in resolved_series:
        lines.append(f"    {ph} = {ph}_by_year[_y]")
    lines.append(f"    _ratios.append({expr})")
    lines.append(f"# Formula: {fk}")
    if unit == "%":
        lines.append("result = round(sum(_ratios) / len(_ratios) * 100, 2)")
        lines.append(f"print(f'{label} ({{len(_years)}}-yr avg): {{result}}%')")
    else:
        lines.append("result = round(sum(_ratios) / len(_ratios), 4)")
        lines.append(f"print(f'{label} ({{len(_years)}}-yr avg): {{result}}')")
    return lines


def _extract_formula_placeholders(expr: str) -> set:
    """Identify identifier tokens in a formula expression that are variable
    placeholders needing a resolved value — NOT builtin functions being
    called. A token immediately followed by "(" is a call, not a placeholder.
    This avoids treating function names as missing placeholders and incorrectly
    failing resolution.
    """
    return {
        m.group(0) for m in re.finditer(r"[a-z_][a-z0-9_]*", expr)
        if expr[m.end():m.end() + 1] != "("
    } - {"years", "math"}


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific value adjustments & derivations (custodial funds, working
# capital, total-row rollforward, adjusted-EBIT-from-EBITDA reconciliation,
# quick-ratio prepaid adjustment, capital-intensity context)
# ─────────────────────────────────────────────────────────────────────────────

_CUSTODIAL_FUNDS_RE = re.compile(r'funds?\s+receivable.*customer', re.IGNORECASE)
_SHORT_TERM_INVESTMENTS_RE = re.compile(r'short[\s-]?term\s+investments?', re.IGNORECASE)


def _adjust_working_capital_for_custodial_funds(
    resolved: Dict[str, float],
    resolved_series: Optional[Dict[str, List[Tuple[float, str]]]],
    extracted_table: Dict[str, Dict],
) -> bool:
    """Adjust working-capital calculation for companies that hold custodial
    customer funds on the balance sheet. Detects a custodial "funds
    receivable/customer accounts" asset paired with a matching liability and
    excludes cash and short-term investments from current_assets while leaving
    the custodial liability in current_liabilities. Mutates resolved (and
    resolved_series if provided) in place. Returns True if the adjustment ran
    so callers can avoid double-applying related adjustments.
    """
    if "current_assets" not in resolved:
        return False
    has_custodial = any(
        _CUSTODIAL_FUNDS_RE.search(v.get("item", "") or "")
        for v in extracted_table.values()
    )
    if not has_custodial:
        return False
    # Recover the YEAR the resolved current_assets value actually came
    # from (by matching it back to its own extracted_table row) so cash/
    # short-term-investments are only ever subtracted for that SAME year,
    # never a different one.
    ca_year = None
    for v in extracted_table.values():
        if v.get("canonical") == "current_assets" and v.get("val") == resolved["current_assets"]:
            ca_year = v.get("year")
            break
    if ca_year is None:
        return False
    cash_val = None
    sti_val = None
    for v in extracted_table.values():
        if v.get("year") != ca_year:
            continue
        if cash_val is None and v.get("canonical") == "cash":
            cash_val = v.get("val", 0.0)
        if sti_val is None and _SHORT_TERM_INVESTMENTS_RE.search(v.get("item", "") or ""):
            sti_val = v.get("val", 0.0)
    deduction = (cash_val or 0.0) + (sti_val or 0.0)
    if not deduction:
        return False
    resolved["current_assets"] = resolved["current_assets"] - deduction
    if resolved_series and resolved_series.get("current_assets"):
        resolved_series["current_assets"] = [
            ((val - deduction) if yr == ca_year else val, yr)
            for val, yr in resolved_series["current_assets"]
        ]
    return True


#: Working-capital conventions can differ by industry: some treatments
#: exclude cash and short-term debt (an operating-only convention), while
#: others (e.g., rate-regulated utilities) include them.
#: Decide convention based on industry signals; mismatching rules leads to incorrect
#: values.
_REGULATED_UTILITY_RE = re.compile(
    r'\bpublic\s+utility\s+commission\b|\bregulated\s+utility\b|\brate\s+case\b|'
    r'\bstate\s+regulatory\s+commission\b|\brate\s+base\b',
    re.IGNORECASE,
)

#: If the question explicitly gives a formula (for example, "net working capital =
#: current assets - current liabilities"), follow that formula rather than applying
#: inferred conventions.
#: This rule prevents applying unrelated operational adjustments when the question
#: specifies the desired measure.
_EXPLICIT_PLAIN_WORKING_CAPITAL_RE = re.compile(
    r'define\w*\s+(?:net\s+)?working\s+capital\s+as\s+total\s+current\s+assets\s+'
    r'(?:less|minus)\s+total\s+current\s+liabilities',
    re.IGNORECASE,
)

_FINANCING_CURRENT_LIAB_RE = re.compile(
    r'short[\s-]?term\s+(?:debt|borrowings)|current\s+portion\s+of\s+long[\s-]?term\s+debt',
    re.IGNORECASE,
)

#: A canonical labeled 'cash' may appear in multiple statements (BS and
#: cash-flow reconciliations) but only the balance-sheet line should be
#: used when deducting current assets.
#: Do not sum multiple 'cash' rows from different statements as a single balance-sheet
#: cash figure.
_CASH_FLOW_STATEMENT_CASH_LABEL_RE = re.compile(
    r'beginning\s+of\s+(?:year|period)|end\s+of\s+(?:year|period)|'
    r'net\s+(?:\(?(?:increase|decrease)\)?\s*)+in\s+cash',
    re.IGNORECASE,
)

#: Distinguish period-ACTIVITY wording from period-END balance for short-term
#: debt/borrowings when adjusting working capital.
#: Adjustments should use end-of-period balances, not cash-flow activity lines.
_CASH_FLOW_STATEMENT_DEBT_ACTION_RE = re.compile(
    r'repayments?\s+of|proceeds\s+from|borrowings?\s+under|issuance\s+of',
    re.IGNORECASE,
)


def _is_regulated_utility_filing(evidence_list: List[Dict[str, Any]]) -> bool:
    for item in evidence_list or []:
        content = str(item.get("content") or item.get("parent_content") or "")
        if _REGULATED_UTILITY_RE.search(content):
            return True
    return False


def _adjust_working_capital_for_financing_items(
    resolved: Dict[str, float],
    resolved_series: Optional[Dict[str, List[Tuple[float, str]]]],
    extracted_table: Dict[str, Dict],
) -> None:
    """
    Operating working capital: current_assets minus cash (+ short-term
    investments), and current_liabilities minus short-term debt / the
    current portion of long-term debt -- see _REGULATED_UTILITY_RE above
    for when this applies. No-op if either side's deduction is zero (a
    plain current_assets - current_liabilities is then already correct),
    same guard pattern as _adjust_working_capital_for_custodial_funds.
    Mutates `resolved` (and `resolved_series`, if given) in place.
    """
    if "current_assets" not in resolved or "current_liabilities" not in resolved:
        return

    def _year_of(canonical: str, target_val: float) -> Optional[str]:
        for v in extracted_table.values():
            if v.get("canonical") == canonical and v.get("val") == target_val:
                return v.get("year")
        return None

    ca_year = _year_of("current_assets", resolved["current_assets"])
    # Canonical taxonomy shortens the field to current_liab rather than
    # current_liabilities.
    # Use the canonical name to avoid missing matches that skip the deduction step.
    cl_year = _year_of("current_liab", resolved["current_liabilities"])

    # Only ONE balance-sheet cash figure and ONE short-term-investments
    # figure ever belong in this deduction -- take the first genuine match
    # of each (same pattern as _adjust_working_capital_for_custodial_funds
    # above), never sum every "cash"-canonical row, which double/triple/
    # quadruple-counts the cash-flow statement's own reconciliation lines
    # for the same balance (see _CASH_FLOW_STATEMENT_CASH_LABEL_RE).
    cash_val = None
    sti_val = None
    if ca_year is not None:
        for v in extracted_table.values():
            if v.get("year") != ca_year:
                continue
            item = v.get("item", "") or ""
            if (
                cash_val is None
                and v.get("canonical") == "cash"
                and not _CASH_FLOW_STATEMENT_CASH_LABEL_RE.search(item)
            ):
                cash_val = v.get("val", 0.0) or 0.0
            elif sti_val is None and _SHORT_TERM_INVESTMENTS_RE.search(item):
                sti_val = v.get("val", 0.0) or 0.0
    ca_deduction = (cash_val or 0.0) + (sti_val or 0.0)

    # Allow two distinct end-of-period rows to sum when they represent different
    # balance-sheet lines (e.g., short-term debt and current portion of long-term debt).
    # Exclude cash-flow action-verb wording and dedupe repeated values by VALUE to avoid
    # double-counting duplicate disclosures.
    cl_deduction = 0.0
    seen_cl_vals = set()
    if cl_year is not None:
        for v in extracted_table.values():
            if v.get("year") != cl_year:
                continue
            item = v.get("item", "") or ""
            if _CASH_FLOW_STATEMENT_DEBT_ACTION_RE.search(item):
                continue
            if _FINANCING_CURRENT_LIAB_RE.search(item):
                val = v.get("val", 0.0) or 0.0
                if val in seen_cl_vals:
                    continue
                seen_cl_vals.add(val)
                cl_deduction += val

    if not ca_deduction and not cl_deduction:
        return

    resolved["current_assets"] = resolved["current_assets"] - ca_deduction
    resolved["current_liabilities"] = resolved["current_liabilities"] - cl_deduction
    if resolved_series:
        if resolved_series.get("current_assets") and ca_deduction:
            resolved_series["current_assets"] = [
                ((val - ca_deduction) if yr == ca_year else val, yr)
                for val, yr in resolved_series["current_assets"]
            ]
        if resolved_series.get("current_liabilities") and cl_deduction:
            resolved_series["current_liabilities"] = [
                ((val - cl_deduction) if yr == cl_year else val, yr)
                for val, yr in resolved_series["current_liabilities"]
            ]


_PREPAID_CURRENT_ASSET_RE = re.compile(r'\bprepaid', re.IGNORECASE)
_OTHER_CURRENT_ASSET_RE = re.compile(r'^\s*other\s+current\s+assets?\b', re.IGNORECASE)
_NONOPERATING_EXCLUDE_RE = re.compile(r'non[\s-]?current|long[\s-]?term|liabilit', re.IGNORECASE)

#: "How many/number of/change in the number of <plural noun>" -- no formula
#: canonical exists for a bare count of physical units (stores, locations...),
#: so this fires only for that specific question shape.
_COUNT_CHANGE_QUERY_RE = re.compile(
    r'\bhow\s+many\b|\bnumber\s+of\b|\bchange\s+in\s+the\s+number\b', re.IGNORECASE
)


def _derive_total_row_period_end_change(
    evidence_list: List[Dict[str, Any]], q_lower: str,
) -> Optional[Tuple[float, float, str, str, str]]:
    """Answer "how many/number of <plural noun> ... did X have, and did it
    change?" from a roll-forward table by reading the row literally labelled
    "Total" and using that row's period-END column values. This preserves
    full column labels (e.g., "End of <period>") which the generic year-only
    extractor would discard. Fires only when the row label is exactly "Total",
    its column names include an "end of" phrase for two fiscal years, and the
    column names refer to the same plural noun asked about. Returns
    (new_val, old_val, new_period_label, old_period_label, noun) or None.
    """
    if not _COUNT_CHANGE_QUERY_RE.search(q_lower):
        return None
    noun_m = re.search(
        r'\b(stores?|locations?|branches?|restaurants?|units?|facilit(?:y|ies)|plants?|offices?)\b',
        q_lower,
    )
    if not noun_m:
        return None
    noun_stem = re.sub(r'(?:y|ies|es|s)$', '', noun_m.group(1).lower())
    for ev in evidence_list:
        content = ev.get("content") or ""
        m = re.search(r'\|\s*Line Item:\s*Total\s*\|(.*)$', content)
        if not m:
            continue
        pairs = []
        for f in m.group(1).split("|"):
            f = f.strip()
            fm = re.match(r'(.+?):\s*(\(?-?[\d,]+\.?\d*\)?)\s*$', f)
            if fm:
                pairs.append((fm.group(1).strip(), fm.group(2)))
        end_of = [
            (k, v) for k, v in pairs
            if re.search(r'(?i)\bend\s+of\b', k) and noun_stem in k.lower()
        ]
        if len(end_of) < 2:
            continue

        def _num(v: str) -> float:
            v = v.strip()
            neg = v.startswith("(") and v.endswith(")")
            v = v.strip("()").replace(",", "")
            val = float(v)
            return -val if neg else val

        def _fy(k: str) -> int:
            ym = re.search(r'(?:19|20)\d{2}', k)
            return int(ym.group(0)) if ym else -1

        end_of.sort(key=lambda kv: _fy(kv[0]), reverse=True)
        if _fy(end_of[0][0]) == _fy(end_of[1][0]):
            continue  # can't tell which is "new" vs "old" without distinct years
        (new_k, new_v), (old_k, old_v) = end_of[0], end_of[1]
        return _num(new_v), _num(old_v), new_k, old_k, noun_m.group(1)
    return None


#: A question naming "Adjusted EBIT" specifically (not plain EBIT/operating
#: income) needs that exact non-GAAP figure, not the GAAP "Operating
#: income (loss)" row interest_coverage's plain "ebit" alias list resolves
#: to by default.
_ADJUSTED_EBIT_QUERY_RE = re.compile(r'\badjusted\s+ebit\b', re.IGNORECASE)
_ADJUSTED_EBITDA_R_LABEL_RE = re.compile(r'(?i)^adjusted\s+ebitda(r)?$')
_DEPRECIATION_LABEL_RE = re.compile(r'(?i)depreciation\s+and\s+amortization')
_RENT_ADDBACK_LABEL_RE = re.compile(r'(?i)triple-?net.*rent|ground\s+lease.*rent|\brent\s+expense\b')


def _derive_adjusted_ebit_from_ebitda_reconciliation(
    evidence_list: List[Dict[str, Any]], target_year: str, q_lower: str,
) -> Optional[Tuple[float, str]]:
    """Derive Adjusted EBIT when a filer provides Adjusted EBITDA or
    Adjusted EBITDAR but not Adjusted EBIT. Use textbook identities
    (EBIT = EBITDA - D&A; EBIT = EBITDAR - D&A - Rent) applied to the
    filers' own reconciliation table rows from the same evidence item.
    This avoids mixing values from different tables. Returns (value,
    source_description) or None when the required rows for the target year
    are not found in the same reconciliation table.
    """
    if not _ADJUSTED_EBIT_QUERY_RE.search(q_lower):
        return None

    def _cell_to_float(cell: str) -> Optional[float]:
        # search (not fullmatch/replace-chain): a "$ 1,473,093"-style cell can
        # have a space right after the "$" that a plain .replace("$", "")
        # leaves behind as a stray leading space, breaking a fullmatch. A
        # digit search on the raw cell sidesteps that (and any other stray
        # whitespace) entirely.
        cell = cell.strip()
        if not cell or cell in ("—", "-", "--"):
            return None
        neg = "(" in cell and ")" in cell
        m = re.search(r'\d[\d,]*\.?\d*', cell)
        if not m:
            return None
        val = float(m.group(0).replace(",", ""))
        return -val if neg else val

    seen_blocks = set()
    for ev in evidence_list:
        content = ev.get("parent_content") or ev.get("content", "")
        if not content or "ebitda" not in content.lower():
            continue
        for block in content.split("\n\n"):
            lines = [l for l in block.splitlines() if l.strip().startswith("|")]
            if len(lines) < 4 or block in seen_blocks:
                continue
            seen_blocks.add(block)
            header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
            if len(header) < 2:
                continue
            # Prefer a column naming the target year that ALSO looks annual
            # (its header does not also say a quarter/short-period code);
            # fall back to any column naming the target year at all.
            col_idx, annual_idx = None, None
            for i, h in enumerate(header[1:], start=1):
                if target_year not in h:
                    continue
                if col_idx is None:
                    col_idx = i
                if not re.search(r'\bQ[1-4]\b|\b[3699]M\b|three\s+months|six\s+months|nine\s+months', h, re.IGNORECASE):
                    annual_idx = i
            col_idx = annual_idx or col_idx
            if col_idx is None:
                continue
            rows: Dict[str, float] = {}
            for l in lines[2:]:
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                if len(cells) <= col_idx or not cells[0]:
                    continue
                val = _cell_to_float(cells[col_idx])
                if val is not None:
                    rows[cells[0]] = val
            ebitda_val, is_ebitdar, ebitda_label = None, False, None
            for label, val in rows.items():
                m = _ADJUSTED_EBITDA_R_LABEL_RE.match(label.strip())
                if m:
                    ebitda_val, is_ebitdar, ebitda_label = val, bool(m.group(1)), label
                    if is_ebitdar:
                        break  # EBITDAR is more specific; prefer it over a same-block EBITDA row
            if ebitda_val is None:
                continue
            dep_val = next((v for k, v in rows.items() if _DEPRECIATION_LABEL_RE.search(k)), None)
            if dep_val is None:
                continue
            rent_val = 0.0
            if is_ebitdar:
                rent_val = next((v for k, v in rows.items() if _RENT_ADDBACK_LABEL_RE.search(k)), None)
                if rent_val is None:
                    continue  # an EBITDAR total without its own rent add-back row can't be bridged safely
            adjusted_ebit = ebitda_val - abs(dep_val) - abs(rent_val)
            source = (
                f"{ebitda_label} {ebitda_val:g} - Depreciation and amortization {dep_val:g}"
                + (f" - Rent expense {rent_val:g}" if is_ebitdar else "")
            )
            return adjusted_ebit, source
    return None

# Apply a materiality threshold when deciding whether to replace the quick-ratio
# shortcut with the strict liquid-asset sum.
# Use a general cutoff (e.g., around 15%) on prepaid+other current assets relative to
# current liabilities to determine which method to use.
_QUICK_RATIO_NONLIQUID_MATERIALITY_THRESHOLD = 0.15


def _adjust_quick_ratio_for_prepaid_and_other(
    resolved: Dict[str, float],
    resolved_series: Optional[Dict[str, List[Tuple[float, str]]]],
    extracted_table: Dict[str, Dict],
) -> None:
    """Handle quick ratio shorthand "(current_assets - inventory) /
    current_liabilities" when a material "prepaid/other current assets"
    bucket exists. Subtracts that bucket from current_assets (and the
    matching year in resolved_series if present) before formula codegen so
    the shorthand aligns with the stricter liquid-assets definition. No-op
    when the bucket is not identifiable or not material. Mutates resolved
    (and resolved_series if given) in place.
    """
    if "current_assets" not in resolved or "current_liabilities" not in resolved:
        return
    cl_val = resolved["current_liabilities"]
    if not cl_val:
        return
    ca_year = None
    for v in extracted_table.values():
        if v.get("canonical") == "current_assets" and v.get("val") == resolved["current_assets"]:
            ca_year = v.get("year")
            break
    if ca_year is None:
        return
    # For prepaid vs other-current-assets buckets, take the single largest matching row,
    # not the sum of all similarly labeled rows.
    # The consolidated total is usually the largest; max-per-bucket avoids double-
    # counting split/disaggregated disclosures.
    prepaid_candidates = []
    other_candidates = []
    for v in extracted_table.values():
        if v.get("year") != ca_year:
            continue
        item = v.get("item", "") or ""
        if _NONOPERATING_EXCLUDE_RE.search(item):
            continue
        val = v.get("val", 0.0)
        if _OTHER_CURRENT_ASSET_RE.search(item):
            other_candidates.append(val)
        elif _PREPAID_CURRENT_ASSET_RE.search(item):
            prepaid_candidates.append(val)
    nonliquid_total = (max(prepaid_candidates) if prepaid_candidates else 0.0) + (
        max(other_candidates) if other_candidates else 0.0
    )
    if nonliquid_total <= 0:
        return
    if nonliquid_total / cl_val < _QUICK_RATIO_NONLIQUID_MATERIALITY_THRESHOLD:
        return
    resolved["current_assets"] = resolved["current_assets"] - nonliquid_total
    if resolved_series and resolved_series.get("current_assets"):
        resolved_series["current_assets"] = [
            ((val - nonliquid_total) if yr == ca_year else val, yr)
            for val, yr in resolved_series["current_assets"]
        ]


def _capital_intensity_context_lines(
    resolved: Dict[str, float], extracted_table: Dict[str, Dict],
) -> List[str]:
    """Emit supplementary diagnostics for capital_intensity_ratio: compute
    CAPEX/Revenue, Fixed-Assets/Total-Assets, and ROA alongside the primary
    assets/revenue ratio without modifying the main result or resolved data.
    Only includes signals whose underlying rows (capex, ppe, net_income)
    resolve from the same year as the ratio's total_assets value, so partial
    outputs are still consistent and more informative than the bare ratio.
    """
    if "total_assets" not in resolved or "revenue" not in resolved:
        return []
    ta_year = None
    for v in extracted_table.values():
        if v.get("canonical") == "total_assets" and v.get("val") == resolved["total_assets"]:
            ta_year = v.get("year")
            break
    if ta_year is None:
        return []

    def _find(canonical: str) -> Optional[float]:
        # When both gross and net PP&E rows are present under the same canonical, prefer
        # the explicit net row for fixed-assets/total-assets calculations.
        # Net carrying value conventionally defines fixed-assets ratios; do not use the
        # gross value if net is available.
        candidates = [
            v for v in extracted_table.values()
            if v.get("canonical") == canonical and v.get("year") == ta_year
        ]
        if not candidates:
            return None
        net_rows = [v for v in candidates if "net" in (v.get("item", "") or "").lower()]
        gross_rows = [v for v in candidates if "gross" in (v.get("item", "") or "").lower()]
        pick = net_rows or [v for v in candidates if v not in gross_rows] or candidates
        return pick[0].get("val")

    capex = _find("capex")
    ppe = _find("ppe")
    net_income = _find("net_income")

    out: List[str] = []
    out.append(
        "# Supplementary capital-intensity signals (FinanceBench gold answers use "
        "DIFFERENT conventions per company -- see all of these, not just the ratio above)"
    )
    if capex is not None:
        out.append(f"_capex_to_revenue_pct = round(abs({capex}) / {resolved['revenue']} * 100, 2)")
        out.append(f"print(f'CapEx/Revenue ({ta_year}): {{_capex_to_revenue_pct}}%')")
    if ppe is not None:
        out.append(f"_fixed_assets_to_total_assets_pct = round({ppe} / {resolved['total_assets']} * 100, 2)")
        out.append(f"print(f'Fixed Assets/Total Assets ({ta_year}): {{_fixed_assets_to_total_assets_pct}}%')")
    if net_income is not None:
        out.append(f"_roa_pct = round({net_income} / {resolved['total_assets']} * 100, 2)")
        out.append(f"print(f'Return on Assets ({ta_year}): {{_roa_pct}}%')")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PoT code generation: single formula (_gen_formula_code)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_formula_code(
    formula_entry: Dict[str, Any],
    resolved: Dict[str, float],
    extracted_table: Dict[str, Dict],
    resolved_series: Optional[Dict[str, List[Tuple[float, str]]]] = None,
    resolved_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    query_years: Optional[List[str]] = None,
    q_lower: str = "",
    is_regulated_utility: bool = False,
) -> List[str]:
    """Generate Python code lines from a formula template.

    For period-average formulas (e.g. "3-year average of X as a % of Y"), compute the
    ratio for each year in the range then average; these need resolved_series (mapping
    placeholders -> list of (value, year)). Other formula shapes use resolved (one value
    per placeholder).

    resolved_meta is used to annotate each variable assignment with its source (which
    evidence item and which line) so incorrect extractions are traceable from generated
    code/logs without re-inspecting raw evidence.
    """
    lines = []
    fk = formula_entry.get("formula_key", "custom")
    label = formula_entry.get("result_label", "Result")
    unit = formula_entry.get("unit", "")
    expr = formula_entry.get("formula_expr", "")
    is_multi_year = formula_entry.get("multi_year", False)
    # A formula being able to compute an N-year average does not imply every multi-year
    # question wants that average.
    # Only use the N-year-average generation when the query explicitly requests an
    # average; otherwise perform a multi-year trend/direction comparison.
    is_period_average = formula_entry.get("period_average", False) and any(
        kw in q_lower for kw in ("average", "avg", "平均")
    )

    if is_period_average:
        return _gen_period_average_code(fk, label, unit, expr, resolved_series or {}, query_years)

    # Prefer a filing's directly disclosed ratio value when present, since it reflects
    # the filer’s own adjustments and rounding.
    # If the filing does not disclose that line, fall back to computing the ratio from
    # subcomponents.
    direct_var = formula_entry.get("direct_lookup_var")
    if direct_var and resolved_series and resolved_series.get(direct_var):
        direct_series = resolved_series[direct_var]
        trend_years = sorted(set(_with_implied_trend_year(query_years, q_lower)))
        by_year = {y: v for v, y in direct_series}
        if len(trend_years) >= 2 and all(y in by_year for y in trend_years):
            lines: List[str] = [f"# Formula: {fk} (filing-reported value)"]
            _emit_multi_year_ratio(
                lines, label, unit,
                [(yr, str(by_year[yr])) for yr in trend_years],
            )
            return lines
        picked_year = preferred_year if preferred_year in by_year else (
            query_years[-1] if query_years and query_years[-1] in by_year else None
        )
        if picked_year is None:
            val, picked_year = direct_series[-1]
        else:
            val = by_year[picked_year]
        return [
            f"# Formula: {fk} (filing-reported value)",
            f"result = round({val}, 2)",
            f"print(f'{label} ({picked_year}): {{result}}{unit}')",
        ]

    if not resolved:
        return []

    if fk == "working_capital":
        custodial_applied = _adjust_working_capital_for_custodial_funds(
            resolved, resolved_series, extracted_table
        )
        explicit_plain_definition = bool(_EXPLICIT_PLAIN_WORKING_CAPITAL_RE.search(q_lower))
        if not custodial_applied and not is_regulated_utility and not explicit_plain_definition:
            _adjust_working_capital_for_financing_items(resolved, resolved_series, extracted_table)

    if fk == "quick_ratio":
        _adjust_quick_ratio_for_prepaid_and_other(resolved, resolved_series, extracted_table)

    # ── Multi-year trend comparison ─────────────────────────────────────────
    # A "did X improve or decline" question needs the SAME ratio computed
    # for two years to actually answer, not one year's snapshot (same
    # rationale as _emit_multi_year_ratio, used by the non-formula-library
    # calculation path for the identical reason). Skipped for is_multi_year
    # formulas (fixed_asset_turnover etc.) — those already compare two
    # years INSIDE a single ratio (average PP&E across years), so a
    # year-over-year comparison of the ratio ITSELF is a different,
    # unrequested question. Only fires when every placeholder the formula
    # needs actually has resolved data for both comparison years — no
    # partial/guessed comparison.
    if not is_multi_year and resolved_series:
        trend_years = sorted(set(_with_implied_trend_year(query_years, q_lower)))
        if len(trend_years) >= 2:
            needed_phs = _extract_formula_placeholders(expr)
            per_year_vals: Dict[str, Dict[str, float]] = {}
            for yr in trend_years:
                ph_vals = {}
                for ph in needed_phs:
                    match = next((v for v, y in resolved_series.get(ph, []) if y == yr), None)
                    if match is None:
                        ph_vals = None
                        break
                    ph_vals[ph] = match
                if ph_vals is not None:
                    per_year_vals[yr] = ph_vals
            if len(per_year_vals) >= 2:
                trend_lines: List[str] = []
                uses_builtin_pct = any(fn in expr for fn in ["yoy(", "cagr(", "* 100", "*100"])
                year_exprs = []
                for yr in sorted(per_year_vals):
                    ph_vals = per_year_vals[yr]
                    for ph, val in ph_vals.items():
                        trend_lines.append(f"{ph}_{yr} = {val}")
                    yr_expr = expr
                    for ph in needed_phs:
                        yr_expr = re.sub(rf"\b{re.escape(ph)}\b", f"{ph}_{yr}", yr_expr)
                    if unit == "%" and not uses_builtin_pct:
                        yr_expr = f"({yr_expr}) * 100"
                    year_exprs.append((yr, yr_expr))
                _emit_multi_year_ratio(trend_lines, label, unit, year_exprs)
                return trend_lines

    for ph, val in resolved.items():
        ph_meta = (resolved_meta or {}).get(ph, {})
        detail = ph_meta.get("source_detail")
        comment = f"  # {ph_meta.get('source', '?')} <- {detail}" if detail else ""
        lines.append(f"{ph} = {val}{comment}")

    # Check all needed placeholders are available
    needed = _extract_formula_placeholders(expr)
    available = set(resolved.keys())
    missing = needed - available
    if missing:
        # Try to fill from linearized table
        for ph in list(missing):
            ph_clean = ph.replace("_new", "")
            for k, v in extracted_table.items():
                # Require exact matches on both the row's canonical tag and the
                # sanitized code_key item-name.
                # Avoid substring matches that can erroneously map a placeholder like
                # "revenue" to unrelated rows such as "cost_of_revenue".
                key_tail = re.sub(r'^val_\d{4}_', '', k)
                key_tail = re.sub(r'_\d+$', '', key_tail)
                if v.get("canonical", "") == ph_clean or key_tail == ph_clean:
                    # This fallback path does not apply special carve-out awareness, so
                    # carve-out or minority-interest rows can match if they share the
                    # same canonical tag.
                    # Be aware such matches may return fractional or subsidiary figures
                    # rather than the parent company's total.
                    if re.search(r'\b(redeemable|noncontrolling|non-controlling|minority)\b',
                                 v.get("item", "").lower()):
                        continue
                    lines.append(f"{ph} = {v['val']}  # from table fallback")
                    available.add(ph)
                    break
        if needed - available:
            return []  # still missing vars → fail

    # Inject 'years' for CAGR formulas
    if "years" in expr:
        yr_matches = re.findall(r"20(\d{2})", " ".join(str(v) for v in resolved.keys()))
        if len(yr_matches) >= 2:
            lines.append(f"years = {abs(int(yr_matches[-1]) - int(yr_matches[0]))}.0")
        else:
            lines.append("years = 1.0")

    if fk == "capital_intensity_ratio":
        lines.extend(_capital_intensity_context_lines(resolved, extracted_table))

    # Some filings report a percent-change line using a short label that prevents alias-
    # substring matching; use a direct percent-change finder for those cases.
    # Only substitute a filing-reported percent when the question does not require a
    # specific rounded computation; otherwise compute the precise value.
    wants_precise_rounding = "decimal place" in q_lower
    if (
        fk == "revenue_yoy" and not wants_precise_rounding
        and "revenue_new" in resolved and "revenue_old" in resolved
    ):
        yoy_new_yr = next(
            (v.get("year") for v in extracted_table.values()
             if v.get("canonical") == "revenue" and v.get("val") == resolved["revenue_new"]),
            None,
        )
        yoy_old_yr = next(
            (v.get("year") for v in extracted_table.values()
             if v.get("canonical") == "revenue" and v.get("val") == resolved["revenue_old"]),
            None,
        )
        if yoy_new_yr and yoy_old_yr:
            direct_pct = _find_direct_pct_change_row(extracted_table, yoy_old_yr, yoy_new_yr)
            if direct_pct is not None:
                lines.append(f"# Formula: {fk} (filing's own reported % change, preferred over recomputed)")
                lines.append(f"result = {direct_pct}")
                lines.append(f"print(f'{label} ({yoy_old_yr}->{yoy_new_yr}) [filing-reported]: {{result}}%')")
                return lines

    lines.append(f"# Formula: {fk}")
    if unit == "%":
        lines.append(f"_raw = {expr}")
        # The formula_expr is a ratio (0-1); multiply by 100 to get percentage
        # UNLESS the expression already handles %, or it calls yoy()/cagr() which
        # already return percentage values
        uses_builtin_pct = any(fn in expr for fn in ["yoy(", "cagr(", "* 100", "*100"])
        if not uses_builtin_pct:
            lines.append("result = round(_raw * 100, 2)")
        else:
            lines.append("result = round(_raw, 2)")
        lines.append(f"print(f'{label}: {{result}}%')")
    elif unit == "x":
        lines.append(f"result = round({expr}, 4)")
        lines.append(f"print(f'{label}: {{result}}')")
    else:
        # Round even here — formulas with no "%"/"x" unit (e.g. DPO's
        # day-count, unadjusted EBITDA's dollar sum) still routinely get
        # asked for with an explicit "round to N decimal places"
        # instruction; 2 decimals is a reasonable general default rather
        # than leaving raw floating-point division noise in the answer.
        lines.append(f"result = round({expr}, 2)")
        lines.append(f"print(f'{label}: {{result}}')")
        # When presenting signed deltas alongside a directional verb, convert to an
        # explicit magnitude-plus-direction phrase to avoid double negatives.
        # Keep the underlying signed result for internal logic, but print a positive
        # magnitude with the correct verb-driven direction for clear narration.
        if fk == "debt_change_yoy":
            lines.append("_magnitude = abs(result)")
            lines.append("_direction_word = 'increased' if result > 0 else ('decreased' if result < 0 else 'stayed flat')")
            lines.append(f"print(f'{{_direction_word}} by {{_magnitude}}')")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Canonical grouping & pair-matching for multi-metric/trend calculations
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_canonical(extracted: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Group extracted variables by their canonical item key."""
    groups: Dict[str, List[Dict]] = {}
    for v in extracted.values():
        c = v.get("canonical", "unknown")
        groups.setdefault(c, []).append(v)
    return groups


# Canonical labels that indicate revenue for YoY/CAGR priority matching
_REVENUE_CANONICALS = {"revenue", "op_income", "gross_profit", "net_income",
                        "ebitda", "fcf", "capex", "eps", "rd_expense",
                        "cost_of_revenue", "sga", "total_assets", "equity",
                        "lt_debt", "current_assets", "current_liab"}

# Explains why this lookup is narrowly scoped to a specific question phrasing
# rather than applied to all "revenue" YoY queries: flattened row labels like "Total"
# can collide with unrelated rows elsewhere in a document, causing incorrect matches.
_HIGH_GROWTH_TRIGGERS = ["high growth", "high-growth", "growth company"]

# Map query keywords → canonical labels (for YoY target item inference)
_QUERY_CANONICAL_HINTS: List[Tuple[List[str], str]] = [
    (["net sales", "revenue", "net revenue", "total revenue", "sales",
      "營業收入", "營收"],                                    "revenue"),
    # Notes that "X margin" must be treated alongside "X profit"/"X income"
    # because margin phrasing is common and otherwise yields no canonical match,
    # which can cause fallback logic to pick an unrelated item with multi-year data.
    (["gross profit", "gross margin", "毛利"],              "gross_profit"),
    (["operating income", "operating profit", "operating margin", "營業利益"],   "op_income"),
    (["net income", "net earnings", "net profit", "net margin", "淨利"],  "net_income"),
    (["ebitda"],                                            "ebitda"),
    (["eps", "earnings per share"],                         "eps"),
    (["capex", "capital expenditure"],                      "capex"),
    (["r&d", "research and development", "研發"],           "rd_expense"),
    (["free cash flow", "fcf"],                             "fcf"),
    (["restructuring charges", "restructuring costs", "restructuring and impairment",
      "restructuring"],                                     "restructuring_costs"),
    # Documents a missing routing hint: direct questions about operating cash flow
    # need to map to the operating_cf canonical, otherwise they can fall back to generic
    # output
    # and miss the correct operating cash flow row already present in extracted
    # evidence.
    (["cash flow from operating activities", "cash from operations",
      "operating cash flow", "cash provided by operating"],  "operating_cf"),
    (["total assets", "資產"],                              "total_assets"),
    # Explains that some canonicals used by formulas were registered for classification
    # but lacked query-hint entries, so direct-lookup questions for those items returned
    # None
    # and fell back to unreliable generic processing instead of using the explicit total
    # row.
    (["total current liabilities", "current liabilities", "流動負債"],
                                                             "current_liab"),
    (["total current assets", "current assets", "流動資產"],
                                                             "current_assets"),
    (["total liabilities", "總負債"],                        "total_liab"),
    # Clarifies ordering of hint checks: specific "dividends" hints must be evaluated
    # before
    # generic "shareholders"/"stockholders" hints because the inference routine returns
    # on first match,
    # preventing misclassification of dividend questions as equity questions.
    (["dividends paid", "cash dividends", "dividends"],       "dividends_paid"),
    (["equity", "shareholders", "stockholders"],            "equity"),
    (["property, plant and equipment", "property, plant, and equipment",
      "property and equipment", "pp&e", "net ppe", "ppne", "fixed assets",
      "不動產、廠房及設備", "固定資產"],                     "ppe"),
    (["cost of revenue", "cost of goods", "cost of sales", "cogs"], "cost_of_revenue"),
    # Records a missing query hint for inventory: classifier produced a target metric of
    # 'inventory'
    # but no corresponding hint existed, so direct-lookup logic never ran and the system
    # fell back
    # to generic evidence selection instead of the explicit inventory row.
    (["inventory", "inventories", "存貨"],                    "inventory"),
    # dividends_paid hint moved above the "equity" entry so it is checked before
    # "shareholders".
    # Missing entirely — same gap class as ppe/inventory above; fallback lookup must
    # handle absent hints to avoid false "not explicitly provided" answers.
    (["accounts receivable", "trade receivables", "receivables", "net ar",
      "應收帳款"],                                             "accounts_rec"),
    # Same gap class again: "accounts_payable" is already a registered
    # _METRIC_KEYWORDS canonical in question_classifier.py, but had no
    # matching hint here at all, so _infer_target_canonical() always
    # returned None for a plain "what is X's accounts payable" question
    # -- the direct-lookup fallback never fired even when the real
    # "Accounts payable" row was sitting right there in extracted_table.
    (["accounts payable", "trade payables", "應付帳款"],       "accounts_payable"),
    # Last-resort hint placed at the end so more specific hints (e.g. "net income") win
    # first.
    # A bare "growth" or "high growth" with no other metric should default to revenue
    # growth.
    (_HIGH_GROWTH_TRIGGERS, "revenue"),
]


def _infer_target_canonical(query_lower: str) -> Optional[str]:
    """Return the most likely canonical label for the item the user is asking about."""
    for triggers, canonical in _QUERY_CANONICAL_HINTS:
        if _kw_match(triggers, query_lower):
            return canonical
    return None


#: Canonicals that are always a signed cash-flow-statement OUTFLOW/expense
#: line in the source filing (parenthesised or minus-prefixed) but whose
#: real-world concept is a positive magnitude ("how much did X pay/spend"),
#: same convention already applied inline in dividend_payout_ratio/
#: retention_ratio/free_cash_flow's formula_expr (abs(dividends_paid),
#: abs(capex)) — extended here to the Direct-lookup path, which builds its
#: own "result = {code_key}" line independently of any formula_expr.
_MAGNITUDE_ONLY_CANONICALS = {"dividends_paid", "dividends_per_share", "capex", "income_tax"}

#: (regex-friendly trigger, multiplier applied to a value that is natively
#: reported in MILLIONS — the near-universal SEC 10-K convention). Detects
#: an EXPLICIT unit instruction in the question itself (e.g. "(in USD
#: billions)") and rescales the extracted figure to match, rather than
#: silently reporting the filing's native millions scale regardless of
#: what was actually asked. Ordered so "billion" is checked before
#: "million" (a query could mention both, e.g. quoting a billion-scale
#: revenue figure while asking about a millions-scale line item — checking
#: the LAST/most specific unit word the query uses would be more precise,
#: but this dataset's questions only ever name one target unit).
_UNIT_SCALE_FROM_MILLIONS: List[Tuple[str, float]] = [
    ("billion", 0.001),
    ("thousand", 1000.0),
    ("million", 1.0),
]


def _detect_unit_scale_multiplier(query_lower: str) -> float:
    """Return a multiplier to rescale a value reported in a native unit (commonly reported
    in millions) into the unit explicitly requested by the question. Returns 1.0 when
    the question names no unit.

    Use cautiously: ensure the evidence's native unit is known before applying scaling
    to avoid incorrect conversions.
    """
    for trigger, multiplier in _UNIT_SCALE_FROM_MILLIONS:
        if trigger in query_lower:
            return multiplier
    return 1.0


def _find_same_item_pair(
    vars_list: List[Dict],
    query_lower: str = "",
    query_years: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Find two variables that share the same canonical item but are from different years.

    If the query specifies exact years, only consider that pair; skip any canonical that
    doesn't cover both requested years. If the query specifies a metric by name that has
    no extractions, return (None, None) instead of falling back to unrelated canonicals.
    """
    # Group by canonical
    groups: Dict[str, List[Dict]] = {}
    for v in vars_list:
        groups.setdefault(v.get("canonical", _sanitize(v["item"])), []).append(v)

    # Determine priority order: put query-relevant canonical first
    target = _infer_target_canonical(query_lower) if query_lower else None
    canonical_order = list(groups.keys())
    if target:
        if target in canonical_order:
            canonical_order.remove(target)
            canonical_order.insert(0, target)
        else:
            return None, None

    specific_years = sorted(set(query_years)) if query_years and len(set(query_years)) >= 2 else None

    for c in canonical_order:
        items = groups[c]
        sorted_items = sorted(items, key=lambda x: str(x["year"]))
        unique_years = {x["year"] for x in sorted_items}

        if specific_years:
            if not set(specific_years).issubset(unique_years):
                continue  # this canonical doesn't cover the years actually asked about
            old_yr, new_yr = specific_years[0], specific_years[-1]
        else:
            if len(unique_years) < 2:
                continue
            # If a query names fewer than 2 years, interpret it as the change into that
            # year from the immediately preceding year (the two most recent consecutive
            # years), not the oldest vs newest in the retrieved table.
            # This prevents using an unintended multi-year span when the statement
            # prints extra comparative years.
            sorted_years = sorted(unique_years)
            old_yr, new_yr = sorted_years[-2], sorted_years[-1]

        old = next(x for x in sorted_items if x["year"] == old_yr)
        new = next(x for x in sorted_items if x["year"] == new_yr)
        return old, new

    return None, None


_DIRECT_GROWTH_LABEL_RE = re.compile(r'^(total|worldwide|consolidated)$', re.IGNORECASE)


def _find_direct_pct_change_row(
    extracted_table: Dict[str, Dict], old_yr: str, new_yr: str,
) -> Optional[float]:
    """Prefer a filer-stated percent-change row for revenue YoY when present, because the
    filing's own percentage may use higher internal precision than recomputing from
    rounded dollar figures.

    Only applied for revenue totals and only trusts rows labeled with a small set of
    total/aggregate terms. Require both years' cells to look like percentages to avoid
    confusing unrelated total-like rows. Return None to fall back to recomputing when no
    suitable row is found.
    """
    # Group by the row's own FAMILY (its code_key with the year stripped
    # out), not just its generic label -- extracted_table routinely holds
    # SEVERAL distinct rows that all happen to be labeled exactly "Total"
    # for the exact same year (a segment subtotal, an asset total, AND
    # the real %-change row can all coexist once retrieval is widened
    # enough to catch this row reliably). Grouping by label alone lets a
    # later same-label/same-year row silently overwrite an earlier one in
    # a plain dict, discarding whichever was assigned first regardless of
    # which one is actually correct. code_key already disambiguates same-
    # label/same-year duplicates with a numeric suffix (_extract_from_
    # linearized_table's own dedup logic) — a chunk's OWN "2022"/"2021"
    # columns are extracted back-to-back in the same loop pass, so they
    # reliably land on the SAME suffix index, letting the two years of
    # the SAME underlying row be paired correctly by that family alone.
    by_family: Dict[str, Dict[str, float]] = {}
    for code_key, v in extracted_table.items():
        item = (v.get("item") or "").strip()
        if not _DIRECT_GROWTH_LABEL_RE.match(item):
            continue
        val = v.get("val")
        yr = v.get("year")
        if val is None or yr is None:
            continue
        family = re.sub(r'^val_(?:19|20)\d{2}_', '', code_key)
        by_family.setdefault(family, {})[yr] = val
    for yr_vals in by_family.values():
        new_val = yr_vals.get(new_yr)
        if new_val is None or abs(new_val) >= 100:
            continue
        old_val = yr_vals.get(old_yr)
        if old_val is not None and abs(old_val) >= 100:
            continue
        return new_val
    return None


_CANONICAL_TO_ALIASES: Dict[str, List[str]] = {c: aliases for c, aliases in _ITEM_TAXONOMY}


#: When a bare label ties against a sibling that explicitly says "net", prefer the net-
#: labeled row since statements often include both gross and net rows and a bare
#: financial question usually means the net figure.
#: This tie-break counters shorter-label bias that would otherwise pick the gross row.
_NET_QUALIFIER_RE = re.compile(r'\bnet\b')


def _pick_best_in_group(
    items: List[Dict],
    canonical: str,
    preferred_year: Optional[str] = None,
) -> Optional[Dict]:
    """When multiple rows normalize to the same canonical, prefer the actual total/subtotal
    row rather than a sub-component.

    Use the same total-row-priority scoring as the extraction routine; break ties by
    preferred year match, then by net-qualifier presence, then shorter label, then first
    occurrence.
    """
    if not items:
        return None
    aliases = _CANONICAL_TO_ALIASES.get(canonical, [canonical])

    def sort_key(x: Dict) -> Tuple[int, int, int, int, int, float]:
        score = _score_row_match(x["item"], aliases)
        year_match = 1 if preferred_year and x["year"] == preferred_year else 0
        is_net = 1 if _NET_QUALIFIER_RE.search(x["item"].lower()) else 0
        # Final tie-break: prefer the larger absolute magnitude only when all earlier
        # tiers tie.
        # This avoids selecting small per-share or reconciliation-table decimals that
        # reuse the same caption as full-statement rows.
        magnitude = abs(x["val"])
        # If a canonical's primary alias includes "total ...", prefer the row that
        # carries the total wording over a shorter plain sibling.
        # This prevents choosing a shorter but non-total line when the canonical refers
        # to a total metric.
        is_total = 1 if aliases and aliases[0].startswith("total") and aliases[0] in x["item"].lower() else 0
        return (score, year_match, is_total, is_net, -len(x["item"]), magnitude)

    return max(items, key=sort_key)


def _find_pair_for_margin(
    groups: Dict[str, List[Dict]],
    num_canonical: str,
    den_canonical: str,
    preferred_year: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find (numerator_var, denominator_var) for a margin calculation.
    Prefers matching the SAME year, and among same-year candidates prefers
    genuine total/subtotal rows over sub-items (same scoring as
    _pick_best_in_group). Falls back to any available year.
    """
    nums = groups.get(num_canonical, [])
    dens = groups.get(den_canonical, [])
    if not nums or not dens:
        return None, None

    num_aliases = _CANONICAL_TO_ALIASES.get(num_canonical, [num_canonical])
    den_aliases = _CANONICAL_TO_ALIASES.get(den_canonical, [den_canonical])

    same_year_pairs = [(n, d) for n in nums for d in dens if n["year"] == d["year"]]
    if same_year_pairs:
        def pair_key(pair: Tuple[Dict, Dict]) -> Tuple[int, int, int]:
            n, d = pair
            year_match = 1 if preferred_year and n["year"] == preferred_year else 0
            score = _score_row_match(n["item"], num_aliases) + _score_row_match(d["item"], den_aliases)
            # Tie-break on label length (shorter = better) when score ties:
            # _score_row_match's substring tier can't distinguish a genuine close
            # label from one where the alias is a tiny fragment of a much longer
            # unrelated label. Prefer shorter labels as they more likely match the
            # intended statement line; avoid selecting long, qualifying footnote rows.
            neg_len = -(len(n["item"]) + len(d["item"]))
            return (year_match, score, neg_len)
        return max(same_year_pairs, key=pair_key)

    # Fallback: no shared year — best candidate for each side independently
    n = _pick_best_in_group(nums, num_canonical, preferred_year)
    d = _pick_best_in_group(dens, den_canonical, preferred_year)
    return n, d


def _pair_for_year(
    groups: Dict[str, List[Dict]],
    num_canonical: str,
    den_canonical: str,
    year: str,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Like the general pair-finding routine but restricted to a single year.

    Used when comparing a ratio or margin across years so each year's numerator and
    denominator are taken only from that year's data.
    """
    nums = [x for x in groups.get(num_canonical, []) if x["year"] == year]
    dens = [x for x in groups.get(den_canonical, []) if x["year"] == year]
    if not nums or not dens:
        return None, None
    return _find_pair_for_margin({num_canonical: nums, den_canonical: dens}, num_canonical, den_canonical, year)


# Calculation builder aligned to a financial QA benchmark
# Provides helpers to construct numeric queries and compute derived metrics for model
# evaluation

# Maps query intent → (numerator_canonical, denominator_canonical, label)
_MARGIN_MAP: List[Tuple[List[str], str, str, str]] = [
    # triggers            numerator_canonical  denominator_canonical  label
    (["毛利率", "gross margin"],    "gross_profit",  "revenue",    "Gross Margin"),
    (["op_margin", "operating margin", "營業利益率"],
                                    "op_income",     "revenue",    "Operating Margin"),
    (["net margin", "淨利率", "net profit margin"],
                                    "net_income",    "revenue",    "Net Profit Margin"),
    (["r&d", "rd", "研發費用佔"],   "rd_expense",    "revenue",    "R&D % of Revenue"),
    (["sg&a", "sga", "推銷"],       "sga",           "revenue",    "SG&A % of Revenue"),
    (["d&a", "depreciation and amortization", "depreciation & amortization",
      "折舊攤銷佔"],                "depreciation",  "revenue",    "D&A % of Revenue"),
    # Recognize phrasing like "X as a % of revenue" / "X as a percentage of revenue":
    # This recurring wording indicates a ratio and must trigger the ratio
    # builder. If not recognized, the fallback returns revenue itself,
    # leading to numeric-type mismatches where a percentage is expected.
    (["cost ratio", "cogs ratio", "cogs margin", "cogs %", "cost of goods sold margin",
      "cost of goods sold %", "cost of goods sold as a % of revenue",
      "cost of goods sold as a percentage of revenue"],
                                    "cost_of_revenue","revenue",   "Cost of Revenue Ratio"),
    (["capex%", "capex ratio"],     "capex",          "revenue",   "CapEx % of Revenue"),
]

def _synthesize_gross_profit(
    code_lines: List[str],
    groups: Dict[str, List[Dict]],
    degraded_notes: Optional[List[str]],
) -> Dict[str, List[Dict]]:
    """Compute gross profit when no explicit subtotal exists.
    If no gross_profit row is present but revenue and one or more cost_of_revenue rows
    exist for the same year, set gross_profit = revenue - sum(cost_of_revenue rows) for
    that year.
    This uses the sum of all cost_of_revenue-tagged rows (to handle split sub-lines) and
    never overrides an existing explicit gross_profit row.
    """
    if groups.get("gross_profit"):
        return groups
    cost_rows = groups.get("cost_of_revenue", [])
    revenue_rows = groups.get("revenue", [])
    if not cost_rows or not revenue_rows:
        return groups
    cost_by_year: Dict[str, List[Dict]] = {}
    for r in cost_rows:
        cost_by_year.setdefault(r["year"], []).append(r)
    revenue_by_year: Dict[str, List[Dict]] = {}
    for r in revenue_rows:
        revenue_by_year.setdefault(r["year"], []).append(r)
    derived: List[Dict] = []
    for yr, rows in cost_by_year.items():
        if yr not in revenue_by_year:
            continue
        rev = _pick_best_in_group(revenue_by_year[yr], "revenue")
        if rev is None:
            continue
        # Do not add RATIO rows as if they were raw-dollar rows, and do not
        # duplicate a subtotal by adding its component plus the subtotal row.
        # Ratio rows represent derived percentages and must remain distinct from
        # monetary line items.
        rows = [r for r in rows if not re.search(r"%|percent|ratio", r.get("item", ""), re.IGNORECASE)]
        if len(rows) >= 3:
            _vals = [abs(r["val"]) for r in rows]
            for _k, _r in enumerate(rows):
                _others = sum(_vals) - _vals[_k]
                if _others > 0 and abs(_vals[_k] - _others) / _others <= 0.02:
                    rows = [_r]  # this row IS the total of the others
                    break
        if not rows:
            continue
        sum_terms = " + ".join(f"abs({r['code_key']})" for r in rows)
        var = f"_derived_gross_profit_{yr}"
        code_lines.append(
            f"{var} = {rev['code_key']} - ({sum_terms})  "
            f"# derived: no explicit Gross Profit line found"
        )
        derived.append({
            "item": "Gross Profit (derived: Revenue - Cost of Revenue)",
            "canonical": "gross_profit",
            "year": yr,
            "val": rev["val"] - sum(abs(r["val"]) for r in rows),
            "code_key": var,
        })
    if derived:
        groups = dict(groups)
        groups["gross_profit"] = derived
        if degraded_notes is not None:
            degraded_notes.append(
                "Gross Profit was not printed in the filing as its own line item -- it was "
                "derived as Revenue minus the filing's own Cost of Revenue sub-line(s) (e.g. "
                "\"Cost of products\" + \"Cost of services\"), which may not exactly match the "
                "filing's true gross profit if other cost components are folded in elsewhere."
            )
    return groups


_ROE_TRIGGERS = ["roe", "return on equity"]
_ROA_TRIGGERS = ["roa", "return on assets"]
_CURRENT_RATIO_TRIGGERS = [
    "current ratio", "working capital ratio", "working capital",
    "流動比率", "營運資金比率",
]
_QUICK_RATIO_TRIGGERS = ["quick ratio", "acid", "速動"]


def _emit_multi_year_ratio(
    code_lines: List[str],
    label: str,
    unit: str,
    year_exprs: List[Tuple[str, str]],
) -> None:
    """Compute and emit a ratio for each of 2+ years plus the numeric delta between the
    first and last year.
    This provides explicit year-over-year values and a direction label
    ("increased"/"decreased"); interpretation of whether a change is good is left to
    later stages.
    year_exprs: [(year, python_expr_string), ...] sorted oldest to newest.
    """
    result_vars: List[Tuple[str, str]] = []
    for yr, expr in year_exprs:
        var = f"{_sanitize(label)}_{yr}"
        code_lines.append(f"{var} = round({expr}, 4)")
        code_lines.append(f"print(f'{label} ({yr}): {{{var}}}{unit}')")
        result_vars.append((yr, var))
    first_yr, first_var = result_vars[0]
    last_yr, last_var = result_vars[-1]
    code_lines.append(f"result = {last_var}")
    code_lines.append(f"_delta = round({last_var} - {first_var}, 4)")
    code_lines.append(
        "_direction = 'increased' if _delta > 0 else ('decreased' if _delta < 0 else 'stayed flat')"
    )
    code_lines.append(
        f"print(f'{label} change ({first_yr}->{last_yr}): {{_direction}} by {{abs(_delta)}}{unit}')"
    )
    # When comparing 3+ consecutive years, also return each adjacent-year delta
    # not just the overall first-to-last change; some reference answers cite
    # an adjacent-pair change rather than the head-to-tail span.
    if len(result_vars) > 2:
        for (prev_yr, prev_var), (yr, var) in zip(result_vars, result_vars[1:]):
            # Keep trailing "_pair" suffix on delta variables: frontend code scans
            # for names ending in a bare "_YYYY" to build result_series. Without
            # _the suffix, a delta can be misidentified as a per-year datapoint and
            # corrupt trend displays; use the suffix to preserve correct series order.
            tag = f"{_sanitize(label)}_{prev_yr}_{yr}_pair"
            code_lines.append(f"_delta_{tag} = round({var} - {prev_var}, 4)")
            code_lines.append(
                f"_dir_{tag} = 'increased' if _delta_{tag} > 0 else "
                f"('decreased' if _delta_{tag} < 0 else 'stayed flat')"
            )
            code_lines.append(
                f"print(f'{label} change ({prev_yr}->{yr}): "
                f"{{_dir_{tag}}} by {{abs(_delta_{tag})}}{unit}')"
            )


#: Toggle for the no-subject YoY guard in _build_calculation_code (kept as a
#: module flag so the change can be A/B-compared across the whole question set).
_YOY_REQUIRES_TARGET = True


def _build_calculation_code(
    code_lines: List[str],
    extracted_table: Dict[str, Dict],
    query: str,
    q_lower: str,
    preferred_year: Optional[str] = None,
    query_years: Optional[List[str]] = None,
    degraded_notes: Optional[List[str]] = None,
    detected_unit: Optional[List[str]] = None,
    evidence_list: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Build the calculation section of the PoT code.
    `degraded_notes`, if given, is appended to whenever this function
    silently substitutes a DIFFERENT formula than the one actually asked
    about (e.g. Current Ratio in place of Quick Ratio, because no
    inventory data — direct or composite — could be found) — never
    overwritten by the caller, only appended to, since generate_and_execute
    passes the same list across the extracted_table AND free-text attempts.
    Returns True if a calculation was successfully generated.
    """
    groups = _group_by_canonical(extracted_table)
    query_years = _with_implied_trend_year(query_years, q_lower)

    # ── CAGR ──────────────────────────────────────────────────────────────────
    if _kw_match(["cagr"], q_lower) or "複合成長率" in q_lower:
        # Bug 3 fix: prefer the canonical the user asked about
        v1, v2 = _find_same_item_pair(list(extracted_table.values()), q_lower, query_years)
        if v1 and v2:
            try:
                yrs = abs(float(v2["year"]) - float(v1["year"])) or 1.0
            except Exception:
                yrs = 1.0
            code_lines.append(f"# CAGR: {v1['item']}")
            code_lines.append(f"result_cagr = cagr({v1['code_key']}, {v2['code_key']}, {yrs})")
            # Round to 4 decimal places here to avoid a double-rounding error when a
            # later step applies the question's requested precision; doing a prior
            # 2-decimal round can flip the final digit if the true value is near a
            # rounding boundary.
            # Do not treat this as proof output — just a precaution to keep downstream
            # rounding correct.
            code_lines.append("result = round(result_cagr, 4)")
            label = v1['item']
            y1, y2 = v1['year'], v2['year']
            code_lines.append(f"print(f'{label} CAGR ({y1}->{y2}): {{result}}%')")
            return True

    # ── YoY Growth ────────────────────────────────────────────────────────────
    yoy_triggers = ["yoy", "year over year", "成長率", "年增", "growth rate", "growth",
                    "변동", "change", "增加多少", "減少多少",
                    # Questions using plain increase/decrease verbs intend a multi-year
                    # change (e.g., two-year change), not a single-year lookup; this
                    # matches the existing verb-class gating used elsewhere.
                    "grow", "drop", "decline", "decrease", "increase", "rise", "fell"]
    # If a question names no recognizable financial metric, do not guess a related
    # metric from available data; the logic should return no-match rather than
    # defaulting to an unrelated canonical metric that happens to have two years of
    # data.
    _yoy_has_subject = (not _YOY_REQUIRES_TARGET) or _infer_target_canonical(q_lower) is not None
    if _kw_match(yoy_triggers, q_lower) and _yoy_has_subject:
        # Bug 3 fix: pass q_lower so _find_same_item_pair prefers the item the user asked about
        v1, v2 = _find_same_item_pair(list(extracted_table.values()), q_lower, query_years)
        if v1 and v2:
            direct_pct = (
                _find_direct_pct_change_row(extracted_table, v1["year"], v2["year"])
                if v1.get("canonical") == "revenue" and _kw_match(_HIGH_GROWTH_TRIGGERS, q_lower)
                else None
            )
            label = v1['item']
            y1, y2 = v1['year'], v2['year']
            if direct_pct is not None:
                code_lines.append(f"# YoY: {label} (filing's own reported % change, preferred over recomputed)")
                code_lines.append(f"result = {direct_pct}")
                code_lines.append(f"print(f'{label} YoY Growth ({y1}->{y2}) [filing-reported]: {{result}}%')")
                return True
            code_lines.append(f"# YoY: {v1['item']}")
            code_lines.append(f"result_yoy = yoy({v1['code_key']}, {v2['code_key']})")
            # 4dp, not 2dp -- same double-rounding rationale as the CAGR
            # branch just above.
            code_lines.append("result = round(result_yoy, 4)")
            code_lines.append(f"print(f'{label} YoY Growth ({y1}->{y2}): {{result}}%')")
            return True

    # Comparison of operating vs investing vs financing cash flows is a 3-way comparison
    # over statement totals and needs a dedicated branch rather than a single arithmetic
    # formula expression; trigger on structural cues mentioning at least two of the
    # three activities plus cash-flow comparison wording.
    _cf_activity_mentions = sum(
        1 for kw in ("operat", "invest", "financ") if kw in q_lower
    )
    if (_cf_activity_mentions >= 2 and "cash flow" in q_lower
            and _kw_match(["most", "least", "which", "brought in", "generated"], q_lower)):
        cf_groups = [
            ("operating activities", groups.get("operating_cf") or []),
            ("investing activities", groups.get("investing_cf") or []),
            ("financing activities", groups.get("financing_cf") or []),
        ]
        candidates: List[Tuple[str, Dict]] = []
        for label, items in cf_groups:
            if not items:
                continue
            sorted_items = sorted(items, key=lambda x: str(x["year"]))
            target_item = None
            if query_years:
                for it in reversed(sorted_items):
                    if it["year"] in query_years:
                        target_item = it
                        break
            if target_item is None:
                target_item = sorted_items[-1]
            candidates.append((label, target_item))
        if len(candidates) >= 2:
            # argmax(val) covers BOTH "brought in the most" (largest
            # positive) and "lost the least" (least-negative, i.e. closest
            # to zero among negatives) in one comparison -- no separate
            # branch needed for the two phrasings.
            best_label, best_item = max(candidates, key=lambda c: c[1]["val"])
            code_lines.append("# Cash flow activity comparison (operating vs investing vs financing)")
            for label, item in candidates:
                yr = item["year"]
                code_lines.append(f"print(f'{label} ({yr}): {{{item['code_key']}}}')")
            code_lines.append(f"result = {best_item['code_key']}")
            best_yr = best_item["year"]
            code_lines.append(
                f"print(f'{best_label} brought in the most cash flow "
                f"({best_yr}): {{result}}')"
            )
            return True

    # ── ROE ───────────────────────────────────────────────────────────────────
    # Bare decimal ratio (e.g. "-0.02"), not a "-2.00%" percentage — see
    # the ROA comment just below for why (same fix, same reasoning).
    if _kw_match(_ROE_TRIGGERS, q_lower):
        n, d = _find_pair_for_margin(groups, "net_income", "equity", preferred_year)
        if n and d:
            code_lines.append(f"# ROE = Net Income / Equity")
            code_lines.append(f"result = round({n['code_key']} / {d['code_key']}, 2)")
            yr = n['year']
            code_lines.append(f"print(f'Return on Equity (ROE) ({yr}): {{result}}')")
            return True

    # Return ROA as a bare decimal ratio (no ×100) to keep consistent with other ratio
    # outputs in this codebase; avoid applying percentage scaling that would produce a
    # 100x mismatch versus ratio conventions.
    if _kw_match(_ROA_TRIGGERS, q_lower):
        n, d = _find_pair_for_margin(groups, "net_income", "total_assets", preferred_year)
        if n and d:
            code_lines.append(f"# ROA = Net Income / Total Assets")
            code_lines.append(f"result = round({n['code_key']} / {d['code_key']}, 2)")
            yr = n['year']
            code_lines.append(f"print(f'Return on Assets (ROA) ({yr}): {{result}}')")
            return True

    # ── Current Ratio ─────────────────────────────────────────────────────────
    if _kw_match(_CURRENT_RATIO_TRIGGERS, q_lower):
        ca_list = groups.get("current_assets", [])
        cl_list = groups.get("current_liab", [])
        if ca_list and cl_list:
            distinct_years = sorted(set(query_years or []))
            if len(distinct_years) >= 2:
                year_exprs = []
                for yr in distinct_years:
                    ca_yr = [x for x in ca_list if x["year"] == yr]
                    cl_yr = [x for x in cl_list if x["year"] == yr]
                    if not ca_yr or not cl_yr:
                        continue
                    ca = _pick_best_in_group(ca_yr, "current_assets")
                    cl = _pick_best_in_group(cl_yr, "current_liab")
                    year_exprs.append((yr, f"{ca['code_key']} / {cl['code_key']}"))
                if len(year_exprs) >= 2:
                    code_lines.append("# Current Ratio = Current Assets / Current Liabilities")
                    _emit_multi_year_ratio(code_lines, "Current Ratio", "", year_exprs)
                    return True
            ca = _pick_best_in_group(ca_list, "current_assets", preferred_year)
            cl = _pick_best_in_group(cl_list, "current_liab", ca["year"])
            code_lines.append(f"# Current Ratio = Current Assets / Current Liabilities")
            code_lines.append(f"result = round({ca['code_key']} / {cl['code_key']}, 4)")
            yr = ca['year']
            code_lines.append(f"print(f'Current Ratio ({yr}): {{result}}')")
            return True

    # ── Quick Ratio ───────────────────────────────────────────────────────────
    if _kw_match(_QUICK_RATIO_TRIGGERS, q_lower):
        ca_list = groups.get("current_assets", [])
        inv_list = groups.get("inventory", [])
        cl_list = groups.get("current_liab", [])
        if ca_list and cl_list:
            distinct_years = sorted(set(query_years or []))
            if len(distinct_years) >= 2:
                year_exprs = []
                any_degraded = False
                for yr in distinct_years:
                    ca_yr = [x for x in ca_list if x["year"] == yr]
                    cl_yr = [x for x in cl_list if x["year"] == yr]
                    if not ca_yr or not cl_yr:
                        continue
                    ca = _pick_best_in_group(ca_yr, "current_assets")
                    cl = _pick_best_in_group(cl_yr, "current_liab")
                    inv_yr = [x for x in inv_list if x["year"] == yr]
                    if inv_yr:
                        inv = _pick_best_in_group(inv_yr, "inventory")
                        year_exprs.append((yr, f"({ca['code_key']} - {inv['code_key']}) / {cl['code_key']}"))
                    else:
                        year_exprs.append((yr, f"{ca['code_key']} / {cl['code_key']}"))
                        any_degraded = True
                if len(year_exprs) >= 2:
                    label = "Quick Ratio (approx, no inventory data)" if any_degraded else "Quick Ratio"
                    code_lines.append("# Quick Ratio = (Current Assets - Inventory) / Current Liabilities")
                    _emit_multi_year_ratio(code_lines, label, "", year_exprs)
                    if any_degraded and degraded_notes is not None:
                        degraded_notes.append(
                            "Quick Ratio could not be computed for at least one year (no "
                            "inventory line item found, even after trying known sub-item "
                            "breakdowns like raw materials + work in process) -- the value(s) "
                            "shown use Current Ratio (Current Assets / Current Liabilities) as "
                            "an approximation, which is NOT the true Quick Ratio and will read "
                            "higher than the real figure."
                        )
                    return True
            ca = _pick_best_in_group(ca_list, "current_assets", preferred_year)
            cl = _pick_best_in_group(cl_list, "current_liab", ca["year"])
            if inv_list:
                inv = _pick_best_in_group(inv_list, "inventory", ca["year"])
                code_lines.append(f"# Quick Ratio = (Current Assets - Inventory) / Current Liabilities")
                code_lines.append(f"result = round(({ca['code_key']} - {inv['code_key']}) / {cl['code_key']}, 4)")
            else:
                code_lines.append(f"# Quick Ratio (approx, no inventory data) = CA / CL")
                code_lines.append(f"result = round({ca['code_key']} / {cl['code_key']}, 4)")
                if degraded_notes is not None:
                    degraded_notes.append(
                        "Quick Ratio could not be computed (no inventory line item found, "
                        "even after trying known sub-item breakdowns like raw materials + "
                        "work in process) -- the value shown uses Current Ratio (Current "
                        "Assets / Current Liabilities) as an approximation, which is NOT the "
                        "true Quick Ratio and will read higher than the real figure."
                    )
            yr = ca['year']
            code_lines.append(f"print(f'Quick Ratio ({yr}): {{result}}')")
            return True

    # ── Margin / Ratio ────────────────────────────────────────────────────────
    if _kw_match(["毛利率", "gross margin"], q_lower):
        groups = _synthesize_gross_profit(code_lines, groups, degraded_notes)
    # When the question asks for an X-year average of a margin, emit a single blended
    # number (mean of each year's ratio) rather than a single-year value; only use
    # multi-year-last-year semantics for explicit trend/delta queries.
    wants_average = any(kw in q_lower for kw in ("average", "avg", "平均"))
    # Check multi-year consistency by using every comparative year the filing tabulates
    # (typically a multi-year income statement), not a single prior-year snapshot.
    # If the query implies a pattern but query_years yields fewer than two distinct
    # years, do not return a single-year ratio; require the full multi-year series to
    # evaluate consistency.
    wants_consistency_check = any(
        kw in q_lower for kw in ("historically consistent", "not fluctuating", "each year", "every year")
    )
    for triggers, num_c, den_c, label in _MARGIN_MAP:
        if _kw_match(triggers, q_lower):
            distinct_years = sorted(set(query_years or []))
            if len(distinct_years) < 2 and wants_consistency_check:
                num_years = {v["year"] for v in groups.get(num_c, [])}
                den_years = {v["year"] for v in groups.get(den_c, [])}
                distinct_years = sorted(num_years & den_years)[-3:]
            if len(distinct_years) >= 2:
                year_exprs = []
                for yr in distinct_years:
                    n_yr, d_yr = _pair_for_year(groups, num_c, den_c, yr)
                    if n_yr and d_yr:
                        year_exprs.append((yr, f"margin({n_yr['code_key']}, {d_yr['code_key']})"))
                if len(year_exprs) >= 2:
                    code_lines.append(f"# {label} = {num_c} / {den_c}")
                    if wants_average:
                        var_names = []
                        for yr, expr in year_exprs:
                            var = f"{_sanitize(label)}_{yr}"
                            code_lines.append(f"{var} = round({expr}, 4)")
                            code_lines.append(f"print(f'{label} ({yr}): {{{var}}}%')")
                            var_names.append(var)
                        code_lines.append(
                            f"result = round(({' + '.join(var_names)}) / {len(var_names)}, 2)"
                        )
                        code_lines.append(
                            f"print(f'{label} ({len(var_names)}-yr avg): {{result}}%')"
                        )
                    else:
                        _emit_multi_year_ratio(code_lines, label, "%", year_exprs)
                    return True
            n, d = _find_pair_for_margin(groups, num_c, den_c, preferred_year)
            if n and d:
                code_lines.append(f"# {label} = {num_c} / {den_c}")
                code_lines.append(f"result = round(margin({n['code_key']}, {d['code_key']}), 2)")
                yr = n['year']
                code_lines.append(f"print(f'{label} ({yr}): {{result}}%')")
                return True

    # ── Direct lookup ─────────────────────────────────────────────────────────
    # Pick the var that best matches the query intent. Uses the same
    # whole-phrase, word-boundary-safe matcher as every other canonical
    # inference in this module (_infer_target_canonical / _kw_match) —
    # NOT a bespoke per-word substring check. The previous version split
    # every canonical's full alias PHRASES into individual words and
    # tested each word as a bare substring, so gross_profit's own alias
    # "gross margin amount" contributed the standalone word "margin" —
    # meaning any query merely containing "margin" (e.g. "COGS % margin",
    # which should resolve to cost_of_revenue) silently matched
    # gross_profit purely because it came first in _ITEM_TAXONOMY, before
    # this fallback even got a chance to look at the metric actually
    # being asked about.
    target_canonical = _infer_target_canonical(q_lower)

    # Do not fall back to the first extracted table row when no matching metric is
    # found; returning a clearly incorrect unrelated line is worse than returning no
    # structured result.
    # If the named metric is absent in structured data, return False so callers can try
    # free-text extraction or emit an explicit "could not find structured financial
    # data" outcome.
    # For Yes/No dividend presence questions, prefer a per-share rate when available;
    # only substitute aggregate cash outflow if the per-share row is genuinely absent.
    if target_canonical == "dividends_paid" and _YESNO_DIVIDEND_QUERY_RE.search(q_lower):
        narrative_year = preferred_year or (query_years[-1] if query_years else None)
        # When a question names a specific quarter, do not use the filing's structured
        # annual "Dividends declared per share" row because it is annual by default.
        # For quarter-specific dividend questions, scan narrative text for explicit
        # quarterly wording and extract the per-share quarterly rate from prose rather
        # than trusting the annual structured row.
        wants_quarter = bool(_QUARTER_QUERY_RE.search(q_lower))
        per_share_group = [] if wants_quarter else (groups.get("dividends_per_share") or [])
        # A stale OTHER YEAR's per-share rate (e.g. a same-company
        # earlier filing's own row, entity-matched loosely across
        # years) is actively misleading here, unlike the aggregate --
        # only trust this canonical when one of its candidates actually
        # covers the year being asked about.
        if per_share_group and (not narrative_year or any(v.get("year") == narrative_year for v in per_share_group)):
            target_canonical = "dividends_per_share"
        elif evidence_list is not None:
            # No clean structured per-share row for the RIGHT
            # year/granularity -- last resort, scan the SAME evidence
            # for a narrative "$X.XX per share" sentence naming the
            # target year. See _extract_narrative_dividend_per_share's
            # docstring.
            found = _extract_narrative_dividend_per_share(evidence_list, narrative_year, prefer_quarterly=wants_quarter)
            if found:
                val, source_detail = found
                code_lines.append(f"result = {val}  # dividends per share (narrative) <- {source_detail}")
                code_lines.append(f"print(f'Dividends per share ({narrative_year}): {{result}}')")
                if detected_unit is not None:
                    detected_unit.append("$")
                return True

    vars_to_use = groups.get(target_canonical, []) if target_canonical else []

    if vars_to_use:
        best = _pick_best_in_group(vars_to_use, target_canonical, preferred_year)
        item_label = best['item']
        yr = best['year']
        expr = best['code_key']
        if target_canonical in _MAGNITUDE_ONLY_CANONICALS:
            expr = f"abs({expr})"
        scale = _detect_unit_scale_multiplier(q_lower)
        if scale != 1.0:
            expr = f"({expr}) * {scale}"
        code_lines.append(f"result = {expr}")
        code_lines.append(f"print(f'{item_label} ({yr}): {{result}}')")
        if detected_unit is not None:
            detected_unit.append("$")
        return True

    return False


#: Inventory-turnover convention should be determined from the filing's inventory note,
#: not hardcoded per company.
#: If the note shows finished goods/merchandise as the main inventory, use COGS /
#: average(inventories) for turnover; if the note shows self-consumed commodities/spare
#: parts without finished goods, using COGS / ending inventory is acceptable.
#: Detect this from the filing's inventory note wording rather than a company allowlist.
_FINISHED_GOODS_INVENTORY_RE = re.compile(
    r'\bfinished\s+goods\b|\bmerchandise\s+inventor(?:y|ies)\b', re.IGNORECASE
)


def _has_finished_goods_inventory(evidence_list: List[Dict[str, Any]]) -> bool:
    for item in evidence_list or []:
        content = str(item.get("content") or item.get("parent_content") or "")
        if _FINISHED_GOODS_INVENTORY_RE.search(content):
            return True
    return False


def _apply_inventory_turnover_convention_override(
    formula_entry: Optional[Dict[str, Any]],
    entity: str,
    evidence_list: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    No-op for every company/question whose own inventory-note evidence
    doesn't mention a finished-goods/merchandise inventory category,
    including every OTHER formula (only swaps when detect_formula already
    picked the plain "inventory_turnover" key) -- so this can't touch any
    other formula's behavior, nor override a question that already
    explicitly asked for "average inventory" itself (that already routes
    straight to inventory_turnover_avg via detect_formula's own keyword
    match, before this override ever runs).
    """
    if not formula_entry or formula_entry.get("formula_key") != "inventory_turnover":
        return formula_entry
    if not _has_finished_goods_inventory(evidence_list or []):
        return formula_entry
    avg_entry = FORMULA_LIBRARY.get("inventory_turnover_avg")
    if not avg_entry:
        return formula_entry
    return {**avg_entry, "formula_key": "inventory_turnover_avg"}


# ─────────────────────────────────────────────────────────────────────────────
# Main Reasoner
# ─────────────────────────────────────────────────────────────────────────────

class ProgramOfThoughtReasoner:
    """A financial reasoning component that generates intermediate reasoning steps aligned
    to a benchmarking style.
    This identifies and structures multi-step calculations for fiscal metrics; avoid
    treating generated chains as authoritative evidence without verifying source data.
    """

    def generate_and_execute(
        self, query: str, evidence_list: List[Dict[str, Any]], entity: str = ""
    ) -> Dict[str, Any]:
        q_lower = query.lower()

        # For "which of operations, investing, financing brought in the most/least cash"
        # questions, treat the task as a categorical comparison, not a request for a
        # single numeric value.
        # Report the winning category as the verdict and avoid presenting one candidate
        # number alone as if it were the complete answer.
        is_cf_activity_comparison = (
            sum(1 for kw in ("operat", "invest", "financ") if kw in q_lower) >= 2
            and "cash flow" in q_lower
            and _kw_match(["most", "least", "which", "brought in", "generated"], q_lower)
        )

        # "Is X a high-growth company?" is a qualitative business
        # characterization, not a request for one specific number -- the
        # classifier still routes it through answer_mode=NUMERIC (a bare
        # "growth" keyword forces that, same as any other YoY-shaped
        # question), so it can't be caught by the frontend's EXPLANATION/
        # ASSESSMENT suppression alone without ALSO rerouting its
        # retrieval strategy (answer_mode also selects which sub-question
        # builder runs -- reclassifying it to ASSESSMENT would silently
        # swap the LLM-decomposition retrieval path this question actually
        # relies on for a different one). Flagged here instead, the same
        # narrow, display-only mechanism as is_cf_activity_comparison
        # above, so the frontend can suppress just the green result card
        # without touching classification or retrieval at all.
        is_qualitative_characterization = _kw_match(_HIGH_GROWTH_TRIGGERS, q_lower)

        # Questions asking whether any product/service category exceeds a threshold
        # require computing and checking every category share, not returning a single
        # total revenue figure.
        # If structured extraction cannot provide per-category shares reliably, skip
        # returning an unverified numeric result and let the textual synthesis enumerate
        # and verify categories instead.
        if _CATEGORY_THRESHOLD_QUERY_RE.search(q_lower) or _no_calculation_path(q_lower):
            return {
                "code": "", "success": True, "result_value": None,
                "output_log": "", "extracted_variables": {},
                "repairs_triggered": 0, "extraction_method": "none",
                "formula_used": None, "is_degraded_formula": False,
                "degraded_note": "", "result_series": [],
                "result_delta": None, "result_direction": None,
                "result_unit": "",
            }

        query_years = _extract_query_years(query)
        preferred_year = query_years[-1] if query_years else None  # latest year mentioned

        # ── Step 1: Detect formula intent ────────────────────────────────────
        formula_entry = detect_formula(query)
        formula_entry = _apply_inventory_turnover_convention_override(formula_entry, entity, evidence_list)

        # Determine whether a referenced percent-of-sales metric increased or decreased
        # when no explicit formula is available; fall back to a generic year-over-year
        # comparison if needed. Avoid attributing the change to unrelated line items;
        # decide direction from the provided evidence text only; do not produce a
        # separate result card when formula is missing.
        if formula_entry is None and _RATIO_DIRECTION_QUERY_RE.search(q_lower):
            return {
                "code": "", "success": True, "result_value": None,
                "output_log": "", "extracted_variables": {},
                "repairs_triggered": 0, "extraction_method": "none",
                "formula_used": None, "is_degraded_formula": False,
                "degraded_note": "", "result_series": [],
                "result_delta": None, "result_direction": None,
                "result_unit": "",
            }

        # ── Step 2: Extract variables from linearized tables ──────────────────
        extracted_table = _extract_from_linearized_table(evidence_list, entity)

        # ── Step 3: Formula-guided extraction ─────────────────────────────────
        resolved_formula: Dict[str, float] = {}
        resolved_formula_series: Dict[str, List[Tuple[float, str]]] = {}
        resolved_formula_meta: Dict[str, Dict[str, Any]] = {}
        duplicate_warnings: List[str] = []
        if formula_entry:
            resolved_formula, resolved_formula_series, resolved_formula_meta = _extract_formula_guided(
                evidence_list, formula_entry, query_years, entity, q_lower
            )
            duplicate_warnings = _detect_and_strip_duplicate_values(
                resolved_formula, resolved_formula_series, resolved_formula_meta
            )
            # When the query explicitly requests an "Adjusted EBIT" metric, require the
            # specific non-GAAP Adjusted EBIT value from a reconciliation table rather
            # than defaulting to the plain GAAP "Operating income" row resolved by the
            # generic "ebit" alias.
            # If no matching reconciliation table is found or the query does not ask for
            # "Adjusted EBIT", fall back to the plain "ebit" result so behavior never
            # degrades unrelated queries.
            if "ebit" in resolved_formula and preferred_year:
                _derived = _derive_adjusted_ebit_from_ebitda_reconciliation(
                    evidence_list, preferred_year, q_lower
                )
                if _derived is not None:
                    _adj_val, _adj_source = _derived
                    resolved_formula["ebit"] = _adj_val
                    resolved_formula_series["ebit"] = [(_adj_val, preferred_year)]
                    resolved_formula_meta["ebit"] = {
                        "source": "derived-from-ebitda-reconciliation",
                        "is_approximate": True,
                        "detail": _adj_source,
                    }

        # ── Step 4: Build code ────────────────────────────────────────────────
        code_lines = [
            "# FinAgent-RAG Program-of-Thought (PoT) Sandbox",
            "# Extracted from retrieved financial evidence",
        ]
        code_lines.extend(duplicate_warnings)

        used_extraction = "formula"
        degraded_notes: List[str] = []
        # This field is set by the Direct-lookup fallback path when no formula_entry
        # matched, so the frontend can show a currency prefix instead of a unit-less
        # number.
        # Only populate this for raw dollar-denominated line items; earlier branches for
        # ratios/percentages return before this path and prevent mislabeling.
        detected_unit: List[str] = []

        if formula_entry and resolved_formula:
            # PATH 1: Formula library
            fk = formula_entry.get("formula_key", "")
            code_lines.append(f"# Formula: {fk} — {formula_entry.get('result_label', '')}")
            is_regulated_utility = fk == "working_capital" and _is_regulated_utility_filing(evidence_list)
            formula_code = _gen_formula_code(
                formula_entry, resolved_formula, extracted_table,
                resolved_formula_series, resolved_formula_meta,
                query_years, q_lower, is_regulated_utility,
            )
            if formula_code:
                code_lines.extend(formula_code)
            else:
                formula_entry = None  # fall through
                used_extraction = "table"

        if not formula_entry or not resolved_formula or "result" not in "\n".join(code_lines):
            # PATH 2: Linearized table + smart calculation
            used_extraction = "table"
            code_lines = [
                "# FinAgent-RAG Program-of-Thought (PoT) Sandbox",
                "# Extracted from retrieved financial evidence",
            ]

            # A "how many/number of <noun>" question with a roll-forward "Total"
            # row (stores opened/closed) skips the generic extractor entirely --
            # see _derive_total_row_period_end_change's docstring for why the
            # generic path can't tell a period's START count from its END count.
            _count_change = (
                None if formula_entry else
                _derive_total_row_period_end_change(evidence_list, q_lower)
            )
            if _count_change is not None:
                used_extraction = "total-row-rollforward"
                new_v, old_v, new_k, old_k, noun = _count_change
                code_lines.append(f"# Total {noun} count: {new_k} vs {old_k}")
                code_lines.append(f"new_total = {new_v}")
                code_lines.append(f"old_total = {old_v}")
                code_lines.append("result = round(new_total - old_total, 4)")
                code_lines.append(
                    f"print(f'Total {noun} count change ({{old_total:g}} -> "
                    f"{{new_total:g}}): {{result}}')"
                )
                if old_v:
                    code_lines.append(
                        "result_pct = round((new_total - old_total) / old_total * 100, 4)"
                    )
                    code_lines.append("print(f'Percent change: {result_pct}%')")
            elif extracted_table:
                # Emit variable assignments
                for v in extracted_table.values():
                    yr_label = v['year']
                    code_lines.append(
                        f"{v['code_key']} = {v['val']}  # {v['item']} ({yr_label})"
                    )
                code_lines.append("")
                code_lines.append("# Calculation")

                # Build the calculation
                success_calc = _build_calculation_code(
                    code_lines, extracted_table, query, q_lower, preferred_year, query_years,
                    degraded_notes, detected_unit, evidence_list,
                )
                if not success_calc:
                    # When evidence contains structured tables but none match the
                    # question, return 0.0 and log an explicit warning rather than
                    # returning a confident number from an unrelated line item.
                    # This prevents silently borrowing values that don't pertain to the
                    # asked metric.
                    code_lines.append("result = 0.0")
                    code_lines.append(
                        "print('WARNING: retrieved evidence did not contain data relevant "
                        "to this specific question -- result is not reliable.')"
                    )
            else:
                # PATH 2.5: free-text keyword extraction for PDF / MD&A narrative chunks
                free_text_extracted = _extract_from_free_text(evidence_list, query_years)
                if free_text_extracted:
                    used_extraction = "free_text"
                    for v in free_text_extracted.values():
                        code_lines.append(
                            f"{v['code_key']} = {v['val']}  # {v['item']} ({v['year']})"
                        )
                    code_lines.append("")
                    code_lines.append("# Calculation (from narrative text)")
                    success_calc = _build_calculation_code(
                        code_lines, free_text_extracted, query, q_lower, preferred_year, query_years,
                        degraded_notes, detected_unit, evidence_list,
                    )
                    if not success_calc:
                        # Same fix as the extracted_table branch above: no
                        # more "grab whichever free-text value came first"
                        # regardless of whether it has anything to do with
                        # the question — an honest 0.0 + warning beats a
                        # confidently wrong unrelated number.
                        code_lines.append("result = 0.0")
                        code_lines.append(
                            "print('WARNING: retrieved evidence did not contain data relevant "
                            "to this specific question -- result is not reliable.')"
                        )
                    # Bug 1 fix: PATH 3 was unconditionally overwriting code_lines here.
                    # It is now correctly in the 'else' branch below.
                else:
                    # PATH 3: raw-number fallback — LAST RESORT
                    # Only reached when BOTH linearised table AND free-text extraction
                    # found nothing. Year-like numbers (1900-2099) are filtered out.
                    used_extraction = "fallback"
                    raw_extracted = _extract_raw_numbers(evidence_list)
                    code_lines = [
                        "# FinAgent-RAG PoT Sandbox",
                        "# WARNING: Could not find structured financial data in evidence.",
                        "# Please upload a structured financial report (CSV/JSON) for accurate results.",
                    ]
                    if raw_extracted:
                        for v in raw_extracted.values():
                            code_lines.append(f"{v['code_key']} = {v['val']}")
                        first = list(raw_extracted.values())[0]
                        code_lines.append(f"result = {first['code_key']}")
                        code_lines.append(f"print(f'Value: {{result}}')")
                    else:
                        code_lines.append("result = 0.0")


        code_str = "\n".join(code_lines)

        # ── Step 5: Execute with repair loop ──────────────────────────────────
        success, res_val, stdout_err, sandbox_locals = execute_pot_code(code_str)
        repair_count = 0
        while not success and repair_count < 2:
            repair_count += 1
            code_lines.append("result = 0.0")
            code_str = "\n".join(code_lines)
            success, res_val, stdout_err, sandbox_locals = execute_pot_code(code_str)

        # Multi-year comparisons always expose per-year series variables (not just the
        # latest-year snapshot) so a change question can present the full before/after
        # values.
        # Emit both names so headlines can show the comparison rather than a lone value.
        result_series: List[Dict[str, Any]] = []
        result_delta = sandbox_locals.get("_delta") if success else None
        result_direction = sandbox_locals.get("_direction") if success else None
        if result_delta is not None:
            # Group multi-year output by a computation-specific prefix (the part before
            # _YYYY) so per-year variables emitted by one formula aren't mixed with
            # same-year variables from other calculations.
            # Filter by the winning block's prefix to avoid pulling unrelated adjacent-
            # year variables into the result series.
            year_re = re.compile(r"^(.*)_((?:19|20)\d{2})$")
            by_prefix: Dict[str, List[Tuple[str, Any]]] = {}
            prefix_order: List[str] = []
            for key, val in sandbox_locals.items():
                m = year_re.match(key)
                if m and isinstance(val, (int, float)) and not isinstance(val, bool):
                    prefix = m.group(1)
                    if prefix not in by_prefix:
                        by_prefix[prefix] = []
                        prefix_order.append(prefix)
                    by_prefix[prefix].append((m.group(2), val))
            # The winning _delta/_direction pair came from whichever
            # multi-year block ran LAST (their fixed names get
            # overwritten by each block in turn) -- that's the last
            # prefix group to have contributed a key, by insertion order.
            if prefix_order:
                winning_prefix = prefix_order[-1]
                result_series = [
                    {"year": y, "value": v}
                    for y, v in sorted(by_prefix[winning_prefix], key=lambda item: item[0])
                ]

        # Build extracted summary — the middle field shows how each value
        # was actually obtained (table-total / table-partial / legacy /
        # free-text / period-average), not a blanket "formula" label, so
        # a suspicious extraction (e.g. table-partial, meaning no real
        # total row was ever found) is visible at a glance rather than
        # indistinguishable from a solid one.
        if formula_entry and resolved_formula:
            extracted_summary = {
                ph: (
                    ph,
                    (
                        resolved_formula_meta.get(ph, {}).get("source", "formula")
                        + ("~approx" if resolved_formula_meta.get(ph, {}).get("is_approximate") else "")
                        + " <- " + resolved_formula_meta.get(ph, {}).get("source_detail", "?")
                    ),
                    val,
                )
                for ph, val in resolved_formula.items()
            }
        else:
            extracted_summary = {
                v["code_key"]: (v["item"], v["year"], v["val"])
                for v in extracted_table.values()
            }

        return {
            "code": code_str,
            "success": success,
            "result_value": res_val,
            "output_log": stdout_err,
            "extracted_variables": extracted_summary,
            "repairs_triggered": repair_count,
            "extraction_method": used_extraction,
            "formula_used": formula_entry.get("formula_key") if formula_entry else None,
            "is_degraded_formula": bool(degraded_notes),
            "degraded_note": " ".join(degraded_notes),
            "result_series": result_series,
            "result_delta": result_delta,
            "result_direction": result_direction,
            "result_unit": (
                (formula_entry.get("unit", "") if formula_entry else "")
                or (detected_unit[0] if detected_unit else "")
                # The generic YoY/CAGR branches print "... Growth (y1->y2): 8.27%"
                # without a formula entry, so no unit was attached and the card
                # showed a bare 8.2721.
                or ("%" if re.search(r"(?:YoY Growth|CAGR)[^:]*:\s*-?[0-9.]+%\s*$", stdout_err or "") else "")
            ),
            "is_comparison_answer": is_cf_activity_comparison,
            "is_qualitative_characterization": is_qualitative_characterization,
        }
