"""
ProgramOfThoughtReasoner (PoT) — v3  FinanceBench-aligned
==========================================================
Variable extraction hierarchy:
  1. Linearized-table (pipe-delimited) — reliable for sample JSON data
  2. Formula-guided (alias search)     — for formula questions
  3. Raw-number fallback               — last resort

Calculation hierarchy:
  1. Formula template (financial_formula_library.py)
  2. Ratio/margin: match numerator + denominator by SAME year
  3. YoY/CAGR: find same item across different years
  4. Direct lookup: first matching item

Key FinanceBench fixes vs v2:
  - Margin calc correctly matches Gross Profit + Revenue for the SAME year
  - Canonical item labels eliminate false matches (e.g. "net revenue" ≠ "net income")
  - Year from query is prioritised over other years in evidence
  - FY prefix is stripped before year matching
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from app.tools.sandbox import execute_pot_code
from app.agent.financial_formula_library import detect_formula, get_variable_aliases
from app.agent.company_line_item_overrides import get_overrides_for_company  # Step 3/4
from app.tools.table_parser import is_markdown_separator_row

# ─────────────────────────────────────────────────────────────────────────────
# Canonical item taxonomy  (order matters — first match wins)
# ─────────────────────────────────────────────────────────────────────────────
_ITEM_TAXONOMY: List[Tuple[str, List[str]]] = [
    # Revenue / top line
    ("revenue",        ["total revenue", "net revenue", "net sales", "revenue",
                         "營業收入", "營收"]),
    # Gross
    ("gross_profit",   ["gross profit", "gross margin amount", "營業毛利", "毛利"]),
    # Operating
    ("cost_of_revenue",["cost of revenue", "cost of goods sold", "cogs",
                         "cost of sales", "營業成本"]),
    ("op_expense",     ["operating expenses", "operating expense", "opex", "營業費用"]),
    ("op_income",      ["operating income", "operating profit", "operating earnings",
                         "ebit", "營業利益", "營業淨利"]),
    ("rd_expense",     ["research & development", "r&d", "research and development",
                         "研發費用"]),
    ("sga",            ["sg&a", "selling general", "selling and marketing",
                         "推銷管理費用", "推銷與管理費用"]),
    # Net income / EPS
    ("net_income_btax",["net income before tax", "income before tax", "pretax income",
                         "稅前淨利"]),
    ("income_tax",     ["income tax", "tax expense", "所得稅費用"]),
    ("net_income",     ["net income", "net profit", "net earnings",
                         "本期淨利", "淨利"]),
    ("eps",            ["eps", "earnings per share", "diluted eps", "基本每股盈餘",
                         "稀釋每股盈餘", "每股盈餘"]),
    # Balance sheet
    ("cash",           ["cash and cash equivalents", "cash & equivalents",
                         "現金及約當現金"]),
    ("total_assets",   ["total assets", "總資產"]),
    ("current_assets", ["total current assets", "current assets", "流動資產"]),
    ("inventory",      ["inventory", "inventories", "存貨"]),
    ("accounts_rec",   ["accounts receivable", "trade receivables", "應收帳款"]),
    ("current_liab",   ["total current liabilities", "current liabilities", "流動負債"]),
    ("total_liab",     ["total liabilities", "總負債"]),
    ("equity",         ["total equity", "total shareholders equity",
                         "stockholders equity", "股東權益總額", "股東權益"]),
    # Cash flow
    ("operating_cf",   ["operating cash flow", "cash from operations",
                         "cash provided by operating", "營業活動現金"]),
    ("capex",          ["capital expenditure", "purchases of ppe",
                         "capital expenditure", "資本支出"]),
    ("depreciation",   ["depreciation", "amortization", "d&a", "折舊"]),
    ("fcf",            ["free cash flow", "fcf", "自由現金流"]),
    # Misc
    ("data_center_rev",["data center revenue", "data center"]),
]

# Pre-build a flat lookup: lowercase alias → canonical key
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in _ITEM_TAXONOMY:
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a.lower()] = _canonical


_NEGATION_PREFIX_RE = re.compile(r'\b(non[- ]?|not\s+|deferred\s+|unearned\s+)$')


def _is_negated_match(item_lower: str, match_start: int) -> bool:
    """
    True if the text immediately before an alias match ends with a prefix
    that turns the alias into a fundamentally different accounting
    concept: a straight negation ("non-", "non ", "not ") — e.g. "current
    assets" matching inside "non-current assets" — or a recognition-timing
    modifier ("deferred ", "unearned ") that turns an income-statement
    line into an unrelated balance-sheet liability — e.g. bare "revenue"
    matching inside "deferred revenues" (a contract-liability line, not
    income; confirmed real case: Activision Blizzard's balance sheet
    "Deferred revenues" row was picked as FY2019 revenue for a fixed-
    asset-turnover calculation). Generic across every canonical/alias
    pair, not specific to any one company or metric.
    """
    return bool(_NEGATION_PREFIX_RE.search(item_lower[:match_start]))


def _get_canonical(item_name: str, company_name: str = "") -> str:
    """
    Map a raw line item label to a canonical key.

    Lookup priority:
      1. Company-specific override aliases (from company_line_item_overrides)
      2. Global _ITEM_TAXONOMY (exact match)
      3. Global _ITEM_TAXONOMY (longest substring match)

    Every substring match (priorities 1 and 3 — an exact match in
    priority 2 can't be accidentally negated, since the whole string
    would have to literally equal the alias) is rejected if it's
    immediately preceded by a negation prefix — see _is_negated_match().

    Args:
        item_name    : raw line item string (e.g. "Total net revenues")
        company_name : company name from evidence (e.g. "activision blizzard")

    Returns:
        canonical metric key (e.g. "revenue") or "unknown"
    """
    item_lower = item_name.lower().strip()

    # ── Priority 1: Company-specific overrides ────────────────────────────────
    if company_name:
        overrides = get_overrides_for_company(company_name)
        for canonical, aliases in overrides.items():
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower == item_lower:
                    return canonical
                idx = item_lower.find(alias_lower)
                if idx != -1 and not _is_negated_match(item_lower, idx):
                    return canonical

    # ── Priority 2: Global taxonomy exact match ──────────────────────────────
    if item_lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[item_lower]

    # ── Priority 3: Global taxonomy longest substring match ────────────────────
    best = ""
    best_len = 0
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        idx = item_lower.find(alias)
        if idx == -1 or len(alias) <= best_len:
            continue
        if _is_negated_match(item_lower, idx):
            continue
        best = canonical
        best_len = len(alias)
    return best or "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(name: str, maxlen: int = 24) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()[:maxlen] or "val"


def _to_float(s: str) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def _normalize_year(y: str) -> str:
    """Strip FY prefix so FY2023 → 2023."""
    return re.sub(r"^FY", "", y.strip())


def _extract_query_years(query: str) -> List[str]:
    """Every distinct fiscal year mentioned in the query, in first-seen
    order. A dash/'to'-joined range ("FY2017-FY2019", "2017 to 2019") is
    expanded to every year in between (inclusive) — a plain year-token
    scan would otherwise return only the two range endpoints, which is
    wrong for an "N-year average" question that needs every year in the
    range (e.g. period_average formulas — see financial_formula_library)."""
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


#: Language implying a two-point comparison, even when the query text
#: only names the LATEST year (e.g. "improving...as of FY2023" implies
#: vs. FY2022 — confirmed real case: "Does AMCOR have an improving gross
#: margin profile as of FY2023?" names only 2023, yet the gold answer is
#: a FY2023-vs-FY2022 comparison). Generic phrasing, not tied to any one
#: metric or company.
_TREND_KEYWORDS = (
    "improv", "declin", "trend", "profile", "increased or decreased",
    "increase or decrease", "compared to", "year over year", "yoy",
)


def _with_implied_trend_year(query_years: Optional[List[str]], q_lower: str) -> List[str]:
    """
    If trend language is present but query_years names only one year,
    append year-1 so multi-year comparison logic has a second year to
    compare against.
    """
    years = list(query_years or [])
    if len(set(years)) == 1 and any(t in q_lower for t in _TREND_KEYWORDS):
        try:
            years = years + [str(int(years[0]) - 1)]
        except ValueError:
            pass
    return years


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: standard Markdown table (header row + |---|---| separator + data rows)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_markdown_table_block(content: str, ev_company: str = "") -> Dict[str, Dict]:
    """
    Parse EVERY standard Markdown pipe table embedded in `content`:
        | Line Item | 2023 | 2024 |
        |---|---|---|
        | Revenue | 2,161.7 | 2,894.3 |
        | Net Income | 567.8 | 601.2 |

    The line immediately above each '|---|---|' separator row is that
    block's own header row (first column = line-item label, remaining
    columns = period/year labels); every line below the separator, up to
    the first blank/non-pipe line, is that block's data rows.

    A real page's parent_content routinely contains SEVERAL such table
    blocks concatenated together (e.g. a cash-flow statement split into
    "Net income / adjustments" and "changes in operating assets /
    Net cash provided by operating activities" blocks by an intervening
    blank line) — a version that stopped after the FIRST block would
    silently never see rows in any later block, no matter how well they
    matched. Confirmed real case: Adobe's "Net cash provided by operating
    activities" row lived in the second block of a two-block cash-flow
    parent_content; a first-block-only parser found nothing for it and
    the formula-guided extraction fell through to an unrelated fallback.
    Each block gets its OWN header/period_headers, since two blocks on
    the same page aren't guaranteed to share identical columns.

    Returns entries in the EXACT same shape as _extract_from_linearized_table()
    ({code_key: {item, canonical, year, val, code_key}}), so downstream
    _find_same_item_pair()/_build_calculation_code() need no changes at all —
    they just see more entries in the same dict shape.
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
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or "|" not in stripped:
                break  # this block ended — outer loop resumes scanning from here

            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 2 or not cells[0]:
                j += 1
                continue
            line_item_name = cells[0]

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

def _extract_from_linearized_table(evidence_list: List[Dict[str, Any]]) -> Dict[str, Dict]:
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

        # Extract company name for override lookup (Step 3/4)
        ev_company = ev.get("company", "")
        if not ev_company:
            import re as _re
            m_co = _re.search(r"Company:\s*([^|]+)", content)
            ev_company = m_co.group(1).strip() if m_co else ""

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
)


def _is_supplementary_schedule(content: str) -> bool:
    """
    True if this evidence chunk looks like a supplementary schedule for a
    SUBSET of the reporting entity rather than the real consolidated
    statements. These routinely reuse the EXACT SAME line-item labels as
    the primary statements ("Net sales", "Gross profit", "Total current
    liabilities", "Inventories") for a much smaller reporting entity —
    a real, confirmed failure mode: _score_row_match() alone can't catch
    this, because the row genuinely IS an exact "Total X" match; it's
    just for the wrong entity. Confirmed on Amcor's real 10-K, where a
    "Deed of Cross Guarantee" schedule's $58M/$377M guarantor-group
    Net sales/Gross profit outranked the real consolidated $14,694M/
    $2,725M by being an equally "clean" exact-label match.
    """
    lower = content.lower()
    return any(m in lower for m in _SUPPLEMENTARY_SCHEDULE_MARKERS)


def _score_row_match(label: str, aliases: List[str]) -> int:
    """
    How well does a table row's own label match one of a variable's
    aliases? Generic across every canonical/alias pair — never special-
    cased to a specific line item.
      2 = the label IS (once normalized) exactly one of the aliases, or
          exactly "total {alias}" — a genuine total/subtotal row.
      1 = the alias appears only as a substring of a longer label — a
          sub-item, an "Other X" line, or a compound "X and Y" label.
          Real, but not the total; callers should flag results built
          from a score-1 match as approximate.
      0 = no real match at all, including a substring match immediately
          preceded by a negation prefix ("non-", "not ") — e.g. "current
          assets" inside "non-current assets" doesn't count.
    When multiple rows compete for the same (placeholder, year), the
    highest score wins — this is what makes "Total net revenues" beat
    "Subscription, licensing, and other revenues" for a revenue lookup,
    and "Total current liabilities" beat "Other current liabilities" for
    a current_liab lookup, purely by comparing the candidates rather than
    hardcoding either company's line-item wording.
    """
    label_norm = re.sub(r'\s+', ' ', label.lower().strip().rstrip(':'))
    for alias in aliases:
        a = alias.lower().strip()
        if not a:
            continue
        idx = label_norm.find(a)
        if idx == -1 or _is_negated_match(label_norm, idx):
            continue
        if label_norm == a or label_norm == f"total {a}":
            return 2
        return 1
    return 0


#: Some formula variables are never reported as a single line item — a
#: filer's own statement structure splits them into several sub-items
#: instead (e.g. Amcor's balance sheet has no "Inventory" row at all,
#: just "Raw materials and supplies" + "Work in process and finished
#: goods"). Keyed by the SAME placeholder name used in
#: financial_formula_library.py's required_vars (the codebase's existing
#: convention for naming a concept consistently across formulas), each
#: value lists the sub-item alias fragments that, summed together for a
#: given year, approximate the composite concept. Generic mechanism, not
#: tied to Amcor or quick_ratio — any formula whose required_vars uses
#: one of these placeholder names benefits automatically.
_COMPOSITE_ITEM_ALIASES: Dict[str, List[str]] = {
    "inventory": [
        "raw materials", "raw materials and supplies",
        "work in process", "work in process and finished goods",
        "finished goods", "merchandise inventory",
        "存貨", "原料", "在製品", "製成品",
    ],
}


def _resolve_composite_item(
    evidence_list: List[Dict[str, Any]],
    sub_aliases: List[str],
) -> Dict[str, Tuple[float, List[str]]]:
    """
    Approximate a composite line item (e.g. "inventory") by summing its
    known sub-items across a filer's own statement structure, when no
    single row matches the composite concept directly.

    Each sub-alias is resolved INDEPENDENTLY using the same per-row/
    per-segment scoring as _extract_formula_guided()'s Priority 1/2 scan
    (so "Total X"-style sub-total rows still win over looser matches),
    reduced to its own single best-scoring row per year — this prevents
    double-counting when the same sub-item legitimately appears on more
    than one page (e.g. a balance-sheet summary AND a detailed note).
    Distinct sub-aliases' per-year values are then summed. Supplementary
    schedules (guarantor/parent-only) are excluded, same as everywhere
    else in this module.

    Returns {year: (summed_value, [line_item_label, ...])} — the label
    list is kept for provenance (what was actually added together).
    """
    # sub_candidates[sub_alias] = {year: (value, score, is_primary, line_item_label)}
    sub_best: Dict[str, Dict[str, Tuple[float, int, bool, str]]] = {a: {} for a in sub_aliases}

    for ev in evidence_list:
        content = ev.get("parent_content") or ev.get("content", "")
        if not content:
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


def _extract_formula_guided(
    evidence_list: List[Dict[str, Any]],
    formula_entry: Dict[str, Any],
    query_years: List[str],
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
    is_period_average = formula_entry.get("period_average", False)

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
            if not content:
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

            # ── Priority 2: legacy "Company: X | ... | Line Item: Y | ──
            # 2017: A | 2016: B" format, ONE SEGMENT PER LINE ITEM.
            #
            # Root cause fix: the previous version checked "does an alias
            # appear ANYWHERE in this whole content string" and separately
            # scanned the WHOLE string for every "year: value"-shaped
            # field, with no link between the two — so an alias matching
            # one line item's label could pull a completely unrelated
            # line item's number out of the same content blob (confirmed
            # real case: Adobe's cash_from_operations alias correctly
            # matched "Net cash provided by operating activities", but
            # the number used, 759,737, actually belonged to a different
            # row, "Maturities of short-term investments", elsewhere in
            # the same page's parent_content). A real 10-K page's
            # parent_content can legitimately contain MANY line items
            # concatenated together, so alias-matching must be scoped to
            # ONE line item's own segment, never the whole blob.
            #
            # "Line Item:" is the fixed marker that starts each line
            # item's own segment in this format — splitting on it (kept
            # via the lookahead) gives one segment per line item, each
            # independently scored via the SAME _score_row_match() used
            # for Priority 1, so a value is never attributed to a line
            # item whose label didn't actually match.
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
    for placeholder in var_aliases:
        best_by_year: Dict[str, Tuple[float, int, str, Tuple[bool, int], int, str]] = {}
        for val, yr, score, source, is_primary, ev_idx, line_item_label in candidates[placeholder]:
            priority = (is_primary, score)
            existing = best_by_year.get(yr)
            if existing is None or priority > existing[3]:
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

    # ── Composite-item fallback: some formula variables are only ever
    # reported as several sub-items, never one line (see
    # _COMPOSITE_ITEM_ALIASES). Tried BEFORE the free-text fallback —
    # summing genuinely structured sub-item rows is more reliable than
    # narrative-text keyword proximity. Never overrides a year that
    # already has a real direct match.
    for placeholder in var_aliases:
        if placeholder not in _COMPOSITE_ITEM_ALIASES:
            continue
        if not _needs_ft_fallback(placeholder, resolved[placeholder]):
            continue
        composite_by_year = _resolve_composite_item(
            evidence_list, _COMPOSITE_ITEM_ALIASES[placeholder]
        )
        existing_years = {y for _, y in resolved[placeholder]}
        for yr, (total_val, labels) in composite_by_year.items():
            if yr in existing_years:
                continue
            resolved[placeholder].append((total_val, yr))
            winner_meta.setdefault(placeholder, {})[yr] = (
                1, "composite", -1, " + ".join(labels)
            )
            existing_years.add(yr)

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
                old_yr, new_yr = sorted_years[0], sorted_years[-1]
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
    """
    Two DIFFERENT variables (different base names, ignoring _new/_old —
    revenue_new legitimately equalling revenue_old happens in a flat-YoY
    evidence set and is not a bug) resolving to the exact same value is a
    strong signal one of them silently borrowed the other's number
    instead of genuinely finding its own — the confirmed real example is
    fixed_asset_turnover's ppe_new landing on exactly revenue's value
    with no real PP&E match anywhere in evidence.

    Detected pairs are dropped back to unresolved IN PLACE (removed from
    resolved_formula and resolved_formula_series) rather than flagged and
    used anyway — this makes _gen_formula_code() fail naturally (it
    already returns [] when a required variable is missing), which falls
    through to generate_and_execute()'s PATH 2 (linearized-table
    extraction), a genuinely different mechanism — the "retry via a
    different path" behaviour, without needing separate re-retrieval
    plumbing. Returns warning lines (as '# ...' comments) so the
    collision is visible in the generated code / sandbox log rather than
    silently disappearing.
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
    # Revenue — all common SEC/10-K wordings
    (["net sales", "net revenues", "total net revenue",
      "revenue", "net revenue", "total revenue", "sales",
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

#: Canonicals whose value IS legitimately a percentage. Every other
#: canonical (revenue, net_income, etc.) needs an absolute dollar figure \u2014
#: a nearby "15%" in prose almost always describes a rate of CHANGE
#: ("decreased 15% to $X"), not the metric's own value, so it must never
#: win as a candidate for a dollar-value canonical (confirmed real case:
#: Activision Blizzard MD&A text "net revenues decreased 15% to $4.9
#: billion" produced revenue=15.0 because 15 was simply the nearest
#: number after the matched year token).
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
            # Standard Markdown table (|---|---|) — also handled by
            # _extract_from_linearized_table via _extract_from_markdown_table_block().
            # Without this check the SAME table content was scanned a
            # second time by this function's crude keyword-proximity
            # heuristic, which has no concept of table structure — a
            # bare "revenue" trigger matched somewhere in a flattened
            # balance-sheet table and grabbed a totally unrelated cell's
            # number (confirmed real case: Activision Blizzard's balance
            # sheet table produced "revenue = 11,174", the value actually
            # belonging to "Additional paid-in capital", just because it
            # was the nearest number after the trigger word once the
            # table was flattened to plain text).
            continue
        if _is_supplementary_schedule(content):
            # A guarantor/parent-only/combining schedule reuses the same
            # line-item labels as the real consolidated statements for a
            # much smaller reporting entity — keyword-proximity matching
            # can't tell the difference the way _score_row_match()'s
            # row-scoped comparison can, so this path must not touch such
            # evidence at all rather than risk silently resolving to the
            # wrong entity's numbers (confirmed real case: Amcor's "Deed
            # of Cross Guarantee" schedule's $58M gross profit vs. the
            # real consolidated $2,725M).
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
) -> List[str]:
    """
    Build code that computes `expr` separately for EVERY year common to
    all placeholders, then averages those per-year results — e.g. for
    capex_to_revenue's "capex / revenue" over FY2017-FY2019, this emits
    the equivalent of:
        capex_by_year = {"2017": ..., "2018": ..., "2019": ...}
        revenue_by_year = {"2017": ..., "2018": ..., "2019": ...}
        _years = sorted(set(capex_by_year) & set(revenue_by_year) & ...)
        _ratios = [capex_by_year[y] / revenue_by_year[y] for y in _years]
        result = round(sum(_ratios) / len(_ratios) * 100, 2)
    Generic over placeholder count/names (whatever `expr` references), so
    this isn't specific to capex_to_revenue — any future "N-year average
    ratio" formula reuses it by setting period_average=True.
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


def _gen_formula_code(
    formula_entry: Dict[str, Any],
    resolved: Dict[str, float],
    extracted_table: Dict[str, Dict],
    resolved_series: Optional[Dict[str, List[Tuple[float, str]]]] = None,
    resolved_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    query_years: Optional[List[str]] = None,
    q_lower: str = "",
) -> List[str]:
    """Generate Python code lines using the formula template.

    period_average formulas (e.g. "3-year average of capex as a % of
    revenue") can't be expressed as a single evaluation of formula_expr
    over one picked value per placeholder — they need the ratio computed
    separately for EVERY year in range, then averaged. `resolved_series`
    (the full {placeholder -> [(value, year), ...]} from
    _extract_formula_guided) carries what that needs; `resolved` (one
    value per placeholder) is enough for every other formula shape.

    `resolved_meta` (also from _extract_formula_guided) is used only to
    annotate each variable assignment with WHERE it came from (which
    evidence item, which line item) as an inline comment — so a wrong
    extraction is traceable straight from the generated code/sandbox log
    instead of needing to re-diagnose against the raw PDF each time.
    """
    lines = []
    fk = formula_entry.get("formula_key", "custom")
    label = formula_entry.get("result_label", "Result")
    unit = formula_entry.get("unit", "")
    expr = formula_entry.get("formula_expr", "")
    is_multi_year = formula_entry.get("multi_year", False)
    is_period_average = formula_entry.get("period_average", False)

    if is_period_average:
        return _gen_period_average_code(fk, label, unit, expr, resolved_series or {})

    if not resolved:
        return []

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
            needed_phs = set(re.findall(r"[a-z_][a-z0-9_]*", expr)) - {"years", "math"}
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
    needed = set(re.findall(r"[a-z_][a-z0-9_]*", expr)) - {"years", "math"}
    available = set(resolved.keys())
    missing = needed - available
    if missing:
        # Try to fill from linearized table
        for ph in list(missing):
            for k, v in extracted_table.items():
                if ph in k or ph.replace("_new", "") in v.get("canonical", ""):
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
        lines.append(f"print(f'{label}: {{result}}x')")
    else:
        lines.append(f"result = {expr}")
        lines.append(f"print(f'{label}: {{result}}')")
    return lines


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

# Map query keywords → canonical labels (for YoY target item inference)
_QUERY_CANONICAL_HINTS: List[Tuple[List[str], str]] = [
    (["net sales", "revenue", "net revenue", "total revenue", "sales",
      "營業收入", "營收"],                                    "revenue"),
    (["gross profit", "毛利"],                              "gross_profit"),
    (["operating income", "operating profit", "營業利益"],   "op_income"),
    (["net income", "net earnings", "net profit", "淨利"],  "net_income"),
    (["ebitda"],                                            "ebitda"),
    (["eps", "earnings per share"],                         "eps"),
    (["capex", "capital expenditure"],                      "capex"),
    (["r&d", "research and development", "研發"],           "rd_expense"),
    (["free cash flow", "fcf"],                             "fcf"),
    (["total assets", "資產"],                              "total_assets"),
    (["equity", "shareholders", "stockholders"],            "equity"),
    (["cost of revenue", "cost of goods", "cost of sales"], "cost_of_revenue"),
]


def _infer_target_canonical(query_lower: str) -> Optional[str]:
    """Return the most likely canonical label for the item the user is asking about."""
    for triggers, canonical in _QUERY_CANONICAL_HINTS:
        for t in triggers:
            if t in query_lower:
                return canonical
    return None


def _find_same_item_pair(
    vars_list: List[Dict],
    query_lower: str = "",
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find two vars with the SAME canonical item but DIFFERENT years.
    Bug 3 fix: prefer the canonical that matches what the user asked about
    (e.g. prefer 'revenue' over 'net_income' when query asks about revenue YoY).
    """
    # Group by canonical
    groups: Dict[str, List[Dict]] = {}
    for v in vars_list:
        groups.setdefault(v.get("canonical", _sanitize(v["item"])), []).append(v)

    # Determine priority order: put query-relevant canonical first
    target = _infer_target_canonical(query_lower) if query_lower else None
    canonical_order = list(groups.keys())
    if target and target in canonical_order:
        canonical_order.remove(target)
        canonical_order.insert(0, target)

    for c in canonical_order:
        items = groups[c]
        sorted_items = sorted(items, key=lambda x: str(x["year"]))
        unique_years = list({x["year"] for x in sorted_items})
        if len(unique_years) >= 2:
            old = next(x for x in sorted_items if x["year"] == sorted(unique_years)[0])
            new = next(x for x in sorted_items if x["year"] == sorted(unique_years)[-1])
            return old, new

    return None, None


_CANONICAL_TO_ALIASES: Dict[str, List[str]] = {c: aliases for c, aliases in _ITEM_TAXONOMY}


def _pick_best_in_group(
    items: List[Dict],
    canonical: str,
    preferred_year: Optional[str] = None,
) -> Optional[Dict]:
    """
    Among several rows that all normalized to the same canonical (e.g. both
    "Total current assets" and "Prepaid expenses and other current assets"
    contain the substring "current assets" and so both get tagged
    canonical="current_assets"), pick the one that's actually the
    total/subtotal line rather than an arbitrary sub-component — using the
    same total-row-priority scoring _extract_formula_guided() already uses
    (_score_row_match: 2 = genuine total row, 1 = sub-item substring match).
    Ties broken by matching preferred_year, then by first occurrence.
    """
    if not items:
        return None
    aliases = _CANONICAL_TO_ALIASES.get(canonical, [canonical])

    def sort_key(x: Dict) -> Tuple[int, int]:
        score = _score_row_match(x["item"], aliases)
        year_match = 1 if preferred_year and x["year"] == preferred_year else 0
        return (score, year_match)

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
        def pair_key(pair: Tuple[Dict, Dict]) -> Tuple[int, int]:
            n, d = pair
            year_match = 1 if preferred_year and n["year"] == preferred_year else 0
            score = _score_row_match(n["item"], num_aliases) + _score_row_match(d["item"], den_aliases)
            return (year_match, score)
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
    """
    Same as _find_pair_for_margin, but scoped to a single specific year —
    used when a question asks to compare a margin/ratio ACROSS years
    (e.g. "did gross margin improve between FY2022 and FY2023"), so each
    year's numerator/denominator must come from that year's own data,
    never bleed in a different year's value as a fallback.
    """
    nums = [x for x in groups.get(num_canonical, []) if x["year"] == year]
    dens = [x for x in groups.get(den_canonical, []) if x["year"] == year]
    if not nums or not dens:
        return None, None
    return _find_pair_for_margin({num_canonical: nums, den_canonical: dens}, num_canonical, den_canonical, year)


# ─────────────────────────────────────────────────────────────────────────────
# FinanceBench-aligned calculation builder
# ─────────────────────────────────────────────────────────────────────────────

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
    (["cost ratio", "cogs ratio"],  "cost_of_revenue","revenue",   "Cost of Revenue Ratio"),
    (["capex%", "capex ratio"],     "capex",          "revenue",   "CapEx % of Revenue"),
]

_ROE_TRIGGERS = ["roe", "return on equity"]
_ROA_TRIGGERS = ["roa", "return on assets"]
_CURRENT_RATIO_TRIGGERS = ["current ratio", "流動比率"]
_QUICK_RATIO_TRIGGERS = ["quick ratio", "acid", "速動"]


def _emit_multi_year_ratio(
    code_lines: List[str],
    label: str,
    unit: str,
    year_exprs: List[Tuple[str, str]],
) -> None:
    """
    Emit code that computes and prints a ratio for EACH of 2+ years, plus
    an explicit numeric delta between the first and last — so a "did X
    improve or decline between year A and year B" question gets both
    real numbers AND an explicit year-over-year comparison to draw on,
    instead of a single year's value with nothing to compare it against
    (confirmed real case: Amcor's gross-margin question only ever
    computed FY2023 alone, so the final answer had no actual FY2022
    figure to compare against and guessed the wrong direction). Reports
    "increased"/"decreased" rather than "improved"/"declined" — whether a
    higher number is actually better differs by metric (a higher current
    ratio is good, a higher SG&A-to-revenue ratio is not), so the neutral
    direction is left for the answer-synthesis stage to interpret.
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


def _build_calculation_code(
    code_lines: List[str],
    extracted_table: Dict[str, Dict],
    query: str,
    q_lower: str,
    preferred_year: Optional[str] = None,
    query_years: Optional[List[str]] = None,
    degraded_notes: Optional[List[str]] = None,
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
    if "cagr" in q_lower or "複合成長率" in q_lower:
        # Bug 3 fix: prefer the canonical the user asked about
        v1, v2 = _find_same_item_pair(list(extracted_table.values()), q_lower)
        if v1 and v2:
            try:
                yrs = abs(float(v2["year"]) - float(v1["year"])) or 1.0
            except Exception:
                yrs = 1.0
            code_lines.append(f"# CAGR: {v1['item']}")
            code_lines.append(f"result_cagr = cagr({v1['code_key']}, {v2['code_key']}, {yrs})")
            code_lines.append("result = round(result_cagr, 2)")
            label = v1['item']
            y1, y2 = v1['year'], v2['year']
            code_lines.append(f"print(f'{label} CAGR ({y1}->{y2}): {{result}}%')")
            return True

    # ── YoY Growth ────────────────────────────────────────────────────────────
    yoy_triggers = ["yoy", "year over year", "成長率", "年增", "growth rate", "growth",
                    "변동", "change", "增加多少", "減少多少"]
    if any(t in q_lower for t in yoy_triggers):
        # Bug 3 fix: pass q_lower so _find_same_item_pair prefers the item the user asked about
        v1, v2 = _find_same_item_pair(list(extracted_table.values()), q_lower)
        if v1 and v2:
            code_lines.append(f"# YoY: {v1['item']}")
            code_lines.append(f"result_yoy = yoy({v1['code_key']}, {v2['code_key']})")
            code_lines.append("result = round(result_yoy, 2)")
            label = v1['item']
            y1, y2 = v1['year'], v2['year']
            code_lines.append(f"print(f'{label} YoY Growth ({y1}->{y2}): {{result}}%')")
            return True

    # ── ROE ───────────────────────────────────────────────────────────────────
    if any(t in q_lower for t in _ROE_TRIGGERS):
        n, d = _find_pair_for_margin(groups, "net_income", "equity", preferred_year)
        if n and d:
            code_lines.append(f"# ROE = Net Income / Equity")
            code_lines.append(f"result = round(margin({n['code_key']}, {d['code_key']}), 2)")
            yr = n['year']
            code_lines.append(f"print(f'Return on Equity (ROE) ({yr}): {{result}}%')")
            return True

    # ── ROA ───────────────────────────────────────────────────────────────────
    if any(t in q_lower for t in _ROA_TRIGGERS):
        n, d = _find_pair_for_margin(groups, "net_income", "total_assets", preferred_year)
        if n and d:
            code_lines.append(f"# ROA = Net Income / Total Assets")
            code_lines.append(f"result = round(margin({n['code_key']}, {d['code_key']}), 2)")
            yr = n['year']
            code_lines.append(f"print(f'Return on Assets (ROA) ({yr}): {{result}}%')")
            return True

    # ── Current Ratio ─────────────────────────────────────────────────────────
    if any(t in q_lower for t in _CURRENT_RATIO_TRIGGERS):
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
                    _emit_multi_year_ratio(code_lines, "Current Ratio", "x", year_exprs)
                    return True
            ca = _pick_best_in_group(ca_list, "current_assets", preferred_year)
            cl = _pick_best_in_group(cl_list, "current_liab", ca["year"])
            code_lines.append(f"# Current Ratio = Current Assets / Current Liabilities")
            code_lines.append(f"result = round({ca['code_key']} / {cl['code_key']}, 4)")
            yr = ca['year']
            code_lines.append(f"print(f'Current Ratio ({yr}): {{result}}x')")
            return True

    # ── Quick Ratio ───────────────────────────────────────────────────────────
    if any(t in q_lower for t in _QUICK_RATIO_TRIGGERS):
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
                    _emit_multi_year_ratio(code_lines, label, "x", year_exprs)
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
            code_lines.append(f"print(f'Quick Ratio ({yr}): {{result}}x')")
            return True

    # ── Margin / Ratio ────────────────────────────────────────────────────────
    for triggers, num_c, den_c, label in _MARGIN_MAP:
        if any(t in q_lower for t in triggers):
            distinct_years = sorted(set(query_years or []))
            if len(distinct_years) >= 2:
                year_exprs = []
                for yr in distinct_years:
                    n_yr, d_yr = _pair_for_year(groups, num_c, den_c, yr)
                    if n_yr and d_yr:
                        year_exprs.append((yr, f"margin({n_yr['code_key']}, {d_yr['code_key']})"))
                if len(year_exprs) >= 2:
                    code_lines.append(f"# {label} = {num_c} / {den_c}")
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
    # Pick the var that best matches the query intent
    target_canonical = None
    for canonical, _ in _ITEM_TAXONOMY:
        aliases_str = " ".join(_a for _c, _als in _ITEM_TAXONOMY if _c == canonical for _a in _als)
        if any(a in q_lower for a in aliases_str.split()):
            target_canonical = canonical
            break

    # No "fall back to whatever's first in extracted_table" beyond this —
    # that used to silently print an ENTIRELY UNRELATED line item as if
    # it answered the question (confirmed real case: a fixed-asset-
    # turnover query with no PP&E/revenue in evidence returned "Provision
    # for inventories: 6.0" as the result, because that happened to be
    # the first value extracted from whatever page WAS retrieved). If the
    # query names a specific metric and nothing extracted matches it,
    # returning False here is the honest outcome — it lets the caller
    # fall through to free-text extraction and, failing that, the
    # explicit "could not find structured financial data" message,
    # instead of confidently stating a number that has nothing to do
    # with what was asked.
    vars_to_use = groups.get(target_canonical, []) if target_canonical else []

    if vars_to_use:
        best = _pick_best_in_group(vars_to_use, target_canonical, preferred_year)
        item_label = best['item']
        yr = best['year']
        code_lines.append(f"result = {best['code_key']}")
        code_lines.append(f"print(f'{item_label} ({yr}): {{result}}')")
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Reasoner
# ─────────────────────────────────────────────────────────────────────────────

class ProgramOfThoughtReasoner:
    """
    Program-of-Thought (PoT) Financial Reasoner — v3 (FinanceBench-aligned)
    """

    def generate_and_execute(
        self, query: str, evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        q_lower = query.lower()
        query_years = _extract_query_years(query)
        preferred_year = query_years[-1] if query_years else None  # latest year mentioned

        # ── Step 1: Detect formula intent ────────────────────────────────────
        formula_entry = detect_formula(query)

        # ── Step 2: Extract variables from linearized tables ──────────────────
        extracted_table = _extract_from_linearized_table(evidence_list)

        # ── Step 3: Formula-guided extraction ─────────────────────────────────
        resolved_formula: Dict[str, float] = {}
        resolved_formula_series: Dict[str, List[Tuple[float, str]]] = {}
        resolved_formula_meta: Dict[str, Dict[str, Any]] = {}
        duplicate_warnings: List[str] = []
        if formula_entry:
            resolved_formula, resolved_formula_series, resolved_formula_meta = _extract_formula_guided(
                evidence_list, formula_entry, query_years
            )
            duplicate_warnings = _detect_and_strip_duplicate_values(
                resolved_formula, resolved_formula_series, resolved_formula_meta
            )

        # ── Step 4: Build code ────────────────────────────────────────────────
        code_lines = [
            "# FinAgent-RAG Program-of-Thought (PoT) Sandbox",
            "# Extracted from retrieved financial evidence",
        ]
        code_lines.extend(duplicate_warnings)

        used_extraction = "formula"
        degraded_notes: List[str] = []

        if formula_entry and resolved_formula:
            # PATH 1: Formula library
            fk = formula_entry.get("formula_key", "")
            code_lines.append(f"# Formula: {fk} — {formula_entry.get('result_label', '')}")
            formula_code = _gen_formula_code(
                formula_entry, resolved_formula, extracted_table,
                resolved_formula_series, resolved_formula_meta,
                query_years, q_lower,
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

            if extracted_table:
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
                    degraded_notes,
                )
                if not success_calc:
                    # The evidence DID contain some structured table data,
                    # but none of it matched what this question actually
                    # asks about — result = 0.0 with an explicit warning
                    # printed to the sandbox log, not a confident-looking
                    # number borrowed from an unrelated line item (the
                    # confirmed real bug: a fixed-asset-turnover question
                    # with no PP&E/revenue in evidence used to silently
                    # return "Provision for inventories: 6.0").
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
                        degraded_notes,
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

        # A multi-year comparison (_emit_multi_year_ratio) always sets
        # exactly these two names — when present, expose the FULL
        # per-year series (not just the latest year's `result`) so a
        # "did X improve or decline between year A and year B" answer's
        # headline number can actually show the comparison, not a lone
        # snapshot (confirmed real case: the frontend's result card showed
        # only Quick Ratio's FY2023 value for a question explicitly asking
        # about the FY2022->FY2023 change).
        result_series: List[Dict[str, Any]] = []
        result_delta = sandbox_locals.get("_delta") if success else None
        result_direction = sandbox_locals.get("_direction") if success else None
        if result_delta is not None:
            year_re = re.compile(r"^(.*)_((?:19|20)\d{2})$")
            series_items = []
            for key, val in sandbox_locals.items():
                m = year_re.match(key)
                if m and isinstance(val, (int, float)) and not isinstance(val, bool):
                    series_items.append((m.group(2), val))
            # Multiple multi-year blocks could in principle coexist; keep
            # only entries sharing the year-set actually used by the
            # winning _delta/_direction pair — approximate by keeping the
            # two most recently assigned per-year values (dict preserves
            # insertion order in the generated code), which are exactly
            # the ones _emit_multi_year_ratio's own delta was computed
            # from.
            if series_items:
                result_series = [{"year": y, "value": v} for y, v in series_items[-2:]]

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
        }
