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


def _get_canonical(item_name: str, company_name: str = "") -> str:
    """
    Map a raw line item label to a canonical key.

    Lookup priority:
      1. Company-specific override aliases (from company_line_item_overrides)
      2. Global _ITEM_TAXONOMY (exact match)
      3. Global _ITEM_TAXONOMY (longest substring match)

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
                # Exact or substring match
                if alias_lower == item_lower or alias_lower in item_lower:
                    return canonical

    # ── Priority 2: Global taxonomy exact match ──────────────────────────────
    if item_lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[item_lower]

    # ── Priority 3: Global taxonomy longest substring match ────────────────────
    best = ""
    best_len = 0
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias in item_lower and len(alias) > best_len:
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
    return [_normalize_year(y) for y in re.findall(r"(?:FY)?(20\d{2})", query)]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: linearized-table (pipe-delimited)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_linearized_table(evidence_list: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Works on pipe-delimited linearized table rows:
      ... | Line Item: 營業收入 (Revenue) | 2023 年 (全年度): 2,161.7 | 2024 年: 2,894.3
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

        # Extract company name for override lookup (Step 3/4)
        ev_company = ev.get("company", "")
        if not ev_company:
            import re as _re
            m_co = _re.search(r"Company:\s*([^|]+)", content)
            ev_company = m_co.group(1).strip() if m_co else ""

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

def _extract_formula_guided(
    evidence_list: List[Dict[str, Any]],
    formula_entry: Dict[str, Any],
    query_years: List[str],
) -> Dict[str, float]:
    """
    For each required variable in formula_entry, search evidence for a chunk whose
    Line Item matches a known alias.  Returns {placeholder -> float}.

    Bug 1 root-cause fix: when evidence is narrative text (not pipe-delimited),
    the pipe-field parser finds nothing and resolved stays empty.  We now fall back
    to _extract_from_free_text to fill any still-empty placeholders.
    """
    var_aliases = get_variable_aliases(formula_entry)
    is_multi_year = formula_entry.get("multi_year", False)
    resolved: Dict[str, List[Tuple[float, str]]] = {k: [] for k in var_aliases}

    for placeholder, aliases in var_aliases.items():
        for ev in evidence_list:
            content = ev.get("parent_content") or ev.get("content", "")
            if not content:
                continue
            content_lower = content.lower()

            # Check if any alias appears in this evidence chunk
            alias_hit = any(a.lower() in content_lower for a in aliases)
            if not alias_hit:
                continue

            # Parse pipe-delimited fields for year: value pairs
            fields = [f.strip() for f in content.split("|")]
            year_col_re = re.compile(
                r"(.*?(?:20\d{2}|FY\d{4})[^:]*?)\s*:\s*([\d,]+\.?\d*)", re.IGNORECASE
            )
            for f in fields:
                m_f = year_col_re.search(f)
                if not m_f:
                    continue
                ym = re.search(r"(20\d{2}|FY\d{4})", m_f.group(1))
                if not ym:
                    continue
                year = _normalize_year(ym.group(1))
                val = _to_float(m_f.group(2))
                if val is not None and val != 0:
                    resolved[placeholder].append((val, year))

            if resolved[placeholder]:
                break  # first chunk match wins

    # ── Free-text fallback: fill any placeholders still empty or single-year ──
    # Triggers when:
    #   (a) a placeholder has no matches at all, OR
    #   (b) formula is multi_year but a placeholder only has ONE distinct year
    #       (the pipe parser grabbed '2022: 34,229' but missed '2021: 35,355')
    def _needs_ft_fallback(ph: str, matches_list: List[Tuple[float, str]]) -> bool:
        if not matches_list:
            return True
        if is_multi_year and len(set(y for _, y in matches_list)) < 2:
            return True
        return False

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
                # Try to match placeholder to a canonical
                base = placeholder.replace("_new", "").replace("_old", "")
                # Find the canonical that best matches any alias
                matched_canonical = None
                for c, items_list in ft_by_canonical.items():
                    if c == base or any(c in a.lower() or a.lower() in c for a in aliases):
                        matched_canonical = c
                        break
                if matched_canonical is None:
                    # Fallback: look for canonical that has entries for our query years
                    for c, items_list in ft_by_canonical.items():
                        if any(yr in [y for _, y in items_list] for yr in query_years):
                            matched_canonical = c
                            break
                if matched_canonical:
                    resolved[placeholder] = ft_by_canonical[matched_canonical]

    # ── Choose the best value for each placeholder ────────────────────────────
    final: Dict[str, float] = {}
    for placeholder, matches in resolved.items():
        if not matches:
            continue
        if is_multi_year:
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
            else:
                best = next((v for v, y in matches if y == new_yr), matches[-1][0])
            final[placeholder] = best
        else:
            # Prefer value from query year
            best = None
            for val, yr in matches:
                if yr in query_years:
                    best = val
                    break
            final[placeholder] = best if best is not None else matches[0][0]

    return final


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
    r"([\d,]+(?:\.\d+)?)\s*(?:%|percent|billion|million|trillion|\u5104|B|M)?",
    re.IGNORECASE,
)
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
            # pipe-delimited table chunk — handled by _extract_from_linearized_table
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

def _gen_formula_code(
    formula_entry: Dict[str, Any],
    resolved: Dict[str, float],
    extracted_table: Dict[str, Dict],
) -> List[str]:
    """Generate Python code lines using the formula template."""
    lines = []
    fk = formula_entry.get("formula_key", "custom")
    label = formula_entry.get("result_label", "Result")
    unit = formula_entry.get("unit", "")
    expr = formula_entry.get("formula_expr", "")
    is_multi_year = formula_entry.get("multi_year", False)

    if not resolved:
        return []

    for ph, val in resolved.items():
        lines.append(f"{ph} = {val}")

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


def _find_pair_for_margin(
    groups: Dict[str, List[Dict]],
    num_canonical: str,
    den_canonical: str,
    preferred_year: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find (numerator_var, denominator_var) for a margin calculation.
    Prefers matching the SAME year.  Falls back to any available year.
    """
    nums = groups.get(num_canonical, [])
    dens = groups.get(den_canonical, [])
    if not nums or not dens:
        return None, None

    # Try same-year match first
    for n in nums:
        for d in dens:
            if n["year"] == d["year"]:
                if preferred_year is None or n["year"] == preferred_year:
                    return n, d

    # Fallback: any year match (prefer preferred_year)
    if preferred_year:
        n = next((x for x in nums if x["year"] == preferred_year), nums[0])
        d = next((x for x in dens if x["year"] == preferred_year), dens[0])
    else:
        n, d = nums[0], dens[0]
    return n, d


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


def _build_calculation_code(
    code_lines: List[str],
    extracted_table: Dict[str, Dict],
    query: str,
    q_lower: str,
    preferred_year: Optional[str] = None,
) -> bool:
    """
    Build the calculation section of the PoT code.
    Returns True if a calculation was successfully generated.
    """
    groups = _group_by_canonical(extracted_table)

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
            ca = next((x for x in ca_list if x["year"] == preferred_year), ca_list[0])
            cl = next((x for x in cl_list if x["year"] == preferred_year), cl_list[0])
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
            ca = next((x for x in ca_list if x["year"] == preferred_year), ca_list[0])
            cl = next((x for x in cl_list if x["year"] == ca["year"]), cl_list[0])
            if inv_list:
                inv = next((x for x in inv_list if x["year"] == ca["year"]), inv_list[0])
                code_lines.append(f"# Quick Ratio = (Current Assets - Inventory) / Current Liabilities")
                code_lines.append(f"result = round(({ca['code_key']} - {inv['code_key']}) / {cl['code_key']}, 4)")
            else:
                code_lines.append(f"# Quick Ratio (approx, no inventory data) = CA / CL")
                code_lines.append(f"result = round({ca['code_key']} / {cl['code_key']}, 4)")
            yr = ca['year']
            code_lines.append(f"print(f'Quick Ratio ({yr}): {{result}}x')")
            return True

    # ── Margin / Ratio ────────────────────────────────────────────────────────
    for triggers, num_c, den_c, label in _MARGIN_MAP:
        if any(t in q_lower for t in triggers):
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

    vars_to_use = groups.get(target_canonical, []) if target_canonical else []
    if not vars_to_use:
        vars_to_use = list(extracted_table.values())

    if vars_to_use:
        # Prefer preferred year
        if preferred_year:
            best = next((v for v in vars_to_use if v["year"] == preferred_year), vars_to_use[0])
        else:
            best = vars_to_use[0]
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
        if formula_entry:
            resolved_formula = _extract_formula_guided(
                evidence_list, formula_entry, query_years
            )

        # ── Step 4: Build code ────────────────────────────────────────────────
        code_lines = [
            "# FinAgent-RAG Program-of-Thought (PoT) Sandbox",
            "# Extracted from retrieved financial evidence",
        ]

        used_extraction = "formula"

        if formula_entry and resolved_formula:
            # PATH 1: Formula library
            fk = formula_entry.get("formula_key", "")
            code_lines.append(f"# Formula: {fk} — {formula_entry.get('result_label', '')}")
            formula_code = _gen_formula_code(formula_entry, resolved_formula, extracted_table)
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
                    code_lines, extracted_table, query, q_lower, preferred_year
                )
                if not success_calc:
                    code_lines.append("result = 0.0")
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
                        code_lines, free_text_extracted, query, q_lower, preferred_year
                    )
                    if not success_calc:
                        first_v = list(free_text_extracted.values())[0]
                        _item_lbl = first_v["item"]
                        _item_yr = first_v["year"]
                        _item_key = first_v["code_key"]
                        code_lines.append(f"result = {_item_key}")
                        code_lines.append(
                            "print(f'" + _item_lbl + " (" + _item_yr + "): {result}')")
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
        success, res_val, stdout_err = execute_pot_code(code_str)
        repair_count = 0
        while not success and repair_count < 2:
            repair_count += 1
            code_lines.append("result = 0.0")
            code_str = "\n".join(code_lines)
            success, res_val, stdout_err = execute_pot_code(code_str)

        # Build extracted summary
        if formula_entry and resolved_formula:
            extracted_summary = {
                ph: (ph, "formula", val)
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
        }
