"""
FinancialFormulaLibrary
========================
Maps question intent → formula → required variable aliases (Chinese & English).
Used by ProgramOfThoughtReasoner to generate semantically correct Python code
instead of blind `num_1, num_2` fallback.

Each entry in FORMULA_LIBRARY contains:
  keywords_zh   : Chinese trigger keywords (substring match on query)
  keywords_en   : English trigger keywords (lowercase substring match)
  formula_expr  : Python expression using named variable placeholders
  required_vars : dict of  placeholder_name -> [alias list in financial reports]
  result_label  : Human-readable label for the printed result
  unit          : "%" | "x" | "" (times, ratio, or raw value)
"""

from typing import Dict, Any, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Core Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

FORMULA_LIBRARY: Dict[str, Dict[str, Any]] = {

    # ── Liquidity Ratios ─────────────────────────────────────────────────────
    "quick_ratio": {
        "keywords_zh": ["速動比率", "速動比", "酸性測試比率"],
        "keywords_en": ["quick ratio", "acid test", "acid-test"],
        "formula_expr": "(current_assets - inventory) / current_liabilities",
        "required_vars": {
            "current_assets":    ["流動資產", "current assets", "total current assets"],
            "inventory":         ["存貨", "inventory", "inventories", "stock"],
            "current_liabilities": ["流動負債", "current liabilities", "total current liabilities"],
        },
        "result_label": "Quick Ratio",
        "unit": "x",
    },
    "current_ratio": {
        "keywords_zh": ["流動比率", "流動比"],
        "keywords_en": ["current ratio"],
        "formula_expr": "current_assets / current_liabilities",
        "required_vars": {
            "current_assets":    ["流動資產", "current assets", "total current assets"],
            "current_liabilities": ["流動負債", "current liabilities", "total current liabilities"],
        },
        "result_label": "Current Ratio",
        "unit": "x",
    },
    "cash_ratio": {
        "keywords_zh": ["現金比率", "現金比"],
        "keywords_en": ["cash ratio"],
        "formula_expr": "(cash + short_term_investments) / current_liabilities",
        "required_vars": {
            "cash":                  ["現金及約當現金", "cash", "cash and cash equivalents", "cash & equivalents"],
            "short_term_investments": ["短期投資", "short-term investments", "marketable securities", "short term investments"],
            "current_liabilities":   ["流動負債", "current liabilities", "total current liabilities"],
        },
        "result_label": "Cash Ratio",
        "unit": "x",
    },
    "operating_cash_flow_ratio": {
        "keywords_zh": ["營業現金流量比率", "營業活動現金比率"],
        "keywords_en": ["operating cash flow ratio", "cash flow from operations ratio",
                         "ocf ratio", "cash from operations ratio"],
        "formula_expr": "cash_from_operations / current_liabilities",
        "required_vars": {
            "cash_from_operations": ["cash from operations", "net cash provided by operating activities",
                                      "cash provided by operating activities",
                                      "net cash from operating activities", "operating cash flow",
                                      "營業活動之現金流量", "營業活動現金流量"],
            "current_liabilities":  ["流動負債", "current liabilities", "total current liabilities"],
        },
        "result_label": "Operating Cash Flow Ratio",
        "unit": "x",
    },

    # ── Profitability Ratios ──────────────────────────────────────────────────
    "gross_margin": {
        "keywords_zh": ["毛利率", "毛利"],
        "keywords_en": ["gross margin", "gross profit margin", "gross profit ratio"],
        "formula_expr": "gross_profit / revenue",
        "required_vars": {
            "gross_profit": ["毛利", "gross profit", "gross income"],
            "revenue":      ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "Gross Margin",
        "unit": "%",
    },
    "operating_margin": {
        "keywords_zh": ["營業利益率", "營業利潤率", "營業利益"],
        "keywords_en": ["operating margin", "operating profit margin", "ebit margin", "operating income margin"],
        "formula_expr": "operating_income / revenue",
        "required_vars": {
            "operating_income": ["營業利益", "operating income", "operating profit", "ebit", "income from operations"],
            "revenue":          ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "Operating Margin",
        "unit": "%",
    },
    "net_margin": {
        "keywords_zh": ["淨利率", "淨利潤率", "純益率"],
        "keywords_en": ["net margin", "net profit margin", "net income margin", "profit margin"],
        "formula_expr": "net_income / revenue",
        "required_vars": {
            "net_income": ["本期淨利", "淨利", "net income", "net profit", "profit after tax", "net earnings"],
            "revenue":    ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "Net Profit Margin",
        "unit": "%",
    },
    "ebitda_margin": {
        "keywords_zh": ["EBITDA 利潤率", "ebitda 利潤率"],
        "keywords_en": ["ebitda margin"],
        "formula_expr": "ebitda / revenue",
        "required_vars": {
            "ebitda":  ["ebitda", "稅息折舊及攤銷前利潤"],
            "revenue": ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "EBITDA Margin",
        "unit": "%",
    },

    # ── Return Ratios ─────────────────────────────────────────────────────────
    "roe": {
        "keywords_zh": ["股東權益報酬率", "權益報酬率", "roe"],
        "keywords_en": ["return on equity", "roe"],
        "formula_expr": "net_income / shareholders_equity",
        "required_vars": {
            "net_income":          ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "shareholders_equity": ["股東權益", "shareholders equity", "stockholders equity", "equity", "total equity"],
        },
        "result_label": "Return on Equity (ROE)",
        "unit": "%",
    },
    "roa": {
        "keywords_zh": ["資產報酬率", "roa"],
        "keywords_en": ["return on assets", "roa"],
        "formula_expr": "net_income / total_assets",
        "required_vars": {
            "net_income":   ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "total_assets": ["總資產", "total assets", "assets"],
        },
        "result_label": "Return on Assets (ROA)",
        "unit": "%",
    },
    "roic": {
        "keywords_zh": ["投入資本報酬率", "roic"],
        "keywords_en": ["return on invested capital", "roic"],
        "formula_expr": "nopat / invested_capital",
        "required_vars": {
            "nopat":           ["稅後淨營業利潤", "nopat", "net operating profit after tax"],
            "invested_capital": ["投入資本", "invested capital"],
        },
        "result_label": "Return on Invested Capital (ROIC)",
        "unit": "%",
    },

    # ── Leverage / Solvency ───────────────────────────────────────────────────
    "debt_to_equity": {
        "keywords_zh": ["負債比率", "負債權益比", "槓桿比率"],
        "keywords_en": ["debt to equity", "debt-to-equity", "leverage ratio", "d/e ratio"],
        "formula_expr": "total_debt / shareholders_equity",
        "required_vars": {
            "total_debt":          ["總負債", "total debt", "total liabilities", "liabilities"],
            "shareholders_equity": ["股東權益", "shareholders equity", "stockholders equity", "equity", "total equity"],
        },
        "result_label": "Debt-to-Equity Ratio",
        "unit": "x",
    },
    "debt_to_assets": {
        "keywords_zh": ["負債資產比", "資產負債率"],
        "keywords_en": ["debt to assets", "debt-to-assets", "debt ratio"],
        "formula_expr": "total_debt / total_assets",
        "required_vars": {
            "total_debt":   ["總負債", "total debt", "total liabilities", "liabilities"],
            "total_assets": ["總資產", "total assets", "assets"],
        },
        "result_label": "Debt-to-Assets Ratio",
        "unit": "%",
    },
    "interest_coverage": {
        "keywords_zh": ["利息保障倍數", "利息覆蓋率"],
        "keywords_en": ["interest coverage", "times interest earned", "interest coverage ratio"],
        "formula_expr": "ebit / interest_expense",
        "required_vars": {
            "ebit":             ["營業利益", "ebit", "operating income", "operating profit"],
            "interest_expense": ["利息費用", "interest expense", "finance costs"],
        },
        "result_label": "Interest Coverage Ratio",
        "unit": "x",
    },

    # ── Efficiency Ratios ─────────────────────────────────────────────────────
    # fixed_asset_turnover MUST be checked before asset_turnover: detect_formula()
    # matches on the FIRST keyword hit in dict order, and "fixed asset turnover"
    # contains "asset turnover" as a substring, so if asset_turnover's entry
    # came first every fixed_asset_turnover question would misfire as a plain
    # asset_turnover match instead.
    "fixed_asset_turnover": {
        # NOT the same required_vars as asset_turnover below: this divides by
        # average net PP&E, not total assets — a company can look
        # completely different on the two ratios (e.g. asset-light
        # software vs. capital-intensive manufacturing), so the two
        # denominators must never share an alias list.
        "keywords_zh": ["固定資產週轉率", "不動產廠房及設備週轉率"],
        "keywords_en": ["fixed asset turnover", "fixed-asset turnover", "net ppe turnover",
                         "ppe turnover", "property plant and equipment turnover"],
        "formula_expr": "revenue / ((ppe_old + ppe_new) / 2)",
        "required_vars": {
            "revenue":  ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
            # Same alias list for both — distinguished purely by which
            # year column matches, same convention as revenue_yoy's
            # revenue_new/revenue_old below.
            "ppe_old": ["property, plant and equipment, net", "property and equipment, net",
                        "net property, plant and equipment", "property, plant and equipment",
                        "property and equipment", "不動產、廠房及設備"],
            "ppe_new": ["property, plant and equipment, net", "property and equipment, net",
                        "net property, plant and equipment", "property, plant and equipment",
                        "property and equipment", "不動產、廠房及設備"],
        },
        "result_label": "Fixed Asset Turnover",
        "unit": "x",
        "multi_year": True,
    },
    "asset_turnover": {
        "keywords_zh": ["資產週轉率", "總資產週轉率"],
        "keywords_en": ["asset turnover", "total asset turnover"],
        "formula_expr": "revenue / total_assets",
        "required_vars": {
            "revenue":      ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
            "total_assets": ["總資產", "total assets", "assets"],
        },
        "result_label": "Asset Turnover",
        "unit": "x",
    },
    "inventory_turnover": {
        "keywords_zh": ["存貨週轉率", "庫存週轉率"],
        "keywords_en": ["inventory turnover"],
        "formula_expr": "cogs / inventory",
        "required_vars": {
            "cogs":      ["銷售成本", "cost of goods sold", "cogs", "cost of revenue", "cost of sales"],
            "inventory": ["存貨", "inventory", "inventories"],
        },
        "result_label": "Inventory Turnover",
        "unit": "x",
    },
    "receivables_turnover": {
        "keywords_zh": ["應收帳款週轉率"],
        "keywords_en": ["receivables turnover", "accounts receivable turnover"],
        "formula_expr": "revenue / accounts_receivable",
        "required_vars": {
            "revenue":             ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
            "accounts_receivable": ["應收帳款", "accounts receivable", "trade receivables"],
        },
        "result_label": "Receivables Turnover",
        "unit": "x",
    },

    # ── Per Share ─────────────────────────────────────────────────────────────
    "eps": {
        "keywords_zh": ["每股盈餘", "eps"],
        "keywords_en": ["eps", "earnings per share"],
        "formula_expr": "net_income / shares_outstanding",
        "required_vars": {
            "net_income":        ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "shares_outstanding": ["流通在外股數", "shares outstanding", "weighted average shares", "diluted shares"],
        },
        "result_label": "Earnings Per Share (EPS)",
        "unit": "",
    },

    # ── Growth Rates ──────────────────────────────────────────────────────────
    "revenue_yoy": {
        "keywords_zh": ["營業收入成長率", "收入成長率", "營收成長", "營收年增"],
        "keywords_en": ["revenue growth", "revenue yoy", "sales growth", "revenue increase"],
        "formula_expr": "(revenue_new - revenue_old) / revenue_old * 100",
        "required_vars": {
            "revenue_new": ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
            "revenue_old": ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "Revenue YoY Growth",
        "unit": "%",
        "multi_year": True,  # signals: need same item across 2 years
    },
    "net_income_yoy": {
        "keywords_zh": ["淨利成長率", "淨利年增", "獲利成長"],
        "keywords_en": ["net income growth", "net profit growth", "earnings growth"],
        "formula_expr": "(net_income_new - net_income_old) / net_income_old * 100",
        "required_vars": {
            "net_income_new": ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "net_income_old": ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
        },
        "result_label": "Net Income YoY Growth",
        "unit": "%",
        "multi_year": True,
    },
    "operating_income_yoy": {
        "keywords_zh": ["營業利益成長率", "營業利益年增"],
        "keywords_en": ["operating income growth", "operating profit growth"],
        "formula_expr": "(operating_income_new - operating_income_old) / operating_income_old * 100",
        "required_vars": {
            "operating_income_new": ["營業利益", "operating income", "operating profit"],
            "operating_income_old": ["營業利益", "operating income", "operating profit"],
        },
        "result_label": "Operating Income YoY Growth",
        "unit": "%",
        "multi_year": True,
    },
    "cagr_generic": {
        "keywords_zh": ["複合成長率", "cagr"],
        "keywords_en": ["cagr", "compound annual growth", "compound growth rate"],
        "formula_expr": "(value_end / value_start) ** (1 / years) - 1",
        "required_vars": {
            "value_end":   [],  # filled dynamically from matched target metric
            "value_start": [],
        },
        "result_label": "CAGR",
        "unit": "%",
        "multi_year": True,
    },

    # ── Multi-Year Average Ratios ────────────────────────────────────────────
    "capex_to_revenue": {
        # "period_average" (unlike "multi_year") isn't a pick-one-of-two-years
        # ratio — it's the average of capex/revenue computed separately for
        # EVERY year the query asks about (e.g. "FY2017-FY2019 3 year
        # average"), which needs the full per-year series, not just an
        # old/new pair. See _extract_formula_guided()/_gen_formula_code() in
        # pot_reasoner.py for how this flag changes extraction/codegen.
        "keywords_zh": ["資本支出佔營收比", "資本支出占營收比", "資本支出營收比"],
        "keywords_en": ["capex to revenue", "capex as a percentage of revenue",
                         "capex % of revenue", "capital expenditures to revenue",
                         "capex as a % of revenue"],
        "formula_expr": "capex / revenue",  # evaluated once per year, then averaged
        "required_vars": {
            "capex":   ["capital expenditures", "capital expenditure", "purchases of property",
                        "purchases of property and equipment", "purchases of property, plant and equipment",
                        "資本支出"],
            "revenue": ["營業收入", "revenue", "net revenue", "net sales", "total revenue", "sales"],
        },
        "result_label": "CapEx to Revenue (period average)",
        "unit": "%",
        "period_average": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Lookup API
# ─────────────────────────────────────────────────────────────────────────────

def detect_formula(query: str) -> Optional[Dict[str, Any]]:
    """
    Scan the FORMULA_LIBRARY for the best matching formula given a user query.
    Returns the formula entry dict (with key 'formula_key' injected), or None.

    Priority: exact Chinese keyword match > English keyword match.
    If multiple match, the first found wins (ordering in FORMULA_LIBRARY matters).
    """
    q_lower = query.lower()

    # First pass: Chinese keywords (higher precision)
    for key, entry in FORMULA_LIBRARY.items():
        for kw in entry.get("keywords_zh", []):
            if kw in query:
                return {**entry, "formula_key": key}

    # Second pass: English keywords
    for key, entry in FORMULA_LIBRARY.items():
        for kw in entry.get("keywords_en", []):
            if kw in q_lower:
                return {**entry, "formula_key": key}

    return None


def get_variable_aliases(formula_entry: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return the required_vars dict: placeholder → alias list."""
    return formula_entry.get("required_vars", {})
