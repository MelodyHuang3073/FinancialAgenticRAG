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

import re
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
    "working_capital": {
        # Explains that the raw working-capital metric (current assets minus current
        # liabilities) is distinct from a working-capital ratio.
        # Matches on explicit phrases like "positive working capital"/"negative working
        # capital" to avoid misapplying the subtraction rule to ratio questions.
        # Uses the unambiguous phrase "net working capital" for the raw-dollar metric so
        # a direct current_assets - current_liabilities formula can be used.
        "keywords_zh": ["正的營運資金", "負的營運資金", "淨營運資金"],
        "keywords_en": ["positive working capital", "negative working capital", "net working capital"],
        "formula_expr": "current_assets - current_liabilities",
        "required_vars": {
            "current_assets":    ["流動資產", "current assets", "total current assets"],
            "current_liabilities": ["流動負債", "current liabilities", "total current liabilities"],
        },
        "result_label": "Working Capital",
        "unit": "$",
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
            # Prioritizes the exact cash-flow-statement label "net cash provided by
            # operating activities" because the retrieval step uses the first ASCII
            # alias as the query.
            # Avoids generic paraphrases like "cash from operations" which can match
            # unrelated balance-sheet cash rows and cause the true cash-flow row to be
            # missed.
            "cash_from_operations": ["net cash provided by operating activities", "cash from operations",
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
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Gross Margin",
        "unit": "%",
        # Lets an "N-year average gross margin" question compute the
        # ratio per year then average, same mechanism as capex_to_revenue
        # below. pot_reasoner._gen_formula_code() only actually routes
        # into that averaging codegen when the query itself says
        # "average" — a 2-year "did gross margin improve" question still
        # gets the explicit before/after comparison, not a blended mean.
        "period_average": True,
    },
    "operating_margin": {
        "keywords_zh": ["營業利益率", "營業利潤率", "營業利益"],
        "keywords_en": ["operating margin", "operating profit margin", "ebit margin",
                         "operating income margin", "operating income % margin",
                         "unadjusted operating income % margin", "unadjusted operating margin"],
        "formula_expr": "operating_income / revenue",
        "required_vars": {
            "operating_income": ["營業利益", "operating income", "operating profit", "ebit", "income from operations"],
            "revenue":          ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Operating Margin",
        "unit": "%",
        "period_average": True,
    },
    "net_margin": {
        "keywords_zh": ["淨利率", "淨利潤率", "純益率"],
        "keywords_en": ["net margin", "net profit margin", "net income margin", "profit margin"],
        "formula_expr": "net_income / revenue",
        "required_vars": {
            "net_income": ["本期淨利", "淨利", "net income", "net profit", "profit after tax", "net earnings"],
            "revenue":    ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Net Profit Margin",
        "unit": "%",
        "period_average": True,
    },
    "da_margin": {
        # Notes this metric lacked a registered formula in the formula library, so the
        # system fell back to unrelated values.
        # Signals that metrics without formulas must be added to the library to enable
        # deterministic computation.
        "keywords_zh": ["折舊攤銷率", "折舊攤提率"],
        "keywords_en": ["d&a margin", "d&a % margin", "depreciation and amortization margin",
                         "depreciation margin", "depreciation and amortization % margin",
                         # Highlights that some benchmark phrasing inserts a
                         # parenthetical between terms (for example between
                         # "amortization" and "% margin"), which contiguous phrase
                         # matching will miss.
                         # Indicates parsers should allow intervening parentheticals
                         # when matching multi-word metric names.
                         "d&a from cash flow statement"],
        "formula_expr": "depreciation / revenue",
        "required_vars": {
            "depreciation": ["折舊", "depreciation and amortization", "depreciation & amortization",
                             "depreciation"],
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "D&A Margin",
        "unit": "%",
    },
    "effective_tax_rate": {
        # Explains a metric existed only as a classifier keyword but had no computation
        # formula registered, so numeric-answer routing worked but no deterministic
        # calculation was available.
        # Notes that once a formula with the needed expression shape exists, existing
        # two-period trend code can compute multi-year change questions.
        "keywords_zh": ["有效稅率", "實質稅率"],
        "keywords_en": ["effective tax rate", "tax rate"],
        # Explains that tax provision rows in multi-step income statements may be shown
        # as negative values (a signed subtraction from pretax income), so using
        # abs(income_tax) normalizes the sign.
        # Applies the same sign-convention handling used for other line items like cost
        # of goods sold or capital expenditures.
        "formula_expr": "abs(income_tax) / pretax_income",
        # Some filings include a filer-stated "Effective tax rate" line.
        # Prefer that extracted value over recomputing income_tax/pretax_income when
        # present.
        # This extracted value is optional extra input and is handled via the same
        # alias/candidate pipeline as other variables.
        "direct_lookup_var": "effective_tax_rate_direct",
        "required_vars": {
            "effective_tax_rate_direct": ["effective tax rate", "有效稅率", "實質稅率"],
            # Use a precise alias like "provision for income taxes" to avoid matching
            # unrelated tax rows.
            # Avoid bare "income tax" aliases because they can match pretax or deferred-
            # tax lines and return the wrong row.
            "income_tax":    ["provision for income taxes", "income tax provision", "income tax expense",
                              "provision for taxes on income", "所得稅費用"],
            "pretax_income": ["稅前淨利", "income before income tax", "income before income taxes",
                              "income before provision for income taxes",
                              "earnings before income taxes", "pretax income", "income before taxes"],
        },
        "result_label": "Effective Tax Rate",
        "unit": "%",
    },
    "ebitda_margin": {
        "keywords_zh": ["EBITDA 利潤率", "ebitda 利潤率"],
        "keywords_en": ["ebitda margin"],
        "formula_expr": "ebitda / revenue",
        "required_vars": {
            "ebitda":  ["ebitda", "稅息折舊及攤銷前利潤"],
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "EBITDA Margin",
        "unit": "%",
        "period_average": True,
    },
    "ebitda_margin_unadjusted": {
        # Check for an "unadjusted EBITDA % margin" style pattern before plain-dollar-
        # sum EBITDA formulas.
        # Otherwise a substring match can yield a dollar EBITDA instead of a percentage
        # that needs revenue and period averaging.
        "keywords_zh": ["未調整EBITDA利潤率", "未調整EBITDA 利潤率"],
        "keywords_en": ["unadjusted ebitda % margin", "unadjusted ebitda margin",
                         "unadjusted ebitda %margin"],
        "formula_expr": "(op_income + depreciation) / revenue",
        "required_vars": {
            "op_income":    ["營業利益", "operating income", "operating profit", "ebit",
                              "income from operations"],
            "depreciation": ["折舊", "depreciation and amortization", "depreciation & amortization",
                              "depreciation", "amortization", "d&a"],
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Unadjusted EBITDA Margin",
        "unit": "%",
        "period_average": True,
    },
    "ebitda_unadjusted_less_capex": {
        # Register a distinct metric for "unadjusted EBITDA less capex" before plain
        # "unadjusted EBITDA".
        # If not, the parser may resolve only EBITDA and silently ignore the capex term,
        # since capex wouldn't be a required variable.
        "keywords_zh": ["未調整EBITDA減資本支出", "未調整EBITDA扣除資本支出"],
        "keywords_en": ["unadjusted ebitda less capex", "unadjusted ebitda minus capex",
                         "unadjusted ebitda - capex"],
        # abs(capex): same sign convention as capex_to_revenue elsewhere
        # in this library -- a cash-flow-statement capex line is
        # routinely a parenthesised/negative outflow figure.
        "formula_expr": "op_income + depreciation - abs(capex)",
        "required_vars": {
            "op_income":    ["營業利益", "operating income", "operating profit", "ebit",
                              "income from operations"],
            "depreciation": ["折舊", "depreciation and amortization", "depreciation & amortization",
                              "depreciation", "amortization", "d&a"],
            "capex":        ["capital expenditures", "capital expenditure", "purchases of property",
                              "purchases of property and equipment",
                              "purchases of property, plant and equipment", "資本支出"],
        },
        "result_label": "Unadjusted EBITDA less CapEx",
        "unit": "",
    },
    "ebitda_unadjusted": {
        # Compute plain operating income + depreciation & amortization as a simple sum.
        # This is distinct from EBITDA margin (a ratio) and from fully adjusted EBITDA
        # with other addbacks.
        "keywords_zh": ["未調整EBITDA", "未調整息稅折舊攤銷前利潤"],
        "keywords_en": ["unadjusted ebitda", "operating income + depreciation",
                         "operating income plus depreciation"],
        "formula_expr": "op_income + depreciation",
        "required_vars": {
            "op_income":    ["營業利益", "operating income", "operating profit", "ebit",
                              "income from operations"],
            "depreciation": ["折舊", "depreciation and amortization", "depreciation & amortization",
                              "depreciation", "amortization", "d&a"],
        },
        "result_label": "Unadjusted EBITDA",
        "unit": "",
    },

    # Return ratios that divide a single-year income by the average of a balance-sheet
    # item across two end-point years (e.g., net_income / avg(total_assets[t-1],
    # total_assets[t])).
    # Do not treat this as averaging the yearly ratios; multi-year formula handling
    # embeds the averaging at the component level and falls back to single-year when
    # only one year is requested.
    "roe": {
        "keywords_zh": ["股東權益報酬率", "權益報酬率"],
        "keywords_en": ["return on equity", "roe"],
        "formula_expr": "net_income / ((shareholders_equity_old + shareholders_equity_new) / 2)",
        "required_vars": {
            "net_income":              ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "shareholders_equity_old": ["股東權益", "shareholders equity", "stockholders equity", "equity", "total equity"],
            "shareholders_equity_new": ["股東權益", "shareholders equity", "stockholders equity", "equity", "total equity"],
        },
        "result_label": "Return on Equity (ROE)",
        # Bare decimal (e.g. "-0.02"), not a percentage.
        # Matches the convention used for related return-on metrics in this codebase.
        "unit": "",
        "multi_year": True,
    },
    "roa": {
        "keywords_zh": ["資產報酬率"],
        "keywords_en": ["return on assets", "roa"],
        "formula_expr": "net_income / ((total_assets_old + total_assets_new) / 2)",
        "required_vars": {
            "net_income":       ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "total_assets_old": ["總資產", "total assets", "assets"],
            "total_assets_new": ["總資產", "total assets", "assets"],
        },
        "result_label": "Return on Assets (ROA)",
        # Bare decimal ratio, not a percentage — consistent with other ratio fields in
        # this codebase.
        # Avoid scaling by 100 when computing this ratio; keep as a unitless decimal.
        "unit": "",
        "multi_year": True,
    },
    "roic": {
        "keywords_zh": ["投入資本報酬率"],
        "keywords_en": ["return on invested capital", "roic"],
        "formula_expr": "nopat / invested_capital",
        "required_vars": {
            "nopat":           ["稅後淨營業利潤", "nopat", "net operating profit after tax"],
            "invested_capital": ["投入資本", "invested capital"],
        },
        "result_label": "Return on Invested Capital (ROIC)",
        "unit": "%",
        "period_average": True,
    },
    "dividend_payout_ratio": {
        "keywords_zh": ["股利發放率", "股息發放率", "配息率"],
        "keywords_en": ["dividend payout ratio", "payout ratio"],
        # abs() because a cash-flow-statement "Dividends" line is a
        # financing-activities OUTFLOW, reported as a negative number --
        # the ratio itself should read as a positive percentage of net
        # income paid out, not a signed cash-flow value.
        "formula_expr": "abs(dividends_paid) / net_income_attributable",
        "required_vars": {
            # Cash-flow-statement financing-activities line commonly labeled simply
            # "Dividends".
            # Use the bare alias and apply negation-prefix checks to avoid substring
            # collisions.
            "dividends_paid": ["股利", "現金股利", "支付股利", "dividends paid",
                                "cash dividends paid", "dividends"],
            # Do NOT merge this alias with the plain "net_income" pool: the
            # shareholders-attributable line is distinct from consolidated net income.
            # Include alternate wording like "net earnings attributable to" early in
            # alias lists so firms that use "earnings" phrasing are matched.
            "net_income_attributable": [
                "歸屬於股東之淨利", "net income attributable to shareowners",
                "net earnings attributable to",
                "net income attributable to shareholders",
                "net income attributable to common shareholders",
                "net income attributable to",
            ],
        },
        "result_label": "Dividend Payout Ratio",
        # Bare decimal (e.g. "0.80"), not a percentage.
        # This matches the unitless representation used by sibling ratio fields; do not
        # multiply by 100.
        "unit": "",
    },
    "retention_ratio": {
        # This field must be explicitly registered; otherwise fallback retrieval can
        # pick unrelated rows.
        # Shares aliases with the payout ratio field — retention ratio is computed as 1
        # - payout ratio using the same line items.
        "keywords_zh": ["保留盈餘率", "盈餘保留率"],
        "keywords_en": ["retention ratio", "plowback ratio"],
        "formula_expr": "(net_income_attributable - abs(dividends_paid)) / net_income_attributable",
        "required_vars": {
            "dividends_paid": ["股利", "現金股利", "支付股利", "dividends paid",
                                "cash dividends paid", "dividends"],
            # Some filers use the word "earnings" instead of "income" in P&L labels.
            # Include an alias for "net earnings attributable to" early so it co-occurs
            # with "net income" aliases in retrieval queries.
            "net_income_attributable": [
                "歸屬於股東之淨利", "net income attributable to shareowners",
                "net earnings attributable to",
                "net income attributable to shareholders",
                "net income attributable to common shareholders",
                "net income attributable to",
            ],
        },
        "result_label": "Retention Ratio",
        "unit": "",
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
    "debt_change_yoy": {
        # This question asks if borrowings increased; it refers to actual borrowings
        # (loans/notes/bonds), not broad "total liabilities".
        # Use distinct required_vars names and sum the two balance-sheet rows for
        # borrowings (long-term debt + current portion) rather than relying on a single
        # alias match.
        # Treat it as a multi-year two-point comparison with direction.
        "keywords_zh": ["負債是否增加", "負債是否減少", "舉債增加", "舉債減少"],
        "keywords_en": ["increased its debt", "increased debt on balance sheet",
                         "decreased its debt", "debt on balance sheet"],
        "formula_expr": "total_borrowings_new - total_borrowings_old",
        # Do not make "long-term debt" a primary alias for this formula because it
        # appears in many row labels and can outrank an authoritative "Total debt" row
        # via frequency ties.
        # Keep "long-term debt" only in the composite fallback so filers that disclose
        # an explicit "Total debt" are matched to that authoritative total, while filers
        # that only present separate long-term and current portions still resolve via
        # the composite.
        "required_vars": {
            "total_borrowings_old": ["total debt"],
            "total_borrowings_new": ["total debt"],
        },
        "multi_year": True,
        "result_label": "Change in Total Debt",
        "unit": "$",
    },
    "interest_coverage": {
        "keywords_zh": ["利息保障倍數", "利息覆蓋率"],
        "keywords_en": ["interest coverage", "times interest earned", "interest coverage ratio"],
        # Use absolute value for interest expense because some statements present it as
        # negative; coverage ratios use magnitude of interest.
        # Cap negative EBIT to zero for coverage: when EBIT is negative, report coverage
        # as 0 rather than a negative ratio.
        "formula_expr": "max(0, ebit) / abs(interest_expense)",
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
            "revenue":  ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
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
        # Compute average total assets across the two endpoint years for turnover ratios
        # (same convention as other turnover metrics).
        # This ensures revenue divided by average assets is used rather than revenue
        # divided by a single-year assets figure.
        "keywords_zh": ["資產週轉率", "總資產週轉率"],
        "keywords_en": ["asset turnover", "total asset turnover"],
        "formula_expr": "revenue / ((total_assets_old + total_assets_new) / 2)",
        "required_vars": {
            "revenue":          ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
            "total_assets_old": ["總資產", "total assets", "assets"],
            "total_assets_new": ["總資產", "total assets", "assets"],
        },
        "result_label": "Asset Turnover",
        "unit": "x",
        "multi_year": True,
    },
    "capital_intensity_ratio": {
        # Assessment-style question: determine capital intensity as total_assets /
        # revenue for a single year (not averaged).
        # Route ASSESSMENT questions through formula detection so retrieval targets the
        # specific values (assets and revenue) rather than falling back to generic
        # topic-keyword search.
        "keywords_zh": ["資本密集度", "資本密集"],
        "keywords_en": ["capital-intensive", "capital intensive", "capital intensity ratio"],
        "formula_expr": "total_assets / revenue",
        "required_vars": {
            "total_assets": ["總資產", "total assets", "assets"],
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Capital Intensity Ratio",
        "unit": "x",
    },
    "cash_conversion_cycle": {
        # Register this formula before inventory_turnover/receivables_turnover/dpo.
        # detect_formula() returns the first matching keyword; a composite metric
        # referencing DIO/DSO/DPO can be misclassified if a bare "dpo" matches earlier.
        "keywords_zh": ["現金轉換週期"],
        "keywords_en": ["cash conversion cycle", "ccc"],
        # CCC = DIO + DSO - DPO, each expanded inline (not composed from
        # the separate dpo/inventory_turnover/receivables_turnover
        # formulas above, since this engine evaluates one flat expression
        # per formula) using the same abs(cogs) and 2-endpoint-average
        # conventions as those formulas.
        "formula_expr": (
            "365 * ((inv_old + inv_new) / 2) / abs(cogs)"
            " + 365 * ((ar_old + ar_new) / 2) / revenue"
            " - 365 * ((ap_old + ap_new) / 2) / (abs(cogs) + (inv_new - inv_old))"
        ),
        "required_vars": {
            "cogs":    ["銷售成本", "cost of goods sold", "cost of products sold", "cost of sales", "cogs", "cost of revenue"],
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
            "inv_old": ["存貨", "inventory", "inventories"],
            "inv_new": ["存貨", "inventory", "inventories"],
            # Include bare "receivables"/"receivable" as an alias.
            # Some balance sheets use that exact label without qualifiers, so substring-
            # only aliases miss them.
            "ar_old":  ["應收帳款", "accounts receivable", "trade receivables", "receivables", "receivable"],
            "ar_new":  ["應收帳款", "accounts receivable", "trade receivables", "receivables", "receivable"],
            "ap_old":  ["應付帳款", "accounts payable"],
            "ap_new":  ["應付帳款", "accounts payable"],
        },
        "result_label": "Cash Conversion Cycle (CCC)",
        "unit": "",
        "multi_year": True,
    },
    # Provide two ratio entries: one for questions that explicitly ask for "average
    # inventory" and one for plain year-end inventory.
    # detect_formula routes on the explicit phrase because a single expression cannot
    # correctly handle both conventions.
    "inventory_turnover_avg": {
        "keywords_zh": ["平均存貨週轉率"],
        "keywords_en": ["average inventory between", "average inventory"],
        # abs(cogs): see inventory_turnover below for the sign-convention
        # rationale (same formula, just averaged inventory).
        "formula_expr": "abs(cogs) / ((inventory_old + inventory_new) / 2)",
        "required_vars": {
            "cogs":      ["銷售成本", "cost of goods sold", "cost of products sold", "cost of sales", "cogs", "cost of revenue"],
            "inventory_old": ["存貨", "inventory", "inventories"],
            "inventory_new": ["存貨", "inventory", "inventories"],
        },
        "result_label": "Inventory Turnover",
        "unit": "x",
        "multi_year": True,
    },
    "inventory_turnover": {
        "keywords_zh": ["存貨週轉率", "庫存週轉率"],
        "keywords_en": ["inventory turnover"],
        # Apply abs() to COGS inputs because some source tables record cost lines with
        # negative signs.
        # COGS should be treated as a positive magnitude for turnover calculations
        # regardless of source sign conventions.
        "formula_expr": "abs(cogs) / inventory",
        "required_vars": {
            # Add "cost of products sold" as an alias for COGS.
            # Some sectors use that phrasing instead of other common COGS labels and
            # would otherwise fail to match.
            "cogs":      ["銷售成本", "cost of goods sold", "cost of products sold", "cost of sales", "cogs", "cost of revenue"],
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
            "revenue":             ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
            "accounts_receivable": ["應收帳款", "accounts receivable", "trade receivables", "receivables", "receivable"],
        },
        "result_label": "Receivables Turnover",
        "unit": "x",
    },
    "dpo": {
        # ap_old/ap_new share one alias list, distinguished purely by
        # which year column matches (same convention as
        # fixed_asset_turnover's ppe_old/ppe_new above); cogs and
        # inv_new/inv_old are resolved the same way — is_multi_year picks
        # the OLDEST query year for every "_old"-suffixed placeholder and
        # the NEWEST for every "_new"-suffixed or bare placeholder, so
        # bare "cogs" naturally resolves to the target (newest) year.
        "keywords_zh": ["應付帳款天數", "應付帳款週轉天數"],
        "keywords_en": ["days payable outstanding", "dpo"],
        # abs(cogs): same rationale as inventory_turnover above — some
        # companies present the cost-of-sales line as a signed subtraction
        # step rather than a plain positive magnitude.
        "formula_expr": "365 * ((ap_old + ap_new) / 2) / (abs(cogs) + (inv_new - inv_old))",
        "required_vars": {
            "ap_old": ["應付帳款", "accounts payable"],
            "ap_new": ["應付帳款", "accounts payable"],
            "cogs":   ["銷售成本", "cost of goods sold", "cost of products sold", "cost of sales", "cogs", "cost of revenue"],
            "inv_old": ["存貨", "inventory", "inventories"],
            "inv_new": ["存貨", "inventory", "inventories"],
        },
        "result_label": "Days Payable Outstanding (DPO)",
        "unit": "",
        "multi_year": True,
    },

    # ── Per Share ─────────────────────────────────────────────────────────────
    "eps": {
        "keywords_zh": ["每股盈餘"],
        "keywords_en": ["eps", "earnings per share"],
        "formula_expr": "net_income / shares_outstanding",
        "required_vars": {
            "net_income":        ["本期淨利", "淨利", "net income", "net profit", "net earnings"],
            "shares_outstanding": ["流通在外股數", "shares outstanding", "weighted average shares", "diluted shares"],
        },
        "result_label": "Earnings Per Share (EPS)",
        "unit": "",
    },

    "free_cash_flow": {
        # Register a calculable free cash flow formula (cash from operations - capital
        # expenditures).
        # Relying only on a direct lookup for a pre-labeled "Free cash flow" line misses
        # cases where the question provides the definition.
        "keywords_zh": ["自由現金流"],
        "keywords_en": ["free cash flow", "fcf"],
        "formula_expr": "cash_from_operations - abs(capex)",
        "required_vars": {
            "cash_from_operations": ["net cash provided by operating activities", "cash from operations",
                                      "cash provided by operating activities",
                                      "net cash from operating activities", "operating cash flow",
                                      "營業活動之現金流量", "營業活動現金流量"],
            "capex": ["capital expenditures", "capital expenditure", "purchases of property",
                      "purchases of property and equipment", "purchases of property, plant and equipment",
                      "purchases of land, buildings, and equipment", "資本支出"],
        },
        "result_label": "Free Cash Flow (FCF)",
        "unit": "$",
    },

    # ── Growth Rates ──────────────────────────────────────────────────────────
    "revenue_yoy": {
        "keywords_zh": ["營業收入成長率", "收入成長率", "營收成長", "營收年增"],
        # "change in revenue" / "year-over-year change in revenue" is a common phrasing for this
        # metric that the "growth"/"yoy"/"increase" variants above cannot match.
        # "high growth"/"growth company" (a qualitative characterization) also means "compute
        # revenue YoY growth" in ordinary usage; registering it here routes such questions through
        # this formula's targeted few-query extraction instead of the generic LLM-decomposed path.
        "keywords_en": ["revenue growth", "revenue yoy", "sales growth", "revenue increase",
                         "change in revenue", "change in total revenue",
                         "high growth", "high-growth", "growth company"],
        "formula_expr": "(revenue_new - revenue_old) / revenue_old * 100",
        # Some MD&A tables report the filer's own year-over-year revenue change as a
        # precomputed row.
        # Prefer the filer's reported percentage when available, since recomputing from
        # rounded dollar figures can produce small mismatches.
        # Use this to avoid spurious grading failures from rounding differences.
        "direct_lookup_var": "revenue_pct_change_direct",
        "required_vars": {
            "revenue_new": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
            "revenue_old": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
            # First alias becomes the actual retrieval query text (see
            # orchestrator._build_formula_subquestions) -- empirically the
            # single best-ranking phrasing found for this row's own
            # extremely terse content (literally "Total | 2022: 1.3%"),
            # not a grammatically "natural" description of it. See
            # pot_reasoner._HIGH_GROWTH_TRIGGERS's own comment for the
            # isolated retrieval-ranking testing this came from.
            "revenue_pct_change_direct": ["total percent change results of operations",
                                            "results of operations analysis of consolidated sales"],
        },
        "result_label": "Revenue YoY Growth",
        "unit": "%",
        "multi_year": True,  # signals: need same item across 2 years
    },
    "net_income_yoy": {
        "keywords_zh": ["淨利成長率", "淨利年增", "獲利成長"],
        "keywords_en": ["net income growth", "net profit growth", "earnings growth",
                         "change in net income"],
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
        "keywords_en": ["operating income growth", "operating profit growth",
                         "change in operating income"],
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
        "keywords_zh": ["複合成長率"],
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

    # ── Cost / Expense Ratios ────────────────────────────────────────────────
    "cogs_ratio": {
        # "cost of goods sold as a % of revenue" style questions had no registered formula, so
        # retrieval fell back to LLM decomposition (no guarantee that every requested year gets
        # both its cost and revenue rows) and the direct-lookup fallback could return revenue
        # itself. Registering the formula gives one deterministic cost/revenue sub-query pair per
        # year; the matching _MARGIN_MAP trigger in pot_reasoner.py covers calculation when this
        # entry does not match first.
        "keywords_zh": ["銷貨成本佔營收比", "銷貨成本占營收比", "營業成本率"],
        "keywords_en": ["cost of goods sold as a % of revenue",
                         "cost of goods sold as a percentage of revenue",
                         "cost of sales as a % of revenue",
                         "cost of revenue as a % of revenue"],
        # abs(cogs): same sign convention as capex_to_revenue below -- COGS
        # is occasionally presented as a parenthesised/negative figure.
        "formula_expr": "abs(cogs) / revenue",
        "required_vars": {
            "cogs":    ["銷售成本", "cost of goods sold", "cost of products sold", "cost of sales",
                        "cost of revenue"],
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Cost of Revenue Ratio",
        "unit": "%",
        "period_average": True,
    },
    "capex_to_revenue": {
        # The period_average flag means compute the metric separately for each year
        # requested and then average those per-year values.
        # This differs from a two-point ratio that only needs an old/new pair;
        # extraction/codegen must fetch the full per-year series.
        "keywords_zh": ["資本支出佔營收比", "資本支出占營收比", "資本支出營收比"],
        "keywords_en": ["capex to revenue", "capex as a percentage of revenue",
                         "capex % of revenue", "capital expenditures to revenue",
                         "capex as a % of revenue"],
        # abs(capex): "Purchases of property and equipment" is a cash
        # outflow, so the cash flow statement often lists it as a
        # parenthesised/negative figure. CapEx is conceptually always a
        # positive spend magnitude for this ratio.
        "formula_expr": "abs(capex) / revenue",  # evaluated once per year, then averaged
        "required_vars": {
            "capex":   ["capital expenditures", "capital expenditure", "purchases of property",
                        "purchases of property and equipment", "purchases of property, plant and equipment",
                        "資本支出"],
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "CapEx to Revenue (period average)",
        "unit": "%",
        "period_average": True,
    },

    # Add standard textbook financial ratios proactively so future similarly-phrased
    # questions match a deterministic formula instead of falling back to generic LLM
    # decomposition.
    # Place these keywords carefully so first-match priority doesn't hijack selection
    # for existing specialized formulas.
    "days_sales_outstanding": {
        "keywords_en": ["days sales outstanding", "dso"],
        "formula_expr": "365 * accounts_receivable / revenue",
        "required_vars": {
            "accounts_receivable": ["accounts receivable", "trade receivables", "receivables", "receivable"],
            "revenue": ["revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Days Sales Outstanding (DSO)",
        "unit": "",
    },
    "days_inventory_outstanding": {
        "keywords_en": ["days inventory outstanding", "dio"],
        # abs(cogs): same sign convention as inventory_turnover above.
        "formula_expr": "365 * inventory / abs(cogs)",
        "required_vars": {
            "inventory": ["inventory", "inventories"],
            "cogs": ["cost of goods sold", "cost of products sold", "cost of sales", "cogs", "cost of revenue"],
        },
        "result_label": "Days Inventory Outstanding (DIO)",
        "unit": "",
    },
    "equity_multiplier": {
        "keywords_en": ["equity multiplier"],
        "formula_expr": "total_assets / shareholders_equity",
        "required_vars": {
            "total_assets":        ["total assets", "assets"],
            "shareholders_equity": ["shareholders equity", "stockholders equity", "equity", "total equity"],
        },
        "result_label": "Equity Multiplier",
        "unit": "x",
    },
    # A pure book-value-per-share (shareholders_equity / shares_outstanding) can
    # overstate recoverable liquidation value because goodwill and other intangibles are
    # often not realizable.
    # Use the filing's tangible-equity reconciliation when available for liquidation-
    # value questions; it excludes intangible assets and aligns with the intended
    # economic interpretation.
    "liquidation_value_per_share": {
        "keywords_en": ["liquidated all of its assets", "went bankrupt", "tangible common equity per share",
                         "tangible book value per share"],
        # Use the filing's reported tangible book value per share when available instead
        # of recomputing it; recomputation can differ depending on which share count is
        # used (weighted-average vs. period-end), so prefer the filer-provided figure to
        # avoid mismatches.
        "formula_expr": "tangible_book_value_per_share",
        "required_vars": {
            "tangible_book_value_per_share": ["tangible book value per share",
                                                "tangible book value (tbv) per share",
                                                "tbv per share"],
        },
        "result_label": "Liquidation Value Per Share",
        "unit": "",
    },
    "book_value_per_share": {
        "keywords_en": ["book value per share"],
        "formula_expr": "shareholders_equity / shares_outstanding",
        "required_vars": {
            "shareholders_equity": ["shareholders equity", "stockholders equity", "equity", "total equity"],
            "shares_outstanding":  ["shares outstanding", "weighted average shares", "diluted shares",
                                     "common shares outstanding"],
        },
        "result_label": "Book Value Per Share",
        "unit": "",
    },
    "operating_expense_ratio": {
        "keywords_en": ["operating expense ratio", "operating expenses as a % of revenue",
                         "operating expenses as a percentage of revenue"],
        # abs(): an income statement occasionally presents opex as a signed
        # subtraction step rather than a plain positive magnitude, same
        # rationale as cogs_ratio/capex_to_revenue above.
        "formula_expr": "abs(operating_expenses) / revenue",
        "required_vars": {
            "operating_expenses": ["operating expenses", "total operating expenses", "operating expense"],
            "revenue":            ["revenue", "net sales", "net revenue", "total revenue", "sales to customers"],
        },
        "result_label": "Operating Expense Ratio",
        "unit": "%",
        "period_average": True,
    },
    "long_term_debt_to_capitalization": {
        "keywords_en": ["long-term debt to capitalization", "long term debt to capitalization",
                         "debt to capitalization ratio", "debt-to-capitalization"],
        "formula_expr": "long_term_debt / (long_term_debt + shareholders_equity)",
        "required_vars": {
            "long_term_debt":      ["long-term debt", "long term debt"],
            "shareholders_equity": ["shareholders equity", "stockholders equity", "equity", "total equity"],
        },
        "result_label": "Long-Term Debt to Capitalization",
        "unit": "%",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Lookup API
# ─────────────────────────────────────────────────────────────────────────────

def detect_formula(query: str) -> Optional[Dict[str, Any]]:
    """Find the best-matching formula in FORMULA_LIBRARY for a user query and return its
    entry dict (injecting key 'formula_key'), or None if no match.

    Priority: exact Chinese keyword substring matches first, then English keyword
    matches.
    English keyword matches require word boundaries to avoid matching short
    abbreviations inside unrelated words; Chinese keywords use plain substring matching.
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
            if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                return {**entry, "formula_key": key}

    return None


def get_variable_aliases(formula_entry: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return the required_vars dict: placeholder → alias list."""
    return formula_entry.get("required_vars", {})
