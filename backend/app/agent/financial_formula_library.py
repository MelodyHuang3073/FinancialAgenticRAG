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
        # The RAW DOLLAR metric (current assets minus current
        # liabilities), not the ratio below — was entirely unregistered,
        # so a "does X have positive working capital" question had no
        # formula to compute it from at all. Deliberately keyed on
        # "positive working capital"/"negative working capital" (matching
        # FinanceBench's own recurring phrasing for this exact question
        # type) rather than the bare phrase "working capital", which would
        # also match inside "working capital RATIO" questions below and
        # wrongly hijack those into a subtraction instead of a division.
        # Confirmed real case: American Water Works FY2022 — the model's
        # text never stated the actual -$1,561M figure at all, just a
        # generic "not a relevant metric for this company" non-answer.
        "keywords_zh": ["正的營運資金", "負的營運資金"],
        "keywords_en": ["positive working capital", "negative working capital"],
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
            # "net cash provided by operating activities" leads the list —
            # it's the literal GAAP cash-flow-statement label almost every
            # 10-K uses, and orchestrator._build_formula_retrieval_steps()
            # picks the FIRST ASCII alias as the actual retrieval query
            # text. The generic paraphrase "cash from operations" used to
            # lead instead: with only "cash" as its distinctive token, it
            # scored higher against unrelated balance-sheet cash rows
            # ("Total cash and cash equivalents", "Total cash, cash
            # equivalents and short-term investments") than against the
            # real cash-flow-statement row, so the real row never made the
            # top_k=3 retrieval cutoff (confirmed real case: Adobe FY2015
            # operating cash flow ratio — the sandbox never found
            # cash_from_operations at all despite the row being retrievable
            # with a more specific query).
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
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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
            "revenue":          ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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
            "revenue":    ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
        },
        "result_label": "Net Profit Margin",
        "unit": "%",
        "period_average": True,
    },
    "da_margin": {
        # Was entirely unregistered — same gap class as effective_tax_rate
        # and free_cash_flow below/above. Confirmed real case: AMD FY2015
        # D&A % margin had no formula to compute it from at all, so the
        # sandbox fell back to an unrelated generic value (17.43% instead
        # of gold's 4.2%).
        "keywords_zh": ["折舊攤銷率", "折舊攤提率"],
        "keywords_en": ["d&a margin", "d&a % margin", "depreciation and amortization margin",
                         "depreciation margin", "depreciation and amortization % margin",
                         # FinanceBench's actual recurring phrasing inserts a
                         # parenthetical between "amortization" and "%
                         # margin" (e.g. "...amortization (D&A from cash
                         # flow statement) % margin"), which none of the
                         # contiguous phrases above can match at all.
                         "d&a from cash flow statement"],
        "formula_expr": "depreciation / revenue",
        "required_vars": {
            "depreciation": ["折舊", "depreciation and amortization", "depreciation & amortization",
                             "depreciation"],
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
        },
        "result_label": "D&A Margin",
        "unit": "%",
    },
    "effective_tax_rate": {
        # Was entirely unregistered in FORMULA_LIBRARY (only a
        # question_classifier.py _METRIC_KEYWORDS entry existed, enough to
        # route answer_mode=NUMERIC but with no formula for the sandbox to
        # actually compute). A bare, non-multi_year, non-period_average
        # ratio like gross_margin/operating_margin above — a "how much has
        # the effective tax rate changed between FY X and FY Y" question
        # is handled by pot_reasoner's existing generic 2-year trend-
        # comparison codegen once ANY formula with this expr shape exists,
        # the same mechanism operating_margin's trend questions already
        # use. Confirmed real case: Corning FY2021->FY2022 effective tax
        # rate change came back as "0.0%, no data available" with no
        # formula to fall back on at all.
        "keywords_zh": ["有效稅率", "實質稅率"],
        "keywords_en": ["effective tax rate", "tax rate"],
        # abs(income_tax): a multi-step income statement often shows the
        # tax provision as a signed subtraction from pretax income (e.g.
        # Corning's "Provision for income taxes" is literally "(411)"),
        # same sign convention as cogs/capex above.
        "formula_expr": "abs(income_tax) / pretax_income",
        "required_vars": {
            # "provision for income taxes" leads — same reasoning as
            # cash_from_operations above: the retrieval query is built
            # from the FIRST ASCII alias, and a too-generic "income tax"
            # scores against every tax-related row in the filing (deferred
            # tax assets/liabilities, tax benefit notes, etc.) instead of
            # the real income-statement provision line. Confirmed real
            # case: Corning's actual "Provision for income taxes" row
            # never made the top_k=3 cutoff under the old ordering.
            "income_tax":    ["provision for income taxes", "income tax provision", "income tax expense",
                              "provision for taxes on income", "income taxes", "income tax",
                              "所得稅費用"],
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
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
        },
        "result_label": "EBITDA Margin",
        "unit": "%",
        "period_average": True,
    },
    "ebitda_unadjusted": {
        # A plain "operating income + D&A" sum — distinct from ebitda_margin
        # above (a ratio) and from a fully-adjusted EBITDA (which would add
        # back other non-recurring items); this only covers the specific,
        # simpler definition FinanceBench-style questions ask for by name.
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

    # ── Return Ratios ─────────────────────────────────────────────────────────
    # ROE/ROA's textbook definition divides a single year's net income by
    # the AVERAGE of the balance-sheet item across its two endpoint years
    # (e.g. "net income / average total assets between FY2016 and
    # FY2017") -- this is a component-level 2-point average, structurally
    # identical to DPO's average-accounts-payable pattern, NOT a "3-year
    # average of the ratio itself" (that's what period_average is for).
    # Confirmed real bug: with period_average previously set here, the
    # mere presence of the word "average" in that textbook phrasing
    # wrongly routed this into _gen_period_average_code(), which computed
    # net_income[2016]/total_assets[2016] and net_income[2017]/
    # total_assets[2017] separately and averaged the two RATIOS (4.48%)
    # instead of net_income[2017] / avg(total_assets[2016,2017]) (the
    # ~1.4% actually asked for). multi_year (the _old/_new suffix
    # convention) embeds the averaging directly in formula_expr instead,
    # so it no longer depends on guessing which sense of "average" the
    # query means -- and degrades gracefully to a plain single-year ratio
    # when only one year is named (old == new).
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
        "unit": "%",
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
        "unit": "%",
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
            # Cash-flow-statement financing-activities line -- real 10-Ks
            # commonly report this as the bare word "Dividends" (a
            # negative/outflow value), not a longer descriptive phrase;
            # confirmed real case: Coca-Cola's FY2022 statement of cash
            # flows literally has a row labeled just "Dividends" with no
            # other qualifier. The bare alias is safe here since nothing
            # else on a cash-flow statement collides with it once
            # negation-prefix checking is applied (e.g. "Equity (income)
            # loss -- net of dividends" only substring-matches, scoring
            # lower than an exact "Dividends" row).
            "dividends_paid": ["股利", "現金股利", "支付股利", "dividends paid",
                                "cash dividends paid", "dividends"],
            # Deliberately NOT sharing the plain "net_income" canonical/
            # alias pool: "net income attributable to shareowners/
            # shareholders" and a bare "Consolidated Net Income" (which
            # includes noncontrolling interests) are two DIFFERENT lines
            # that commonly appear on the same statement, and the
            # question asks specifically for the shareholders-attributable
            # figure -- confirmed real case: Coca-Cola's FY2022 income
            # statement has both "Consolidated Net Income" (9,571) and
            # "Net Income Attributable to Shareowners of The Coca-Cola
            # Company" (9,542) as separate rows.
            # "net EARNINGS attributable to" — General Mills (and other
            # companies that use "earnings" rather than "income"
            # throughout their P&L) label this row "Net earnings
            # attributable to General Mills", which none of the
            # "net income attributable..." variants below can match as a
            # substring at all.
            # "net earnings attributable to" placed 2nd (not last): the
            # orchestrator combines the first TWO ascii aliases into the
            # retrieval query, so this must sit early enough to actually
            # be included alongside the "net income..." family — otherwise
            # a company using "earnings" phrasing (General Mills) never
            # gets its own wording into the query at all.
            "net_income_attributable": [
                "歸屬於股東之淨利", "net income attributable to shareowners",
                "net earnings attributable to",
                "net income attributable to shareholders",
                "net income attributable to common shareholders",
                "net income attributable to",
            ],
        },
        "result_label": "Dividend Payout Ratio",
        "unit": "%",
    },
    "retention_ratio": {
        # Was entirely unregistered — fell to a generic evidence-dump
        # fallback that landed on an unrelated row each time (confirmed
        # real case: General Mills FY2022 retention ratio came back as
        # 6.2, built from "Net earnings attributable to redeemable and
        # noncontrolling interests" — an unrelated line the fallback
        # happened to land on). Shares dividends_paid/net_income_attributable
        # aliases with dividend_payout_ratio above — retention ratio is
        # just 1 - payout ratio, same underlying line items.
        "keywords_zh": ["保留盈餘率", "盈餘保留率"],
        "keywords_en": ["retention ratio", "plowback ratio"],
        "formula_expr": "(net_income_attributable - abs(dividends_paid)) / net_income_attributable",
        "required_vars": {
            "dividends_paid": ["股利", "現金股利", "支付股利", "dividends paid",
                                "cash dividends paid", "dividends"],
            # "net EARNINGS attributable to" — General Mills (and other
            # companies that use "earnings" rather than "income"
            # throughout their P&L) label this row "Net earnings
            # attributable to General Mills", which none of the
            # "net income attributable..." variants below can match as a
            # substring at all.
            # "net earnings attributable to" placed 2nd (not last): the
            # orchestrator combines the first TWO ascii aliases into the
            # retrieval query, so this must sit early enough to actually
            # be included alongside the "net income..." family — otherwise
            # a company using "earnings" phrasing (General Mills) never
            # gets its own wording into the query at all.
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
            "revenue":  ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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
            "revenue":      ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
            "total_assets": ["總資產", "total assets", "assets"],
        },
        "result_label": "Asset Turnover",
        "unit": "x",
    },
    "cash_conversion_cycle": {
        # Must be registered BEFORE inventory_turnover/receivables_turnover/
        # dpo below: detect_formula() returns the FIRST formula whose own
        # keyword matches, and a CCC question's text spells out DIO/DSO/DPO
        # as part of defining CCC itself — so the bare "dpo" keyword would
        # otherwise hijack formula selection away from CCC entirely.
        # Confirmed real case: General Mills FY2019 CCC question got
        # answered as if it were plain DPO ("92.7 days"), silently
        # dropping the DIO/DSO terms, because "dpo" matched first.
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
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
            "inv_old": ["存貨", "inventory", "inventories"],
            "inv_new": ["存貨", "inventory", "inventories"],
            # Bare "receivables"/"receivable" — General Mills' balance
            # sheet just says "Receivables", with no "accounts"/"trade"
            # qualifier, which neither of the other two aliases can match
            # as a substring.
            "ar_old":  ["應收帳款", "accounts receivable", "trade receivables", "receivables", "receivable"],
            "ar_new":  ["應收帳款", "accounts receivable", "trade receivables", "receivables", "receivable"],
            "ap_old":  ["應付帳款", "accounts payable"],
            "ap_new":  ["應付帳款", "accounts payable"],
        },
        "result_label": "Cash Conversion Cycle (CCC)",
        "unit": "",
        "multi_year": True,
    },
    # Two entries for the same ratio, split purely on whether the question
    # itself asks for an AVERAGE inventory base. FinanceBench isn't
    # internally consistent here: some inventory-turnover questions spell
    # out "average inventory between FY X and FY Y" (e.g. Kraft Heinz) and
    # expect the 2-endpoint average like dpo/roa/roe use; others give no
    # such instruction and their own gold answer only lines up against
    # plain year-end inventory (e.g. AES Corporation: 9.5x only matches
    # cogs / ending_inventory, not cogs / average_inventory). Since a
    # single formula_expr can't honour both conventions at once, detect_formula
    # routes on the explicit "average inventory" phrase — this entry MUST
    # stay ordered before the plain one below so its more specific keyword
    # wins the "first match" scan.
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
        # abs() on cogs: some companies' income statements present cost
        # lines as a subtraction step with a parenthesised/negative value
        # (e.g. AES Corporation's "Total cost of sales" row is literally
        # "(10,069)" in the source table) rather than a plain positive
        # magnitude. COGS is conceptually always a positive cost magnitude
        # for this ratio regardless of how one company's statement signs
        # it, so this must be a general abs(), not a per-company patch.
        # Confirmed real case: AES FY2022 inventory turnover computed as
        # -9.54x instead of +9.5x purely from this sign convention.
        "formula_expr": "abs(cogs) / inventory",
        "required_vars": {
            # "cost of products sold" — the phrasing pharma/consumer-health
            # companies (e.g. Johnson & Johnson) use instead of "cost of
            # goods sold"/"cost of sales" — was missing here, so their real
            # COGS row never matched any alias at all and the formula fell
            # through to the generic evidence-dump fallback instead of a
            # targeted calculation.
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
            "revenue":             ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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
        # Was entirely unregistered — "free cash flow" only existed as a
        # _ITEM_TAXONOMY direct-lookup target in pot_reasoner.py, which
        # only works when a filing already prints its own "Free cash flow"
        # line. When a question instead gives its own definition ("cash
        # from operations - capex", the standard FinanceBench phrasing),
        # there was no formula to compute it from, so the sandbox fell
        # back to a generic evidence dump and picked an unrelated value.
        # Confirmed real case: General Mills FY2020 FCF fell back to
        # "Capital expenditures (2020)" alone (460.8) as the "result".
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
        # "change in revenue" / "year-over-year change in revenue" is
        # FinanceBench's own recurring phrasing for this exact metric —
        # none of the "growth"/"yoy"/"increase" variants above can match
        # it at all. Confirmed real case: Amazon FY2016->FY2017 revenue
        # change had no formula match, so the sandbox never computed it
        # despite "Net sales" being cleanly retrievable for both years.
        "keywords_en": ["revenue growth", "revenue yoy", "sales growth", "revenue increase",
                         "change in revenue", "change in total revenue"],
        "formula_expr": "(revenue_new - revenue_old) / revenue_old * 100",
        "required_vars": {
            "revenue_new": ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
            "revenue_old": ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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
        # abs(capex): "Purchases of property and equipment" is a cash
        # outflow, so the cash flow statement often lists it as a
        # parenthesised/negative figure. CapEx is conceptually always a
        # positive spend magnitude for this ratio.
        "formula_expr": "abs(capex) / revenue",  # evaluated once per year, then averaged
        "required_vars": {
            "capex":   ["capital expenditures", "capital expenditure", "purchases of property",
                        "purchases of property and equipment", "purchases of property, plant and equipment",
                        "資本支出"],
            "revenue": ["營業收入", "revenue", "net sales", "net revenue", "total revenue"],
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

    English keywords are matched on a WORD BOUNDARY, not a bare substring
    — several formula abbreviations are short enough to appear inside
    ordinary English words (confirmed real case: "roa" — the keyword for
    Return on Assets — matched inside "Approach the question asked by...",
    a boilerplate instruction sentence with no relation to ROA at all,
    causing the whole formula, and thus the whole calculation, to be
    wrong). Chinese keywords are left as plain substring matches since
    Chinese text has no whitespace word boundaries to anchor on, and the
    library's Chinese keywords are multi-character phrases unlikely to
    collide with unrelated text the way 3-letter English abbreviations do.
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
