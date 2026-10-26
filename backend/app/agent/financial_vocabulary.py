"""Financial vocabulary -- the single source of truth for metric names.

Two different kinds of vocabulary live here, kept deliberately separate
because they answer different questions:

1. QUESTION_METRICS -- what the USER is asking about. Keyed by a question-
   level metric (which may be a derived ratio such as gross_margin or roe).
   Each record holds:
     keywords      how a question phrases the metric (English and Chinese --
                   users may ask in either language)
     statement     which financial statement is the metric's primary source
     search_terms  how FILINGS word it, used to build retrieval queries
     components    for a derived metric, the base metrics it is built from

2. LINE_ITEM_ALIASES / COMPOSITE_LINE_ITEM_ALIASES -- how a base line item
   is LABELED in a filing's own tables. English only: every filing is an
   English SEC document. Order is significant (pot_reasoner's canonical
   matcher resolves ties by first occurrence).

Consumers derive their own lookup views from these tables instead of
keeping private copies: question_classifier (keywords, statement,
search_terms), decomposer (search_terms, components), pot_reasoner
(line-item aliases), lexical_retriever (alias groups, via pot_reasoner).

Formula-specific variable bindings stay in financial_formula_library.py:
a formula's variable can intentionally use a narrower or broader alias set
than the generic line item (e.g. one specific balance-sheet caption), so
those are part of each formula's own definition, not shared vocabulary.
"""
import os
from typing import Any, Dict, List


QUESTION_METRICS: Dict[str, Dict[str, Any]] = {
    'revenue': {
        'keywords': ['營業收入', 'revenue', 'sales', 'net sales', 'total revenue', '營收'],
        'statement': 'income_statement',
        'search_terms': 'Revenue Net Revenue Total Revenue',
    },
    'gross_profit': {
        'keywords': ['營業毛利', 'gross profit', '毛利'],
        'statement': 'income_statement',
        'search_terms': 'Gross Profit Cost of Products Sold Cost of Goods Sold Cost of Sales COGS',
    },
    'gross_margin': {
        'keywords': ['毛利率', 'gross margin'],
        'statement': 'income_statement',
        'search_terms': 'Gross Margin Gross Profit Revenue Cost of Products Sold Cost of Goods Sold Cost of Sales COGS',
        'components': ['gross_profit', 'revenue'],
    },
    'op_income': {
        'keywords': ['營業利益', 'operating income', 'operating profit', '營業淨利'],
        'statement': 'income_statement',
        'search_terms': 'Operating Income Operating Profit',
    },
    'op_expense': {
        'keywords': ['營業費用', 'operating expense', 'opex'],
        'statement': 'income_statement',
        'search_terms': 'Operating Expense',
    },
    'op_margin': {
        'keywords': ['營業利益率', 'operating margin', 'op margin'],
        'statement': 'income_statement',
        'search_terms': 'Operating Margin Operating Income Revenue',
        'components': ['op_income', 'revenue'],
    },
    'net_income': {
        'keywords': ['本期淨利', 'net income', 'net profit', '淨利', '淨利潤'],
        'statement': 'income_statement',
        'search_terms': 'Net Income Net Profit',
    },
    'eps': {
        'keywords': ['每股盈餘', 'eps', 'earnings per share'],
        'statement': 'income_statement',
        'search_terms': 'EPS Earnings Per Share',
    },
    'rd_expense': {
        'keywords': ['研發費用', 'r&d', 'research', '研發'],
        'statement': 'income_statement',
        'search_terms': 'R&D Research Development',
    },
    'sga': {
        'keywords': ['推銷與管理費用', 'sg&a', 'selling general'],
        'statement': 'income_statement',
        'search_terms': 'SG&A Selling General Administrative',
    },
    'cost_revenue': {
        'keywords': ['營業成本', 'cost of revenue', 'cogs', 'cost of goods sold', 'cost of sales'],
        'statement': 'income_statement',
        'search_terms': 'Cost of Revenue COGS Cost of Goods Sold',
    },
    'total_assets': {
        'keywords': ['總資產', 'total assets'],
        'statement': 'balance_sheet',
        'search_terms': 'Total Assets',
    },
    'total_liab': {
        'keywords': ['總負債', 'total liabilities'],
        'statement': 'balance_sheet',
        'search_terms': 'Total Liabilities',
    },
    'equity': {
        'keywords': ['股東權益', 'shareholders equity', 'stockholders equity', 'equity'],
        'statement': 'balance_sheet',
        'search_terms': 'Shareholders Equity Total Equity',
    },
    'cash': {
        'keywords': ['現金及約當現金', 'cash and cash equivalents', 'cash & equivalents', 'cash equivalents'],
        'statement': 'balance_sheet',
        'search_terms': 'Cash Equivalents',
    },
    'capex': {
        'keywords': ['資本支出', 'capital expenditure', 'capex'],
        'statement': 'cash_flow',
        'search_terms': 'Capital Expenditure CapEx Capital Spending',
    },
    'ppe': {
        'keywords': ['pp&e', 'ppne', 'property, plant and equipment', 'property, plant, and equipment', 'property and equipment', 'fixed assets', '不動產、廠房及設備', '固定資產'],
        'statement': 'balance_sheet',
        'search_terms': 'Property Plant and Equipment PP&E Fixed Assets',
    },
    'depreciation': {
        'keywords': ['折舊', 'depreciation', 'amortization'],
        'statement': 'cash_flow',
        'search_terms': 'Depreciation Amortization',
    },
    'ebitda': {
        'keywords': ['ebitda'],
        'statement': 'income_statement',
        'search_terms': 'EBITDA',
        'components': ['op_income', 'depreciation'],
    },
    'roe': {
        'keywords': ['roe', 'return on equity'],
        'statement': 'income_statement',
        'search_terms': 'Return on Equity ROE Net Income Equity',
        'components': ['net_income', 'equity'],
    },
    'roa': {
        'keywords': ['roa', 'return on assets'],
        'statement': 'income_statement',
        'search_terms': 'Return on Assets ROA Net Income Total Assets',
        'components': ['net_income', 'total_assets'],
    },
    'dividend': {
        'keywords': ['股利', 'dividend', 'dividends'],
        'statement': 'income_statement',
        'search_terms': 'Dividend',
    },
    'data_center': {
        'keywords': ['data center', '資料中心'],
        'search_terms': 'Data Center Revenue',
    },
    'fcf': {
        'keywords': ['自由現金流', 'free cash flow', 'fcf'],
        'statement': 'cash_flow',
        'search_terms': 'Free Cash Flow FCF',
        'components': ['operating_cf', 'capex'],
    },
    'quick_ratio': {
        'keywords': ['quick ratio', 'acid-test ratio', 'acid test'],
        'statement': 'balance_sheet',
        'search_terms': 'Quick Ratio Acid Test Current Assets Inventory Current Liabilities',
        'components': ['current_assets', 'inventory', 'current_liab'],
    },
    'current_ratio': {
        'keywords': ['current ratio', '流動比率'],
        'statement': 'balance_sheet',
        'search_terms': 'Current Ratio Current Assets Current Liabilities',
        'components': ['current_assets', 'current_liab'],
    },
    'current_assets': {
        'keywords': ['current assets', '流動資產'],
        'statement': 'balance_sheet',
        'search_terms': 'Current Assets',
    },
    'current_liab': {
        'keywords': ['current liabilities', '流動負債'],
        'statement': 'balance_sheet',
        'search_terms': 'Current Liabilities',
    },
    'inventory': {
        'keywords': ['inventory', '存貨', 'inventories'],
        'statement': 'balance_sheet',
        'search_terms': 'Inventory Inventories',
    },
    'accounts_rec': {
        'keywords': ['accounts receivable', '應收帳款', 'receivables', 'net ar'],
        'statement': 'balance_sheet',
        'search_terms': 'Accounts Receivable Receivables',
    },
    'accounts_payable': {
        'keywords': ['accounts payable', '應付帳款', 'payables'],
        'statement': 'balance_sheet',
        'search_terms': 'Accounts Payable',
    },
    'dpo': {
        'keywords': ['days payable outstanding', 'dpo'],
        'statement': 'balance_sheet',
        'components': ['accounts_payable', 'cost_revenue', 'inventory'],
    },
    'dividends_paid': {
        'keywords': ['dividends paid', 'cash dividends paid', 'dividends', '股利', '現金股利'],
        'statement': 'cash_flow',
    },
    'net_income_attributable': {
        'keywords': ['net income attributable to shareowners', 'net income attributable to shareholders', 'net income attributable to'],
        'statement': 'income_statement',
    },
    'dividend_payout_ratio': {
        'keywords': ['dividend payout ratio', 'payout ratio', '股利發放率'],
        'statement': 'income_statement',
    },
    'net_debt': {
        'keywords': ['net debt', '淨負債'],
        'statement': 'balance_sheet',
        'search_terms': 'Net Debt',
    },
    'debt': {
        'keywords': ['debt', 'long-term debt', 'short-term debt', '借款', '負債'],
        'statement': 'balance_sheet',
        'search_terms': 'Debt Long-term Debt Short-term Debt Borrowings',
    },
    'working_capital': {
        'keywords': ['working capital', '營運資金'],
        'statement': 'balance_sheet',
        'search_terms': 'Working Capital',
        'components': ['current_assets', 'current_liab'],
    },
    'fixed_asset_turnover': {
        'keywords': ['fixed asset turnover', 'fixed-asset turnover', 'net ppe turnover'],
        'statement': 'balance_sheet',
        'search_terms': 'Property Plant and Equipment PP&E Fixed Assets Revenue',
        'components': ['revenue', 'ppe'],
    },
    'operating_cash_flow_ratio': {
        'keywords': ['operating cash flow ratio', 'cash flow ratio', 'ocf ratio'],
        'statement': 'cash_flow',
        'search_terms': 'Cash from Operations Operating Cash Flow Current Liabilities',
        'components': ['operating_cf', 'current_liab'],
    },
    'debt_equity': {
        'keywords': ['debt-to-equity', 'd/e ratio', 'leverage', 'financial leverage'],
        'statement': 'balance_sheet',
        'search_terms': 'Debt-to-Equity Leverage Financial Leverage',
        'components': ['total_liab', 'equity'],
    },
    'interest_coverage': {
        'keywords': ['interest coverage', 'times interest earned'],
        'statement': 'balance_sheet',
        'search_terms': 'Interest Coverage Times Interest Earned',
        'components': ['op_income'],
    },
    'roce': {
        'keywords': ['roce', 'return on capital employed'],
        'statement': 'income_statement',
        'search_terms': 'ROCE Return on Capital Employed',
    },
    'roic': {
        'keywords': ['roic', 'return on invested capital'],
        'statement': 'income_statement',
        'search_terms': 'ROIC Return on Invested Capital',
    },
    'net_margin': {
        'keywords': ['net margin', 'net profit margin', 'profit margin'],
        'statement': 'income_statement',
        'search_terms': 'Net Margin Net Profit Margin Profit Margin',
        'components': ['net_income', 'revenue'],
    },
    'tax_rate': {
        'keywords': ['effective tax rate', 'tax rate'],
        'statement': 'income_statement',
    },
    'operating_cf': {
        'keywords': ['operating cash flow', 'cash from operations', 'cfo', 'cash from operating', 'cash provided by operating', 'operating activities', 'generate from operations', 'generated by operating', 'operating cash'],
        'statement': 'cash_flow',
        'search_terms': 'Operating Cash Flow Cash from Operations CFO',
    },
    'investing_cf': {
        'keywords': ['investing activities', 'investing cash flow'],
        'statement': 'cash_flow',
        'search_terms': 'Investing Activities Investing Cash Flow',
    },
    'financing_cf': {
        'keywords': ['financing activities', 'financing cash flow'],
        'statement': 'cash_flow',
        'search_terms': 'Financing Activities Financing Cash Flow',
    },
    'restructuring_costs': {
        'keywords': ['restructuring charges', 'restructuring costs', 'restructuring and impairment charges', '重組費用', '重組成本'],
        'statement': 'income_statement',
        'search_terms': 'Restructuring Charges Restructuring Costs Restructuring and Impairment Charges',
    },
    'rd_pct_revenue': {
        'components': ['rd_expense', 'revenue'],
    },
    'sga_pct_revenue': {
        'components': ['sga', 'revenue'],
    },
    'cost_ratio': {
        'components': ['cost_revenue', 'revenue'],
    },
    'capex_pct_revenue': {
        'components': ['capex', 'revenue'],
    },
    'inventory_turnover': {
        'components': ['cost_revenue', 'inventory'],
    },
}


LINE_ITEM_ALIASES: Dict[str, List[str]] = {
    'revenue': ['total revenue', 'net revenue', 'net sales', 'revenue', 'sales to customers', 'total revenues', 'total net revenues', 'net revenues', 'total net revenue', 'total net sales', 'total sales'],
    'gross_profit': ['gross profit', 'gross margin amount'],
    'cost_of_revenue': ['cost of revenue', 'cost of goods sold', 'cogs', 'cost of sales', 'cost of products', 'cost of services', 'cost of products sold'],
    'op_expense': ['operating expenses', 'operating expense', 'opex'],
    'op_income': ['operating income', 'operating profit', 'operating earnings', 'income from operations', 'earnings from operations', 'ebit'],
    'rd_expense': ['research & development', 'r&d', 'research and development', 'research and development expense', 'research and development expenses'],
    'sga': ['sg&a', 'selling general', 'selling and marketing'],
    'restructuring_costs': ['restructuring and impairment charges', 'restructuring charges', 'restructuring costs'],
    'net_income_btax': ['net income before tax', 'income before tax', 'pretax income'],
    'income_tax': ['income tax', 'tax expense'],
    'net_income': ['net income', 'net profit', 'net earnings'],
    'eps': ['eps', 'earnings per share', 'diluted eps'],
    'shares_outstanding': ['shares outstanding', 'weighted average shares', 'diluted shares', 'common shares outstanding'],
    'cash': ['cash and cash equivalents', 'cash & equivalents'],
    'short_term_investments': ['short-term investments', 'short term investments', 'marketable securities'],
    'total_assets': ['total assets'],
    'current_assets': ['total current assets', 'current assets'],
    'inventory': ['inventory', 'inventories'],
    'accounts_rec': ['accounts receivable', 'trade receivables', 'receivables', 'receivable'],
    'accounts_payable': ['accounts payable', 'trade payables', 'payables', 'payable'],
    'current_liab': ['total current liabilities', 'current liabilities'],
    'total_liab': ['total liabilities'],
    'long_term_debt': ['long-term debt', 'long term debt'],
    'interest_expense': ['interest expense'],
    'goodwill': ['goodwill'],
    'intangible_assets': ['intangible assets', 'other intangible assets'],
    'retained_earnings': ['retained earnings', 'accumulated deficit'],
    'ppe': ['property, plant and equipment, net', 'property, plant, and equipment, net', 'property and equipment, net', 'property, plant and equipment', 'property, plant, and equipment', 'property and equipment', 'net ppe', 'pp&e', 'fixed assets'],
    'equity': ['total equity', 'total shareholders equity', 'stockholders equity'],
    'operating_cf': ['operating cash flow', 'cash from operations', 'cash provided by operating', 'cash provided (used) by operations', 'cash provided (used) by operating activities'],
    'investing_cf': ['cash used in investing', 'cash provided by investing', 'investing activities'],
    'financing_cf': ['cash used in financing', 'cash provided by financing', 'financing activities'],
    'capex': ['capital expenditure', 'purchases of ppe', 'purchases of property and equipment', 'purchases of property, plant and equipment', 'additions to property and equipment', 'capital spending'],
    'depreciation': ['depreciation and amortization', 'depreciation & amortization', 'depreciation', 'amortization', 'd&a'],
    'fcf': ['free cash flow', 'fcf'],
    'dividends_paid': ['dividends paid', 'cash dividends paid', 'dividends'],
    'dividends_per_share': ['dividends declared per share', 'dividends declared per common share', 'cash dividends declared per share', 'dividends per common share', 'dividends per share', 'dividend per share'],
    'data_center_rev': ['data center revenue', 'data center'],
}


#: A base quantity some filings report only as several sub-lines (e.g.
#: inventory split into raw materials / work in process / finished goods),
#: summed when no single total row exists. Keyed by formula placeholder.
COMPOSITE_LINE_ITEM_ALIASES: Dict[str, List[str]] = {
    'inventory': ['raw materials', 'raw materials and supplies', 'work in process', 'work in process and finished goods', 'finished goods', 'merchandise inventory'],
    'total_borrowings_old': ['long-term debt', 'current portion of long-term debt', 'short-term borrowings', 'current maturities of long-term debt'],
    'total_borrowings_new': ['long-term debt', 'current portion of long-term debt', 'short-term borrowings', 'current maturities of long-term debt'],
}


#: The primary financial statement each line item belongs to. When several
#: evidence rows carry the same label, the row from this statement is the
#: one a financial ratio means; a same-labeled row in a note or a
#: supplementary table (segment, investee or schedule figures) is not.
#: Values match the page `section` tags assigned at ingestion. Line items
#: that are not presented on a primary statement are left out.
LINE_ITEM_STATEMENT: Dict[str, str] = {
    'revenue': 'income_statement',
    'cost_of_revenue': 'income_statement',
    'gross_profit': 'income_statement',
    'op_expense': 'income_statement',
    'rd_expense': 'income_statement',
    'sga': 'income_statement',
    'restructuring_costs': 'income_statement',
    'op_income': 'income_statement',
    'interest_expense': 'income_statement',
    'net_income_btax': 'income_statement',
    'income_tax': 'income_statement',
    'net_income': 'income_statement',
    'eps': 'income_statement',
    'cash': 'balance_sheet',
    'short_term_investments': 'balance_sheet',
    'accounts_rec': 'balance_sheet',
    'inventory': 'balance_sheet',
    'current_assets': 'balance_sheet',
    'ppe': 'balance_sheet',
    'goodwill': 'balance_sheet',
    'intangible_assets': 'balance_sheet',
    'total_assets': 'balance_sheet',
    'accounts_payable': 'balance_sheet',
    'current_liab': 'balance_sheet',
    'long_term_debt': 'balance_sheet',
    'total_liab': 'balance_sheet',
    'retained_earnings': 'balance_sheet',
    'equity': 'balance_sheet',
    'operating_cf': 'cash_flow',
    'investing_cf': 'cash_flow',
    'financing_cf': 'cash_flow',
    'capex': 'cash_flow',
    'depreciation': 'cash_flow',
    'dividends_paid': 'cash_flow',
}


def metric_view(field: str) -> Dict[str, Any]:
    """{metric: value} for every QUESTION_METRICS record that defines
    `field`, in QUESTION_METRICS order."""
    return {k: v[field] for k, v in QUESTION_METRICS.items() if field in v}


if os.environ.get("ABLATE_DOMAIN") == "1":
    QUESTION_METRICS.clear()
    LINE_ITEM_ALIASES.clear()
    COMPOSITE_LINE_ITEM_ALIASES.clear()
    LINE_ITEM_STATEMENT.clear()
