"""
company_line_item_overrides.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Company-specific financial line-item alias overrides.

Format:
    { "<company_name_lowercase>": { "<canonical_metric>": ["alias1", "alias2"] } }

These override the global _ITEM_TAXONOMY in pot_reasoner.py when a specific
company is identified in the evidence.  Company names are matched case-
insensitively as substrings (e.g. "activision blizzard 2022 10-k" still
matches the "activision blizzard" key).

HOW TO EXTEND:
  Add an entry for the company (key must be lowercase), then add canonical
  metric keys mapping to a list of company-specific aliases.
  No code changes needed elsewhere — pot_reasoner picks this up automatically.
"""
from typing import Dict, List

# Primary override table
COMPANY_LINE_ITEM_OVERRIDES: Dict[str, Dict[str, List[str]]] = {

    # Activision Blizzard
    "activision blizzard": {
        "revenue":      ["total net revenues", "net revenues"],
        "capex":        ["capital expenditures", "purchases of property and equipment"],
        "operating_cf": ["net cash provided by operating activities"],
        "net_income":   ["net income attributable to activision blizzard"],
    },
    # 3M Company
    "3m": {
        "revenue":   ["net sales"],
        "op_income": ["operating income"],
        "capex":     ["purchases of property, plant and equipment"],
    },
    # Apple
    "apple": {
        "revenue":    ["net sales", "total net sales"],
        "capex":      ["purchases of property, plant and equipment"],
        "rd_expense": ["research and development"],
    },
    # Microsoft
    "microsoft": {
        "revenue":    ["total revenue", "revenue"],
        "capex":      ["additions to property and equipment"],
        "rd_expense": ["research and development"],
    },
    # Amazon
    "amazon": {
        "revenue":   ["net product sales", "net service sales", "total net sales"],
        "op_income": ["income from operations"],
        "capex":     ["purchases of property and equipment", "capital expenditures"],
    },
    # TSMC
    "tsmc": {
        "revenue":   ["net revenue", "net revenues", "\u71df\u696d\u6536\u5165"],
        "capex":     ["capital expenditures", "\u8cc7\u672c\u652f\u51fa"],
        "op_income": ["income from operations", "operating income", "\u71df\u696d\u5229\u76ca"],
    },
    # Pfizer
    "pfizer": {
        "revenue":    ["total revenues", "revenues"],
        "rd_expense": ["research and development expenses"],
    },
    # JPMorgan Chase
    "jpmorgan": {
        "revenue":    ["total net revenue", "net revenue"],
        "net_income": ["net income applicable to common equity"],
    },
    # Johnson & Johnson
    "johnson": {
        "revenue":    ["total sales", "sales to customers"],
        "rd_expense": ["research and development expense"],
    },
    # Caterpillar
    "caterpillar": {
        "revenue": ["total sales and revenues",
                    "sales of machinery, energy and transportation"],
        "capex":   ["capital expenditures for equipment"],
    },
    # Costco
    "costco": {
        "revenue":   ["net sales", "total revenue"],
        "op_income": ["operating income"],
    },
    # Boeing
    "boeing": {
        "revenue":   ["total revenues", "revenues"],
        "op_income": ["earnings from operations"],
    },
    # Tesla
    "tesla": {
        "revenue": ["total revenues", "automotive revenues"],
        "capex":   ["purchases of property and equipment"],
    },
    # Walmart
    "walmart": {
        "revenue":   ["net sales", "total revenues"],
        "op_income": ["operating income"],
    },
}


def get_overrides_for_company(company_name: str) -> Dict[str, List[str]]:
    """
    Return the alias overrides dict for the given company name.
    Matching is case-insensitive substring: "3M_2022_10K" -> matches "3m".
    Returns an empty dict if no overrides are registered.
    """
    name_lower = company_name.lower().strip()
    for key, overrides in COMPANY_LINE_ITEM_OVERRIDES.items():
        if key in name_lower:
            return overrides
    return {}
