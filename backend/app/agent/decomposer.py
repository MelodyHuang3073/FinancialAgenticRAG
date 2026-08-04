"""
QueryDecomposer (FinAgent-RAG Section 3.3)

Decomposes a complex financial question into sequential retrieval + computation steps.
Each retrieval step targets a specific (metric × year) pair so the orchestrator can
retrieve the right table rows one-by-one, rather than sending one big generic query.

Key fix: ratio/formula metrics are expanded to their component line items so that
ALL required rows are retrieved (e.g. "gross margin" → fetch Gross Profit + Revenue).
"""

import re
from typing import List, Dict, Any, Optional

# Canonical metric → display name used in retrieval queries
# Keeps both English and Chinese variants so BM25 can match linearized table content
_METRIC_NAMES: Dict[str, str] = {
    "revenue":       "Revenue Net Revenue Total Revenue 營業收入 營收",
    "gross_profit":  "Gross Profit 營業毛利 毛利",
    "gross_margin":  "Gross Margin 毛利率 Gross Profit Revenue",
    "op_income":     "Operating Income Operating Profit 營業利益 營業淨利",
    "op_expense":    "Operating Expense 營業費用",
    "op_margin":     "Operating Margin 營業利益率 Operating Income Revenue",
    "net_income":    "Net Income Net Profit 本期淨利 淨利",
    "eps":           "EPS Earnings Per Share 每股盈餘",
    "rd_expense":    "R&D Research Development 研發費用",
    "sga":           "SG&A Selling General Administrative 推銷管理費用",
    "cost_revenue":  "Cost of Revenue COGS Cost of Goods Sold 營業成本",
    "total_assets":  "Total Assets 總資產",
    "total_liab":    "Total Liabilities 總負債",
    "equity":        "Shareholders Equity 股東權益 Total Equity",
    "cash":          "Cash Equivalents 現金及約當現金",
    "capex":         "Capital Expenditure CapEx 資本支出",
    "depreciation":  "Depreciation Amortization 折舊",
    "ebitda":        "EBITDA",
    "roe":           "Return on Equity ROE Net Income Equity",
    "roa":           "Return on Assets ROA Net Income Total Assets",
    "dividend":      "Dividend 股利",
    "data_center":   "Data Center Revenue",
    "fcf":           "Free Cash Flow FCF 自由現金流",
    # balance sheet / liquidity
    "current_assets": "Current Assets 流動資產",
    "current_liab":   "Current Liabilities 流動負債",
    "inventory":      "Inventory Inventories 存貨",
    "accounts_rec":   "Accounts Receivable 應收帳款 Receivables",
    "operating_cf":   "Operating Cash Flow Cash from Operations CFO",
    "investing_cf":   "Investing Activities Investing Cash Flow",
    "financing_cf":   "Financing Activities Financing Cash Flow",
}

# Maps a composite ratio/margin metric → the individual line-item metrics that
# must EACH be retrieved before the calculation can be performed.
# This is the critical table that drives multi-step retrieval for ratio questions.
_RATIO_COMPONENTS: Dict[str, List[str]] = {
    "gross_margin":       ["gross_profit", "revenue"],
    "op_margin":          ["op_income", "revenue"],
    "net_margin":         ["net_income", "revenue"],
    "roe":                ["net_income", "equity"],
    "roa":                ["net_income", "total_assets"],
    "quick_ratio":        ["current_assets", "inventory", "current_liab"],
    "current_ratio":      ["current_assets", "current_liab"],
    "debt_equity":        ["total_liab", "equity"],
    "ebitda":             ["op_income", "depreciation"],
    "fcf":                ["operating_cf", "capex"],
    "working_capital":    ["current_assets", "current_liab"],
    "interest_coverage":  ["op_income"],  # interest often in MD&A notes
    "rd_pct_revenue":     ["rd_expense", "revenue"],
    "sga_pct_revenue":    ["sga", "revenue"],
    "cost_ratio":         ["cost_revenue", "revenue"],
    "capex_pct_revenue":  ["capex", "revenue"],
}


class QueryDecomposer:
    """
    Decomposes a complex financial question into an ordered list of sub-tasks.
    For ratio/margin metrics, automatically expands to component line-items
    so that ALL required data rows are fetched before PoT calculation.
    """

    def decompose(
        self,
        query: str,
        target_metrics: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
        entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # ── Defaults / normalisation ──────────────────────────────────────────
        if not years:
            # Strip optional "FY" prefix so "FY2023" → "2023" matches corpus data
            years = sorted(set(re.findall(r'(?:FY)?(20\d\d)', query)))
        else:
            years = [re.sub(r'^FY', '', y) for y in years]

        target_metrics = list(target_metrics or [])
        entity = (entity or "").strip()

        # ── Expand ratio/formula metrics to component line items ─────────────
        # KEY FIX: "gross_margin" → ["gross_profit", "revenue"]
        # This ensures we fetch BOTH rows needed to compute the margin, not just one.
        expanded_metrics: List[str] = []
        for m in target_metrics:
            components = _RATIO_COMPONENTS.get(m)
            if components:
                for c in components:
                    if c not in expanded_metrics:
                        expanded_metrics.append(c)
            else:
                if m not in expanded_metrics:
                    expanded_metrics.append(m)

        if not expanded_metrics:
            expanded_metrics = target_metrics

        steps: List[Dict[str, Any]] = []
        step_num = 0

        # ── Case 1: Known metrics AND years → one step per (metric × year) ──
        if expanded_metrics and years:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                for year in years:
                    step_num += 1
                    q = f"{entity} {metric_display} {year}".strip()
                    steps.append({
                        "step": step_num,
                        "type": "retrieval",
                        "query": q,
                        "target_metric": metric,
                        "target_year": year,
                    })

        # ── Case 2: Known metrics, no specific year ───────────────────────────
        elif expanded_metrics:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                step_num += 1
                q = f"{entity} {metric_display}".strip()
                steps.append({
                    "step": step_num,
                    "type": "retrieval",
                    "query": q,
                    "target_metric": metric,
                    "target_year": None,
                })

        # ── Case 3: Years present, no explicit metric → keyword fallback ──────
        elif years:
            q_lower = query.lower()
            if "cagr" in q_lower or "複合成長率" in q_lower:
                y1, y2 = years[0], years[-1]
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y1}".strip(),
                               "target_metric": None, "target_year": y1})
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y2}".strip(),
                               "target_metric": None, "target_year": y2})
            else:
                for y in years:
                    step_num += 1
                    steps.append({"step": step_num, "type": "retrieval",
                                  "query": f"{entity} {query} {y}".strip(),
                                  "target_metric": None, "target_year": y})

        # ── Case 4: Fully generic ─────────────────────────────────────────────
        else:
            step_num += 1
            steps.append({
                "step": step_num,
                "type": "retrieval",
                "query": query,
                "target_metric": None,
                "target_year": None,
            })

        # ── Final computation step (always appended) ──────────────────────────
        step_num += 1
        steps.append({
            "step": step_num,
            "type": "computation",
            "query": "Compute the final answer from accumulated retrieved financial data.",
            "target_metric": None,
            "target_year": None,
        })

        return steps
