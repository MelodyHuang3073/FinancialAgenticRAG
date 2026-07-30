"""
QueryDecomposer (FinAgent-RAG Section 3.3)

Decomposes a complex financial question into sequential retrieval + computation steps.
Each retrieval step targets a specific (metric × year) pair so the orchestrator can
retrieve the right table rows one-by-one, rather than sending one big generic query.
"""

import re
from typing import List, Dict, Any, Optional

# Canonical metric → display name used in retrieval queries
# Keeps both English and Chinese variants so BM25 can match linearized table content
_METRIC_NAMES: Dict[str, str] = {
    "revenue":       "Revenue Net Revenue 營業收入 營收",
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
    "equity":        "Shareholders Equity 股東權益",
    "cash":          "Cash Equivalents 現金及約當現金",
    "capex":         "Capital Expenditure CapEx 資本支出",
    "depreciation":  "Depreciation Amortization 折舊",
    "ebitda":        "EBITDA",
    "roe":           "Return on Equity ROE Net Income Equity",
    "roa":           "Return on Assets ROA Net Income Total Assets",
    "dividend":      "Dividend 股利",
    "data_center":   "Data Center Revenue",
    "fcf":           "Free Cash Flow FCF 自由現金流",
}


class QueryDecomposer:
    """
    Decomposes a complex financial question into an ordered list of sub-tasks:

    * retrieval  — search the vector store for a specific (metric, year) slice
    * computation — aggregate the retrieved data and compute the final answer

    Unlike the old keyword-only decomposer, this one accepts structured context
    (target_metrics, years, entity) from the FinanceBenchClassifier so that every
    retrieval step is *precise* and directly maps to a table row.
    """

    def decompose(
        self,
        query: str,
        target_metrics: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
        entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        query          : Original user question (used as fallback query text).
        target_metrics : Canonical metric keys from FinanceBenchClassifier.
        years          : Year strings extracted from the question (e.g. ["2021","2022"]).
        entity         : Company/entity name resolved by the orchestrator.

        Returns
        -------
        List of step dicts, each with keys: step, type, query, target_metric, target_year.
        Last element is always a "computation" step.
        """
        # ── Defaults ──────────────────────────────────────────────────────────
        if not years:
            years = sorted(set(re.findall(r'20\d\d', query)))
        target_metrics = target_metrics or []
        entity = (entity or "").strip()

        steps: List[Dict[str, Any]] = []
        step_num = 0

        # ── Case 1: Known metrics AND years → one step per (metric × year) ──
        if target_metrics and years:
            for metric in target_metrics:
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

        # ── Case 2: Known metrics, no specific year ────────────────────────
        elif target_metrics:
            for metric in target_metrics:
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

        # ── Case 3: Years present, no explicit metric → keyword fallback ──
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

        # ── Case 4: Fully generic ─────────────────────────────────────────
        else:
            step_num += 1
            steps.append({
                "step": step_num,
                "type": "retrieval",
                "query": query,
                "target_metric": None,
                "target_year": None,
            })

        # ── Final computation step (always appended) ──────────────────────
        step_num += 1
        steps.append({
            "step": step_num,
            "type": "computation",
            "query": "Compute the final answer from accumulated retrieved financial data.",
            "target_metric": None,
            "target_year": None,
        })

        return steps
