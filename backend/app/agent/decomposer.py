"""
QueryDecomposer (FinAgent-RAG Section 3.3 / FinanceBench SKILL Stage 3)

Decomposes a complex financial question into sequential retrieval + computation steps.
Each retrieval step targets a specific (metric × year) pair so the orchestrator can
retrieve the right table rows one-by-one.

Decomposition modes (in priority order):
  1. LLM (OpenAI / Gemini) — structured sub-queries via FinanceBench SOP prompt
  2. Rule-based fallback — deterministic (metric × year) expansion
"""

import json
import os
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
    "ppe":           "Property Plant and Equipment PP&E Fixed Assets 不動產廠房及設備 固定資產",
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
    "fixed_asset_turnover":    ["revenue", "ppe"],
    "operating_cash_flow_ratio": ["operating_cf", "current_liab"],
}


class QueryDecomposer:
    """
    Decomposes a complex financial question into an ordered list of retrieval sub-tasks.

    Uses OpenAI / Gemini when available (FinanceBench SOP Stage 3),
    otherwise falls back to the deterministic rule-based expansion.
    """

    # ── LLM system prompt aligned with FinanceBench SKILL.md Stage 3 ──────────
    _SYSTEM_PROMPT = (
        "You are a financial analysis assistant following the FinanceBench SOP.\n"
        "Your task (Stage 3 — Multi-Step Retrieval Decomposition):\n"
        "Break the user's financial question into atomic search sub-queries.\n"
        "Each sub-query must target ONE specific (metric, year) pair.\n"
        "For ratio/margin questions, include sub-queries for BOTH numerator and denominator.\n"
        "Output ONLY a JSON array, no explanation. Format:\n"
        "[\n"
        "  {\"step\": 1, \"query\": \"<company> <metric> <year>\", "
        "\"target_metric\": \"<canonical_metric>\", \"target_year\": \"<YYYY>\"},\n"
        "  ...\n"
        "]\n"
        "Use short, precise English queries. Include company name in every query."
    )

    def __init__(self) -> None:
        self._llm_client = None
        self._llm_model: Optional[str] = None
        self._llm_enabled: Optional[bool] = None  # None = not checked yet

    def _get_llm_client(self):
        """Lazy-init: try OpenAI first, then Gemini, then return None."""
        if self._llm_enabled is not None:
            return self._llm_client  # already resolved

        # Load .env if present
        try:
            from dotenv import load_dotenv
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            env_path = os.path.join(base_dir, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
        except Exception:
            pass

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=openai_key)
                self._llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self._llm_enabled = True
                return self._llm_client
            except Exception:
                pass

        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            try:
                from google import genai
                self._llm_client = genai.Client(api_key=google_key)
                self._llm_model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")
                self._llm_enabled = True
                return self._llm_client
            except Exception:
                pass

        self._llm_enabled = False
        return None

    def _llm_decompose(
        self,
        query: str,
        entity: str,
        target_metrics: List[str],
        years: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Call the LLM to generate structured sub-queries aligned with SKILL.md Stage 3.
        Returns None if LLM is unavailable or output cannot be parsed.
        """
        client = self._get_llm_client()
        if not client:
            return None

        # Build a concise context block for the LLM
        context_lines = [f"Question: {query}"]
        if entity and entity != "company":
            context_lines.append(f"Company: {entity}")
        if target_metrics:
            context_lines.append(f"Identified metrics: {', '.join(target_metrics)}")
        if years:
            context_lines.append(f"Years involved: {', '.join(years)}")
        context_lines.append(
            "\nGenerate the minimum set of sub-queries needed to retrieve all required data. "
            "Max 8 sub-queries."
        )
        user_content = "\n".join(context_lines)

        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                # OpenAI
                response = client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
            else:
                # Gemini
                full_prompt = self._SYSTEM_PROMPT + "\n\n" + user_content
                response = client.models.generate_content(
                    model=self._llm_model, contents=full_prompt
                )
                raw = getattr(response, "text", "") or ""

            # Extract JSON array from the response
            json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not json_match:
                return None

            parsed = json.loads(json_match.group())
            if not isinstance(parsed, list):
                return None

            # Validate and normalise each step
            steps: List[Dict[str, Any]] = []
            for i, item in enumerate(parsed, 1):
                if not isinstance(item, dict):
                    continue
                q = str(item.get("query", "")).strip()
                if not q:
                    continue
                steps.append({
                    "step": item.get("step", i),
                    "type": "retrieval",
                    "query": q,
                    "target_metric": item.get("target_metric") or None,
                    "target_year": str(item.get("target_year", "")).strip() or None,
                    "source": "llm",
                })

            return steps if steps else None

        except Exception:
            return None

    def _rule_decompose(
        self,
        query: str,
        target_metrics: List[str],
        years: List[str],
        entity: str,
    ) -> List[Dict[str, Any]]:
        """Deterministic (metric × year) expansion — used when LLM is unavailable."""
        # Expand ratio/formula metrics to component line items
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

        if expanded_metrics and years:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                for year in years:
                    step_num += 1
                    steps.append({
                        "step": step_num,
                        "type": "retrieval",
                        "query": f"{entity} {metric_display} {year}".strip(),
                        "target_metric": metric,
                        "target_year": year,
                        "source": "rule",
                    })
        elif expanded_metrics:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                step_num += 1
                steps.append({
                    "step": step_num,
                    "type": "retrieval",
                    "query": f"{entity} {metric_display}".strip(),
                    "target_metric": metric,
                    "target_year": None,
                    "source": "rule",
                })
        elif years:
            q_lower = query.lower()
            if "cagr" in q_lower or "複合成長率" in q_lower:
                y1, y2 = years[0], years[-1]
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y1}".strip(),
                               "target_metric": None, "target_year": y1, "source": "rule"})
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y2}".strip(),
                               "target_metric": None, "target_year": y2, "source": "rule"})
            else:
                for y in years:
                    step_num += 1
                    steps.append({"step": step_num, "type": "retrieval",
                                  "query": f"{entity} {query} {y}".strip(),
                                  "target_metric": None, "target_year": y, "source": "rule"})
        else:
            step_num += 1
            steps.append({
                "step": step_num,
                "type": "retrieval",
                "query": query,
                "target_metric": None,
                "target_year": None,
                "source": "rule",
            })

        return steps

    def decompose(
        self,
        query: str,
        target_metrics: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
        entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Decompose a financial query into ordered retrieval + computation steps."""
        # ── Normalise inputs ──────────────────────────────────────────────────
        if not years:
            years = sorted(set(re.findall(r'(?:FY)?(20\d\d)', query)))
        else:
            years = [re.sub(r'^FY', '', y) for y in years]

        target_metrics = list(target_metrics or [])
        entity = (entity or "").strip()

        # ── Try LLM first (FinanceBench SOP Stage 3) ─────────────────────────
        llm_steps = self._llm_decompose(query, entity, target_metrics, years)

        if llm_steps:
            retrieval_steps = llm_steps
            decompose_source = "llm"
        else:
            retrieval_steps = self._rule_decompose(query, target_metrics, years, entity)
            decompose_source = "rule"

        # ── Renumber steps sequentially ───────────────────────────────────────
        for i, s in enumerate(retrieval_steps, 1):
            s["step"] = i

        # ── Always append final computation step ──────────────────────────────
        retrieval_steps.append({
            "step": len(retrieval_steps) + 1,
            "type": "computation",
            "query": "Compute the final answer from accumulated retrieved financial data.",
            "target_metric": None,
            "target_year": None,
            "source": decompose_source,
        })

        return retrieval_steps
