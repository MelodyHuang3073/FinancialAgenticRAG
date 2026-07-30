import re
from typing import Any, Dict, List


class FinancialQuestionUnderstanding:
    @staticmethod
    def build(query: str) -> Dict[str, Any]:
        q = query.strip()
        lowered = q.lower()

        entity = FinancialQuestionUnderstanding._extract_entity(q)
        metric = FinancialQuestionUnderstanding._extract_metric(lowered)
        years = re.findall(r"20\d{2}", q)
        intent = FinancialQuestionUnderstanding._infer_intent(lowered)
        financial_primer = FinancialQuestionUnderstanding._build_financial_primer(metric)

        retrieval_queries: List[str] = []
        if metric == "gross margin":
            retrieval_queries.extend([
                f"{entity} gross profit cost of sales margin trend {years[-1] if years else ''}".strip(),
                f"{entity} gross margin driver reason change in cost structure {years[-1] if years else ''}".strip(),
            ])
        elif metric == "operating margin":
            retrieval_queries.extend([
                f"{entity} operating expense operating income margin trend {years[-1] if years else ''}".strip(),
                f"{entity} operating margin driver reason SG&A R&D expense {years[-1] if years else ''}".strip(),
            ])
        elif metric == "revenue growth":
            retrieval_queries.extend([
                f"{entity} revenue growth period comparison {years[-1] if years else ''}".strip(),
                f"{entity} revenue driver reason product mix geography {years[-1] if years else ''}".strip(),
            ])
        else:
            retrieval_queries.extend([
                f"{entity} financial statement {metric} {years[-1] if years else ''}".strip(),
                f"{entity} {metric} driver reason change {years[-1] if years else ''}".strip(),
            ])

        return {
            "entity": entity or "company",
            "metric": metric or "financial metric",
            "years": years,
            "intent": intent,
            "retrieval_queries": retrieval_queries,
            "financial_primer": financial_primer,
            "reasoning_plan": [
                "Identify the target company and metric from the question",
                "Map the metric to the relevant financial statement and driver concepts",
                "Search for supporting data and business explanations before drafting the answer",
            ],
        }

    @staticmethod
    def _extract_entity(query: str) -> str:
        candidates = ["tsmc", "apple", "microsoft", "amazon", "google", "meta", "nvidia", "jpmorgan", "bank of america"]
        lowered = query.lower()
        for candidate in candidates:
            if candidate in lowered:
                return candidate.upper() if candidate not in {"bank of america"} else "Bank of America"
        match = re.search(r"for\s+([A-Za-z0-9&.\- ]+?)(?: in | on | for | was|is|did|why|what|how|\?)", query, flags=re.I)
        if match:
            return match.group(1).strip()
        return "company"

    @staticmethod
    def _infer_intent(query: str) -> str:
        if any(k in query for k in ["exclude", "excluding", "without", "adjusted", "organic", "acquisition", "merger"]):
            return "EXCLUSION"
        if any(k in query for k in ["why", "cause", "driver", "reason", "explain", "drove"]):
            return "EXPLANATION"
        if any(k in query for k in ["is", "should", "whether", "useful", "appropriate", "suitable"]):
            return "ASSESSMENT"
        return "NUMERIC"

    @staticmethod
    def _build_financial_primer(metric: str) -> str:
        primers = {
            "gross margin": "Gross margin is primarily driven by revenue mix, pricing, and cost of sales; changes often reflect product mix or input cost shifts.",
            "operating margin": "Operating margin is shaped by operating leverage, SG&A, R&D, and changes in revenue scale.",
            "revenue growth": "Revenue growth should be interpreted through volume, pricing, product mix, and geographic exposure.",
            "roe": "ROE depends on profitability, asset efficiency, and leverage.",
            "ebitda": "EBITDA is a proxy for operating performance and should be interpreted alongside cash flow and capital intensity.",
        }
        return primers.get(metric, "Financial analysis should connect the metric to the relevant statement line items and business drivers.")

    @staticmethod
    def _extract_metric(query: str) -> str:
        if "gross margin" in query or "毛利率" in query:
            return "gross margin"
        if "operating margin" in query or "營業利益率" in query:
            return "operating margin"
        if "revenue growth" in query or "營收成長" in query or "growth" in query:
            return "revenue growth"
        if "roe" in query or "return on equity" in query:
            return "ROE"
        if "ebitda" in query:
            return "EBITDA"
        return "financial metric"
