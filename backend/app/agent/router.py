import re
from typing import Dict, Any

from app.agent.financial_knowledge import FinancialQuestionUnderstanding


class AdaptiveStrategyRouter:
    NUMERIC_KEYWORDS = [
        "cagr", "yoy", "qoq", "成長率", "毛利率", "比較", "變動", "比率",
        "roe", "趨勢", "幾倍", "百分點", "增加多少", "減少多少", "margin",
        "growth", "compare", "ratio", "change", "difference", "%", "percent"
    ]

    EXPLANATION_KEYWORDS = [
        "why", "what drove", "what caused", "driver", "drove", "explain",
        "原因", "驅動", "帶動", "造成", "解釋", "說明", "變化原因", "margin change"
    ]

    ASSESSMENT_KEYWORDS = [
        "is", "are", "was", "were", "capital-intensive", "capital intensive",
        "useful metric", "not a useful metric", "should", "whether", "是否", "適用", "有沒有意義"
    ]

    EXCLUSION_KEYWORDS = [
        "exclude", "excluding", "without", "adjusted for", "m&a", "ma ", "merger",
        "acquisition", "organic", "剔除", "排除", "不含", "併購", "併入"
    ]

    def _contains_any(self, text: str, keywords) -> bool:
        for keyword in keywords:
            is_chinese = any('\u4e00' <= c <= '\u9fff' for c in keyword)
            if is_chinese or len(keyword) > 2 or not keyword.isalnum():
                if keyword in text:
                    return True
            else:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return True
        return False

    def route(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        understanding = FinancialQuestionUnderstanding.build(query)

        has_multiple_years = len(re.findall(r'20\d\d', query)) >= 2
        has_numeric_keyword = self._contains_any(q_lower, self.NUMERIC_KEYWORDS)
        has_explanation_keyword = self._contains_any(q_lower, self.EXPLANATION_KEYWORDS)
        has_assessment_keyword = self._contains_any(q_lower, self.ASSESSMENT_KEYWORDS)
        has_exclusion_keyword = self._contains_any(q_lower, self.EXCLUSION_KEYWORDS)
        has_domain_intent = understanding.get("intent") in {"EXCLUSION", "EXPLANATION", "ASSESSMENT"}

        if has_exclusion_keyword or understanding.get("intent") == "EXCLUSION":
            answer_mode = "EXCLUSION"
        elif has_explanation_keyword or understanding.get("intent") == "EXPLANATION":
            answer_mode = "EXPLANATION"
        elif has_assessment_keyword or understanding.get("intent") == "ASSESSMENT":
            answer_mode = "ASSESSMENT"
        elif has_numeric_keyword or has_multiple_years or has_domain_intent:
            answer_mode = "NUMERIC"
        else:
            answer_mode = "EXPLANATION"

        if answer_mode == "NUMERIC" and (has_multiple_years or has_numeric_keyword):
            return {
                "complexity": "COMPLEX",
                "answer_mode": answer_mode,
                "reason": f"根據問題中的財務指標與意圖識別為數值計算路徑，指標為 {understanding.get('metric', 'financial metric')}。"
            }
        else:
            return {
                "complexity": "COMPLEX" if has_exclusion_keyword or has_explanation_keyword or has_assessment_keyword or has_multiple_years else "SIMPLE",
                "answer_mode": answer_mode,
                "reason": {
                    "EXCLUSION": "檢測到需要排除特定因素或調整後比較的問題，分配至條件排除分析路徑。",
                    "ASSESSMENT": "檢測到指標是否適用 / 公司特性判斷問題，分配至判斷與解釋路徑。",
                    "EXPLANATION": "檢測到原因 / 驅動因子 / 變動來源問題，分配至原因分析路徑。",
                    "NUMERIC": f"檢測到明確數值比較需求，並基於 {understanding.get('metric', 'financial metric')} 的財務知識分配至數值計算路徑。",
                }.get(answer_mode, "檢測為單次數據查表或單一細項目查詢，分配至快速路徑。")
            }
