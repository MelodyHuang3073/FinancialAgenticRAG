from typing import List, Dict, Any

class QueryRefiner:
    """
    Query Refiner (FinAgent-RAG Paper Section 3.5)
    Refines queries based on Self-Verifier failure feedback to perform targeted re-retrieval.
    """

    def refine(self, query: str, verifier_output: Dict[str, Any], attempt_iter: int) -> str:
        checks = verifier_output.get("checks", {})
        cross_detail = checks.get("nu_cross", {}).get("detail", "")
        suff_detail = checks.get("nu_suff", {}).get("detail", "")

        if "缺漏特定年份" in cross_detail:
            # Extract missing year hint
            return f"{query} 併列詳細綜合損益表與合併財務報表附註"
        elif not checks.get("nu_suff", {}).get("passed", False):
            return f"{query} 營業收入 營業費用 毛利 明細數據"
        else:
            return f"{query} 財務報告 完整數據與經營討論"
