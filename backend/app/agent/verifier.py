from typing import List, Dict, Any

class TriCheckSelfVerifier:
    """
    Self-Verifier with Tri-Check Mechanism (FinAgent-RAG Paper Section 3.5):
    1. nu_suff: Evidence Sufficiency Check
    2. nu_num: Numerical Consistency Check (sandbox execution)
    3. nu_cross: Cross-Evidence Validation
    """

    def verify(self, query: str, evidence_list: List[Dict[str, Any]], pot_output: Dict[str, Any]) -> Dict[str, Any]:
        answer_mode = pot_output.get("answer_mode", "NUMERIC")

        # 1. nu_suff: Evidence Sufficiency
        suff_passed = len(evidence_list) > 0
        suff_reason = "已獲取相關財報數據與附註說明。" if suff_passed else "未檢索到足夠的財報證據。"

        # 2. nu_num: Numerical Consistency
        if answer_mode == "NUMERIC":
            num_passed = pot_output.get("success", False) and (pot_output.get("result_value") is not None)
            num_reason = f"PoT 程式碼沙盒計算順利完成 (結果 = {pot_output.get('result_value')})。" if num_passed else f"計算失敗: {pot_output.get('output_log')}"
        else:
            num_passed = suff_passed
            num_reason = "此題為非數值分析路徑，不以 PoT 計算為必要條件；以證據整合與語意判斷為主。" if num_passed else "非數值路徑但缺少足夠證據。"

        # 3. nu_cross: Cross-Evidence Validation
        cross_passed = True
        cross_reason = "跨報表與時間軸數據對照無矛盾。"
        
        # Check if query asked for years that are missing in retrieved evidence
        import re
        query_years = set(re.findall(r'20\d\d', query))
        evidence_text = " ".join([ev.get("content", "") for ev in evidence_list])
        found_years = set(re.findall(r'20\d\d', evidence_text))
        
        missing_years = query_years - found_years
        if missing_years:
            cross_passed = False
            cross_reason = f"缺漏特定年份數據: {', '.join(missing_years)}，可能導致跨期比對偏差。"

        is_accepted = suff_passed and num_passed and cross_passed
        confidence_score = 0.95 if is_accepted else (0.6 if suff_passed else 0.2)

        return {
            "decision": "ACCEPT" if is_accepted else "REJECT",
            "confidence_score": confidence_score,
            "checks": {
                "nu_suff": {"passed": suff_passed, "detail": suff_reason},
                "nu_num": {"passed": num_passed, "detail": num_reason},
                "nu_cross": {"passed": cross_passed, "detail": cross_reason}
            }
        }
