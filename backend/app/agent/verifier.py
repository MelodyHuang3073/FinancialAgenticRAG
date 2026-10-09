import re
from typing import List, Dict, Any, Optional, Tuple

#: Strips a formula placeholder's year-role suffix so "revenue_new" and
#: "revenue_old" collapse to the same base name -- see
#: _suspected_duplicate_placeholder_pair's docstring.
_PLACEHOLDER_SUFFIX_RE = re.compile(r'_(new|old)$')

#: Boilerplate/filler tokens stripped before comparing a document's company
#: metadata against a target entity string -- same normalization shape as
#: hybrid_retriever.HybridFinancialRetriever._normalise_company, kept as an
#: independent, self-contained copy here (deliberately not imported) so the
#: verifier has no dependency on the retriever module.
_COMPANY_FILLER_RE = re.compile(r'(?:19|20)\d{2}(?:q[1-4])?$|^10k$|^10q$', re.IGNORECASE)


def _company_words(name: str) -> set:
    name = re.sub(r'\.(pdf|csv|txt|xlsx?|json)$', '', name or '', flags=re.IGNORECASE)
    name = re.sub(r'[_\-]+', ' ', name)
    return {
        w.lower() for w in name.split()
        if len(w) >= 2 and not _COMPANY_FILLER_RE.fullmatch(w)
    }


def _evidence_matches_entity(doc_company: str, entity_words: set) -> bool:
    """Lenient word-overlap check: True if doc_company plausibly names the
    same company as entity_words (or either side has nothing to compare).
    Deliberately coarse (this exists only to catch a GROSS mismatch -- e.g.
    evidence from a completely different company -- not to reproduce
    retrieval's own fine-grained company scoring, which stays untouched)."""
    doc_words = _company_words(doc_company)
    if not doc_words or not entity_words:
        return True  # nothing to contradict the target
    if doc_words & entity_words:
        return True
    doc_collapsed = ''.join(sorted(doc_words))
    ent_collapsed = ''.join(sorted(entity_words))
    return any(w in doc_collapsed for w in entity_words) or any(w in ent_collapsed for w in doc_words)


def _suspected_duplicate_placeholder_pair(
    extracted_variables: Dict[str, Tuple]
) -> Optional[Tuple[str, str]]:
    """Detect a formula's own paired placeholders (e.g. revenue_new /
    revenue_old) that BOTH resolved via a free-text (unstructured keyword-
    proximity) match to the IDENTICAL value -- a strong signal the free-text
    fallback found only one real match and silently reused it for both
    years, rather than two genuinely independent matches that happen to tie
    (a legitimate flat year-over-year value). Deliberately narrower than,
    and does not replace, pot_reasoner._detect_and_strip_duplicate_values,
    which intentionally SKIPS same-base-name pairs for exactly that reason
    -- this only flags when NEITHER side has any real structured backing
    (both free-text), never merely because two values happen to be equal.

    `extracted_variables` is pot_output["extracted_variables"] on the
    formula-matched path: {placeholder: (placeholder, source_string, val)},
    where source_string already encodes how the value was obtained (see
    pot_reasoner.generate_and_execute's own extracted_summary construction)
    -- read as-is here, nothing new added to pot_reasoner's return shape.
    """
    by_base: Dict[str, List[str]] = {}
    for ph in extracted_variables:
        base = _PLACEHOLDER_SUFFIX_RE.sub('', ph)
        by_base.setdefault(base, []).append(ph)

    for phs in by_base.values():
        if len(phs) < 2:
            continue
        for i in range(len(phs)):
            for j in range(i + 1, len(phs)):
                entry_a, entry_b = extracted_variables[phs[i]], extracted_variables[phs[j]]
                if len(entry_a) < 3 or len(entry_b) < 3:
                    continue
                source_a, val_a = entry_a[1], entry_a[2]
                source_b, val_b = entry_b[1], entry_b[2]
                if (
                    isinstance(source_a, str) and isinstance(source_b, str)
                    and source_a.startswith("free-text") and source_b.startswith("free-text")
                    and val_a == val_b
                ):
                    return phs[i], phs[j]
    return None


class TriCheckSelfVerifier:
    """
    Self-Verifier with Tri-Check Mechanism (FinAgent-RAG Paper Section 3.5):
    1. nu_suff: Evidence Sufficiency Check
    2. nu_num: Numerical Consistency Check (sandbox execution)
    3. nu_cross: Cross-Evidence Validation
    """

    def verify(
        self, query: str, evidence_list: List[Dict[str, Any]], pot_output: Dict[str, Any],
        entity: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        cross_reasons: List[str] = []

        # Check if query asked for years that are missing in retrieved evidence
        query_years = set(re.findall(r'20\d\d', query))
        evidence_text = " ".join([ev.get("content", "") for ev in evidence_list])
        found_years = set(re.findall(r'20\d\d', evidence_text))

        missing_years = query_years - found_years
        if missing_years:
            cross_passed = False
            cross_reasons.append(f"缺漏特定年份數據: {', '.join(missing_years)}，可能導致跨期比對偏差。")

        # Entity-identity cross-check: if a target entity is given and there
        # IS evidence, but not a single evidence item's own company metadata
        # plausibly names that entity, the retrieval likely ran unscoped
        # (e.g. drifted onto a different company's filing entirely) rather
        # than merely being imprecise -- a materially different failure mode
        # from "right company, imperfect passage" that the existing checks
        # never looked at. Skipped (no signal either way) when there's no
        # entity to check against, no evidence at all (nu_suff already
        # covers that), or no evidence item carries company metadata.
        entity_words = _company_words(entity or "") if entity and entity.lower() not in ("company", "unknown", "") else set()
        if entity_words and evidence_list:
            companies_present = [ev.get("company", "") for ev in evidence_list if ev.get("company")]
            if companies_present and not any(
                _evidence_matches_entity(c, entity_words) for c in companies_present
            ):
                cross_passed = False
                cross_reasons.append(
                    f"檢索到的證據公司與目標實體「{entity}」不符，可能是實體辨識或檢索範圍偏移。"
                )

        # Formula-variable provenance cross-check: see
        # _suspected_duplicate_placeholder_pair's docstring -- catches a
        # same-metric year-pair (e.g. revenue_new/revenue_old) that both
        # silently reused one free-text match, which otherwise computes a
        # confident-looking but fabricated "0% change" result.
        dup_pair = None
        if pot_output.get("formula_used"):
            dup_pair = _suspected_duplicate_placeholder_pair(pot_output.get("extracted_variables") or {})
            if dup_pair:
                cross_passed = False
                cross_reasons.append(
                    f"變數 '{dup_pair[0]}' 與 '{dup_pair[1]}' 皆為未經結構化比對的自由文字匹配，"
                    f"且數值相同，疑似誤將同一筆資料套用到兩個年度。"
                )

        cross_reason = " ".join(cross_reasons) if cross_reasons else "跨報表與時間軸數據對照無矛盾。"

        is_accepted = suff_passed and num_passed and cross_passed
        confidence_score = 0.95 if is_accepted else (0.6 if suff_passed else 0.2)

        return {
            "decision": "ACCEPT" if is_accepted else "REJECT",
            "confidence_score": confidence_score,
            # Structured (not prose-parsed) signal for callers that want to
            # react specifically to this failure mode -- e.g. the
            # orchestrator's retry logic issuing a targeted re-search for
            # exactly these two placeholders instead of (or in addition to)
            # its generic query-refinement retry. None when not applicable.
            "duplicate_placeholder_pair": dup_pair,
            "checks": {
                "nu_suff": {"passed": suff_passed, "detail": suff_reason},
                "nu_num": {"passed": num_passed, "detail": num_reason},
                "nu_cross": {"passed": cross_passed, "detail": cross_reason}
            }
        }
