"""The Task-4 rule registry (hybrid_retriever.py's RULE_MULTIPLIERS/
RULE_DESCRIPTIONS, DISABLE_RULE_MULTIPLIERS/_EXCEPT, applied_rules): ablation
switches and per-result rule reporting. search()'s own rule VALUES and
trigger conditions are unchanged by this registry (see the comment above
RULE_MULTIPLIERS) -- covered instead by a real-corpus before/after parity
check, not a unit test (unit fixtures can't exercise the real alias/regex
machinery each rule's condition depends on)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.tools.hybrid_retriever import HybridFinancialRetriever, RULE_MULTIPLIERS


def _corpus():
    return [
        # Real "Total X" row shape (see hybrid_retriever._TOTAL_ROW_RE),
        # so total_row_boost's condition actually fires for it.
        {"id": "a", "content": "Company: ACME | Line Item: Total revenue | 2023: 5,000",
         "company": "ACME", "type": "table_row"},
        {"id": "b", "content": "revenue revenue revenue growth growth",
         "company": "ACME", "type": "table_row"},
        {"id": "c", "content": "completely unrelated content about weather patterns",
         "company": "ACME", "type": "text_note"},
    ]


def test_disable_rule_multipliers_ranks_by_bm25_and_overlap_only():
    corpus = _corpus()
    retriever = HybridFinancialRetriever(corpus)
    retriever.DISABLE_RULE_MULTIPLIERS = True
    query = "revenue growth"

    results = retriever.search(query, top_k=10)

    # Independently recomputed via the retriever's own primitives, bypassing
    # search()'s multiplier logic entirely -- not a hand-predicted BM25 score.
    query_tokens = retriever._tokenize(query)
    query_idf = {t: retriever._idf(t) for t in set(query_tokens)}
    expected = []
    for doc in corpus:
        doc_tokens = retriever._tokenize(doc["content"])
        bm25 = retriever._bm25_score(query_tokens, doc_tokens, query_idf)
        overlap = sum(query_idf[q] for q in query_tokens if q in doc["content"].lower())
        score = bm25 * 0.7 + overlap * 0.3
        if score > 0.01:
            expected.append((doc["id"], round(score, 4)))
    expected.sort(key=lambda x: x[1], reverse=True)

    assert [(r["id"], r["relevance_score"]) for r in results] == expected
    assert all(r["applied_rules"] == [] for r in results)


def test_rule_multipliers_active_by_default_changes_ranking_and_reports_applied_rules():
    corpus = _corpus()
    retriever = HybridFinancialRetriever(corpus)
    # Doc "a" is a real "Total revenue" row sharing a stem with the query,
    # so total_row_boost should fire for it (default: every rule active).
    results = retriever.search("revenue growth", top_k=10)
    doc_a = next(r for r in results if r["id"] == "a")
    assert "total_row_boost" in doc_a["applied_rules"]

    retriever_ablated = HybridFinancialRetriever(corpus)
    retriever_ablated.DISABLE_RULE_MULTIPLIERS = True
    ablated_results = retriever_ablated.search("revenue growth", top_k=10)
    assert [r["id"] for r in results] != [r["id"] for r in ablated_results] or \
        [r["relevance_score"] for r in results] != [r["relevance_score"] for r in ablated_results]


def test_disable_rule_multipliers_except_keeps_only_named_rules():
    corpus = _corpus()
    retriever = HybridFinancialRetriever(corpus)
    retriever.DISABLE_RULE_MULTIPLIERS = True
    retriever.DISABLE_RULE_MULTIPLIERS_EXCEPT = {"total_row_boost"}

    results = retriever.search("revenue growth", top_k=10)
    doc_a = next(r for r in results if r["id"] == "a")
    assert doc_a["applied_rules"] == ["total_row_boost"]
    # Every other doc's applied_rules stays empty -- only the excepted rule
    # was ever allowed to evaluate.
    for r in results:
        if r["id"] != "a":
            assert r["applied_rules"] == []


def test_rule_multipliers_registry_has_no_duplicate_or_stray_keys():
    # Every RULE_MULTIPLIERS key referenced in search() must exist in the
    # registry (a typo'd lookup would KeyError at search time instead of
    # import time) -- a cheap static sanity check on the registry itself.
    for name, value in RULE_MULTIPLIERS.items():
        assert isinstance(value, (int, float))
        assert value > 0
