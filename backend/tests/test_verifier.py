"""TriCheckSelfVerifier.verify()'s nu_cross checks: the pre-existing
year-coverage check, plus two additions motivated by real out-of-sample
failures -- entity/company-identity coverage (a 3M question whose retrieval
drifted onto Netflix's evidence) and formula-variable provenance (a Boeing/
American Water Works revenue-growth question whose free-text fallback
silently reused one match for both year placeholders, computing a
confident-looking but fabricated 0% change)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.verifier import TriCheckSelfVerifier


def _pot(answer_mode="NUMERIC", **kw):
    base = {
        "answer_mode": answer_mode, "success": True, "result_value": 1.0,
        "output_log": "", "extracted_variables": {}, "formula_used": None,
    }
    base.update(kw)
    return base


def test_existing_year_coverage_check_still_rejects_missing_year():
    v = TriCheckSelfVerifier()
    res = v.verify(
        "What was revenue in 2022?",
        [{"content": "Revenue in 2021 was $100 million.", "company": "ACME"}],
        _pot(),
    )
    assert res["decision"] == "REJECT"
    assert not res["checks"]["nu_cross"]["passed"]
    assert "2022" in res["checks"]["nu_cross"]["detail"]


def test_accepts_when_nothing_is_wrong():
    v = TriCheckSelfVerifier()
    res = v.verify(
        "What was ACME's revenue in 2022?",
        [{"content": "ACME revenue in 2022 was $100 million.", "company": "ACME_2022_10K"}],
        _pot(),
        entity="ACME_2022_10K",
    )
    assert res["decision"] == "ACCEPT"
    assert res["checks"]["nu_cross"]["passed"]


def test_entity_mismatch_rejects_when_no_evidence_matches_target_company():
    # Mirrors the real 3M case: entity resolved correctly, but every piece
    # of retrieved evidence is tagged to a totally different company.
    v = TriCheckSelfVerifier()
    res = v.verify(
        "What is 3M's FY2025 net profit margin?",
        [{"content": "Netflix net income...", "company": "NETFLIX_2025_10K"}],
        _pot(),
        entity="3M_2025_10K",
    )
    assert res["decision"] == "REJECT"
    assert not res["checks"]["nu_cross"]["passed"]
    assert "3M" in res["checks"]["nu_cross"]["detail"] or "實體" in res["checks"]["nu_cross"]["detail"]


def test_entity_check_does_not_fire_when_company_plausibly_matches():
    v = TriCheckSelfVerifier()
    res = v.verify(
        "What is 3M's FY2025 net profit margin?",
        [{"content": "3M net income in FY2025 was...", "company": "3M_2025_10K"}],
        _pot(),
        entity="3M_2025_10K",
    )
    assert res["checks"]["nu_cross"]["passed"]


def test_entity_check_skipped_when_no_entity_or_untagged_evidence():
    v = TriCheckSelfVerifier()
    # No entity given at all.
    res = v.verify("some question", [{"content": "x", "company": "ANY"}], _pot())
    assert res["checks"]["nu_cross"]["passed"]
    # Entity given but evidence carries no company metadata to contradict it.
    res2 = v.verify("some question", [{"content": "x"}], _pot(), entity="3M_2025_10K")
    assert res2["checks"]["nu_cross"]["passed"]


def test_duplicate_free_text_placeholder_pair_rejects():
    # Mirrors the real Boeing/AWW case: revenue_new and revenue_old both
    # resolved via free-text with the identical (wrong) value.
    v = TriCheckSelfVerifier()
    pot = _pot(
        formula_used="revenue_yoy",
        extracted_variables={
            "revenue_new": ("revenue_new", "free-text <- free-text match on \"revenue\"", 19.0),
            "revenue_old": ("revenue_old", "free-text <- free-text match on \"revenue\"", 19.0),
        },
    )
    res = v.verify("What was the revenue growth rate?", [{"content": "x", "company": "BOEING"}], pot)
    assert res["decision"] == "REJECT"
    assert not res["checks"]["nu_cross"]["passed"]
    assert "revenue_new" in res["checks"]["nu_cross"]["detail"]


def test_duplicate_check_does_not_fire_on_genuinely_structured_matches():
    v = TriCheckSelfVerifier()
    pot = _pot(
        formula_used="revenue_yoy",
        extracted_variables={
            "revenue_new": ("revenue_new", "table-total <- evidence[2] Line Item \"Total revenue\"", 100.0),
            "revenue_old": ("revenue_old", "table-total <- evidence[5] Line Item \"Total revenue\"", 90.0),
        },
    )
    res = v.verify("What was the revenue growth rate?", [{"content": "x", "company": "ACME"}], pot)
    assert res["checks"]["nu_cross"]["passed"]


def test_duplicate_check_does_not_fire_on_a_legitimate_flat_value():
    # Two genuinely independent STRUCTURED matches that happen to tie (a
    # real flat YoY) must never be flagged -- only both-free-text pairs are.
    v = TriCheckSelfVerifier()
    pot = _pot(
        formula_used="revenue_yoy",
        extracted_variables={
            "revenue_new": ("revenue_new", "table-total <- evidence[2] Line Item \"Total revenue\"", 100.0),
            "revenue_old": ("revenue_old", "table-total <- evidence[5] Line Item \"Total revenue\"", 100.0),
        },
    )
    res = v.verify("What was the revenue growth rate?", [{"content": "x", "company": "ACME"}], pot)
    assert res["checks"]["nu_cross"]["passed"]


def test_duplicate_check_skipped_when_no_formula_was_used():
    v = TriCheckSelfVerifier()
    pot = _pot(formula_used=None, extracted_variables={
        "foo": ("a", "2022", 5.0), "bar": ("b", "2023", 5.0),
    })
    res = v.verify("some question", [{"content": "x", "company": "ACME"}], pot)
    assert res["checks"]["nu_cross"]["passed"]
