"""PoT derivations that read a table's OWN full row/column structure directly
(not the generic year-only extractor) to answer a question no formula alias
covers on its own -- app/agent/pot_reasoner.py's
_derive_adjusted_ebit_from_ebitda_reconciliation() and
_derive_total_row_period_end_change()."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.pot_reasoner import ProgramOfThoughtReasoner, _derive_total_row_period_end_change


def test_adjusted_ebit_derived_from_ebitdar_reconciliation_and_floored_at_zero():
    """MGM-shaped case: no 'Adjusted EBIT' line exists, only an 'Adjusted
    EBITDAR' reconciliation; derive EBIT = EBITDAR - D&A - Rent, and the
    interest-coverage ratio floors a negative result at 0 (a company can't
    have negative 'coverage')."""
    parent = (
        "Company: TESTCO_2022_10K | Document: TESTCO_2022_10K.pdf | Page: 9\n\n"
        "| Line Item | 2022 (3M) | 2021 (3M) | 2022 (12M) | 2021 (12M) |\n"
        "|---|---|---|---|---|\n"
        "| Interest expense, net of amounts capitalized | 137,132 | 201,477 | 594,954 | 799,593 |\n"
        "| Operating income (loss) | (1,896) | 368,847 | 1,439,372 | 2,278,699 |\n"
        "| Depreciation and amortization | 1,421,637 | 297,031 | 3,482,050 | 1,150,610 |\n"
        "| Triple-net operating lease and ground lease rent expense | 600,467 | 262,307 | 1,950,566 | 833,158 |\n"
        "| Adjusted EBITDAR | $ 957,307 |  | $ 3,497,254 |  |\n"
    )
    evidence = [{
        "company": "TESTCO_2022_10K", "page_number": 9, "type": "table_row",
        "content": "Line Item: Operating income (loss) | 2022 (12M): 1,439,372",
        "parent_content": parent,
    }]
    q = "What was TestCo's interest coverage ratio using FY2022 Adjusted EBIT as the numerator and annual Interest Expense as the denominator?"
    res = ProgramOfThoughtReasoner().generate_and_execute(q, evidence, entity="TESTCO_2022_10K")
    assert res.get("result_value") == 0.0
    assert "ebit = -1935362.0" in res.get("code", "")
    assert "max(0, ebit)" in res.get("code", "")


def test_total_row_rollforward_gives_deterministic_store_count_change():
    """A 'how many/number of <noun>' question with a roll-forward 'Total' row
    (stores opened/closed) must compute END-of-period counts, not whichever
    row/column the LLM happens to read, and must not confuse the Beginning
    count with the End count (both get tagged with the same bare year by the
    generic extractor)."""
    evidence = [{
        "company": "TESTCO_2024Q2_10Q", "page_number": 17, "type": "table_row",
        "content": (
            "Company: TESTCO_2024Q2_10Q | Report: TESTCO.pdf (Page 17) | Line Item: Total | "
            "Fiscal 2024 Total Stores at Beginning of Second Quarter: 966 | "
            "Fiscal 2024 Stores Opened: 5 | Fiscal 2024 Stores Closed: (2) | "
            "Fiscal 2024 Total Stores at End of Second Quarter: 969 | "
            "Fiscal 2023 Total Stores at Beginning of Second Quarter: 977 | "
            "Fiscal 2023 Stores Opened: 7 | Fiscal 2023 Stores Closed: (2) | "
            "Fiscal 2023 Total Stores at End of Second Quarter: 982"
        ),
    }]
    q = "Was there any change in the number of TestCo stores between Q2 of FY2024 and FY2023?"
    res = ProgramOfThoughtReasoner().generate_and_execute(q, evidence, entity="TESTCO_2024Q2_10Q")
    assert res.get("result_value") == -13.0
    assert res.get("extraction_method") == "total-row-rollforward"


def test_total_row_rollforward_does_not_fire_without_a_matching_noun():
    """A 'Total' roll-forward row for an unrelated noun (e.g. warehouses) must
    not be used to answer a question about a different noun (stores)."""
    content = (
        "Company: TESTCO | Line Item: Total | "
        "Fiscal 2024 Total Warehouses at Beginning of Second Quarter: 10 | "
        "Fiscal 2024 Total Warehouses at End of Second Quarter: 11 | "
        "Fiscal 2023 Total Warehouses at Beginning of Second Quarter: 9 | "
        "Fiscal 2023 Total Warehouses at End of Second Quarter: 10"
    )
    res = _derive_total_row_period_end_change(
        [{"content": content}], "was there any change in the number of testco stores?"
    )
    assert res is None
