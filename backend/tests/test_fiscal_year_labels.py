"""The question's fiscal year follows the filing's own labels; the year ending in the
asked year is used only when no period carries the asked number.
Sentences are synthetic (a fictional retailer), not wording from any filing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.answer_directives import fiscal_year_directive


def test_label_equal_to_the_asked_number_is_never_shifted():
    # the filing calls the year ended in early 2025 "fiscal 2025": nothing to shift
    ev = "Net sales for fiscal 2025, the 52 weeks ended February 1, 2025, rose. Fiscal 2024 ended February 3, 2024."
    assert fiscal_year_directive("What was net income in fiscal 2025?", ev) == ""


def test_label_equal_to_the_asked_number_wins_even_if_it_ends_in_the_next_year():
    # "fiscal 2025" ends in January 2026: the asked number exists, so it is that period
    ev = "Fiscal 2025 ended January 31, 2026. Fiscal 2024 ended February 1, 2025."
    assert fiscal_year_directive("What was net income in fiscal 2025?", ev) == ""
    ev = "The year ended January 31, 2026 (fiscal 2025) and the year ended February 1, 2025 (fiscal 2024)."
    assert fiscal_year_directive("What was net income in FY2025?", ev) == ""


def test_labels_the_whole_filing_uses_also_prevent_a_shift():
    ev = "The year ended February 1, 2025 and fiscal 2024 compare."
    assert fiscal_year_directive("What were net sales in FY2025?", ev) != ""
    assert fiscal_year_directive("What were net sales in FY2025?", ev, {2025}) == ""


def test_start_year_naming_is_shifted_when_no_period_has_the_asked_number():
    ev = "52 weeks ended January 28, 2023 (fiscal 2022). The prior year was fiscal 2021."
    assert "fiscal 2022" in fiscal_year_directive("What share of repurchases was in Q4 of FY2023?", ev)


def test_no_year_end_information_means_no_directive():
    assert fiscal_year_directive("What was net income in FY2023?", "Revenue was higher in 2023.") == ""


def test_a_52_53_week_calendar_year_ending_in_early_january_is_not_shifted():
    # the year that ended January 2, 2022 is the company's calendar 2021: no directive
    ev = "The fiscal year ended January 2, 2022 (fiscal 2021) and the fiscal year ended January 3, 2021 (fiscal 2020)."
    assert fiscal_year_directive("What drove the gross margin change in FY2022?", ev) == ""


def test_a_year_ending_in_late_january_is_still_shifted():
    ev = "52 weeks ended January 28, 2023 (fiscal 2022). The prior year was fiscal 2021."
    assert fiscal_year_directive("What drove the gross margin change in FY2023?", ev) != ""
