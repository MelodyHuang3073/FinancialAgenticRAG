"""The 'no single computed answer' decision (pot_reasoner.is_no_calculation_question):
the five question shapes it recognises, its overlap with a library metric, and that
generate_and_execute honours the caller's own decision.

Example sentences are synthetic shapes with placeholders, not wording taken from
any filing or benchmark question."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.financial_formula_library import detect_formula
from app.agent.pot_reasoner import ProgramOfThoughtReasoner, is_no_calculation_question


def test_the_five_no_calculation_shapes_are_recognised():
    shapes = [
        "Which unit had the highest value?",                  # selection
        "What is the outlook for the next period?",           # guidance
        "How does the domestic figure compare to the international one?",  # group comparison
        "Were there any events that changed the result?",     # existence
        "Does any category exceed N% of the total?",         # category threshold
    ]
    assert all(is_no_calculation_question(q) for q in shapes)


def test_ordinary_calculation_questions_are_not_flagged():
    for q in ("What is the gross margin?", "What was the total in year A compared to year B?"):
        assert not is_no_calculation_question(q)


def test_a_no_calculation_question_can_also_name_a_library_metric():
    """The two detectors look at different things: the formula library at metric
    keywords, this one at the question's wording. When both match, the
    no-calculation shape decides whether the calculation is skipped."""
    q = "Is the gross margin expected to grow next period?"
    assert detect_formula(q) is not None
    assert is_no_calculation_question(q)


def test_generate_and_execute_uses_the_callers_decision():
    reasoner = ProgramOfThoughtReasoner()
    q = "What is the gross margin?"
    skipped = reasoner.generate_and_execute(q, [], "Acme", library_formula_entry=None, no_calculation=True)
    assert skipped["calculation_skipped"] is True and skipped["result_value"] is None
    default = reasoner.generate_and_execute(q, [], "Acme", library_formula_entry=None)
    assert not default.get("calculation_skipped")
