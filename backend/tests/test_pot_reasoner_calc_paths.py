"""Question shapes the PoT sandbox deliberately has NO calculation path for
(selection/guidance/group-comparison/ratio-direction questions skip PoT
entirely, result_value=None, card hidden) -- app/agent/pot_reasoner.py's
_no_calculation_path() and _RATIO_DIRECTION_QUERY_RE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.pot_reasoner import _no_calculation_path, _RATIO_DIRECTION_QUERY_RE


def test_no_calculation_path_shapes():
    assert _no_calculation_path("which unit of acme corp reported the smallest net revenue for the first quarter?")
    assert _no_calculation_path("by how many percentage points did acme corp lift its annual guidance for adjusted eps growth?")
    assert _no_calculation_path("how did acme corp's domestic sales growth compare to its overseas sales growth last year?")
    assert _no_calculation_path("were there any potential events that lifted acme corp's net income in the prior year?")
    # ordinary numeric questions keep their PoT path
    assert not _no_calculation_path("what is the cash conversion cycle of acme corp for the latest fiscal year?")
    assert not _no_calculation_path("did cash balances decline between the prior year end and the second quarter of the current year?")


def test_ratio_direction_question_has_no_calculation_path():
    q1 = "Did Acme Corp's payroll cost as a percent of net sales rise or fall this year?".lower()
    q2 = "Did Acme Corp's net earnings, as a percent of sales, rise in the second quarter versus a year ago?".lower()
    assert _RATIO_DIRECTION_QUERY_RE.search(q1)
    assert _RATIO_DIRECTION_QUERY_RE.search(q2)
    # a computed 3-year average must keep its calculation path
    q3 = "What is the three-year average of capital spending as a share of revenue for Acme Corp?".lower()
    assert not _RATIO_DIRECTION_QUERY_RE.search(q3)
