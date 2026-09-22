"""Question shapes the PoT sandbox deliberately has NO calculation path for
(selection/guidance/group-comparison/ratio-direction questions skip PoT
entirely, result_value=None, card hidden) -- app/agent/pot_reasoner.py's
_no_calculation_path() and _RATIO_DIRECTION_QUERY_RE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.pot_reasoner import _no_calculation_path, _RATIO_DIRECTION_QUERY_RE


def test_no_calculation_path_shapes():
    assert _no_calculation_path("which of jpm's business segments had the lowest net revenue in 2021 q1?")
    assert _no_calculation_path("by how many percentage points did pepsico raise full year guidance in respect of core eps growth?")
    assert _no_calculation_path("how did jnj's us sales growth compare to international sales growth in fy2022?")
    assert _no_calculation_path("were there any potential events that increased net income in 2019?")
    # ordinary numeric questions keep their PoT path
    assert not _no_calculation_path("what is the fy2019 cash conversion cycle for general mills?")
    assert not _no_calculation_path("was there any drop in cash & cash equivalents between fy 2023 and q2 of fy2024?")


def test_ratio_direction_question_has_no_calculation_path():
    q1 = "Did Ulta Beauty's wages expense as a percent of net sales increase or decrease in FY2023?".lower()
    q2 = "Did JnJ's net earnings as a percent of sales increase in Q2 of FY2023 compared to Q2 of FY2022?".lower()
    assert _RATIO_DIRECTION_QUERY_RE.search(q1)
    assert _RATIO_DIRECTION_QUERY_RE.search(q2)
    # a computed 3-year average must keep its calculation path
    q3 = "What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard?".lower()
    assert not _RATIO_DIRECTION_QUERY_RE.search(q3)
