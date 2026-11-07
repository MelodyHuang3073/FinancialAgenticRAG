"""grading_check: the mechanical label checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_results"))

import grading_check as g  # noqa: E402


def _rows(n=300, label="C", rule=None):
    return [{"id": f"train-{i}", "label": label, "rule": rule} for i in range(n)]


def test_a_well_formed_group_has_no_problems():
    rows = _rows()
    rows[3] = {"id": "train-3", "label": "S", "rule": 1}
    assert g.validate(rows) == []


def test_non_c_label_needs_a_rule_and_the_rule_must_allow_it():
    rows = _rows()
    rows[0] = {"id": "train-0", "label": "P", "rule": None}
    rows[1] = {"id": "train-1", "label": "S", "rule": 2}       # row 2 is P
    rows[2] = {"id": "train-2", "label": "W", "rule": "9.4-5"}  # 9.4-5 is P
    rows[3] = {"id": "train-3", "label": "P", "rule": 99}
    msgs = g.validate(rows)
    assert len(msgs) == 4 and "no rule" in msgs[0] and "allows" in msgs[1] and "unknown rule" in msgs[3]


def test_size_duplicates_and_bad_labels_are_reported():
    rows = _rows(299)
    rows.append({"id": "train-0", "label": "X", "rule": None})
    msgs = g.validate(rows)
    assert any("duplicate" in m for m in msgs) and any("not C/S/P/W" in m for m in msgs)


def test_differing_lists_only_questions_whose_labels_disagree():
    a, b = _rows(), _rows()
    b[5] = {"id": "train-5", "label": "P", "rule": 18}
    diff = g.differing({"a": a, "b": b})
    assert list(diff) == ["train-5"] and diff["train-5"]["b"] == ("P", 18)
