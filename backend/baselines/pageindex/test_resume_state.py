import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_questions import resume_state  # noqa: E402


def test_max_turns_error_is_kept_and_not_retried_other_errors_are_retried():
    results = [
        {"id": "train-1", "error": None},
        {"id": "train-2", "error": "PageIndexAPIError('The agent did not finish within max_turns (the default limit).')"},
        {"id": "train-3", "error": "TimeoutError('read timed out')"},
    ]
    keep, done = resume_state(results)
    assert [r["id"] for r in keep] == ["train-1", "train-2"]
    assert done == {"train-1", "train-2"}


if __name__ == "__main__":
    test_max_turns_error_is_kept_and_not_retried_other_errors_are_retried()
    print("ok")
