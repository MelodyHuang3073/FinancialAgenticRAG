"""Retry rules of the Gemini call: a per-minute 429 is retried, a daily-quota 429 is not."""
from types import SimpleNamespace

import pytest

from app.agent import gemini_call


class _Client:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise Exception(self.errors.pop(0))
        return SimpleNamespace(text="ok")


def test_per_minute_429_is_retried(monkeypatch):
    monkeypatch.setattr(gemini_call.time, "sleep", lambda s: None)
    c = _Client(["429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel-FreeTier retry in 3.2s"])
    assert gemini_call.generate_content(c, "m", "x").text == "ok"
    assert c.calls == 2


def test_daily_429_fails_at_once(monkeypatch):
    monkeypatch.setattr(gemini_call.time, "sleep", lambda s: pytest.fail("must not wait"))
    c = _Client(["429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
    with pytest.raises(Exception):
        gemini_call.generate_content(c, "m", "x")
    assert c.calls == 1


def test_other_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(gemini_call.time, "sleep", lambda s: pytest.fail("must not wait"))
    c = _Client(["400 INVALID_ARGUMENT bad request"])
    with pytest.raises(Exception):
        gemini_call.generate_content(c, "m", "x")
    assert c.calls == 1
