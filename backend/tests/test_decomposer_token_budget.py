"""The decomposer's output budget for reasoning models, and its fallback when the reply is empty.

A reasoning model spends part of max_completion_tokens on invisible reasoning; with too small a
budget the reply comes back empty and the decomposer silently uses the rule path. The budget is
REASONING_MAX_COMPLETION_TOKENS; the fallback itself is unchanged.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.agent import decomposer as dec_mod  # noqa: E402


class _Client:
    """Records the arguments of every chat completion call and answers with `content`."""

    def __init__(self, content=""):
        self.calls = []
        self.content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def _decomposer(model, client):
    d = dec_mod.QueryDecomposer()
    d._llm_client, d._llm_model, d._llm_enabled = client, model, True
    return d


def test_reasoning_model_gets_the_large_budget_in_both_calls():
    assert dec_mod.REASONING_MAX_COMPLETION_TOKENS == 16000
    client = _Client()
    d = _decomposer("gpt-5-mini", client)
    d._llm_decompose("What was 3M's FY2018 capital expenditure?", "3M", ["capex"], ["2018"])
    d.suggest_narrative_topic_query("What legal battles did 3M face?", "3M")
    assert [c["max_completion_tokens"] for c in client.calls] == [16000, 16000]
    assert all("temperature" not in c for c in client.calls)


def test_non_reasoning_model_keeps_its_small_budgets_and_temperature_zero():
    client = _Client()
    d = _decomposer("gpt-4o-mini", client)
    d._llm_decompose("What was 3M's FY2018 capital expenditure?", "3M", ["capex"], ["2018"])
    d.suggest_narrative_topic_query("What legal battles did 3M face?", "3M")
    assert [c["max_completion_tokens"] for c in client.calls] == [768, 100]
    assert all(c["temperature"] == 0 for c in client.calls)


def test_an_empty_reply_still_falls_back_to_the_rule_decomposition():
    d = _decomposer("gpt-5-mini", _Client(content=""))
    steps, formula = d.decompose("What was 3M's FY2018 capital expenditure?", ["capital_expenditure"], ["2018"], "3M")
    assert formula is None
    assert steps and all(s.get("source") == "rule" for s in steps if s["type"] == "computation")
