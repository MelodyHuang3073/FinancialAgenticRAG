"""The Anthropic call path of the answer generator and the usage ledger's token names.
No network: a fake client stands in for anthropic.Anthropic."""
from types import SimpleNamespace

from app.agent import llm_client
from app.agent.usage_tracker import _usage_numbers


class _FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="  The ratio is 1.71.  ")],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
        )


class _FakeAnthropic:
    def __init__(self):
        self.messages = _FakeMessages()


def test_usage_numbers_reads_anthropic_names():
    n = _usage_numbers(SimpleNamespace(input_tokens=1200, output_tokens=80))
    assert (n["prompt"], n["completion"], n["total"]) == (1200, 80, 1280)


def test_answer_generator_uses_messages_api(monkeypatch):
    recorded = []
    monkeypatch.setattr(llm_client, "record_usage", lambda *a, **k: recorded.append(a))
    gen = llm_client.LLMAnswerGenerator()
    gen._client = _FakeAnthropic()
    gen._model = "claude-haiku-5-5"
    out = gen.generate_answer("What is the current ratio?", "NUMERIC", [], {})
    assert out == "The ratio is 1.71."
    kw = gen._client.messages.kwargs
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["system"] == llm_client._ANSWER_SYSTEM_PROMPT
    assert kw["max_tokens"] >= 4000
    assert kw["messages"][0]["role"] == "user"
    assert recorded and recorded[0][1] == "answer"
