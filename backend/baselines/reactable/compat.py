"""Adapter that lets ReAcTable (github.com/yunjiazhang/ReAcTable, commit aefc7a3; setup.py says MIT, there is no
LICENSE file) run on gpt-5-mini. The third-party source is NOT vendored and no file of it is edited: point
REACTABLE_SRC at the cloned folder; this module replaces its model call at run time.

The method itself is the released one: CodexAnswerCOTExecutor_HighTemperaturMajorityVote with the settings of the
authors' WikiTQ notebook (template original-sql-py-no-intermediate, 5 demonstrations from WikiTQ-sql-py.json, 10
table rows in the prompt, at most 5 steps, 5 runs and a majority vote over the answer strings).

Differences, to be reported in the thesis:
M1. Model gpt-5-mini instead of gpt-4 / Codex. Their GptCompletion only recognises old chat model names, and its
    other branch calls the API with arguments openai 1.x rejects, so it is replaced.
M2. gpt-5 reasoning models reject temperature (they use 0.6), frequency_penalty and stop, and max_tokens=128 would
    be used up by reasoning, so none of these is sent. The work of the stop strings (end the step at the closing
    fence) is done by M4. The model's default temperature is 1.
M3. One branch of the released code reads the response as a dictionary and the others as an object; the reply object
    returned here supports both.
M4. Reading the reply: the released code decides the next step from the text before the first ":" ("SQL", "Python"
    or "Answer") and takes the code from inside the fence. gpt-5-mini often writes a sentence first, adds a language
    tag to the fence ("```sql") or sends bare code, so its steps could not run (a first 3-question trial without
    M4 gave code or commentary as every voted answer). Each reply is rewritten into the shape the demonstrations
    show: its first fenced block (or its bare SQL), without the language tag, labelled by the label right before the
    fence, else by the tag, else by the content; a reply with no code is an answer. When the code forces an answer
    (the prompt already ends with 'Answer: ```'), the bare text is returned, which is what a completion continuing
    that prompt would give. The prompt, the demonstrations and every other setting are unchanged.
M5. The forced answer. When the released code gives up on further steps (step limit, repeated or failing code) it
    ends the prompt with "Answer: ```" for a completion model to continue. A chat model cannot continue a prefilled
    answer and kept writing code (in a trial with M4, 4 of 5 votes of one question were SQL). Only in these calls,
    one user message is appended: "Answer the question directly now. Do not write any code." This is the one change
    to what the model is told; every other call is sent exactly as the released conversion builds it.
D1. Input: one long table per filing built from our parser's rows (see build_inputs.py); no text input.
E1. The code runs the Python the model writes with exec(). It runs inside a Docker container that sees only this
    adapter, the usage ledger, the ReAcTable source (read-only), the tables and the output folder. Python 3.9 and
    the package versions follow requirements.txt (see Dockerfile).
"""
import os
import re
import sys
import threading

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_STOP = ["```.", "``` "]                # the released default; not sent (M2)
_tally = threading.local()


def counters():
    if not hasattr(_tally, "value"):
        reset_counters()
    return _tally.value


def reset_counters():
    _tally.value = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "empty_replies": 0, "raw_replies": [], "forced_calls": []}


FENCE = re.compile(r"```[ \t]*(?:(sqlite|sql|python|py)\b[ \t]*)?(.*?)(?:```|\Z)", re.IGNORECASE | re.DOTALL)
LABEL = re.compile(r"(SQL|Python|Answer)\s*:\s*\Z", re.IGNORECASE)
LEADING_LABEL = re.compile(r"\A\s*(SQL|Python|Answer)\s*:\s*", re.IGNORECASE)
NAMES = {"sql": "SQL", "python": "Python", "answer": "Answer"}
FORCED_SUFFIX = "Answer: ```"
FORCED_INSTRUCTION = "Answer the question directly now. Do not write any code."      # M5
RAW_KEEP = 8000                                                                         # characters of a reply kept


def looks_like_sql(code):
    body = re.sub(r"\A(?:\s*--[^\n]*\n|\s*/\*.*?\*/)*\s*", "", code, flags=re.DOTALL)
    return bool(re.match(r"(?i)(select|with)\b", body))


def looks_like_python(code):
    return bool(re.search(r"(?m)^\s*(import |from \w+ import |def |DF\w*\s*(\[.*?\])?\s*=[^=])", code))


def split_reply(text):
    """(kind, body) of a reply, kind being "SQL", "Python" or "Answer" (M4)."""
    text = (text or "").strip()
    leading = LEADING_LABEL.match(text)
    if leading and NAMES[leading.group(1).lower()] == "Answer":
        body = text[leading.end():]
        fenced = FENCE.match(body)
        return "Answer", (fenced.group(2) if fenced else body).strip().rstrip(".").strip("`").strip()
    block = FENCE.search(text)
    if block:
        tag, body = (block.group(1) or "").lower(), block.group(2).strip().strip("`").strip()
        label = LABEL.search(text[:block.start()].rstrip())
        if label:
            return NAMES[label.group(1).lower()], body
        if tag:
            return ("Python" if tag.startswith("py") else "SQL"), body
        if looks_like_sql(body):
            return "SQL", body
        if looks_like_python(body):
            return "Python", body
        return "Answer", body
    body = (text[leading.end():] if leading else text).strip()
    if leading:
        return NAMES[leading.group(1).lower()], body
    if looks_like_sql(body):
        return "SQL", body
    return "Answer", body


def normalise(text, forced):
    """The reply as the released parser expects it: 'SQL: ```<code>' / 'Python: ```<code>' / 'Answer: ```<text>',
    or only the text when the answer was forced."""
    kind, body = split_reply(text)
    return body if forced else f"{kind}: ```{body}"


class Reply(dict):
    """M3: reply['choices'][0]['text'] and reply.choices[0].message.content give the same string."""

    def __init__(self, text):
        super().__init__(choices=[{"text": text}])
        message = type("Message", (), {"content": text})()
        self.choices = [type("Choice", (), {"message": message, "text": text})()]


def make_completion(to_messages, client_factory):
    state = {}

    def completion(engine, prompt, suffix=None, max_tokens=128, temperature=0, top_p=1, n=1, stream=False,
                   logprobs=None, stop=DEFAULT_STOP, **ignored):
        from app.agent import usage_tracker
        if "client" not in state:
            state["client"] = client_factory()
        messages = to_messages(prompt.strip(" "))                      # their prompt -> chat turns conversion
        forced = prompt.rstrip().endswith(FORCED_SUFFIX)
        if forced:
            messages = messages + [{"role": "user", "content": FORCED_INSTRUCTION}]
        response = state["client"].chat.completions.create(model=MODEL, messages=messages)
        usage = getattr(response, "usage", None)
        usage_tracker.record_usage(MODEL, "reactable", usage, len(prompt))
        tally = counters()
        tally["calls"] += 1
        tally["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        tally["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        raw_text = response.choices[0].message.content or ""
        if not raw_text:
            tally["empty_replies"] += 1
        tally["raw_replies"].append(raw_text[:RAW_KEEP])
        if forced:
            tally["forced_calls"].append(len(tally["raw_replies"]) - 1)
        return Reply(normalise(raw_text, forced=forced))

    return completion


def _openai_client():
    from openai import OpenAI
    return OpenAI(max_retries=5, timeout=600)


def load(src_dir=None, client_factory=_openai_client):
    """Import the released modules, replace their model call, and return the majority-vote class."""
    src_dir = src_dir or os.environ.get("REACTABLE_SRC", "")
    if not os.path.isdir(os.path.join(src_dir, "tabqa")):
        raise FileNotFoundError("set REACTABLE_SRC to the cloned yunjiazhang/ReAcTable folder")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import tabqa.GptConnector as connector
    import tabqa.GptCOTPrompter as cot
    import tabqa.GptCOTPrompter_BeamSeach as vote
    completion = make_completion(connector.prompt2messages, client_factory)
    for module in (connector, cot, vote):                              # M1: the name is star-imported into each
        module.GptCompletion = completion
    return vote.CodexAnswerCOTExecutor_HighTemperaturMajorityVote
