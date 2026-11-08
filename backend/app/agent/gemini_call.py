"""One Gemini call with retry, for the Google GenAI fallback path.

The answer generator and the decomposer fall back to the Google GenAI client when no
OpenAI key is set. Free-tier Gemini keys answer 429 / 503 under bursts, and both callers
turn any exception into "no answer", so a rate limit would silently degrade a whole run.
Transient errors are retried with a growing wait (or the wait the API asks for); anything
else, a daily-quota error, or the last failed attempt, is raised unchanged. Every retry and
every final failure is printed to stderr, so a run's .err file shows what happened.
"""
import re
import sys
import time
from typing import Any, Optional

_TRANSIENT = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
              "overloaded", "timed out", "timeout", "Server disconnected")


def generate_content(client: Any, model: str, contents: str,
                     system_instruction: Optional[str] = None, tries: int = 8) -> Any:
    kwargs = {"model": model, "contents": contents}
    if system_instruction:
        from google.genai import types
        kwargs["config"] = types.GenerateContentConfig(system_instruction=system_instruction)
    delay = 4.0
    for attempt in range(tries):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:  # noqa: BLE001 - the SDK raises its own error types
            text = str(e)
            daily = "PerDay" in text and "PerMinute" not in text
            if daily or attempt == tries - 1 or not any(k.lower() in text.lower() for k in _TRANSIENT):
                # a daily quota will not clear within a run, and a non-transient error will not
                # either: fail loudly instead of waiting
                print(f"[GEMINI FAIL] attempt {attempt + 1}/{tries} {type(e).__name__}: {text[:240]}",
                      file=sys.stderr, flush=True)
                raise
            m = re.search(r"retry in ([\d.]+)s", text, re.I)
            wait = min(float(m.group(1)) + 1 if m else delay, 90.0)
            print(f"[GEMINI RETRY] attempt {attempt + 1}/{tries} in {wait:.0f}s: {text[:120]}",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
