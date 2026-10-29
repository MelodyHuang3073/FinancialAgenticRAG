"""One Gemini call with retry, for the Google GenAI fallback path.

The answer generator and the decomposer fall back to the Google GenAI client when no
OpenAI key is set. Free-tier Gemini keys answer 429 / 503 under bursts, and both callers
turn any exception into "no answer", so a rate limit would silently degrade a whole run.
Transient errors are retried with a growing wait (or the wait the API asks for); anything
else, a daily-quota error, or the last failed attempt, is raised unchanged.
"""
import re
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
            if "PerDay" in text or "per day" in text.lower():
                raise  # a daily quota will not clear within a run: fail loudly instead of waiting
            if attempt == tries - 1 or not any(k.lower() in text.lower() for k in _TRANSIENT):
                raise
            m = re.search(r"retry in ([\d.]+)s", text, re.I)
            time.sleep(min(float(m.group(1)) + 1 if m else delay, 90.0))
            delay = min(delay * 2, 60.0)
