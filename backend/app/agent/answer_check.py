"""Checks on the finished answer text, run after synthesis.

The Tri-Check verifier (verifier.py) judges the evidence and the calculation
BEFORE the answer is written. These checks read the answer itself, for the
failures that only show there. Each returns a failure type that the retry
controller (orchestrator._plan_retry) maps to a new retrieval:

  non_answer         the answer says the filing does not give the information
  wrong_company      the answer says its evidence is another company's filing

Tested and left out: a "figure not found in the evidence" check. Evidence holds so
many numbers that almost any figure is the difference or ratio of two of them, so
the check flagged nothing it should not and could not tell a wrong figure from a
computed one.
"""
import re
from typing import Any, Dict, List, Optional, Set

_NON_ANSWER_RE = re.compile(
    r"(?:does|do|did)\s+not\s+(?:disclose|provide|give|state|specify)"
    r"|(?:is|are|was|were)\s+not\s+(?:disclosed|provided|reported|available|stated|specified|mentioned)"
    r"|not\s+(?:enough|sufficient)\s+(?:information|evidence)"
    r"|cannot\s+(?:be\s+)?(?:determine|determined|calculate|calculated|answer|confirm|confirmed|verify|verified)"
    r"|can(?:'|’)?t\s+(?:be\s+)?(?:determine|determined|calculate|calculated)"
    r"|unable\s+to\s+(?:determine|calculate|find|answer)"
    r"|no\s+(?:information|data|evidence)\s+(?:is\s+|was\s+)?(?:available|provided|found|given)"
    r"|(?:insufficient|missing)\s+(?:information|evidence|data)"
    r"|(?:evidence|filing|document|excerpt)s?\s+(?:provided\s+)?(?:contains?|has|have)\s+no\b",
    re.IGNORECASE,
)

#: "<Company>'s 10-K / filing / report / annual report" -- the company a sentence
#: attributes its source to.
_SOURCE_COMPANY_RE = re.compile(
    r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3})(?:'s|’s)\s+"
    r"(?:\d{4}\s+)?(?:Form\s+)?(?:10-?K|10-?Q|8-?K|annual\s+report|filing|report|data|evidence)",
)

_NUMBER_RE = re.compile(r"(?<![\w.])\(?\$?\s?(-?\d[\d,]*(?:\.\d+)?)\s?(%|[KkMmBb]\b|million|billion|thousand)?")
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_GENERIC_SUBJECTS = {"the", "company", "our", "this", "its", "corporation", "inc", "group"}
_FILLER = re.compile(r"[^a-z0-9 ]")


def _words(text: str) -> Set[str]:
    return {w for w in _FILLER.sub(" ", (text or "").lower()).split() if len(w) >= 3}


def check_answer(
    query: str, answer: str, evidence: List[Dict[str, Any]], pot_res: Optional[Dict[str, Any]],
    entity: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The first failure found in `answer`, as {"failure": type, "detail": text}, or None."""
    text = answer or ""
    if not text.strip():
        return {"failure": "non_answer", "detail": "empty answer"}

    entity_words = _words(entity or "") if entity and entity.lower() not in ("company", "unknown") else set()
    evidence_companies = {ev.get("company") for ev in evidence if ev.get("company")}
    allowed_words = set(entity_words)
    for c in evidence_companies:
        allowed_words |= _words(str(c))
    for m in _SOURCE_COMPANY_RE.finditer(text):
        named = _words(m.group(1))
        if named and allowed_words and not (named & allowed_words) and not (named & _words(query))                 and not (named & _GENERIC_SUBJECTS):
            return {"failure": "wrong_company", "detail": m.group(1)}

    # Only the opening of the answer: a verdict given first and a caveat later
    # ("... the filing does not disclose the prior-year figure") is an answer.
    hit = _NON_ANSWER_RE.search(re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0])
    if hit:
        return {"failure": "non_answer", "detail": hit.group(0)}

    return None
