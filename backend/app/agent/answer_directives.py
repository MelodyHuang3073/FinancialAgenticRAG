"""Directives about the SHAPE of a question, inserted into the answer prompt only
when the question matches.

Each one used to be a standing line of the prompt that every question paid for
in tokens. A line the answer model must obey is worded as a short CRITICAL
note placed right after the evidence, and is built here only for the question
shapes that need it, so the standing prompt stays short.

  ranking_directive       "which X performed best / had the biggest drop"
  fiscal_year_directive   a question's FY label vs a January/February year-end
"""
import re
from typing import Optional, Set

# "performed the best/worst", "top performer": no stated basis -> size AND growth
_BEST_WORST_RE = re.compile(
    r'\bperform\w*\s+(?:the\s+)?(?:best|worst)\b|\b(?:top|best|worst)[\s-]perform\w*'
    r'|\b(?:best|worst)\s+(?:\w+\s+){0,2}perform\w*',
    re.IGNORECASE,
)
_GROWTH_WORD_RE = re.compile(r'\b(?:growth|grew|grow|increase\w*|decline\w*|drop\w*|decrease\w*)\b', re.IGNORECASE)

# "which region/segment/category ... biggest/highest/lowest ..."
_WHICH_RANK_RE = re.compile(
    r'\bwhich\s+(?:\S+\s+){0,4}?(?:regions?|segments?|divisions?|categor(?:y|ies)|business(?:es)?|products?|types?)\b'
    r'[^?.]{0,120}?\b(?:biggest|largest|highest|lowest|greatest|smallest|most|least|worst|best)\b',
    re.IGNORECASE,
)

# a fiscal year ending in January/February, and the question's own FY label
_EARLY_YEAR_END_RE = re.compile(r'\bended\s+(January|February)\s+(\d{1,2}),\s*(20\d\d)', re.IGNORECASE)
# A 52/53-week year of a calendar-year company ends on the day nearest December 31,
# so it can end in the first days of January; that is still its calendar year, not a
# January/February fiscal year.
_CALENDAR_YEAR_SPILL_LAST_DAY = 10
_QUESTION_FY_RE = re.compile(r'\b(?:FY|fiscal\s+(?:year\s+)?)\s*(20\d\d)\b', re.IGNORECASE)


def ranking_directive(query: str) -> str:
    parts = []
    if _BEST_WORST_RE.search(query) and not _GROWTH_WORD_RE.search(query):
        parts.append(
            "this question does not say whether \"best/worst\" means size or growth, so the "
            "first sentence MUST name BOTH: the item with the largest amount AND the item "
            "with the fastest growth rate (find the growth percentages in the evidence), "
            "each with its figure. An answer with only one of the two is wrong."
        )
    if _WHICH_RANK_RE.search(query):
        parts.append(
            "to say WHICH item has the biggest, highest or lowest value, rank only the "
            "finest-level rows the table shows, never an aggregate row (such as a regional "
            "total) that is made up of listed sub-rows; compare every one of them and name "
            "the winning row with its figure."
        )
    return "\n⚠️ CRITICAL: " + " Also, ".join(parts) if parts else ""


# "fiscal 2025 ... ended February 1, 2025" and "ended January 31, 2026 (fiscal 2025)":
# a period's own label next to its January/February end date.
_LABEL_THEN_END_RE = re.compile(
    r'\bfiscal\s+(?:year\s+)?(20\d\d)\b[^.]{0,120}?\bended\s+(?:on\s+)?(?:January|February)\s+\d{1,2},\s*(20\d\d)',
    re.IGNORECASE,
)
_END_THEN_LABEL_RE = re.compile(
    r'\bended\s+(?:on\s+)?(?:January|February)\s+\d{1,2},\s*(20\d\d)[^.]{0,80}?\bfiscal\s+(?:year\s+)?(20\d\d)\b',
    re.IGNORECASE,
)


def labels_the_filing_uses(evidence_text: str) -> set:
    """The fiscal-year numbers the filing itself attaches to a period that ends
    in January or February."""
    labels = {int(label) for label, _ in _LABEL_THEN_END_RE.findall(evidence_text)}
    labels |= {int(label) for _, label in _END_THEN_LABEL_RE.findall(evidence_text)}
    return labels


def _early_year_end_years(evidence_text: str) -> set:
    """Years in which the filing says a period ended in January or February, leaving
    out the first days of January (the end of a 52/53-week calendar year)."""
    years = set()
    for month, day, year in _EARLY_YEAR_END_RE.findall(evidence_text):
        if month.lower() == "january" and int(day) <= _CALENDAR_YEAR_SPILL_LAST_DAY:
            continue
        years.add(int(year))
    return years


def fiscal_year_directive(query: str, evidence_text: str, filing_labels: Optional[Set[int]] = None) -> str:
    """For a fiscal year that ends in January or February. The question's year
    follows the filing's own labels: when the filing names a period with that
    number, that is the period asked about and nothing is shifted. Only when no
    period carries the asked number and the year that ENDS in the asked year is
    labelled with the year before ("fiscal 2022" for the year ended early 2023)
    is the question's year the year that ends in it. `filing_labels` are the
    numbers the whole filing attaches to a period (not only the retrieved
    passages), so a filing that uses the asked number somewhere is never shifted."""
    q_year = _QUESTION_FY_RE.search(query)
    if not q_year:
        return ""
    asked = int(q_year.group(1))
    if asked in labels_the_filing_uses(evidence_text) or asked in (filing_labels or ()):
        return ""
    if asked not in _early_year_end_years(evidence_text):
        return ""
    if not re.search(rf'\bfiscal\s+(?:year\s+)?{asked - 1}\b', evidence_text, re.IGNORECASE):
        return ""
    return (
        f"\n⚠️ CRITICAL: this filing's fiscal year ends in early {asked}, and the company labels "
        f"that year \"fiscal {asked - 1}\". The question's FY{asked} IS that year: use the "
        f"figures the filing labels \"fiscal {asked - 1}\" (and its quarters) for it. Do not "
        f"say a figure is missing because it carries the label \"fiscal {asked - 1}\"."
    )


# "is growth expected to accelerate / slow down": a verdict about the direction of a rate
_ACCELERATION_RE = re.compile(
    r'\b(?:accelerat\w*|decelerat\w*|slow(?:s|ing)?\s+down|speed(?:s|ing)?\s+up)\b', re.IGNORECASE
)


def acceleration_directive(query: str) -> str:
    """Guidance is often printed on more than one basis (reported, adjusted,
    operational / constant-currency), and the verdict can differ between them."""
    if not _ACCELERATION_RE.search(query):
        return ""
    return (
        "\n⚠️ CRITICAL: this question compares a growth rate with the rate before it. If the "
        "evidence gives the outlook on more than one basis (for example reported, adjusted, "
        "operational or constant-currency), state, for EACH basis, the outlook growth (midpoint "
        "if a range) and the prior year's growth on that same basis, and say whether the "
        "verdict is the same on every basis. Never answer from one basis only."
    )


# A filer's own customer-concentration disclosure: "Revenues from <customer> ... represented
# N% of consolidated revenues". Scanned over the whole retrieved pool, because the page that
# holds it is not always among the passages that fit the prompt.
_CUSTOMER_CONCENTRATION_RE = re.compile(
    r'[Rr]evenues?\s+from\s+(?:the\s+)?([^,()\n]{2,80}?)(?:\s*\([^)]{0,160}\))?[^.]{0,120}?'
    r'\brepresented\s+(\d+(?:\.\d+)?)\s*%[^.]{0,120}?consolidated\s+revenues?',
    re.IGNORECASE,
)


def customer_concentration_directive(query: str, evidence_text: str) -> str:
    """For a "who are the customers" question: when the filing states how much of its
    consolidated revenue one customer represents, that sentence is the answer's headline
    fact, not a revenue split across the company's own segments or locations."""
    if "customer" not in query.lower():
        return ""
    found = _CUSTOMER_CONCENTRATION_RE.search(evidence_text)
    if not found:
        return ""
    sentence = " ".join(found.group(0).split())[:300]
    return (
        "\n⚠️ CRITICAL: the evidence explicitly states \"" + sentence + "\" -- the filer's own "
        "customer-concentration disclosure and the direct answer to a \"who are the customers\" "
        "question. You MUST name that customer and give its share of consolidated revenue, "
        + found.group(2) + "% (the first figure listed is the latest year), as a headline fact, "
        "and also name the other customer groups the filing identifies. "
        "A different percentage for the same customer in a segment table, or a share that "
        "describes how revenue splits across the company's own segments, products or "
        "locations, is not the company-wide share and must not replace it."
    )


def question_directives(query: str, evidence_text: str, filing_labels: Optional[Set[int]] = None) -> str:
    return (
        ranking_directive(query)
        + acceleration_directive(query)
        + customer_concentration_directive(query, evidence_text)
        + fiscal_year_directive(query, evidence_text, filing_labels)
    )
