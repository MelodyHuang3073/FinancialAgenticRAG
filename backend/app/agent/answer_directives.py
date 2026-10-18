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

# "performed the best/worst", "top performer": no stated basis -> size AND growth
_BEST_WORST_RE = re.compile(
    r'\bperform\w*\s+(?:the\s+)?(?:best|worst)\b|\b(?:top|best|worst)[\s-]perform\w*',
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
_EARLY_YEAR_END_RE = re.compile(r'\bended\s+(January|February)\s+\d{1,2},\s*(20\d\d)', re.IGNORECASE)
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


def fiscal_year_directive(query: str, evidence_text: str) -> str:
    """For a fiscal year that ends in January or February: the filing labels it
    with the year it STARTS in ("fiscal 2022" for the year ended early 2023),
    while a question's "FY2023" means the year that ENDS in 2023."""
    q_year = _QUESTION_FY_RE.search(query)
    if not q_year:
        return ""
    asked = int(q_year.group(1))
    if not any(int(y) == asked for _, y in _EARLY_YEAR_END_RE.findall(evidence_text)):
        return ""
    if not re.search(rf'\bfiscal\s+(?:year\s+)?{asked - 1}\b', evidence_text, re.IGNORECASE):
        return ""
    return (
        f"\n⚠️ CRITICAL: this filing's fiscal year ends in early {asked}, and the company labels "
        f"that year \"fiscal {asked - 1}\". The question's FY{asked} IS that year: use the "
        f"figures the filing labels \"fiscal {asked - 1}\" (and its quarters) for it. Do not "
        f"say a figure is missing because it carries the label \"fiscal {asked - 1}\"."
    )


def question_directives(query: str, evidence_text: str) -> str:
    return ranking_directive(query) + fiscal_year_directive(query, evidence_text)
