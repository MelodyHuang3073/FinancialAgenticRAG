"""Directives about the SHAPE of a question, inserted into the answer prompt only
when the question matches.

Each one used to be a standing line of the prompt that every question paid for
in tokens. A line the answer model must obey is worded as a short CRITICAL
note placed right after the evidence, and is built here only for the question
shapes that need it, so the standing prompt stays short.

  ranking_directive       "which X performed best / had the biggest drop"
  fiscal_year_directive   a question's FY label vs a January/February year-end
"""
import os
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
    r'\bfiscal\s+(?:year\s+)?(20\d\d)\b(?:(?!\bfiscal\b)[^.]){0,120}?\bended\s+(?:on\s+)?(?:January|February)\s+\d{1,2},\s*(20\d\d)',
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


class FilingPeriods:
    """What a whole filing says about its own fiscal years.

    labels  the fiscal-year numbers the filing names for its periods
    ends    {label: (year of the period's end date, "Month d, yyyy")} for the periods the
            filing ties to an end date in a sentence ("fiscal 2025 ... ended January 31, 2026")
    end_years  years in which the filing says a period ended in January or February
    """

    def __init__(self, labels: Optional[Set[int]] = None, ends: Optional[dict] = None,
                 end_years: Optional[Set[int]] = None):
        self.labels = set(labels or ())
        self.ends = dict(ends or {})
        self.end_years = set(end_years or ())

    def __contains__(self, year: int) -> bool:
        return year in self.labels

    def __iter__(self):
        return iter(self.labels)

    def __bool__(self) -> bool:
        return bool(self.labels or self.ends)


_LABEL_END_DATE_RE = re.compile(
    r'\bfiscal\s+(?:year\s+)?(20\d\d)\b(?:(?!\bfiscal\b)[^.]){0,120}?\bended\s+(?:on\s+)?((?:January|February)\s+\d{1,2},\s*(20\d\d))',
    re.IGNORECASE,
)
_END_DATE_LABEL_RE = re.compile(
    r'\bended\s+(?:on\s+)?((?:January|February)\s+\d{1,2},\s*(20\d\d))(?:(?!\bfiscal\b)[^.]){0,80}?\bfiscal\s+(?:year\s+)?(20\d\d)\b',
    re.IGNORECASE,
)
# "fiscal 2023, fiscal 2022 and fiscal 2021": consecutive years named together
_LABEL_SERIES_RE = re.compile(
    r'\bfiscal\s+(?:year\s+)?(20\d\d)\b[^.\n]{0,40}?\bfiscal\s+(?:year\s+)?(20\d\d)\b', re.IGNORECASE
)
# a table row key as the extractor prints it: "| 2024: 103"
_ROW_YEAR_RE = re.compile(r'\|\s*(20\d\d):')


def filing_periods(filing_text: str) -> FilingPeriods:
    """Read a filing's own fiscal-year naming from its whole text (any filing: it
    finds nothing when no period ends in January or February)."""
    from collections import Counter
    seen: dict = {}
    for label, date, year in _LABEL_END_DATE_RE.findall(filing_text):
        seen.setdefault(int(label), Counter())[(int(year), re.sub(r"\s+", " ", date))] += 1
    for date, year, label in _END_DATE_LABEL_RE.findall(filing_text):
        seen.setdefault(int(label), Counter())[(int(year), re.sub(r"\s+", " ", date))] += 1
    ends = {}
    for label, counts in seen.items():
        plausible = [(pair, n) for pair, n in counts.most_common() if pair[0] in (label, label + 1)]
        if plausible:
            ends[label] = plausible[0][0]
    labels = set(ends)
    for a, b in _LABEL_SERIES_RE.findall(filing_text):
        if int(a) == int(b) + 1:
            labels |= {int(a), int(b)}
    # Table columns headed by the company's own names: the newest column's year is
    # BEFORE the year the newest period ended in, so the headings are not end years.
    row_counts = Counter(int(y) for y in _ROW_YEAR_RE.findall(filing_text))
    ended = _early_year_end_years(filing_text)
    if row_counts and ended:
        # the columns of the annual statements head hundreds of rows; a forecast or
        # maturity table heads a few, so they do not set the newest column year
        columns = {y for y, n in row_counts.items() if n >= max(row_counts.values()) / 2}
        if max(columns) < max(ended):
            labels |= columns
    return FilingPeriods(labels, ends, ended)


def fiscal_column_directive(query: str, periods: Optional[FilingPeriods]) -> str:
    """The filing names a period "fiscal N" but that period ends in a later year, and its
    tables are headed by dates, which the extractor turns into the end date's year: the
    column headed N is then the year BEFORE fiscal N. Says which column is which."""
    if not isinstance(periods, FilingPeriods) or not periods.ends:
        return ""
    q_year = _QUESTION_FY_RE.search(query)
    if not q_year:
        return ""
    asked = int(q_year.group(1))
    end = periods.ends.get(asked)
    if not end or end[0] == asked:
        return ""
    prior = periods.ends.get(asked - 1)
    prior_text = f" (fiscal {asked - 1} ended {prior[1]})" if prior else ""
    return (
        f"\n⚠️ CRITICAL: in this filing \"fiscal {asked}\" is the period ended {end[1]}{prior_text}. "
        f"The extracted tables head date-headed columns with the YEAR OF THE END DATE, so fiscal "
        f"{asked} is the column headed {end[0]} and fiscal {asked - 1} is the column headed "
        f"{end[0] - 1}; the column headed {asked} is fiscal {asked - 1}. Answer a fiscal {asked} "
        f"question from the column headed {end[0]}. (A table whose headings print \"fiscal\" "
        f"names is read by those names; the newest column of an annual table is the newest "
        f"fiscal year.)"
    )


def fiscal_year_directive(query: str, evidence_text: str, filing_labels=None) -> str:
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
    if isinstance(filing_labels, FilingPeriods) and asked in filing_labels.ends:
        return ""
    # The end dates come from the whole filing, not only from the passages retrieved
    # this time: which passages are retrieved varies, the filing's fiscal year does not.
    whole_filing = filing_labels.end_years if isinstance(filing_labels, FilingPeriods) else set()
    if asked not in (_early_year_end_years(evidence_text) | whole_filing):
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
        "operational or constant-currency), list the outlook growth (midpoint if a range) of "
        "EACH basis, never only one, and compare each with the prior year's growth on the SAME "
        "basis. Reported and operational growth differ by the currency effect, so a verdict "
        "about speeding up or slowing down is judged on the operational basis (currency "
        "excluded). If the prior year's growth is printed on only one basis, reason from the "
        "filing's own statement of that year's currency effect (a headwind means the prior "
        "year's operational growth was at least its reported growth) and state that this "
        "comparison is across bases. Give the verdict, the basis it rests on, and the other "
        "bases' verdicts."
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


def question_directives(query: str, evidence_text: str, filing_labels=None) -> str:
    if os.environ.get("ABLATE_DOMAIN") == "1":
        # Question-type hints removed; the fiscal-year/column mapping stays
        # (it tells the model which column of THIS filing is the asked year).
        return (
            fiscal_year_directive(query, evidence_text, filing_labels)
            + fiscal_column_directive(query, filing_labels)
        )
    return (
        ranking_directive(query)
        + acceleration_directive(query)
        + customer_concentration_directive(query, evidence_text)
        + fiscal_year_directive(query, evidence_text, filing_labels)
        + fiscal_column_directive(query, filing_labels)
    )
