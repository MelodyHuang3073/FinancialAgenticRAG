"""The accounting conventions the answer model is told to apply, in one place.

A convention reaches the model in up to three forms, deliberately:
  1. ACCOUNTING_CONVENTIONS -- the definitions, shown in the prompt's glossary.
  2. a directive about THIS evidence -- built by a detector below when the
     question and the retrieved passages match the convention's shape, placed
     right after the evidence in CRITICAL wording.
  3. a line in the prompt's closing checklist (llm_client.py).
The repetition is intentional: a convention stated only once in the middle of a
long prompt was not followed by the answer model in testing, while a directive
next to the evidence and a reminder at the end were.
"""
import re


ACCOUNTING_CONVENTIONS = '''⑥ Accounting conventions (definitions -- apply them literally, even where a
sentence could be read another way)
- A figure in parentheses is negative or an outflow, not zero.
- "Incurred" means a cost has already occurred. In a disclosure of costs of
  approximately $X "of which approximately Y% has been incurred", $X is the
  amount already incurred (the Y% portion), so the implied total is
  $X / (Y/100) and the amount still to come is that total minus $X. An
  activity (a separation, restructuring or integration) whose costs are
  still being incurred in the current period is still in progress, even if
  its main transaction closed earlier. A question asking whether such an
  activity is under way is therefore answered "Yes" while its costs are
  still being incurred; this needs no calculation.
- "Leverage" of an expense means it fell as a percent of net sales;
  "deleverage" means it rose.
- "Operational" or "constant-currency" figures exclude currency translation
  and are the basis for judging underlying growth; "reported" and plain
  "adjusted" figures include it. A verdict on whether growth speeds up or
  slows down is judged on that underlying basis: compare the outlook's
  operational growth with the prior year's growth as the filing prints it (if
  the prior year is printed only on the reported or adjusted basis, use that
  figure and say so), and give the other basis's outlook as context.
- Capital intensity is judged primarily by return on assets (ROA): a low ROA
  (well under about 10%) means the business needs a large asset base for
  the profit it earns (capital-intensive); a healthy ROA (about 10% or more)
  means it is not. Capex/revenue and fixed assets/total assets are
  supporting context; total assets/revenue alone overstates intensity when
  assets are mostly goodwill, intangibles or financial assets.
- A question's fiscal year follows the filing's own labels: if the filing names
  a period "fiscal N", that period is the question's "FY N" (or "fiscal N"),
  whichever month it ends in. Only when the filing names no period N, and the
  year that ends in January or February of N is labelled N-1 (a company that
  names its year by the calendar year it STARTS in), is the question's FY N
  that year ending in N: say so in one clause.
- Banks, card issuers and insurers have no cost-of-goods-sold or gross-profit
  line, so gross or operating margin is not how they are measured; if asked,
  say so first and why, then give the closest meaningful measure.
'''


#: A question about an initiative's status or remaining cost (a separation,
#: spin-off, restructuring or integration) and the disclosure shape "we expect
#: to incur costs of approximately $X million ..., of which approximately Y%
#: has been incurred ... through <period>". Accounting convention: "incurred"
#: means already occurred, so $X is the already-incurred Y% portion (implied
#: total = X / (Y/100), still to come = implied total - X), and an activity
#: whose costs are still being incurred this period is still in progress.
#: Applied as a directive about THIS evidence because the convention stated
#: only in the glossary was not followed.
_INITIATIVE_QUERY_RE = re.compile(
    r'\bspin(?:ning)?[\s-]?off\w*|\bseparat\w+\b|\brestructur\w+|\bintegrat\w+', re.IGNORECASE
)
_HOW_MUCH_QUERY_RE = re.compile(r'\bhow\s+much\b', re.IGNORECASE)
_INCURRED_COST_SENTENCE_RE = re.compile(
    r'expects?\s+to\s+incur\s+costs?\s+of\s+approximately\s+\$?\s*([\d,]+(?:\.\d+)?)\s*'
    r'(million|billion)\b[^.]{0,120}?of\s+which\s+approximately\s+(\d+(?:\.\d+)?)\s*%\s+'
    r'has\s+been\s+incurred',
    re.IGNORECASE | re.DOTALL,
)


def incurred_cost_directive(query: str, evidence_text: str) -> str:
    """The directive for an initiative-cost disclosure found in the evidence
    (see _INCURRED_COST_SENTENCE_RE), or "" when the question or the evidence
    does not match."""
    if not _INITIATIVE_QUERY_RE.search(query):
        return ""
    match = _INCURRED_COST_SENTENCE_RE.search(evidence_text)
    if not match:
        return ""
    amount_str, unit, pct_str = match.group(1), match.group(2), match.group(3)
    if _HOW_MUCH_QUERY_RE.search(query):
        amount, pct = float(amount_str.replace(",", "")), float(pct_str)
        if pct <= 0:
            return ""
        implied_total = amount / (pct / 100.0)
        remaining = implied_total - amount
        return (
            f"\n⚠️ CRITICAL: the evidence states an already-incurred cost of "
            f"approximately {amount_str} {unit} ({pct_str}% of the total, not the total itself). "
            f"Implied total = {amount_str} / ({pct_str}/100) = {implied_total:,.2f} {unit}; "
            f"still to come = implied total - {amount_str} = {remaining:,.2f} {unit}. "
            f"You MUST use these exact figures (implied total {implied_total:,.2f} {unit}, "
            f"still to come {remaining:,.2f} {unit}) -- do NOT read the disclosed "
            f"{amount_str} {unit} as the total with {100 - pct:g}% remaining; that is the "
            f"wrong direction."
        )
    return (
        "\n⚠️ CRITICAL: the evidence states costs of an initiative that are "
        "still being incurred in the current period (\"we expect to incur costs of "
        "approximately $X ..., of which approximately Y% has been incurred ... through "
        "<period>\"). \"Incurred\" means already occurred, so the initiative is still in "
        "progress even if its main transaction closed earlier: the completed transaction and "
        "the ongoing cost of finishing it are two different facts. Your VERDICT on a "
        "question whether the company is still doing it MUST be YES."
    )


#: Earnings-release idiom "leverage of X" / "deleverage of X": X FELL (leverage)
#: or ROSE (deleverage) as a percent of net sales, with no percentage printed
#: for X alone. A question about whether a cost rose or fell as a percent of
#: sales is answered from this direction, so when a question of that shape
#: meets such sentences the pairs are handed to the model as a directive about
#: THIS evidence (the glossary convention alone was not applied).
_LEVERAGE_ITEM_RE = re.compile(
    r'\b(de)?leverage\s+(?:of|in)\s+([^,.;]+?)'
    r'(?=\s+due\s+to\s+|[,.;]|\s+and\s+deleverage\b|\s+and\s+leverage\b|$)',
    re.IGNORECASE,
)
_PERCENT_OF_SALES_QUERY_RE = re.compile(r'(percent|%)[^.?]{0,40}\b(sales|revenue)\b', re.IGNORECASE)


def leverage_directive(query: str, evidence_text: str) -> str:
    """The directive for "(de)leverage of <item>" sentences in the evidence
    when the question asks about a cost as a percent of sales, else ""."""
    if not _PERCENT_OF_SALES_QUERY_RE.search(query):
        return ""
    seen, pairs = set(), []
    for de_prefix, item in _LEVERAGE_ITEM_RE.findall(evidence_text):
        item = " ".join(item.split())
        key = ("rose" if de_prefix else "fell", item)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    if not pairs:
        return ""
    listed = "; ".join(f'"{item}" {direction}' for direction, item in pairs)
    return (
        "\n⚠️ CRITICAL: the evidence states, with the earnings-release idiom \"leverage\" / "
        "\"deleverage\" of a cost line (leverage = that cost FELL as a percent of net sales; "
        "deleverage = it ROSE), that these items moved as follows as a percent of net sales: "
        f"{listed}. If the question asks whether a cost increased or decreased as a percent "
        "of sales and it is one of these items or means the same thing (match by meaning, the "
        "filing may word it differently), you MUST answer from this stated direction -- do NOT "
        "say it cannot be determined just because no percentage is printed for that item alone."
    )
