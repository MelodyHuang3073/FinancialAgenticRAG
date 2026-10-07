import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from app.tools.table_parser import is_markdown_separator_row
from app.agent.evidence_selection import select_with_quota

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if load_dotenv is not None and os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


# ── Daily free-token-quota tracking ─────────────────────────────────────────
# Every model call in the project reports to app/agent/usage_tracker.py (an
# append-only ledger, safe with parallel shards). This module reports its answer
# calls as caller="answer"; the decomposer reports as caller="decomposer".
from app.agent.usage_tracker import record_usage, DAILY_FREE_TOKEN_CAP  # noqa: E402,F401

# How many retrieved evidence items (sorted by relevance_score) are formatted into the
# prompt the LLM actually sees.
# Expose as a named module constant so other components can report the same subset as
# the LLM-facing evidence rather than the full retrieval buffer.
EVIDENCE_PROMPT_CAP = 16

# Higher evidence cap was raised to support narrative/explanation questions where the
# LLM must see more context; numeric questions that rely on a verified sandbox result do
# not require the same window.
# This feature is off by default and enabled via an environment override.
RELIABLE_POT_EVIDENCE_CAP = int(os.getenv("RELIABLE_POT_EVIDENCE_CAP", "0") or "0")


def is_reliable_pot_result(pot_res: Optional[Dict[str, Any]]) -> bool:
    """True only for a NUMERIC question whose PoT sandbox produced a real,
    trustworthy result_value -- excludes the "result is not reliable"
    placeholder (pot_res.get("result_value") is the sandbox's literal 0.0
    at this point, NOT yet converted to None -- that conversion happens
    later, only in orchestrator.py's frontend-facing return dict, not in
    what's passed to generate_answer), a degraded/approximated formula,
    and the no-calculation-path skip (result_value is already None
    there)."""
    if not pot_res:
        return False
    if pot_res.get("result_value") is None:
        return False
    if pot_res.get("is_degraded_formula"):
        return False
    if "result is not reliable" in (pot_res.get("output_log") or ""):
        return False
    return True


def _is_reasoning_model(model_name: str) -> bool:
    """
    True for OpenAI's reasoning-tier models (gpt-5 family, o1/o3/o4), which
    reject any non-default `temperature` value and require
    `max_completion_tokens` instead of `max_tokens` -- calling them with
    the same params used for gpt-4o-mini raises a 400 BadRequestError
    ("Unsupported value: 'temperature' does not support 0.2 with this
    model. Only the default (1) value is supported"), which the broad
    `except Exception: return None` around every LLM call here silently
    swallows into an empty answer instead of surfacing the real cause.
    Duplicated in decomposer.py (same rationale as this module's other
    small duplicated helpers) since that module builds its own OpenAI
    call independently and there's no shared client-config module to
    import from.
    """
    m = (model_name or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


#: A spin-off/separation STATUS question ("is X still separating Y") and an
#: AMOUNT question ("how much is still expected") need different handling:
#: only the status shape gets the "answer YES" directive below.
_SEPARATION_STATUS_QUERY_RE = re.compile(
    r'\bspin(?:ning)?[\s-]?off\w*|\bseparat\w+\b', re.IGNORECASE
)
_HOW_MUCH_QUERY_RE = re.compile(r'\bhow\s+much\b', re.IGNORECASE)

#: Detector for the disclosure shape "we expect to incur costs of approximately
#: $X million ..., of which approximately Y% has been incurred ... through
#: <period>". Accounting convention applied here: "incur" means the cost has
#: already occurred, so $X is the already-incurred Y% portion (implied total =
#: X / (Y/100), remaining = implied total - X), and costs still being incurred
#: in the current period mean the initiative is still in progress. Captures $X
#: and Y so the derivation is computed in code, and is injected as a directive
#: about THIS evidence because a numbered prose rule alone was not followed.
_SEPARATION_COST_SENTENCE_RE = re.compile(
    r'expects?\s+to\s+incur\s+costs?\s+of\s+approximately\s+\$?\s*([\d,]+(?:\.\d+)?)\s*'
    r'(million|billion)\b[^.]{0,120}?of\s+which\s+approximately\s+(\d+(?:\.\d+)?)\s*%\s+'
    r'has\s+been\s+incurred',
    re.IGNORECASE | re.DOTALL,
)

#: Detector for a filer's own dividend-increase streak ("the Nth consecutive
#: year of dividend increases"). The prompt asks for it in prose, but a numbered
#: rule among many is not followed reliably, so when the phrase is in the
#: evidence it is also injected as a directive about that specific text.
_DIVIDEND_STREAK_RE = re.compile(
    r'(\d+(?:st|nd|rd|th))\s+consecutive\s+year\s+of\s+dividend\s+increases?',
    re.IGNORECASE,
)

#: Detector for a filer's own customer-concentration disclosure ("Revenues from
#: <entity> ... represented N% ... of consolidated revenues"). Injected as a
#: directive only when this shape is present, so it cannot affect unrelated
#: questions. (A static prose rule for the same purpose was tried and removed:
#: it was visible on every question and interfered with unrelated ones.)
_CUSTOMER_CONCENTRATION_RE = re.compile(
    r'[Rr]evenues?\s+from\s+(?:the\s+)?([^,()\n]{2,80}?)\s*'
    r'(?:\([^)]{0,120}\)\s*)?,?\s*'
    r'(?:primarily\s+recorded\s+at\s+[^,\n]{0,60},?\s*)?'
    r'represented\s+(\d+(?:\.\d+)?)\s*%[^.\n]{0,80}?consolidated\s+revenues?',
    re.IGNORECASE,
)

#: Extractor for a filing's own acquisitions-note company sub-headings, each
#: followed by "On <date>, we acquired ...". Used to hand the model the
#: filing's own list of acquired companies, so that "which companies were
#: acquired" is answered from the filing's structure rather than from the
#: model's memory of the filer's acquisition history (a grounding failure
#: seen when the note was retrieved correctly). Matched directly against the
#: full evidence text, WITHOUT first locating an enclosing "Acquisitions"
#: section: retrieved passages for one filing's note can land in the
#: evidence out of page order (sorted by relevance, not position), which
#: broke an earlier version that required the section heading and its
#: sub-headings to be contiguous in the right order. The verb itself
#: ("acquired" / "completed the acquisition of") already excludes a
#: differently-worded neighboring disclosure (e.g. a divestiture's "completed
#: the sale of ..."), so no section boundary is needed to avoid picking those
#: up. No separate topic gate is needed: the sub-heading pattern itself
#: (a short title-case line immediately followed by a dated "we acquired /
#: completed the acquisition of" sentence) is already specific enough on its
#: own not to fire on unrelated evidence.
_ACQUISITION_SUBHEADING_RE = re.compile(
    r'\n([A-Z][A-Za-z0-9&.,\' \-]{1,40})\n\s*On\s+\w+\s+\d{1,2},\s*\d{4},\s*we\s+'
    r'(?:acquired|completed\s+the\s+acquisition\s+of)',
)


def _extract_acquisition_note_companies(evidence_text: str) -> list:
    """Returns the company names named as their own sub-heading immediately
    before an "On <date>, we acquired/completed the acquisition of ..."
    sentence anywhere in the evidence, in the order they first appear, or
    [] if no such structured disclosure is found."""
    names = _ACQUISITION_SUBHEADING_RE.findall("\n" + evidence_text)
    # De-dupe while preserving order (a company can recur, e.g. a
    # measurement-period-adjustment paragraph naming it again).
    seen = set()
    ordered = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


#: Detector for a filing's own "completed the acquisition of N% equity/
#: ownership/membership interest in <target>" or "acquired N% of the
#: outstanding shares of <target>" disclosure -- the common variants a 10-K's
#: acquisitions note uses to state the ownership stake obtained (confirmed
#: across several unrelated filers' own wording: "100% of the outstanding
#: shares of" / "100% of the outstanding equity interests in" / "85% of the
#: equity interests in" / "the remaining 50% ownership interest in" / "100%
#: of the membership interests of"). A numbered prose rule already instructs
#: the model to state the ownership stake when listing acquisitions, but a
#: rule among 30+ others is not followed reliably when the model's own
#: narrative focuses on dollar figures instead -- surfacing each actually-
#: matched percentage/target pair as a directive (like the dividend-streak
#: and customer-concentration detectors above) makes the omission far less
#: likely regardless of which company or how many acquisitions are listed.
_ACQUISITION_EQUITY_INTEREST_RE = re.compile(
    r'(?:completed\s+(?:the|its)\s+acquisition\s+of|acquired)\s+'
    r'(?:an?\s+)?(?:additional\s+|the\s+remaining\s+)?(\d+(?:\.\d+)?)%\s+'
    r'(?:'
    r'(?:of\s+(?:the\s+)?)?(?:outstanding\s+)?(?:equity|ownership|membership)\s+interests?'
    r'|'
    r'of\s+the\s+outstanding\s+shares(?:\s+and\s+voting\s+interests)?'
    r')\s+'
    r'(?:in|of)\s+'
    r'([^.,\n]{3,100})',
    re.IGNORECASE,
)


def _extract_acquisition_equity_interests(evidence_text: str) -> list:
    """Returns (percent, target_description) pairs for every "acquisition of
    N% equity interest in <target>" disclosure found anywhere in the
    evidence, in order of first appearance, deduplicated."""
    matches = _ACQUISITION_EQUITY_INTEREST_RE.findall(evidence_text)
    seen = set()
    ordered = []
    for pct, target in matches:
        target = " ".join(target.split())
        key = (pct, target)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


#: Detector for the earnings-release idiom "leverage of X" / "deleverage of
#: X" (also "X leveraged" is covered by the general prose rule, not this
#: regex). This is standard SEC earnings-release vocabulary for describing a
#: cost line's change as a percent of net sales WITHOUT printing an explicit
#: number for that line alone: "leverage" means the cost FELL as a percent of
#: sales, "deleverage" means it ROSE. A numbered prose rule already explains
#: this convention, but a rule among 30+ others is not followed reliably --
#: confirmed case: a filer's own SG&A paragraph named several contributing
#: sub-items this way with no percentage printed for any single one of them,
#: and the model concluded the direction "cannot be determined" for a
#: sub-item asked about individually, even though the evidence states its
#: direction explicitly via this idiom. Surfacing each actually-matched
#: item/direction pair as a directive (like the other detectors above) makes
#: that omission far less likely regardless of which cost item or filer.
_LEVERAGE_ITEM_RE = re.compile(
    r'\b(de)?leverage\s+(?:of|in)\s+([^,.;]+?)'
    r'(?=\s+due\s+to\s+|[,.;]|\s+and\s+deleverage\b|\s+and\s+leverage\b|$)',
    re.IGNORECASE,
)


def _extract_leverage_items(evidence_text: str) -> list:
    """Returns (direction, item_description) pairs -- direction is "increased"
    (deleverage) or "decreased" (leverage) -- for every "leverage/deleverage
    of <item>" disclosure found anywhere in the evidence, in order of first
    appearance, deduplicated."""
    matches = _LEVERAGE_ITEM_RE.findall(evidence_text)
    seen = set()
    ordered = []
    for de_prefix, item in matches:
        item = " ".join(item.split())
        direction = "increased" if de_prefix else "decreased"
        key = (direction, item)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _truncate_evidence_content(content: str, max_chars: int = 600, max_table_rows: int = 10) -> str:
    """
    Truncate one evidence item's content for the LLM prompt.

    A plain character-count slice ([:max_chars]) can land in the middle of
    a Markdown table row, corrupting it (e.g. cutting a '|---|---|'
    separator or a data row in half) and confusing the LLM about which
    number belongs to which column/period. So:
      - If `content` contains a Markdown table (detected via a '|---|'
        separator row), it is never character-truncated. Instead it's
        row-truncated: the header row + separator row + up to
        `max_table_rows` data rows are kept, with an
        "...(more rows omitted)" marker appended if rows were dropped.
        Anything before the header (e.g. a "Company: X | Report: Y |
        Period: Z" metadata prefix) is preserved as-is.
      - Otherwise (ordinary narrative text), the original [:max_chars]
        behaviour is unchanged.
    """
    lines = content.split("\n")
    sep_idx = next(
        (i for i, l in enumerate(lines) if i > 0 and is_markdown_separator_row(l)),
        None,
    )
    if sep_idx is None:
        return content[:max_chars]

    # When keeping table content, retain all table blocks (each row truncated) rather
    # than only the first block, to avoid omitting later rows from the LLM's view.
    # A total budget still applies once the item is large.
    table_budget = max_chars * 2
    out: List[str] = list(lines[:sep_idx - 1])
    cur: Optional[int] = sep_idx
    first_block = True
    while cur is not None:
        data_lines: List[str] = []
        j = cur + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or "|" not in stripped:
                break
            data_lines.append(lines[j])
            j += 1
        block = [lines[cur - 1], lines[cur]] + data_lines[:max_table_rows]
        if len(data_lines) > max_table_rows:
            block.append("...(more rows omitted)")
        if not first_block and len("\n".join(out)) + len("\n".join(block)) > table_budget:
            out.append("...(more tables omitted)")
            break
        if not first_block:
            out.append("")
        out.extend(block)
        first_block = False
        cur = next(
            (k for k in range(j, len(lines)) if k > 0 and is_markdown_separator_row(lines[k])),
            None,
        )
    return "\n".join(out)


class LLMAnswerGenerator:
    def __init__(self) -> None:
        self._client = None
        self._model = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                from openai import OpenAI
            except Exception:
                openai_api_key = None
            else:
                self._client = OpenAI(api_key=openai_api_key)
                self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                return self._client

        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not google_api_key:
            return None

        try:
            from google import genai
        except Exception:
            return None

        try:
            self._client = genai.Client(api_key=google_api_key)
            self._model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")
            return self._client
        except Exception:
            return None

    def generate_answer(
        self,
        query: str,
        answer_mode: str,
        evidence: List[Dict[str, Any]],
        route_res: Dict[str, Any],
        pot_res: Optional[Dict[str, Any]] = None,
        verification_res: Optional[Dict[str, Any]] = None,
        sub_questions: Optional[List[Dict[str, Any]]] = None,
        external_context: Optional[List[Dict[str, Any]]] = None,
        understanding: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        client = self._get_client()
        if not client:
            return None

        # Pass parent_content (the full page/table) to the LLM, not just the matched
        # fragment, so multi-chunk notes/tables are seen in context.
        # Sort selected items by retriever relevance_score across sub-queries; do not
        # rely on the concatenated retrieval order.
        sorted_evidence = select_with_quota(evidence, EVIDENCE_PROMPT_CAP)
        # Use a larger per-query evidence item count for multi-page narrative topics
        # since relevant content can be spread across many pages that score similarly.
        # Numeric questions validated by the sandbox do not need this wider window; that
        # behavior is configurable.
        effective_cap = EVIDENCE_PROMPT_CAP
        if (
            RELIABLE_POT_EVIDENCE_CAP > 0
            and answer_mode == "NUMERIC"
            and is_reliable_pot_result(pot_res)
        ):
            effective_cap = RELIABLE_POT_EVIDENCE_CAP
        evidence_text = "\n".join(
            f"- [{item.get('company', 'Company')} / {item.get('table_name', 'Source')}] "
            f"{_truncate_evidence_content(item.get('parent_content') or item.get('content', ''), max_chars=4000, max_table_rows=30)}"
            for item in sorted_evidence[:effective_cap]
        )

        # These detectors scan the FULL pre-cap `evidence` list, not just the
        # EVIDENCE_PROMPT_CAP window shown in the prompt: a fact that ranked just
        # outside the cap is as invisible to the model as one never retrieved.
        full_evidence_text = "\n".join(
            item.get("parent_content") or item.get("content", "")
            for item in evidence
        )

        dividend_streak_summary = ""
        streak_match = _DIVIDEND_STREAK_RE.search(full_evidence_text)
        if streak_match and "dividend" in query.lower():
            dividend_streak_summary = (
                f"\n⚠️ CRITICAL: the evidence above explicitly states the "
                f"\"{streak_match.group(0)}\" -- an official, filer-disclosed "
                "streak of consecutive annual dividend increases. This is the "
                "single strongest piece of evidence for any question about "
                "whether the dividend trend is stable/growing/consistent. You "
                "MUST state this streak explicitly in your answer (not only a "
                "computed multi-year per-share trend from other evidence) -- "
                "an official streak claim is more authoritative than an "
                "independently recomputed trend."
            )

        customer_concentration_summary = ""
        cust_match = _CUSTOMER_CONCENTRATION_RE.search(full_evidence_text)
        if cust_match and "customer" in query.lower():
            cust_entity = cust_match.group(1).strip()
            cust_pct = cust_match.group(2)
            customer_concentration_summary = (
                f"\n⚠️ CRITICAL: the evidence above explicitly states "
                f"\"Revenue from {cust_entity} represented {cust_pct}% of "
                "consolidated revenues\" -- this is a direct customer-"
                "concentration disclosure and the definitive answer to a "
                "\"who are the customers\" question. You MUST cite this "
                f"entity ({cust_entity}) and this exact percentage "
                f"({cust_pct}%) as a headline fact. Do NOT substitute a "
                "different-shaped percentage describing how revenue splits "
                "across the company's OWN business segments or product "
                "lines (e.g. \"Segment X represented Y% of revenue\") -- "
                "that answers a different question (what the company "
                "sells), not who buys it. Also do NOT substitute a "
                f"different percentage for {cust_entity} found elsewhere in "
                "the evidence (e.g. a segment/customer-type breakdown table) "
                "even if it names the same entity -- a different table can "
                "use a narrower or different scope definition for the same-"
                "looking label, and this explicit sentence is the one "
                "directly answering the question asked."
            )

        acquisition_note_summary = ""
        if "acqui" in query.lower():
            acq_companies = _extract_acquisition_note_companies(full_evidence_text)
            if acq_companies:
                names_str = ", ".join(acq_companies)
                acquisition_note_summary = (
                    f"\n⚠️ CRITICAL: the evidence above contains the filing's "
                    f"OWN dedicated \"Acquisitions\" note, which names exactly "
                    f"these compan{'y' if len(acq_companies)==1 else 'ies'} as "
                    f"their own sub-heading with transaction details: "
                    f"{names_str}. If asked which/how many companies were "
                    f"acquired, your answer MUST be drawn ONLY from this list "
                    f"-- do NOT add any other company, even one you recognize "
                    f"as a real acquisition this filer made in some OTHER "
                    f"year, unless it also appears as its own sub-heading "
                    f"here. This filing's own note is the authoritative "
                    f"source, not your general knowledge of this company's "
                    f"acquisition history."
                )

        acquisition_equity_interest_summary = ""
        if "acqui" in query.lower():
            equity_interests = _extract_acquisition_equity_interests(full_evidence_text)
            if equity_interests:
                pairs_str = "; ".join(f"{pct}% equity interest in {target}" for pct, target in equity_interests)
                acquisition_equity_interest_summary = (
                    f"\n⚠️ CRITICAL: the evidence above explicitly states the ownership "
                    f"stake acquired in each of these transactions: {pairs_str}. When "
                    f"describing or listing these acquisitions, you MUST state this "
                    f"percentage for each one, not only the dollar amount -- the "
                    f"ownership stake (full vs. partial) is a materially different fact "
                    f"from the purchase price, and a description that gives only the "
                    f"price is incomplete."
                )

        leverage_item_summary = ""
        leverage_items = _extract_leverage_items(full_evidence_text)
        if leverage_items:
            pairs_str = "; ".join(f'"{item}" {direction}' for direction, item in leverage_items)
            leverage_item_summary = (
                f"\n⚠️ CRITICAL: the evidence above states, using the standard "
                f"earnings-release idiom \"leverage\"/\"deleverage\" of a cost line "
                f"(leverage = that cost FELL as a percent of net sales; deleverage = "
                f"that cost ROSE as a percent of net sales), that the following items "
                f"moved this way: {pairs_str}. If the question asks whether one of "
                f"these items (or a close match, e.g. \"wages\"/\"payroll\" for "
                f"\"store payroll and benefits\") increased or decreased as a percent "
                f"of net sales, you MUST answer from this stated direction -- do NOT "
                f"say the direction cannot be determined just because no explicit "
                f"percentage is printed for that item alone."
            )

        separation_status_summary = ""
        sep_cost_match = (
            _SEPARATION_COST_SENTENCE_RE.search(evidence_text)
            if _SEPARATION_STATUS_QUERY_RE.search(query) else None
        )
        if sep_cost_match and not _HOW_MUCH_QUERY_RE.search(query):
            separation_status_summary = (
                "\n⚠️ CRITICAL: the evidence above contains a sentence disclosing "
                "separation/spin-off costs that are still being incurred as of the "
                "current reporting period (an \"we expect to incur costs of approximately "
                "$X million ..., of which approximately Y% has been incurred ... through "
                "<period>\" sentence). In accounting, \"incurred\" means the cost has "
                "already occurred; costs still being incurred in this period mean the "
                "separation is STILL IN PROGRESS. Your VERDICT on any question asking "
                "whether the company is still separating/spinning off that business MUST "
                "be YES, even when another sentence says the legal transaction closed "
                "earlier: the completed transaction and the ongoing cost of finishing it "
                "are two different facts."
            )
        elif sep_cost_match and _HOW_MUCH_QUERY_RE.search(query):
            # In accounting "incur" means already occurred: $X is the already-incurred
            # Y% portion, so implied total = X / (Y/100) and remaining = implied total - X.
            # Computed here so the arithmetic direction is not left to the model.
            amount_str, unit, pct_str = sep_cost_match.group(1), sep_cost_match.group(2), sep_cost_match.group(3)
            amount = float(amount_str.replace(",", ""))
            pct = float(pct_str)
            if pct > 0:
                implied_total = amount / (pct / 100.0)
                remaining = implied_total - amount
                separation_status_summary = (
                    f"\n⚠️ CRITICAL: the evidence above states an already-incurred "
                    f"separation/spin-off cost of approximately {amount_str} {unit} "
                    f"({pct_str}% of the total, NOT the total itself): in accounting, "
                    f"\"incur\" means the cost has already occurred. Implied total = "
                    f"{amount_str} / ({pct_str}/100) ≈ {implied_total:,.2f} {unit}; "
                    f"remaining (future) amount = implied total - {amount_str} ≈ "
                    f"{remaining:,.2f} {unit}. You MUST use these exact computed figures "
                    f"(implied total ≈ {implied_total:,.2f} {unit}, remaining ≈ "
                    f"{remaining:,.2f} {unit}) -- do NOT read the disclosed {amount_str} "
                    f"{unit} as the total with {100 - pct:g}% remaining; that is the "
                    f"wrong direction."
                )

        pot_summary = ""
        if pot_res:
            result_value = pot_res.get("result_value")
            # Include the sandbox's own variable assignments as the authoritative record
            # of which numeric inputs were used in calculations.
            # Expose those assignments to the LLM so supporting prose cites the exact
            # inputs the computation used, avoiding accidental reference to other raw
            # figures.
            pot_code_text = pot_res.get("code", "") or ""
            output_log_text = pot_res.get("output_log", "") or ""
            # Explains why the sandbox fallback sets result_value to 0.0 with a generic
            # "no relevant structured data found" message; this is a deliberate
            # placeholder to avoid returning a confidently wrong numeric value when no
            # computed fact exists.
            # Avoid treating this placeholder as a real finding when generating
            # headlines or leading answers.
            is_unreliable_fallback = "result is not reliable" in output_log_text
            pot_summary = (
                f"\nPoT result: {result_value}\n"
                f"PoT calculation code (these are the EXACT input values actually used):\n"
                f"{pot_code_text[:1200]}\n"
                f"Sandbox output: {output_log_text[:600]}"
            )
            if is_unreliable_fallback:
                pot_summary += (
                    "\n⚠️ NOTE: The sandbox could NOT find structured data in the retrieved "
                    "evidence that actually matches what this question asks about -- the 0.0 "
                    "result above is a meaningless placeholder, NOT a real computed answer. "
                    "Do NOT cite \"0\"/\"0.0\" anywhere in your answer as if it were a genuine "
                    "figure or headline result. Instead, answer directly from the qualitative "
                    "evidence text below (e.g. explaining what actually drove a change, or why "
                    "this metric isn't a meaningful one for this company), the same way you "
                    "would if no PoT result had been computed at all."
                )
            elif result_value is not None:
                pot_summary += (
                    f"\n⚠️ CRITICAL: The PoT result above ({result_value}) was computed by a "
                    "verified Python sandbox, NOT by you. You MUST quote this exact number -- "
                    "in your HEADLINE/first-line answer, not only in supporting detail -- "
                    "(reformatted for units/rounding exactly as the question asks, but not "
                    "recalculated) as your answer. Do NOT redo the arithmetic yourself from "
                    "the raw evidence figures below -- independent re-derivation has produced "
                    "wrong numbers before even when every input you cited was correct. When "
                    "citing ANY figure, headline or supporting (e.g. \"net income of $X "
                    "million\", \"a dividend of $X million\"), you MUST quote the exact value "
                    "assigned to that variable in the PoT calculation code above -- NOT a "
                    "different, more detailed-looking number for the same line item that "
                    "appears in the raw evidence below, even if that other number comes with "
                    "an appealing extra detail (a per-share rate, a specific date) that makes "
                    "it look more complete or authoritative. The code's variables are ground "
                    "truth for what was used; the raw evidence commonly contains OTHER rows "
                    "sharing the exact same line-item label but naming a DIFFERENT fiscal year "
                    "in nearby text (e.g. a multi-year rollforward statement repeats \"Dividends "
                    "declared and paid to common shareholders\" once per year, each with its own "
                    "number) -- these were NOT the ones the sandbox used and must not replace "
                    "the PoT result."
                )
                # Notes that a non-zero PoT result should be interpreted as a direct
                # affirmative answer to a yes/no question (e.g., whether a filer did X),
                # since numeric conventions in source rows (like parentheses for
                # negatives) can be misread when shown together.
                # Emphasize aligning the natural-language conclusion with the computed
                # result_value to avoid contradictory statements.
                if result_value not in (0, 0.0):
                    pot_summary += (
                        f"\nNote: a raw evidence row may show this figure in "
                        f"parentheses, e.g. \"({abs(result_value):g})\" -- that is "
                        "standard financial-statement notation for a negative "
                        "amount or cash outflow, NOT zero or \"did not occur\". "
                        f"The PoT result ({result_value}) being non-zero means "
                        "the event/amount the question asks about DID occur -- "
                        "for a yes/no question, this generally means the answer "
                        "is Yes."
                    )
            if pot_res.get("is_degraded_formula"):
                pot_summary += (
                    f"\n⚠️ CRITICAL: {pot_res.get('degraded_note', '')} "
                    "You MUST explicitly state this limitation in your answer -- do not "
                    "present the shown number as the exact metric the question asked for."
                )
        # The generic "quote the PoT result as your headline" instruction conflicts
        # with a verdict that must follow a different signal: the capital-intensity
        # ratio (total assets / revenue) is context only for a "is X capital-
        # intensive" verdict, which follows ROA.
        if pot_res and pot_res.get("formula_used") == "capital_intensity_ratio":
            pot_summary += (
                "\n⚠️ CRITICAL OVERRIDE: the PoT result above (the capital-intensity "
                "RATIO, total assets / revenue) is supporting context only -- it is NOT "
                "the signal that decides a capital-intensive verdict, and must not be "
                "quoted as if it were the answer. Find or compute ROA (net income / "
                "average total assets) from the evidence and let ROA's direction decide "
                "Yes/No: a low ROA (well under ~10%) means capital-intensive (Yes), a "
                "healthy ROA (~10%+) means it is not (No). Do NOT let this ratio, or a "
                "low fixed-assets share of total assets, override the ROA-based verdict."
            )

        verification_summary = ""
        if verification_res:
            checks = verification_res.get("checks", {})
            verification_summary = "\nVerification summary: " + ", ".join(
                f"{k}={'passed' if v.get('passed') else 'needs review'}"
                for k, v in checks.items()
            )

        external_text = ""
        if external_context:
            external_text = "\nExternal web evidence:\n" + "\n".join(
                f"- {item.get('source', 'web')} | {item.get('title', 'External')} | {str(item.get('content', ''))[:800]}"
                for item in external_context[:2]
            )

        understanding_context = ""
        if understanding:
            understanding_context = (
                f"\nFinancial question understanding:\n"
                f"Entity: {understanding.get('entity', 'company')}\n"
                f"Metric: {understanding.get('metric', 'financial metric')}\n"
                f"Intent: {understanding.get('intent', 'NUMERIC')}\n"
                f"Financial primer: {understanding.get('financial_primer', '')}\n"
            )

        prompt = f"""You are a professional financial analysis assistant. Please answer the question in English.

Question: {query}
Answer Mode: {answer_mode}
Routing Reason: {route_res.get('reason', '')}
{understanding_context}

Available Evidence:
{evidence_text}
{pot_summary}
{dividend_streak_summary}
{customer_concentration_summary}
{acquisition_note_summary}
{acquisition_equity_interest_summary}
{leverage_item_summary}
{separation_status_summary}
{verification_summary}

【RESPONSE FORMAT REQUIREMENTS】:
1. The first line MUST be a direct, conclusive answer (1-2 sentences) with key
   numbers and percentages. Two failure shapes to avoid: (a) burying the
   single most identifying fact (the acquired company's name, the specific
   driver) at the END of the answer instead of attaching it to its FIRST
   mention -- if the opening sentence says "acquisition-related" or "a
   driver", name the deal/company right there, not several sentences
   later; (b) opening with a figure that doesn't answer what was literally
   asked (e.g. leading a "what products does X sell" answer with X's total
   revenue) -- the opening sentence's numbers must be the ones the
   question asked for, not other notable figures from the same evidence.
2. Highlight key figures/results in **bold**.
3. Keep total response under 150 words.
4. Do NOT repeat raw evidence verbatim or list variable names.
5. If evidence is insufficient, state clearly what data is missing.
6. If the question asks which SECURITIES are registered to trade on a national
   exchange, the authoritative source is the filing's own "Securities
   registered pursuant to Section 12(b)/12(g)" cover-page table -- trust it
   even if it lists only common stock and no debt securities. A separate
   "Long-Term Debt" footnote answers a different question (financing amount)
   and must never stand in for the exchange-registration answer.
7. A company whose fiscal year ends in January/February is still often
   labeled by the LATER calendar year. If the
   question names a year no evidence item is literally labeled with, but
   the evidence has data for that company's adjacent fiscal year, use it,
   state the assumption in one clause, and answer -- do not refuse. Never
   apply this to a December fiscal year-end, where labels are unambiguous.
8. A 10-K states its own fiscal year on its cover. If the evidence comes
   from the filing for the year asked about, answer from it -- never say
   that year's disclosure is missing just because of the filename; a 10-K
   also carries the prior year's comparatives.
9. Before concluding the evidence does NOT contain something asked for (an
   acquisition, a litigation category, a note by name), scan every evidence
   item first -- there are up to 16, and the right one is not always the
   first or most prominent. Never say "not disclosed in the excerpts" while
   an item literally contains that fact.
10. If asked WHICH region/segment had the biggest drop/highest growth/etc.,
    and the evidence lists both an aggregate row (e.g. "International") and
    the finer rows making it up, rank the FINEST rows -- an aggregate averages away the
    extreme sub-item. Compare every non-overlapping finest-level row for
    the same period.
11. If asked how MANY of something a company has (stores, locations,
    employees) with no brand/banner named, and the evidence shows per-brand
    rows plus a "Total" row, answer with the Total row -- even when one
    brand row happens to share the company's own name, that row is still
    only one banner, not the whole count.
12. If asked what SHARE/percent/contribution of the company-level total a
    segment made, divide by the company-level CONSOLIDATED figure the
    filing reports (after corporate/other and eliminations), not by the sum
    of the segment figures shown.
13. A business-segment table's figures belong to the segment named in the
    section heading of the SAME page -- never attribute them to a different
    segment just because the question is about that one. If the page
    doesn't clearly say which segment a table is for, omit the figure.
14. If asked WHICH segment/business had the highest or lowest value of some
    measure, first list the value for EVERY segment shown on the page
    before choosing -- a segment table is often split into several blocks,
    each naming different segments, and the answer may sit in a later
    block. A negative value is lower than any positive one. Never treat the
    firm-total column as a segment.
15. If asked what GEOGRAPHIES/regions a company operates in: expand an
    internal segment code that is itself defined as multiple places into those actual
    places -- an abbreviation is not itself a geography. When the filing
    reports revenue by its OWN named geographic segments, use those names -- optionally with
    the figures given -- as the answer's structure, not a flat unordered
    list of every place mentioned anywhere. Do NOT lead with employee/
    headcount geography -- the question is about where revenue/operations
    are, not where staff are located.

【ENUMERATION -- LIST EVERY ITEM THE EVIDENCE SUPPORTS, NOT JUST THE FIRST ONE OR TWO】:
16. If asked what DROVE or CAUSED a change, and the evidence describes
    MULTIPLE distinct contributing factors (several segments, products, or
    line items each with their own stated reason), name ALL of them the
    evidence supports -- do not stop after the first one or two that seem
    sufficient.
17. If asked to LIST items (acquisitions, legal matters, products,
    geographies), enumerate EVERY item the evidence names -- do not stop at
    a plausible-looking subset. NEVER name a specific company, transaction,
    amount or event unless the evidence EXPLICITLY connects it to the exact
    period/year asked -- a name matching something you recognize appearing
    ANYWHERE in the evidence is not enough; check what the surrounding
    sentence actually says (a stray mention, e.g. an executive's past
    employer, is not evidence of relevance to the asked period). When
    listing ACQUISITIONS specifically, also state the OWNERSHIP STAKE
    acquired if the evidence gives it.
18. If the verdict on LEGAL MATTERS is "Yes, materially important" and the
    evidence names multiple DISTINCT categories (e.g. a filing's own named
    sub-headings), give a one-sentence summary of what is actually ALLEGED
    in EACH category, not just its name -- including a category with no
    dollar figure disclosed yet. This enumerate-every-category instruction
    applies ONLY to a "Yes" verdict. If the verdict is "No" (ordinary-course
    matters, assessed immaterial), state "No" plainly with at most one
    brief clause noting ordinary-course matters exist -- do not enumerate
    the individual immaterial matters; heavy enumeration on a "No" reads as
    hedging. Decide the Yes/No verdict from the NATURE and SEVERITY actually
    described (fatal accidents, large settlement/remediation figures, an
    active bankruptcy proceeding, multiple concurrent suits) -- not from
    whether the filing happens to use the literal word "material"; that
    word's absence does not make the question unanswerable, and evidence
    naming specific serious matters is sufficient basis for a confident
    verdict without it.
19. If asked for the MAIN/MAJOR companies a filer ACQUIRED, use the
    acquisition/business-combination disclosures (purchase price, closing
    date, accounting) for the periods the question covers -- never a
    company only mentioned in passing elsewhere (old litigation history, an
    executive's biography).
20. When you list legal matters, acquisitions or similar events, also state
    the dollar amount the filing discloses for each one (settlement cap,
    fees, purchase price) next to that item -- not a different, only
    loosely related total.

【AMBIGUOUS "BEST/WORST" COMPARISONS -- these are the categories most often answered incompletely, double-check them】:
21. If asked which item "performed the best/worst" (or was the "top"
    performer) WITHOUT saying whether that means the largest amount or the
    fastest growth, give BOTH readings in one short answer: the item with
    the largest figure (value + share) AND the item with the highest growth
    rate (with that percentage), each with its period. Never give only one
    reading.
22. For a CAPITAL-INTENSIVE verdict when the PoT output shows ROA alongside
    assets/revenue, CapEx/Revenue and Fixed-Assets/Total-Assets: treat ROA
    as the PRIMARY signal, not the bare assets/revenue ratio. A healthy ROA
    (~10%+) argues AGAINST capital-intensive even if assets exceed revenue;
    a low ROA (well under ~10%) with a real fixed-asset base argues FOR it.
    A low Fixed-Assets/Total-Assets % does not override a low-ROA verdict
    if the total asset base is mostly goodwill/intangibles from
    acquisitions. Whenever ROA is available, your answer text MUST state
    that ROA percentage explicitly as a number -- do not only reason from
    it internally.
23. If asked what DROVE a margin change and the evidence has BOTH routine/
    recurring factors (input-cost inflation, pricing/volume/mix, FX,
    productivity) AND one-off/special items (litigation, impairment,
    restructuring, a named exit), lead the explanation with the one-off
    items -- even when a routine factor's stated impact is numerically
    larger. Routine factors are secondary context, not the headline.
24. If asked about a margin/profitability TREND (gross margin, operating
    margin, or similar) for a bank, card issuer, insurer or other financial
    institution (no COGS/gross-profit/ordinary-operating-income line) --
    even when the question is phrased as a plain "does X have an improving
    <metric> profile" or "what drove <metric> change" with NO explicit "if
    this isn't a useful metric" wording of its own -- START your answer by
    saying this metric isn't how such a company's performance is measured,
    and why, BEFORE any other computed verdict (e.g. net income margin).
    This applies purely based on the COMPANY'S business type (no COGS/
    gross-profit line in its own financials) -- do not require the
    question to use this rule's own trigger wording ("if not a useful
    metric") before applying it; a plain-phrased question about a
    financial institution's margin still needs this framing led first.

【FORWARD-LOOKING / GUIDANCE / ONE-TIME EVENTS】:
25. If asked whether a dividend is STABLE/growing/consistent and the
    evidence states an explicit streak (e.g. "the Nth consecutive year of
    dividend increases"), state that streak -- it is the strongest evidence
    of a stable trend.
26. If asked whether a growth rate is expected to accelerate/slow, and the
    evidence gives forward guidance on MORE THAN ONE basis (e.g. "Adjusted
    EPS" vs "Adjusted Operational EPS", or reported vs constant-currency),
    state the guidance midpoint + prior-year figure for EACH basis, then
    give the verdict and say which basis it rests on. If the bases point in
    different directions, LEAD the verdict with the operational/constant-
    currency basis (it removes currency noise), mentioning the other basis
    as a caveat -- never open with a plain "Yes"/"No" that only holds for
    one line.
27. Earnings-release wording "leverage of X" (or "X leveraged") means
    expense X FELL as a percent of net sales; "deleverage of X" means X
    ROSE as a percent of net sales. When a question asks whether a cost's
    percent of net sales increased/decreased and the evidence uses this
    wording for it, answer from it directly -- do not say the information
    is missing just because no explicit percentage figure is printed.
28. If an initiative's cost disclosure says "we expect to incur costs of
    approximately $X million ..., of which approximately Y% has been incurred
    ... through <period>", apply the accounting meaning of "incur" (the cost
    has already occurred): read $X as the amount ALREADY incurred (the Y%
    portion), not the total. Implied total = $X / (Y/100); remaining
    (future) amount = implied total minus $X. State the implied total, the
    amount already incurred and the remaining amount. If costs are still
    being incurred as of the reporting period, the initiative is still in
    progress, even when the legal transaction itself closed earlier.
29. When a question asks whether an unusual/non-recurring/one-time event
    affected a result, NAME the event using the filing's own line-item
    wording, not only its amount (e.g. "the gain on completion of [named
    transaction] ($X million)", not just "a one-time gain of $X million").
    The same applies to a margin/income/expense driver that comes from an
    acquisition, merger or divestiture -- name the other company or deal
    (e.g. "amortization of intangibles from the [Company Y] acquisition"),
    never just "acquisition-related".

【NAMING AND NUMBER-FORMAT CONVENTIONS】:
30. If asked whether the company paid/declared DIVIDENDS and the evidence
    has both a PER-SHARE rate and an aggregate dollar total, state BOTH --
    the per-share rate is usually the more specific fact being asked for.
31. For a ratio/multiple result (turnover ratio, current ratio, quick
    ratio), state the number alone -- do not append a
    trailing "x". Percentages still get a trailing "%".
32. If asked what each shareholder could receive in a bankruptcy/
    liquidation, headline the TANGIBLE book value per share (goodwill and
    other intangibles are not distributable); mention plain book value per
    share only as context.
33. If asked about the nature, composition or purpose of a liability or
    other total the evidence breaks into components, give each component's
    amount AND its percentage of the total (e.g. "[component] $X million,
    about Y% of the $Z million total"). Likewise, if asked which category/
    type is the largest or smallest among items that add up to a total,
    name it and state its share of that total.
34. If asked what PERCENT/SHARE of a FULL-PERIOD total (e.g. a full fiscal
    year) occurred during a SPECIFIC SHORTER SUB-PERIOD within it (e.g. one
    quarter), and the evidence contains BOTH a full-period total AND a
    separate sub-period disclosure for the SAME line item -- even if they
    come from two different evidence items (e.g. a balance-sheet-note total
    and a separate "Issuer Purchases of Equity Securities" quarterly
    table) -- compute the ratio directly (sub-period ÷ full-period). Do not
    say the breakdown "cannot be determined" just because the two numbers
    were not printed next to each other in the same table; this is a
    common failure mode confirmed live even when both numbers were plainly
    present in the evidence actually provided.

【FINAL CHECK before you answer -- these failure modes have been observed live, more than once, even when the rule above already covers them】:
35. Does the evidence contain an explicit "Nth consecutive year of dividend
    increases" streak claim? If so, and the question asks about dividend
    stability/trend, did you state that streak explicitly rather than only
    a self-computed multi-year trend (rule 25)? Did the question ask what
    percent/share of a full-period total happened in a specific sub-period
    (e.g. Q4 of the full year)? If so, did you check whether evidence
    contains BOTH the full-period total and a separate sub-period
    disclosure before concluding the breakdown "cannot be determined" --
    the two numbers are often in different evidence items, not the same
    table (rule 34).
36. Did the question ask "best/worst/top performer" with no stated basis? If
    so, does your answer give BOTH the largest-amount reading AND the
    fastest-growth reading (rule 21)? Did the question ask about
    accelerating/decelerating growth with more than one guidance basis
    available? If so, are BOTH bases stated, with the verdict leading on
    the operational/constant-currency basis (rule 26)? Did the question ask
    whether the company is CAPITAL-INTENSIVE (or a similarly-framed asset-
    heavy/asset-light verdict) and does your evidence/PoT output include
    ROA? If so, does your VERDICT follow ROA's direction -- a low ROA
    (well under ~10%) means capital-intensive, a healthy ROA (~10%+) means
    it is not -- rather than the bare CapEx/Revenue or Fixed-Assets/Total-
    Assets ratio (rule 22)? Does the evidence contain a sentence shaped "we
    expect to incur costs of approximately $X million ..., of which
    approximately Y% has been incurred ... through <period>"? If so, did
    you read $X as the amount already incurred and state the implied total
    and the remaining amount (rule 28)?
"""

        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                create_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "You are a professional financial report analysis assistant. Respond in clear English with a result-first format."},
                        {"role": "user", "content": prompt},
                    ],
                }
                if not _is_reasoning_model(self._model):
                    create_kwargs["temperature"] = 0.2
                response = client.chat.completions.create(**create_kwargs)
                record_usage(self._model, "answer", getattr(response, "usage", None), len(prompt))
                if response and getattr(response, "choices", None):
                    first_choice = response.choices[0]
                    message = getattr(first_choice, "message", None)
                    content = getattr(message, "content", None)
                    if content:
                        return str(content).strip()
            else:
                response = client.models.generate_content(model=self._model, contents=prompt)
                record_usage(self._model, "answer", getattr(response, "usage_metadata", None), len(prompt))
                if hasattr(response, "text") and response.text:
                    return str(response.text).strip()
                if hasattr(response, "candidates") and response.candidates:
                    first = response.candidates[0]
                    if hasattr(first, "content") and hasattr(first.content, "parts"):
                        parts = []
                        for part in first.content.parts:
                            if hasattr(part, "text") and part.text:
                                parts.append(part.text)
                        if parts:
                            return "".join(parts).strip()
        except Exception as e:
            # Surface a quota/rate-limit failure loudly instead of silently
            # returning None like every other failure here -- the caller's
            # fallback behavior is unchanged (still just gets None either
            # way), but without this, a daily free-tier quota running out
            # mid-run (e.g. gpt-5-mini's own daily cap) looked identical to
            # any other transient LLM hiccup: answers quietly got worse/
            # emptier for the REST of a long test run with no visible sign
            # of the actual cause in the console output. Detected via the
            # OpenAI SDK's own error code/type when available, falling
            # back to a substring check on the error text for whichever
            # provider client is in use (openai vs. Google GenAI both
            # raise their own exception types here).
            err_code = getattr(e, "code", None) or getattr(getattr(e, "body", None), "get", lambda *_: None)("code")
            err_text = str(e).lower()
            is_quota_or_rate_limit = (
                err_code in ("insufficient_quota", "rate_limit_exceeded")
                or type(e).__name__ in ("RateLimitError",)
                or "insufficient_quota" in err_text
                or "quota" in err_text
                or "rate_limit" in err_text
                or "rate limit" in err_text
                or "429" in err_text
            )
            if is_quota_or_rate_limit:
                print(
                    f"[LLM QUOTA/RATE-LIMIT] model='{self._model}' call failed: {e}",
                    file=sys.stderr, flush=True,
                )
            return None

        return None
