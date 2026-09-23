import json
import os
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

# How many of the retrieved evidence items (sorted by relevance_score)
# actually get formatted into the prompt text the LLM sees -- see
# generate_answer()'s own use of this below for the full history/
# reasoning. Exposed as a named module constant (not just a literal
# slice index) so orchestrator.py can import it and report to the
# frontend EXACTLY this same subset as "evidence_sources", instead of
# the full unfiltered evidence_buffer (which can hold up to
# retrieved items) -- the frontend's own Source Evidence
# panel was showing candidates the LLM never actually saw, making it
# impossible to tell from the UI alone whether an answer's evidence
# panel and its actual grounding agreed.
# 16, not 12: a real-LLM run showed the one correct page sitting just past
# the 12-item cut -- Amcor's Q2 FY2023 restructuring note (score-rank 15,
# holds the "87% employee liabilities" fact) and Verizon's FY2021
# expected-benefit-payments page (score-rank 14, holds the 2024 figures)
# were both retrieved but never reached the prompt.
EVIDENCE_PROMPT_CAP = 16

# Every one of the real cases that justified raising EVIDENCE_PROMPT_CAP
# above (12->16, and originally 4->6->12, see the trail of comments below
# where it's sliced) was a NARRATIVE/EXPLANATION question -- legal
# proceedings, customer concentration, gross-margin drivers -- never a
# NUMERIC question where the PoT sandbox already produced a verified,
# non-degraded result_value. For THAT narrow case the LLM's job is mostly
# to phrase pot_summary's own number in prose, not to search a wide
# evidence pool for a fact PoT never found -- so a much smaller window is
# a reasonable token-saving trim there specifically, without touching any
# of the documented narrative cases above. Off by default (0 = disabled,
# same as leaving EVIDENCE_PROMPT_CAP alone) until verified on real
# metrics-style questions; set RELIABLE_POT_EVIDENCE_CAP in the
# environment (e.g. "6") to enable.
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

    # ALL table blocks are kept (each row-truncated), not just the first:
    # a real statement page routinely splits one statement into several
    # blank-line-separated blocks, and keeping only the first silently hid
    # everything after it. Confirmed real case: Adobe FY2022's cash-flow
    # page has five blocks; only the first (depreciation/stock comp) reached
    # the LLM, so "Net cash provided by operating activities" and "Purchases
    # of property and equipment" (capex) were invisible and the model said
    # capex was not disclosed. A total budget stops extra blocks once the
    # item is already large; the first block is always kept.
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

        # parent_content (the FULL page/table this chunk came from), not
        # content (just the one matched row/paragraph fragment) — mirrors
        # what the frontend's own Source Evidence panel already does (see
        # orchestrator._build_evidence_info's docstring) and what
        # pot_reasoner's extraction already relies on. Confirmed real case:
        # a narrative note spanning several child chunks on one page (e.g.
        # Amcor's FY2023 "Note 5" listing three separate acquisitions, each
        # named in a different chunk) had the LLM see only whichever single
        # ~150-char fragment happened to be evidence[0] — usually just the
        # first item named — even though the full page (now available via
        # parent_content, see parser._chunk_text_to_passages) already
        # covers all of them. max_chars raised from the function's 600
        # default to comfortably fit a full single-page note/table rather
        # than just a fragment of one -- and raised AGAIN from 2000 to
        # 4000 once parser.py's own chunk_size grew from 800 to 3000 (see
        # that constant's own docstring): a single retrieved chunk can now
        # be up to 3000 chars on its own, and parent_content (the whole
        # page) is routinely longer still, so 2000 chars often cut off
        # BEFORE reaching content the retrieval step deliberately
        # surfaced. Confirmed real case: AMD's FY2022 "What drove revenue
        # change" question retrieved the correct page (containing "driven
        # by a 64% increase in Data Center segment revenue... EPYC...")
        # but that sentence sat past character 2000 of the page's own
        # parent_content, so the LLM's answer cited the OTHER two drivers
        # it could still see (Gaming, Xilinx/Embedded) while silently
        # omitting the one that got truncated away.
        # Sorted by the retriever's own relevance_score, NOT the order
        # items happen to sit in `evidence` -- for a non-numeric question
        # with multiple retrieval sub-queries (see orchestrator.py's
        # non-numeric loop), `evidence` is several sub-queries' hit lists
        # concatenated in whichever order those sub-queries happened to
        # run, so a plain evidence[:4] slice is really "the first
        # sub-query's own top few candidates", not "the 4 most relevant
        # items across every sub-query". Confirmed real case: AMD's FY2022
        # "What drove revenue change" question's OWN keyword sub-query
        # ("AMD Revenue Net Revenue...") ran first and doesn't mention a
        # year at all, so AMD's unrelated FY2015 filing content filled its
        # own top slots on equal footing with the real FY2022 content --
        # the one passage that actually named the Data Center/EPYC driver
        # ranked 5th within THAT sub-query alone and never reached the
        # unsorted evidence[:4] cut, even though it clearly outranks the
        # FY2015 content by score once every sub-query's results are
        # considered together. A local copy -- `evidence` itself is left
        # untouched for any other consumer (e.g. the Source Evidence
        # panel) that may rely on its original order.
        # Score-sorted, with each sub-query keeping its own top passages
        # (BM25 scores of different queries are not comparable) -- see
        # evidence_selection.select_with_quota.
        sorted_evidence = select_with_quota(evidence, EVIDENCE_PROMPT_CAP)
        # 12, not 6 -- a genuinely multi-page narrative topic (e.g. a
        # litigation/legal-proceedings discussion, or a list of several
        # acquisitions each described on its own page) routinely has its
        # relevant content spread across MORE than 4 distinct pages, each
        # scoring close to the others. Confirmed real case (the ORIGINAL
        # reason this was already raised from 4 to 6): Boeing's FY2022
        # "materially important ongoing legal battles" question has
        # relevant evidence on pages 4, 19, 113, 128, 146, 148, and 149 --
        # the one page naming the Lion Air/Ethiopian Airlines litigation
        # specifically (page 113) ranked 5th by score, just outside a
        # 4-item cut.
        #
        # Raised again from 6 to 12 for the SAME reason, a rank further
        # out: two more confirmed real cases where the one genuinely
        # correct passage scored close to, but just past, a 6-item cut --
        # Boeing's OWN "who are Boeing's primary customers" question (the
        # sentence stating "Revenues from the U.S. government... 40%...
        # of consolidated revenues" ranked #9, edged out by several
        # higher-scoring but topically-adjacent passages -- e.g. a
        # DIFFERENT true statistic, "non-U.S. customers = 41% of
        # revenues", answering a related but distinct question); and
        # Johnson & Johnson's "what drove gross margin change" question
        # (the passage listing the actual named drivers -- "One-time
        # COVID-19 vaccine manufacturing exit related costs...", "driven
        # by:" -- shares no literal "gross margin" wording at all, so it
        # depends entirely on OTHER matched terms to rank, landing well
        # outside a 6-item window even after a companion retrieval-layer
        # fix (see orchestrator.py's RETRIEVAL_TOP_K_NARRATIVE) got it
        # into the evidence buffer in the first place -- raising THIS cap
        # too was still needed since a passage present in the buffer but
        # cut from the prompt here is exactly as invisible to the LLM as
        # one retrieval never found. Each item can be up to 4000 chars
        # (see max_chars above), so 12 items is a still-reasonable ~48K-
        # char evidence budget for a single LLM call on a modern context
        # window.
        #
        # See RELIABLE_POT_EVIDENCE_CAP's own docstring above: a NUMERIC
        # question whose PoT sandbox already produced a verified result
        # doesn't need the full narrative-sized window -- none of the
        # documented cases above are this shape. Off (0) by default.
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

        pot_summary = ""
        if pot_res:
            result_value = pot_res.get("result_value")
            # The sandbox's own variable assignments (e.g. "net_income =
            # 1182.0  # table-partial <- evidence[9] Line Item ...") are
            # the ONLY authoritative record of which specific number among
            # several same-labeled candidates the calculation actually
            # used. Without this, the model has no way to tell which
            # figure was used when it writes supporting prose ("this
            # figure is derived from net income of $X million") and ends
            # up re-picking a plausible-looking but DIFFERENT number
            # straight out of the raw evidence text below instead —
            # confirmed real case: a ROA answer's headline result (1.35%)
            # was correctly computed from net_income=1182 (quoted
            # verbatim per the instruction below), but the SAME answer's
            # supporting sentence separately cited "$1,248 million" as
            # the net income, because only the final ratio, never the
            # inputs that produced it, was ever shown to the model.
            pot_code_text = pot_res.get("code", "") or ""
            output_log_text = pot_res.get("output_log", "") or ""
            # The sandbox's own generic "no relevant structured data found"
            # fallback (pot_reasoner.py's _build_calculation_code, both the
            # extracted_table and free_text branches) always sets
            # result_value to a bare 0.0 alongside this exact warning text
            # -- a deliberately meaningless placeholder, not a genuine
            # computed fact, printed so an honest "couldn't compute this"
            # beats a confidently wrong number borrowed from an unrelated
            # line item. Without this check, the unconditional "MUST quote
            # this exact number in your HEADLINE" instruction below applied
            # here too, making the model literally open its answer with
            # "0.0 --" or "PoT result: 0.0" as if that were a real finding.
            # Confirmed real case: American Express's own "Does AMEX have an
            # improving operating margin profile...?" and "What drove gross
            # margin change...for American Express?" questions (gold: the
            # metric simply isn't measured for a financial institution) --
            # the model's reasoning and conclusion were already correct, but
            # the answer opened with an oddly out-of-place "0.0" because
            # this instruction told it to.
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
                # A non-zero PoT result directly answers a "Has company X
                # done Y?" yes/no question (paid dividends, reported
                # restructuring costs, etc.) -- spelled out explicitly so
                # the model doesn't need to infer direction from a raw
                # evidence row's own accounting notation (parentheses
                # around a number mean a negative amount/cash outflow,
                # NOT zero or "nothing happened", but that convention is
                # easy to misread when several evidence rows are shown
                # together). Confirmed real case: MGM Resorts' FY2022
                # dividend question -- result_value correctly computed as
                # 4048 (from a cash-flow-statement row reading "(4,048)"),
                # yet the model's answer stated "MGM did not pay
                # dividends", contradicting its own PoT result.
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
{verification_summary}

【RESPONSE FORMAT REQUIREMENTS】:
1. The first line MUST be a direct, conclusive answer (1-2 sentences) with key numbers and percentages.
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
   labeled by the LATER calendar year (e.g. year ended Jan 28, 2023 = the
   company's own "fiscal 2022" but commonly called "FY2023"). If the
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
    the finer rows making it up (e.g. "Developed Europe", "Emerging
    Markets"), rank the FINEST rows -- an aggregate averages away the
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
    internal segment code that is itself defined as multiple places (e.g.
    "AMESA" = "Africa, the Middle East and South Asia") into those actual
    places -- an abbreviation is not itself a geography. When the filing
    reports revenue by its OWN named geographic segments (e.g. "United
    States", "EMEA", "APAC", "LACC"), use those names -- optionally with
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
24. If the question says "if <metric> is not a useful metric, state that and
    explain why" and the company is a bank, card issuer, insurer or other
    financial institution (no COGS/gross-profit/ordinary-operating-income
    line), START by saying this metric isn't how such a company's
    performance is measured, and why -- do not open with a plain Yes/No
    about the metric itself.

【FORWARD-LOOKING / GUIDANCE / ONE-TIME EVENTS】:
25. If asked whether a dividend is STABLE/growing/consistent and the
    evidence states an explicit streak ("the 65th consecutive year of
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
28. If a spin-off/divestiture/separation question's evidence states that
    separation-related costs are STILL being incurred as of (or through)
    the reporting period (e.g. "we expect to incur costs of approximately
    $X million ..., of which Y% has been incurred ... through <the current
    quarter>"), answer YES -- the ongoing, unfinished cost of completing
    the separation means it is still in progress, even when another
    sentence says the legal transaction itself closed earlier. Only answer
    NO when the evidence shows no such ongoing separation costs at all.
    Relatedly, when evidence gives a separation/spin-off cost sentence
    shaped "we expect to incur costs of approximately $X million ..., of
    which approximately Y% has been incurred ... through <period>", read $X
    as the amount ALREADY incurred (the Y% portion), not the total --
    compute the implied total as $X / (Y/100), and the remaining (future)
    amount as implied total minus $X. State the implied total, the amount
    already incurred, and the remaining amount.
29. When a question asks whether an unusual/non-recurring/one-time event
    affected a result, NAME the event using the filing's own line-item
    wording, not only its amount (e.g. "the gain on completion of the
    Consumer Healthcare JV transaction ($8,107 million)", not just "a
    one-time gain of $8,107 million"). The same applies to a margin/income/
    expense driver that comes from an acquisition, merger or divestiture --
    name the other company or deal (e.g. "amortization of intangibles from
    the Xilinx acquisition"), never just "acquisition-related".

【NAMING AND NUMBER-FORMAT CONVENTIONS】:
30. If asked whether the company paid/declared DIVIDENDS and the evidence
    has both a PER-SHARE rate and an aggregate dollar total, state BOTH --
    the per-share rate is usually the more specific fact being asked for.
31. For a ratio/multiple result (turnover ratio, current ratio, quick
    ratio), state the number alone (e.g. "17.98") -- do not append a
    trailing "x". Percentages still get a trailing "%".
32. If asked what each shareholder could receive in a bankruptcy/
    liquidation, headline the TANGIBLE book value per share (goodwill and
    other intangibles are not distributable); mention plain book value per
    share only as context.
33. If asked about the nature, composition or purpose of a liability or
    other total the evidence breaks into components, give each component's
    amount AND its percentage of the total (e.g. "employee-related $81
    million, about 87% of the $93 million liability").

【FINAL CHECK before you answer -- these two failure modes have been observed live, more than once】:
34. Did the question ask "best/worst/top performer" with no stated basis? If
    so, does your answer give BOTH the largest-amount reading AND the
    fastest-growth reading (rule 21)? Did the question ask about
    accelerating/decelerating growth with more than one guidance basis
    available? If so, are BOTH bases stated, with the verdict leading on
    the operational/constant-currency basis (rule 26)?
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
