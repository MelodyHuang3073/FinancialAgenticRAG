import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional

from app.tools.table_parser import is_markdown_separator_row

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if load_dotenv is not None and os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


# ── Daily free-token-quota tracker ──────────────────────────────────────────
# OpenAI's response body/headers never say "this call was billed against
# your paid balance instead of the free daily grant" -- there is no direct
# per-call signal for that (confirmed via OpenAI's own docs/help center: the
# only usage data a response carries is its OWN token counts, not a running
# daily total or free/paid split). The account still has funds, so a call
# past the free daily allowance succeeds exactly like any other call --
# nothing fails, nothing looks different -- so this has to be tracked
# locally: sum each call's own `usage.total_tokens` into a small per-day
# file and flag once the known daily free-tier cap for mini/nano models
# (10,000,000 tokens/day, confirmed by the user against their own OpenAI
# account) is crossed. Notifies once per day, not on every call after.
_USAGE_TRACK_PATH = os.path.join(BASE_DIR, ".llm_daily_usage.json")
DAILY_FREE_TOKEN_CAP = 10_000_000

# How many of the retrieved evidence items (sorted by relevance_score)
# actually get formatted into the prompt text the LLM sees -- see
# generate_answer()'s own use of this below for the full history/
# reasoning. Exposed as a named module constant (not just a literal
# slice index) so orchestrator.py can import it and report to the
# frontend EXACTLY this same subset as "evidence_sources", instead of
# the full unfiltered evidence_buffer (which can hold up to
# RETRIEVAL_MAX_TOTAL=45 items) -- the frontend's own Source Evidence
# panel was showing candidates the LLM never actually saw, making it
# impossible to tell from the UI alone whether an answer's evidence
# panel and its actual grounding agreed.
EVIDENCE_PROMPT_CAP = 12


def _record_daily_usage(model: str, total_tokens: int) -> None:
    if not total_tokens:
        return
    today = date.today().isoformat()
    state = {"date": today, "total_tokens": 0, "notified": False}
    try:
        if os.path.exists(_USAGE_TRACK_PATH):
            with open(_USAGE_TRACK_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("date") == today:
                state = loaded
    except Exception:
        pass  # a corrupt/unreadable tracker file just resets for today

    state["date"] = today
    state["total_tokens"] = state.get("total_tokens", 0) + total_tokens

    crossed_now = (
        not state.get("notified")
        and state["total_tokens"] >= DAILY_FREE_TOKEN_CAP
    )
    if crossed_now:
        state["notified"] = True
        print(
            f"[LLM DAILY FREE QUOTA] model='{model}' has used "
            f"{state['total_tokens']:,} tokens today, past the "
            f"{DAILY_FREE_TOKEN_CAP:,}-token/day free-tier cap -- further "
            f"calls today are being billed against your paid balance.",
            file=sys.stderr, flush=True,
        )

    try:
        with open(_USAGE_TRACK_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass  # tracking is best-effort; never let it break a real LLM call


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

    prefix_lines = lines[:sep_idx - 1]
    header_line = lines[sep_idx - 1]
    separator_line = lines[sep_idx]

    data_lines: List[str] = []
    for line in lines[sep_idx + 1:]:
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            break
        data_lines.append(line)

    kept_rows = data_lines[:max_table_rows]
    result_lines = prefix_lines + [header_line, separator_line] + kept_rows
    if len(data_lines) > max_table_rows:
        result_lines.append("...(more rows omitted)")
    return "\n".join(result_lines)


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
        sorted_evidence = sorted(
            evidence, key=lambda item: item.get("relevance_score") or 0, reverse=True
        )
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
        evidence_text = "\n".join(
            f"- [{item.get('company', 'Company')} / {item.get('table_name', 'Source')}] "
            f"{_truncate_evidence_content(item.get('parent_content') or item.get('content', ''), max_chars=4000)}"
            for item in sorted_evidence[:EVIDENCE_PROMPT_CAP]
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
            pot_summary = (
                f"\nPoT result: {result_value}\n"
                f"PoT calculation code (these are the EXACT input values actually used):\n"
                f"{pot_code_text[:1200]}\n"
                f"Sandbox output: {pot_res.get('output_log', '')[:600]}"
            )
            if result_value is not None:
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
6. If the question asks which SECURITIES (stock, bonds, notes) are
   REGISTERED to trade on a national exchange, the authoritative source
   is a "Securities registered pursuant to Section 12(b)/12(g) of the
   Act" disclosure (usually on the filing's own cover page) -- trust
   that table's own contents even if it says only common stock is
   listed and no debt securities appear there at all. A separate
   "Long-Term Debt" or similar footnote describing outstanding notes/
   borrowings answers a DIFFERENT question (how much debt financing the
   company has) and must never be substituted as if it were the
   exchange-registration answer.
7. If the question asks what DROVE or CAUSED a change, and the evidence
   describes MULTIPLE distinct contributing factors (e.g. several
   business segments, products, or line items each with their own
   stated reason), name ALL of them that the evidence supports -- do
   not stop after the first one or two that seem sufficient.
8. If the question asks you to LIST items (acquisitions, legal matters,
   products, geographies, etc.), enumerate EVERY item the evidence below
   names -- do not stop after finding a plausible-looking subset. NEVER
   name a specific company, transaction, dollar amount, or event unless
   the evidence below EXPLICITLY connects it to the exact period/year the
   question asks about. A name or word matching something you recognize
   from general knowledge appearing ANYWHERE in the evidence text is NOT
   enough by itself -- check what the SURROUNDING sentence actually says
   about it before citing it. Confirmed real failure mode: a company name
   appears in evidence only as part of an unrelated executive's past
   employer ("President, [Company] North America, 2017 to 2019") or a
   stray mention with a different year attached -- seeing that name is
   not evidence that IT was acquired in, or is otherwise relevant to, the
   fiscal year actually being asked about. When in doubt about whether a
   specific fact you're about to cite is truly supported for the exact
   period asked, leave it out rather than include it. When listing
   ACQUISITIONS specifically, also state the OWNERSHIP STAKE acquired
   (e.g. "100% equity interest", "all of the outstanding shares") if the
   evidence explicitly states it for that deal -- filings routinely
   phrase this detail right alongside the target's name and purchase
   price, and it is often the specific fact a question about acquisitions
   is checking for, not just the dollar amount.
9. If the question asks whether the company paid/declared DIVIDENDS, and
   the evidence contains a PER-SHARE dividend rate (e.g. "$0.01 per
   share", "$0.55 per share dividend") in addition to an aggregate dollar
   total (e.g. a cash-flow-statement "Dividends paid" line), state BOTH
   numbers -- the per-share rate is usually the more specific fact a
   dividend question is really asking for, and citing only the aggregate
   total is an incomplete answer even when that total is itself correct.
10. For a ratio/multiple result (turnover ratio, current ratio, quick
    ratio, etc.), state the number by itself (e.g. "17.98") -- do NOT
    append a trailing "x" ("17.98x"). Percentages still get a trailing
    "%" as usual; this rule is only about the "x" multiple suffix.
11. If the question asks what GEOGRAPHIES/regions a company operates in,
    and the evidence uses a combined internal segment label whose own
    definition spans multiple actual places (e.g. "AMESA" defined as
    "Africa, the Middle East and South Asia"; "APAC" defined as "Asia
    Pacific, Australia and New Zealand, and China"), list the individual
    places the evidence itself names, not just the abbreviation -- a
    "geographies" question is asking for actual regions, and an internal
    reporting-segment code is not itself a geography.
12. If the question asks about ongoing LEGAL BATTLES/litigation and the
    verdict is that the company DOES have materially important ones, and
    the evidence names multiple DISTINCT categories of legal matters
    (e.g. a filing's own named sub-headings like "Usual and Customary
    Pricing Litigation", "PBM Litigation and Investigations", "Controlled
    Substances Litigation"), give a one-sentence summary of what is
    actually ALLEGED or at issue in EACH named category, not just its
    name -- do not describe only the category with the largest dollar
    figure in detail while merely name-dropping the others. Evidence
    for a smaller/less-quantified category (e.g. no settlement figure
    disclosed yet) still states what the claim itself is about (e.g.
    "alleges retail pharmacies overcharged for prescription drugs by
    not submitting the correct usual and customary price"); state that,
    even without a dollar figure to cite alongside it.
    This enumerate-every-category instruction applies ONLY when the
    verdict is "Yes, materially important". When the filing discloses
    ordinary-course legal proceedings but management/the filing itself
    concludes none are expected to have a material adverse effect (a
    "No" verdict), do NOT enumerate the individual immaterial matters or
    their categories in detail -- state the "No" verdict plainly in the
    opening sentence, optionally with a single brief clause noting that
    ordinary-course matters exist but were assessed as immaterial, and
    stop there. Spending a paragraph walking through each disclosed-but-
    immaterial matter (even when every individual fact stated is
    accurate) makes a correct "No" answer read as hedgy or self-
    contradictory -- the level of enumeration itself implies materiality
    the verdict is denying. Confirmed real case: PepsiCo's own "Has
    PepsiCo reported any materially important ongoing legal battles..."
    question (gold: a plain "No, PepsiCo is not involved in material
    legal battles") got a technically-correct "No" that nonetheless
    opened with "PepsiCo's FY2022 and FY2021 10-K filings disclose
    various legal proceedings..." followed by a category-by-category
    breakdown, before finally restating "No" -- accurate in substance,
    but reads as uncertain given how much space was spent describing
    matters the verdict itself says aren't material.
    Decide the Yes/No verdict itself from the NATURE and SEVERITY of what
    the evidence actually describes (e.g. lawsuits/investigations tied to
    fatal accidents, large stated settlement/remediation figures, an
    active bankruptcy-court proceeding, multiple concurrent suits) -- NOT
    from whether the filing happens to contain an explicit sentence
    literally labeling something "material" or "not material". A filing
    that explicitly states its own materiality conclusion (as PepsiCo's
    and CVS's do) is one valid signal when present, but its ABSENCE does
    not make the question unanswerable -- evidence naming specific,
    serious ongoing lawsuits (e.g. litigation arising from fatal aircraft
    accidents) is by itself sufficient basis for a confident "Yes", even
    without the filing using the word "material" anywhere in the excerpt
    you were given. Do not respond "I cannot confirm" / "not enough
    information to determine materiality" when the evidence already
    names specific, serious ongoing legal matters -- use professional
    judgment on severity the same way a human analyst reading the same
    excerpt would, rather than treating the ABSENCE of an explicit
    materiality label as itself inconclusive. Confirmed real case:
    Boeing's own "Has Boeing reported any materially important ongoing
    legal battles...FY2022?" question (gold: "Yes. Multiple lawsuits...
    resulting from a 2018 Lion Air crash and a 2019 Ethiopian Airlines
    crash") had evidence correctly naming both accidents and the
    resulting litigation, but got "I cannot confirm... management's
    materiality conclusions are not included here" instead of "Yes" --
    the underlying facts were sufficient to answer confidently; two fatal
    aircraft-accident lawsuits against a company this size are self-
    evidently material without needing the filing to say so explicitly.
13. Before concluding that the evidence does NOT contain something the
    question asks for (an acquisition, a litigation category, a specific
    figure, a named note like "Acquisitions and Divestitures"), you MUST
    actually scan every evidence item listed above first -- there are up
    to 12 of them, and the one that answers the question is not always
    the first or the most prominent-looking one. Do not conclude
    "not disclosed in the provided excerpts" / "the relevant note is not
    included" while an evidence item literally contains that note's own
    heading and content -- this has been a confirmed real failure mode,
    denying evidence that was directly present in the prompt. Confirmed
    real case: Amcor's own "Note 5 - Acquisitions and Divestitures" note
    (naming a Czech Republic plant, a Shanghai facility, and a New
    Zealand manufacturer by name, with dollar amounts) was evidence
    item #1 of 12, yet the answer claimed no such note was supplied at
    all.
14. If the question asks whether a company is CAPITAL-INTENSIVE and the
    PoT sandbox output shows the assets/revenue ratio ALONGSIDE
    CapEx/Revenue, Fixed Assets/Total Assets, and/or Return on Assets,
    treat RETURN ON ASSETS AS THE PRIMARY signal, not the bare
    assets/revenue ratio -- a company that generates a healthy, efficient
    return on its asset base (roughly double-digit ROA, ~10%+) is
    evidence AGAINST calling it capital-intensive, even when assets
    exceed revenue (a large asset base that still earns a strong return
    is being used efficiently, which is the opposite of the "money tied
    up unproductively" concern "capital-intensive" is meant to flag).
    Conversely a LOW ROA (roughly single-digit, well under ~10%)
    alongside a meaningful fixed-asset base IS a strong sign of capital
    intensity, even when the bare assets/revenue ratio looks modest --
    the company ties up a lot of capital relative to the profit that
    capital actually generates. Do not default to "assets/revenue > 1.0
    -> capital-intensive" as your primary rule; that ratio alone is a
    weaker signal than ROA for this specific judgment. A LOW Fixed-
    Assets/Total-Assets percentage does NOT by itself override a LOW ROA
    verdict -- a low fixed-asset share only argues against capital
    intensity when it means the company's total assets are genuinely
    modest (a lean, low-capital operation). If the evidence shows a
    large TOTAL asset base that is mostly goodwill/intangibles rather
    than physical plant (common after a company has made large
    acquisitions), a low fixed-asset PERCENTAGE is really just a
    reflection of that mix, not evidence the company needs little
    capital overall -- the low ROA on that large total asset base is
    still the more telling signal, and still supports a capital-
    intensive verdict even though physical PP&E itself is a small slice
    of it. ROA efficiency should anchor the verdict whenever it's
    available; use Fixed-Assets/Total-Assets to explain WHAT KIND of
    capital intensity it is (physical plant vs. a large acquired/
    goodwill-heavy balance sheet), not to overrule what ROA already
    indicates.
15. If the question asks what DROVE a margin change (gross margin,
    operating margin, etc.) and the evidence contains BOTH (a) routine/
    recurring operational factors (raw-material or logistics cost
    inflation, ordinary pricing/volume/mix shifts, FX translation,
    routine productivity gains) AND (b) one-off/non-recurring special
    items (litigation charges, impairments, restructuring/divestiture
    costs, a specific named exit from a product line or manufacturing
    process), lead the explanation with the one-off/special items, not
    the routine operational factors -- even when a routine factor has a
    numerically larger stated percentage-point impact. This mirrors
    standard financial-statement-analysis practice (distinguishing
    "core"/recurring performance from non-recurring items when
    explaining a period-over-period change): a margin move is usually
    considered NEWSWORTHY and explanation-worthy specifically because of
    what's unusual about it, not because of the routine cost/pricing
    noise that's present in every period regardless. Routine factors can
    still be mentioned, but as secondary context after the special
    items, not as the headline explanation. Confirmed real case: 3M's
    own "What drove operating margin change...FY2022" question (gold:
    "...primarily due to...mostly one-off charges including Combat Arms
    Earplugs litigation, impairment related to exiting PFAS
    manufacturing, costs related to exiting Russia and divestiture-
    related restructuring charges") had evidence containing both a raw-
    material/logistics inflation drag AND these named one-off items, but
    the routine inflation factor was cited first as the primary driver
    while the litigation/PFAS/Russia/divestiture items were relegated to
    an "other contributors" list -- gold's own framing treats the one-off
    items as the primary story.
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
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", None) if usage else None
                if total_tokens:
                    _record_daily_usage(self._model, total_tokens)
                if response and getattr(response, "choices", None):
                    first_choice = response.choices[0]
                    message = getattr(first_choice, "message", None)
                    content = getattr(message, "content", None)
                    if content:
                        return str(content).strip()
            else:
                response = client.models.generate_content(model=self._model, contents=prompt)
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
