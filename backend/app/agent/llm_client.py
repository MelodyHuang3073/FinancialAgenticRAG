import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from app.tools.table_parser import is_markdown_separator_row
from app.agent.evidence_selection import select_with_quota
from app.agent.answer_directives import question_directives
from app.agent.accounting_conventions import (
    ACCOUNTING_CONVENTIONS, incurred_cost_directive, leverage_directive,
)

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


def _evidence_display_text(item: Dict[str, Any], max_chars: int = 4000) -> str:
    """The text the answer prompt shows for one retrieved passage.

    A table row is shown with its whole page: a lone row carries no column
    headers or units, and the calculation needs the table around it.
    A narrative chunk is shown with its page too, unless the page is longer
    than the per-passage cap AND the retrieved chunk lies past the point where
    head-truncation would cut: then the very sentences that were retrieved are
    lost, so the shown text is a window of the page that contains the whole
    chunk, centred on it, within the same cap. Every other passage is shown
    exactly as before.
    """
    parent = item.get("parent_content") or ""
    content = item.get("content", "")
    if item.get("type") != "text_note" or len(parent) <= max_chars:
        return parent or content
    body = content.split("Content:", 1)[-1].strip()
    pos = parent.find(body[:80]) if body else -1
    if pos < 0:
        return parent
    end = pos + len(body)
    if end <= max_chars:
        return parent
    spare = max_chars - len(body)
    if spare <= 0:
        return body[:max_chars]
    after = min(len(parent) - end, spare // 2)
    before = min(pos, spare - after)
    after = min(len(parent) - end, spare - before)
    return parent[pos - before:end + after]


_ANSWER_SYSTEM_PROMPT = (
    "You are a professional financial report analysis assistant. Respond in clear English with a result-first format, always in complete sentences -- never reply with only a number."
)


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
                self._model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
                return self._client

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=anthropic_api_key, max_retries=6)
                self._model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
                return self._client
            except Exception:
                pass

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
        filing_labels: Optional[set] = None,
    ) -> Optional[str]:
        client = self._get_client()
        if not client:
            return None

        # A table row is passed with its whole page (parent_content); a narrative
        # chunk is passed as the retrieved chunk itself (_evidence_display_text).
        # Sort selected items by retriever relevance_score across sub-queries; do not
        # rely on the concatenated retrieval order. Pin whichever passages the PoT
        # sandbox actually used (pot_res["used_evidence_ids"]) so a passage backing
        # the computed number can't rank just outside the window the model reads.
        pinned_ids = set(pot_res.get("used_evidence_ids") or []) if pot_res else None
        sorted_evidence = select_with_quota(evidence, EVIDENCE_PROMPT_CAP, pinned_ids=pinned_ids)
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
            f"{_truncate_evidence_content(_evidence_display_text(item), max_chars=4000, max_table_rows=30)}"
            for item in sorted_evidence[:effective_cap]
        )

        computed_summary = ""
        if pot_res:
            result_value = pot_res.get("result_value")
            calc_code_text = pot_res.get("code", "") or ""
            output_log_text = pot_res.get("output_log", "") or ""
            implausible_reason = pot_res.get("implausible_result_reason")
            # The calculation code's variable assignments are the authoritative
            # record of which input values were used, so they are shown in full.
            computed_summary = (
                f"\nComputed value: {result_value}\n"
                f"Calculation (its variable assignments are the exact inputs used):\n"
                f"{calc_code_text[:1200]}\n"
                f"Calculation output: {output_log_text[:600]}"
            )
            # Each status note states what the computed value is and how far it
            # can be relied on; how to use it is the general principle ⑤ below.
            if "result is not reliable" in output_log_text:
                computed_summary += (
                    "\nStatus: no matching structured data was found, so the computed "
                    "value above is a placeholder, not a result."
                )
            elif implausible_reason:
                computed_summary += (
                    f"\nStatus: after every retrieval attempt the computed value is still "
                    f"outside the plausible range for this metric ({implausible_reason}), "
                    f"so it is not reliable."
                )
            elif result_value is not None:
                computed_summary += (
                    "\nStatus: computed deterministically from the evidence inputs shown."
                )
            if pot_res.get("is_degraded_formula"):
                computed_summary += (
                    f"\nLimitation: {pot_res.get('degraded_note', '')}"
                )

        # The detectors scan the FULL retrieved pool, not just the passages that fit
        # the prompt window: a fact ranked just outside the window is as invisible
        # to the model as one never retrieved.
        pool_text = "\n".join(item.get("parent_content") or item.get("content", "") for item in evidence)
        incurred_cost_summary = (
            incurred_cost_directive(query, pool_text) + leverage_directive(query, pool_text) + question_directives(query, pool_text, filing_labels)
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
{incurred_cost_summary}
{computed_summary}
{verification_summary}

【ANSWERING PRINCIPLES】

① Answer format
1. Open with a direct, conclusive answer (1-2 sentences) whose figures are
   the ones the question literally asks for. Attach the identifying fact
   (the named company, deal, segment or driver) to its first mention.
   Always answer in complete sentences, never a lone number: for a yes/no
   question begin with "Yes" or "No"; for a "which" question name the item;
   for a change between periods give each period's value as well as the
   change.
2. Bold the key figures. Keep the whole answer under 150 words. Do not paste
   raw evidence or list variable names.
3. State a ratio or multiple without a trailing "x"; percentages keep "%".
4. If the evidence is insufficient, say exactly which data is missing.

② The filing is the source of truth
5. A fact the filing states explicitly (a figure, a named list, a table, a
   stated trend) outranks your general knowledge and outranks any figure you
   would work out yourself from other rows. (A computed value supplied
   above is not one of yours -- see principle 15.)
6. Before concluding something is not disclosed, check every evidence item;
   the relevant one is not always the first. A 10-K covers the fiscal year
   on its cover and also carries the prior year's comparatives.
7. Name a company, transaction, amount or event only when the evidence ties
   it to the period asked; a passing mention elsewhere is not enough.

③ Correct level of aggregation
8. Answer a company-wide amount or count with the company-level total, not
   one brand or segment row. Compute a share against the consolidated total
   the filing reports, not the sum of the segments shown.
9. To pick the highest/lowest/fastest item, compare every comparable item at
   the finest level shown before choosing (a table may be split across
   several blocks; an aggregate row averages away extremes; a firm-total
   column is not a segment).
10. Attribute a segment table's figures only to the segment its own page
    names.
11. When asked about the composition of a total, give each component's
    amount and its share of the total. Figures needed for one ratio may sit
    in different evidence items; combine them.

④ Completeness
12. When asked what drove a change, or to list items (acquisitions, legal
    matters, products, geographies), cover every item the evidence supports,
    each with the specifics the filing discloses for it (amount, ownership
    stake, per-share rate) and named in the filing's own wording -- name the
    deal or counterparty, not just its category. A plan or forecast the
    filing states without a figure is an item too; list it with its wording.
13. When the question's basis is ambiguous, answer every reasonable reading,
    never only one. A "best/worst/top performer" question that does not say
    whether it means size or growth gets BOTH: the item with the largest
    amount (value and share) and the item with the fastest growth (with its
    rate), each with its period. A figure the filing reports on more than one
    basis (reported, adjusted, operational/constant-currency) gets each
    basis. Then give the verdict and say which reading it rests on.
14. Decide a Yes/No judgment from the nature and severity the evidence
    describes, not from whether a particular keyword appears, and state the
    verdict first.

⑤ Computed values
15. A computed value above was produced deterministically from the
    evidence, not by you. If its status says it was computed from the
    inputs shown, use it as the headline figure (reformatted only for the
    units or rounding the question asks), do not recompute it, and cite its
    inputs with exactly the values the calculation used -- the evidence may
    contain same-labeled rows for other years. It answers the metric the
    question asks for under that metric's standard definition, so it leads
    even when the filing also prints a related but different measure (which
    is context, not the answer). When supporting indicators are computed for
    a judgment question, state them and base the verdict on the accounting
    conventions below. If its status says it is a
    placeholder, do not cite it: use the same metric if the filing states it
    directly. Otherwise, if the question needs only simple arithmetic (a
    share, difference or ratio) and every input figure is stated explicitly
    in the evidence, work it out yourself, listing each input figure with
    where it comes from and the operation, and say it was calculated from
    those figures. If any input is not stated, say it could not be
    determined. If its status says
    it is not reliable, do not cite it and do not compute or substitute the
    figure yourself: state plainly that it could not be reliably computed
    from the filing. If a limitation is given, state it.

{ACCOUNTING_CONVENTIONS}
【OUTPUT FORMAT】
Reply in 2-5 complete English sentences, never a bare number. The first
sentence states the conclusion together with its key figure(s) and what they
measure (metric, period, unit) -- "Yes"/"No" first for a yes/no question, the
item's name for a "which" question, each period's value for a change. The
following sentences give the supporting figures or reasons.
Back the verdict with a short quotation (a few words in quotation marks) from
the filing sentence it rests on.

【CHECK BEFORE YOU ANSWER】
- Before saying something is "not disclosed", look through the evidence for a
  sentence that states it in other words (a direction such as "deleverage", a
  cost still being incurred, a plan without a figure), and quote it if found.
- For a question whether an activity (a separation, restructuring, integration)
  is under way: if the evidence shows its costs still being incurred in the
  period, the answer is Yes, whatever else the filing says about its main
  transaction. If a CRITICAL note above says the verdict MUST be YES, your
  first word is "Yes".
- Did you list every item the evidence names, including plans and forecasts
  stated without a figure? Check the whole list, not only the items that carry
  numbers.
- ⚠️ CRITICAL: if a change is driven by an acquisition, merger or divestiture,
  you MUST name the other company or the deal in the first sentence that
  mentions it, never only "acquisition-related".
- ⚠️ CRITICAL: asked WHERE a company operates (geographies, regions): build the
  answer on the regions the filing itself names (typically in its geographic or
  segment note) and name each one. Do not answer with employee headcount or
  with a list of individual countries when the filing names regions.
- ⚠️ CRITICAL: asked which or what companies were acquired: list every company
  the filing's acquisitions note names for the periods asked, using the note's
  own names, and add none that the note does not name for those periods.
- The question may word an item differently from the filing. Match them by
  meaning, then apply the accounting conventions in ⑥ exactly as written.
"""

        try:
            if hasattr(client, "messages") and not hasattr(client, "chat"):
                # Anthropic: system prompt is its own argument, max_tokens is required
                response = client.messages.create(
                    model=self._model,
                    max_tokens=8000,
                    system=_ANSWER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                record_usage(self._model, "answer", getattr(response, "usage", None), len(prompt))
                text = "".join(getattr(b, "text", "") for b in (getattr(response, "content", None) or []))
                if text.strip():
                    return text.strip()
            elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
                create_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
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
