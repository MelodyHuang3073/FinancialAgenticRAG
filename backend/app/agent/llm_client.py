import os
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
        # 6, not 4 -- a genuinely multi-page narrative topic (e.g. a
        # litigation/legal-proceedings discussion, or a list of several
        # acquisitions each described on its own page) routinely has its
        # relevant content spread across MORE than 4 distinct pages, each
        # scoring close to the others. Confirmed real case: Boeing's FY2022
        # "materially important ongoing legal battles" question has
        # relevant evidence on pages 4, 19, 113, 128, 146, 148, and 149 --
        # the one page naming the Lion Air/Ethiopian Airlines litigation
        # specifically (page 113) ranked 5th by score, just outside a
        # 4-item cut, even though every one of those pages is genuinely
        # about the same legal-proceedings topic. Each item can be up to
        # 4000 chars (see max_chars above), so 6 items is still a modest
        # ~24K-char evidence budget for a single LLM call.
        evidence_text = "\n".join(
            f"- [{item.get('company', 'Company')} / {item.get('table_name', 'Source')}] "
            f"{_truncate_evidence_content(item.get('parent_content') or item.get('content', ''), max_chars=4000)}"
            for item in sorted_evidence[:6]
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
                    "verified Python sandbox, NOT by you. You MUST quote this exact number "
                    "(reformatted for units/rounding exactly as the question asks, but not "
                    "recalculated) as your answer. Do NOT redo the arithmetic yourself from "
                    "the raw evidence figures below -- independent re-derivation has produced "
                    "wrong numbers before even when every input you cited was correct. When "
                    "citing ANY supporting figure (e.g. \"net income of $X million\"), you MUST "
                    "quote the exact value assigned to that variable in the PoT calculation "
                    "code above -- NOT a different number for the same line item that appears "
                    "in the raw evidence below, even if that other number looks equally "
                    "plausible. The code's variables are ground truth for what was used; the "
                    "raw evidence may contain other same-labeled figures that were NOT used."
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
"""

        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a professional financial report analysis assistant. Respond in clear English with a result-first format."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
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
        except Exception:
            return None

        return None
