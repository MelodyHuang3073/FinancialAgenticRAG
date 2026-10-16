"""QueryDecomposer

Decomposes a complex financial question into sequential retrieval and computation steps.
Each retrieval step targets a specific metric and year so the orchestrator can fetch
relevant table rows in order.

Decomposition modes (priority order):
1. LLM — structured sub-queries via the SOP prompt
2. Rule-based fallback — deterministic metric×year expansion
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from app.agent.usage_tracker import record_usage
from app.agent.financial_vocabulary import metric_view


# ─────────────────────────────────────────────────────────────────────────────
# LLM-extracted formula validation
# ─────────────────────────────────────────────────────────────────────────────
# A question's own inline formula definition ("X is defined as: A / B") is
# extracted by the SAME LLM call that already decomposes the question into
# retrieval sub-queries (see _SYSTEM_PROMPT) -- not a second call, and not a
# regex parser. The LLM's JSON output is still untrusted input, though: this
# validates and sanitizes it into the exact shape FORMULA_LIBRARY entries use
# (formula_expr / required_vars / result_label / unit / multi_year) before
# pot_reasoner's existing, already-proven extraction/codegen engine
# (_extract_formula_guided / _gen_formula_code) ever sees it. A formula that
# fails any check here is dropped (returns None) rather than guessed at --
# the caller then falls back to the same no-formula behavior as a question
# that never defined one, never to a partially-trusted guess.
#
# formula_expr is only ever built from pieces validated here before being
# embedded in generated Python code and executed in pot_reasoner's sandbox
# (app/tools/sandbox.py, which itself AST-checks all generated code
# regardless of where it came from) -- this character whitelist is an
# additional, narrower check specific to this being the first time
# formula_expr is LLM-sourced rather than developer-authored.
_SAFE_FORMULA_EXPR_RE = re.compile(r'^[A-Za-z0-9_\s+\-*/().]+$')
_FORMULA_IDENTIFIER_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_FORMULA_EXPR_SAFE_FUNCS = {"abs", "max", "min", "round"}
_VALID_FORMULA_UNITS = {"%", "x", ""}


def _validate_llm_formula(raw: Any) -> Optional[Dict[str, Any]]:
    """The LLM's "formula" field, sanitized into a FORMULA_LIBRARY-shaped
    entry, or None if it's absent (the question didn't define one -- the
    overwhelmingly common case) or fails any structural/safety check."""
    if not isinstance(raw, dict):
        return None

    formula_expr = raw.get("formula_expr")
    required_vars = raw.get("required_vars")
    if not isinstance(formula_expr, str) or not formula_expr.strip():
        return None
    if not isinstance(required_vars, dict) or not required_vars:
        return None
    if not _SAFE_FORMULA_EXPR_RE.match(formula_expr):
        return None

    clean_vars: Dict[str, List[str]] = {}
    for var_name, aliases in required_vars.items():
        if not isinstance(var_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', var_name):
            return None
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list) or not aliases:
            return None
        clean_aliases = [str(a).strip().lower() for a in aliases if str(a).strip()]
        if not clean_aliases:
            return None
        clean_vars[var_name] = clean_aliases

    # Every identifier formula_expr actually references must be either a
    # declared variable or a whitelisted helper function name -- an
    # expression referencing something undeclared (a hallucinated or typo'd
    # variable name) is rejected outright rather than silently producing a
    # NameError deep in codegen.
    referenced = set(_FORMULA_IDENTIFIER_RE.findall(formula_expr))
    unknown = referenced - set(clean_vars.keys()) - _FORMULA_EXPR_SAFE_FUNCS
    if unknown:
        return None

    unit = raw.get("unit") if raw.get("unit") in _VALID_FORMULA_UNITS else ""
    result_label = str(raw.get("result_label") or "Computed Ratio").strip() or "Computed Ratio"
    multi_year = bool(raw.get("multi_year"))

    return {
        "formula_key": "llm_extracted",
        "result_label": result_label,
        "unit": unit,
        "formula_expr": formula_expr,
        "required_vars": clean_vars,
        "multi_year": multi_year,
    }


def _is_reasoning_model(model_name: str) -> bool:
    """
    True for OpenAI's reasoning-tier models (gpt-5 family, o1/o3/o4), which
    reject any non-default `temperature` value and spend part of
    `max_completion_tokens` on internal reasoning before the visible
    answer -- calling them with gpt-4o-mini's params (temperature=0,
    max_tokens=512) either 400s on temperature or, if temperature were
    dropped but the token budget left at 512, can return an EMPTY message
    (reasoning alone consumed the whole budget). Duplicated from
    llm_client.py (same rationale as this module's other small duplicated
    helpers) since there's no shared client-config module to import from.
    """
    m = (model_name or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))

# Both views come from financial_vocabulary.QUESTION_METRICS:
# search_terms = how filings word each metric (used to phrase retrieval queries);
# components = for a derived ratio, the base metrics that must EACH be retrieved
# before it can be computed.
_METRIC_NAMES: Dict[str, str] = metric_view("search_terms")
_RATIO_COMPONENTS: Dict[str, List[str]] = metric_view("components")


class QueryDecomposer:
    """Decomposes a complex financial question into an ordered list of retrieval sub-tasks.
    Prefers an LLM when available; otherwise uses a deterministic rule-based expansion.
    """

    _SYSTEM_PROMPT = (
        "You are a financial analysis assistant following the FinanceBench SOP.\n"
        "Your task (Stage 3 — Multi-Step Retrieval Decomposition):\n"
        "Break the user's financial question into atomic search sub-queries.\n"
        "Each sub-query must target ONE specific (metric, year) pair.\n"
        "For ratio/margin questions, include sub-queries for BOTH numerator and denominator.\n"
        "\n"
        "Also check: does the question define its OWN formula inline (e.g. \"X is defined "
        "as: A / B\", or \"X (A / B)\")? This only applies to a formula the question spells "
        "out itself in words — not a metric you merely recognize by name (gross margin, ROE, "
        "etc. without an explicit definition in the question text). If it does, ALSO output a "
        "\"formula\" object describing exactly that formula, using short variable names (op0, "
        "op1, ...) for each named financial quantity, so it can be computed later from "
        "retrieved data. If the question does not define its own formula inline, set "
        "\"formula\" to null.\n"
        "\n"
        "A formula object looks like:\n"
        "{\n"
        "  \"result_label\": \"<short metric name from the question>\",\n"
        "  \"unit\": \"%\" | \"x\" | \"\",\n"
        "  \"formula_expr\": \"<arithmetic expression using ONLY the variable names below, "
        "+ - * / ( ) and numeric literals — no other text>\",\n"
        "  \"required_vars\": {\"<var_name>\": [\"<line item name this variable refers to>\"]},\n"
        "  \"multi_year\": true | false\n"
        "}\n"
        "For a quantity that needs the AVERAGE of a line item between two fiscal years (e.g. "
        "\"average total assets between FY2021 and FY2022\"), use TWO variables with the SAME "
        "base name suffixed _old and _new (e.g. op1_old, op1_new, both mapped to the SAME "
        "line-item alias), combine them as ((op1_old + op1_new) / 2) inside formula_expr, and "
        "set multi_year to true. Use the same _old/_new pattern for a \"change in X between "
        "year1 and year2\" quantity, combined as (op_new - op_old).\n"
        "\n"
        "Output ONLY a JSON object, no explanation. Format:\n"
        "{\n"
        "  \"sub_queries\": [\n"
        "    {\"step\": 1, \"query\": \"<company> <metric> <year>\", "
        "\"target_metric\": \"<canonical_metric>\", \"target_year\": \"<YYYY>\"},\n"
        "    ...\n"
        "  ],\n"
        "  \"formula\": <formula object, or null>\n"
        "}\n"
        "Use short, precise English queries. Include company name in every query."
    )

    def __init__(self) -> None:
        self._llm_client = None
        self._llm_model: Optional[str] = None
        self._llm_enabled: Optional[bool] = None  # None = not checked yet

    def _get_llm_client(self):
        """Lazy-init: try OpenAI first, then Gemini, then return None."""
        if self._llm_enabled is not None:
            return self._llm_client  # already resolved

        # Load .env if present
        try:
            from dotenv import load_dotenv
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            env_path = os.path.join(base_dir, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
        except Exception:
            pass

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=openai_key)
                self._llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self._llm_enabled = True
                return self._llm_client
            except Exception:
                pass

        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            try:
                from google import genai
                self._llm_client = genai.Client(api_key=google_key)
                self._llm_model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")
                self._llm_enabled = True
                return self._llm_client
            except Exception:
                pass

        self._llm_enabled = False
        return None

    def _llm_decompose(
        self,
        query: str,
        entity: str,
        target_metrics: List[str],
        years: List[str],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """
        Call the LLM to generate structured sub-queries for the decomposition
        step AND, in the same call, extract any formula the question defines
        inline (see _SYSTEM_PROMPT and _validate_llm_formula). One call does
        both jobs -- no separate LLM round-trip for formula understanding.

        Returns (steps, formula); either half is None on its own terms (no
        formula defined → formula is None; LLM unavailable or its output
        couldn't be parsed at all → both are None).
        """
        client = self._get_llm_client()
        if not client:
            return None, None

        # Build a concise context block for the LLM
        context_lines = [f"Question: {query}"]
        if entity and entity != "company":
            context_lines.append(f"Company: {entity}")
        if target_metrics:
            context_lines.append(f"Identified metrics: {', '.join(target_metrics)}")
        if years:
            context_lines.append(f"Years involved: {', '.join(years)}")
        context_lines.append(
            "\nGenerate the minimum set of sub-queries needed to retrieve all required data. "
            "Max 8 sub-queries."
        )
        user_content = "\n".join(context_lines)

        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                # OpenAI
                reasoning = _is_reasoning_model(self._llm_model)
                create_kwargs: Dict[str, Any] = {
                    "model": self._llm_model,
                    "messages": [
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    # Reasoning models spend part of this budget on
                    # invisible internal reasoning tokens before the JSON
                    # output -- can be entirely consumed by reasoning alone,
                    # leaving an empty message. Generous since these tokens
                    # are free under the daily complimentary-token
                    # allowance. Raised from 512 now that a reply can also
                    # carry a formula object alongside the sub-query array.
                    "max_completion_tokens": 768 if not reasoning else 2000,
                }
                if not reasoning:
                    create_kwargs["temperature"] = 0
                response = client.chat.completions.create(**create_kwargs)
                record_usage(self._llm_model, "decomposer", getattr(response, "usage", None), len(user_content))
                raw = response.choices[0].message.content or ""
            else:
                # Gemini
                full_prompt = self._SYSTEM_PROMPT + "\n\n" + user_content
                response = client.models.generate_content(
                    model=self._llm_model, contents=full_prompt
                )
                record_usage(self._llm_model, "decomposer", getattr(response, "usage_metadata", None), len(full_prompt))
                raw = getattr(response, "text", "") or ""

            # Extract the JSON object from the response (the prompt asks for
            # {"sub_queries": [...], "formula": {...}|null} specifically so
            # this is an object, not the bare array the prompt used before).
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                return None, None

            parsed = json.loads(json_match.group())
            if not isinstance(parsed, dict):
                return None, None

            raw_steps = parsed.get("sub_queries")
            steps: List[Dict[str, Any]] = []
            if isinstance(raw_steps, list):
                for i, item in enumerate(raw_steps, 1):
                    if not isinstance(item, dict):
                        continue
                    q = str(item.get("query", "")).strip()
                    if not q:
                        continue
                    steps.append({
                        "step": item.get("step", i),
                        "type": "retrieval",
                        "query": q,
                        "target_metric": item.get("target_metric") or None,
                        "target_year": str(item.get("target_year", "")).strip() or None,
                        "source": "llm",
                    })

            formula = _validate_llm_formula(parsed.get("formula"))

            return (steps if steps else None), formula

        except Exception:
            return None, None

    def suggest_narrative_topic_query(self, query: str, entity: str) -> Optional[str]:
        """
        LLM-suggested extra retrieval query for narrative/qualitative questions
        (topics that are not statement line items): asks which disclosure
        section / accounting vocabulary would hold the answer, in the filing's
        own wording (e.g. "Legal Proceedings" for "legal battles"). Used as ONE
        extra query next to the classifier's own queries, never a replacement;
        a failed or empty response is a silent no-op.
        """
        client = self._get_llm_client()
        if not client:
            return None

        prompt = (
            "A user asked this question about a company's SEC filing "
            f"(10-K/10-Q/8-K): \"{query}\"\n"
            f"Company: {entity or 'the company'}\n\n"
            "What specific disclosure topic, SEC filing section/item "
            "name, or accounting vocabulary would most likely contain "
            "the answer? Give 4-8 concise search KEYWORDS/PHRASES (not a "
            "sentence), using the filing's own typical wording (the "
            "actual SEC item heading or standard accounting term), "
            "separated by spaces. Output ONLY the keywords, nothing else."
        )
        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                reasoning = _is_reasoning_model(self._llm_model)
                create_kwargs: Dict[str, Any] = {
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    # Increase model token budget beyond 800 for complex reasoning
                    # prompts.
                    # Internal reasoning tokens count toward completion limits and can
                    # exhaust output on nontrivial questions.
                    "max_completion_tokens": 100 if not reasoning else 2000,
                }
                if not reasoning:
                    create_kwargs["temperature"] = 0
                response = client.chat.completions.create(**create_kwargs)
                record_usage(self._llm_model, "decomposer", getattr(response, "usage", None), len(prompt))
                raw = (response.choices[0].message.content or "").strip()
            else:
                response = client.models.generate_content(model=self._llm_model, contents=prompt)
                record_usage(self._llm_model, "decomposer", getattr(response, "usage_metadata", None), len(prompt))
                raw = (getattr(response, "text", "") or "").strip()
            return raw or None
        except Exception:
            return None

    def _rule_decompose(
        self,
        query: str,
        target_metrics: List[str],
        years: List[str],
        entity: str,
    ) -> List[Dict[str, Any]]:
        """Deterministic (metric × year) expansion — used when LLM is unavailable."""
        # Expand ratio/formula metrics to component line items
        expanded_metrics: List[str] = []
        for m in target_metrics:
            components = _RATIO_COMPONENTS.get(m)
            if components:
                for c in components:
                    if c not in expanded_metrics:
                        expanded_metrics.append(c)
            else:
                if m not in expanded_metrics:
                    expanded_metrics.append(m)

        if not expanded_metrics:
            expanded_metrics = target_metrics

        steps: List[Dict[str, Any]] = []
        step_num = 0

        if expanded_metrics and years:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                for year in years:
                    step_num += 1
                    steps.append({
                        "step": step_num,
                        "type": "retrieval",
                        "query": f"{entity} {metric_display} {year}".strip(),
                        "target_metric": metric,
                        "target_year": year,
                        "source": "rule",
                    })
        elif expanded_metrics:
            for metric in expanded_metrics:
                metric_display = _METRIC_NAMES.get(metric, metric)
                step_num += 1
                steps.append({
                    "step": step_num,
                    "type": "retrieval",
                    "query": f"{entity} {metric_display}".strip(),
                    "target_metric": metric,
                    "target_year": None,
                    "source": "rule",
                })
        elif years:
            q_lower = query.lower()
            if "cagr" in q_lower or "複合成長率" in q_lower:
                y1, y2 = years[0], years[-1]
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y1}".strip(),
                               "target_metric": None, "target_year": y1, "source": "rule"})
                step_num += 1
                steps.append({"step": step_num, "type": "retrieval",
                               "query": f"{entity} {query} {y2}".strip(),
                               "target_metric": None, "target_year": y2, "source": "rule"})
            else:
                for y in years:
                    step_num += 1
                    steps.append({"step": step_num, "type": "retrieval",
                                  "query": f"{entity} {query} {y}".strip(),
                                  "target_metric": None, "target_year": y, "source": "rule"})
        else:
            step_num += 1
            steps.append({
                "step": step_num,
                "type": "retrieval",
                "query": query,
                "target_metric": None,
                "target_year": None,
                "source": "rule",
            })

        return steps

    def decompose(
        self,
        query: str,
        target_metrics: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
        entity: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Decompose a financial query into ordered retrieval + computation
        steps, and (second return value) any formula the question defines
        inline, extracted by the same LLM call -- see _llm_decompose. None
        when the LLM path didn't run (unavailable, or fell back to rule-based
        decomposition) or the question doesn't define one of its own."""
        # ── Normalise inputs ──────────────────────────────────────────────────
        if not years:
            years = sorted(set(re.findall(r'(?:FY)?(20\d\d)', query)))
        else:
            years = [re.sub(r'^FY', '', y) for y in years]

        target_metrics = list(target_metrics or [])
        entity = (entity or "").strip()

        llm_steps, llm_formula = self._llm_decompose(query, entity, target_metrics, years)

        if llm_steps:
            retrieval_steps = llm_steps
            decompose_source = "llm"
        else:
            retrieval_steps = self._rule_decompose(query, target_metrics, years, entity)
            decompose_source = "rule"
            llm_formula = None  # the rule-based path has no formula-understanding ability

        # ── Renumber steps sequentially ───────────────────────────────────────
        for i, s in enumerate(retrieval_steps, 1):
            s["step"] = i

        # ── Always append final computation step ──────────────────────────────
        retrieval_steps.append({
            "step": len(retrieval_steps) + 1,
            "type": "computation",
            "query": "Compute the final answer from accumulated retrieved financial data.",
            "target_metric": None,
            "target_year": None,
            "source": decompose_source,
        })

        return retrieval_steps, llm_formula
