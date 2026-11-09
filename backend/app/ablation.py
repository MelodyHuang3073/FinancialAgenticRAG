"""Ablation switch (EXPERIMENT_PLAN.md section 4): one environment variable, ABLATE_ROW.

    ABLATE_ROW unset or 0   the full system, unchanged
    ABLATE_ROW=1            C1 off  corpus: plain 1000-character chunks, no fields, no label, no page expansion
    ABLATE_ROW=2a           C2 + C3 off  answer prompt, question classifier, vocabulary; every question forced onto
                            the numeric path (answer mode NUMERIC)
    ABLATE_ROW=2b           the same, but every question forced onto the explanation path (answer mode EXPLANATION)
    ABLATE_ROW=3            C4 off  question decomposer and formula library
    ABLATE_ROW=4            C5 off  system retriever (rank_bm25, one concatenated query, 16 chunks)
    ABLATE_ROW=5            C6 + C7 off  program-of-thought calculation, Tri-Check verification and retry

One variable, not one flag per component, so two rows can never be on at once. The value is read once when
the process starts; an unknown value stops the process instead of silently running the full system. Every
hook elsewhere is `if ablation.off("C<n>")` around the component's own code, so with ABLATE_ROW unset no
line of the full system changes behaviour.
"""
import os
from typing import Any, Dict, List

_OFF_BY_ROW = {
    "0": frozenset(),
    "1": frozenset({"C1"}),
    "2a": frozenset({"C2", "C3"}),
    "2b": frozenset({"C2", "C3"}),
    "3": frozenset({"C4"}),
    "4": frozenset({"C5"}),
    "5": frozenset({"C6", "C7"}),
}


def _read_row() -> str:
    raw = (os.environ.get("ABLATE_ROW") or "0").strip()
    if raw not in _OFF_BY_ROW:
        raise RuntimeError(f"ABLATE_ROW={raw!r} is not one of 0 (full system), 1, 2a, 2b, 3, 4, 5")
    return raw


ROW: str = _read_row()


def forced_answer_mode() -> str:
    """Row 2 (C3 off): the one answer mode every question is forced onto -- 2a numeric, 2b explanation."""
    return "EXPLANATION" if ROW == "2b" else "NUMERIC"


def off(component: str) -> bool:
    """True when this component is switched off in the current process."""
    return component in _OFF_BY_ROW[ROW]


def describe() -> Dict[str, Any]:
    """What this process runs, written into every result record and the run folder's config file."""
    return {"ablate_row": ROW, "components_off": sorted(_OFF_BY_ROW[ROW])}


# ── C1 off: the corpus is plain text ─────────────────────────────────────────────────────────────────
def plain_entries(company: str, chunks: List[str]) -> List[Dict[str, Any]]:
    """Corpus entries for a filing kept as plain-text chunks (no table rows, fields, labels or parent page).
    Only the fields every consumer needs to run at all are present, all empty except `content`."""
    return [
        {
            "id": f"plain_{company}_{i}", "company": company, "table_name": "", "period": "",
            "section": "", "page_number": "", "content": text, "type": "text_note",
            "parent_id": "", "parent_content": "",
        }
        for i, text in enumerate(chunks)
    ]


# ── C6 + C7 off: what stands in for the PoT result and the Tri-Check verdict ───────────────────────────
def empty_pot_result(answer_mode: str) -> Dict[str, Any]:
    """The PoT result of a question with no calculation (the shape the non-numeric path already uses)."""
    return {
        "code": "", "success": True, "result_value": None, "output_log": "",
        "extracted_variables": {}, "answer_mode": answer_mode,
    }


def accepted_verdict() -> Dict[str, Any]:
    """A verdict that accepts the first attempt (so there is no retry) and carries no checks, hence no
    Verification summary line in the answer prompt."""
    return {
        "decision": "ACCEPT", "confidence_score": None, "failure_type": None,
        "missing_years": [], "duplicate_placeholder_pair": None, "checks": {},
    }
