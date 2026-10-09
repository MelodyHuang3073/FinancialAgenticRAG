"""Dumps every retrieval re-weighting rule hybrid_retriever.HybridFinancialRetriever
applies (name, multiplier, trigger condition) as a Markdown table, for the
paper's methodology appendix. Reads RULE_MULTIPLIERS/RULE_DESCRIPTIONS
directly -- values/descriptions can never drift out of sync with the actual
search() implementation, since there is only one copy of each.

Usage:
    python tests/tools/dump_retrieval_rules.py [output.md]
Defaults to stdout when no output path is given.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.tools.hybrid_retriever import RULE_MULTIPLIERS, RULE_DESCRIPTIONS

#: Rules whose value is scaling-formula/method-based rather than a flat
#: constant in RULE_MULTIPLIERS -- see hybrid_retriever.py's
#: RULE_MULTIPLIERS docstring for why.
_NON_FLAT_RULES = {
    "company_match": "0.05x - 2.0x (scoring method, see _company_match_score)",
    "line_item_match": "1.0x or 1.5x (scoring method, see _line_item_match_score)",
    "geographic_region_density": (
        f"{RULE_MULTIPLIERS['geographic_region_density_base']}x - "
        f"{RULE_MULTIPLIERS['geographic_region_density_cap']}x "
        f"(scales with region count, see RULE_MULTIPLIERS['geographic_region_density_*'])"
    ),
}


def _value_display(name: str) -> str:
    if name in _NON_FLAT_RULES:
        return _NON_FLAT_RULES[name]
    return f"{RULE_MULTIPLIERS[name]}x"


def build_markdown() -> str:
    # RULE_DESCRIPTIONS is the single source of truth for which rule names
    # exist (it covers the two scoring-method-based rules RULE_MULTIPLIERS
    # itself doesn't); every name in it gets one row.
    names = sorted(RULE_DESCRIPTIONS.keys())
    lines = [
        "# Retrieval re-weighting rules",
        "",
        "Every rule `hybrid_retriever.HybridFinancialRetriever.search()` can apply to a "
        "candidate passage's score, generated from `RULE_MULTIPLIERS`/`RULE_DESCRIPTIONS` "
        "so this table can never drift out of sync with the implementation.",
        "",
        "| Rule | Multiplier | Trigger |",
        "|---|---|---|",
    ]
    for name in names:
        lines.append(f"| `{name}` | {_value_display(name)} | {RULE_DESCRIPTIONS[name]} |")
    lines.append("")
    lines.append(
        "Two pairs are mutually exclusive (if/elif, only the first matching one in each "
        "pair applies): `geographic_section_boost`/`geographic_region_density`, and "
        "`year_match_boost`/`recent_year_mismatch_penalty`. Every other rule is "
        "independent and stacks multiplicatively with any other rule that also fires."
    )
    lines.append("")
    lines.append(
        "Set `DISABLE_RULE_MULTIPLIERS = True` on a `HybridFinancialRetriever` instance "
        "(or the class) to ablate every rule at once (ranking driven purely by "
        "`bm25 * 0.7 + overlap_count * 0.3`), or additionally set "
        "`DISABLE_RULE_MULTIPLIERS_EXCEPT = {\"rule_name\", ...}` to keep only specific "
        "rules active while ablating the rest."
    )
    return "\n".join(lines) + "\n"


def main():
    md = build_markdown()
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
