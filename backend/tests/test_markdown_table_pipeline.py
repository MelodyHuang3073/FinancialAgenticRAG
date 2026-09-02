"""
Tests for the standard-Markdown-table storage format upgrade:

1. to_markdown_table() (app.tools.table_parser) — the single canonical
   table formatter, now shared by app.rag.parser (PDF table extraction)
   and linearize_financial_table() (sample/CSV table data).
2. pot_reasoner._extract_from_markdown_table_block() / the new dispatch
   inside _extract_from_linearized_table() — parses standard Markdown
   tables into the same {code_key: {item, canonical, year, val, code_key}}
   shape the legacy single-line parser already produces, so downstream
   _find_same_item_pair()/_build_calculation_code() need no changes.
3. Legacy single-line "Line Item: X | Year: Val" format must keep working
   unchanged (regression).
4. llm_client._truncate_evidence_content() — row-based truncation for
   Markdown table content instead of a raw character slice that could cut
   a table row in half.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.table_parser import to_markdown_table
from app.rag.chunker import _is_table_line
from app.agent.pot_reasoner import _extract_from_linearized_table, _find_same_item_pair
from app.agent.llm_client import _truncate_evidence_content


# ─────────────────────────────────────────────────────────────────────────
# 1. to_markdown_table()
# ─────────────────────────────────────────────────────────────────────────

def test_to_markdown_table_lines_are_all_valid_table_lines():
    md = to_markdown_table(
        headers=["Line Item", "2023", "2024"],
        rows=[["Revenue", "2,161.7", "2,894.3"], ["Net Income", "567.8", "601.2"]],
    )
    lines = md.split("\n")
    assert len(lines) == 4  # header + separator + 2 data rows
    for line in lines:
        assert _is_table_line(line), f"not recognised as a table line: {line!r}"

    assert lines[0] == "| Line Item | 2023 | 2024 |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| Revenue | 2,161.7 | 2,894.3 |"
    assert lines[3] == "| Net Income | 567.8 | 601.2 |"


def test_to_markdown_table_does_not_reformat_numbers():
    """Cell values must pass through byte-for-byte — no rounding, no
    reformatting. Formatting cleanup is the extraction stage's job."""
    md = to_markdown_table(
        headers=["Line Item", "FY2023"],
        rows=[["Weird value", "  (1,234.500)  "]],
    )
    assert "(1,234.500)" in md
    assert "1234.5" not in md and "1,235" not in md


# ─────────────────────────────────────────────────────────────────────────
# 2. New Markdown-table parser in pot_reasoner, wired into
#    _extract_from_linearized_table()
# ─────────────────────────────────────────────────────────────────────────

def _testco_markdown_evidence():
    md = to_markdown_table(
        headers=["Line Item", "FY2023", "FY2024"],
        rows=[
            ["Revenue", "2,161.7", "2,894.3"],
            ["Net Income", "567.8", "601.2"],
        ],
    )
    content = "Company: TESTCO | Report: Income Statement | Period: 2023-2024\n\n" + md
    return [{"content": content, "company": "TESTCO"}]


def test_markdown_table_extraction_gets_correct_revenue_values():
    extracted = _extract_from_linearized_table(_testco_markdown_evidence())

    revenue_vals = {v["year"]: v["val"] for v in extracted.values() if v["canonical"] == "revenue"}
    assert revenue_vals == {"2023": 2161.7, "2024": 2894.3}


def test_markdown_table_extraction_does_not_mix_up_revenue_and_net_income():
    extracted = _extract_from_linearized_table(_testco_markdown_evidence())

    net_income_vals = {v["year"]: v["val"] for v in extracted.values() if v["canonical"] == "net_income"}
    assert net_income_vals == {"2023": 567.8, "2024": 601.2}

    # The revenue-growth calculation must pick Revenue's pair, not Net
    # Income's — this is what "沒有抓錯科目" actually means end-to-end:
    # both rows get extracted correctly, but the query-relevant one wins.
    old, new = _find_same_item_pair(list(extracted.values()), "testco revenue growth from 2023 to 2024")
    assert old is not None and new is not None
    assert old["canonical"] == "revenue" and new["canonical"] == "revenue"
    assert old["val"] == 2161.7
    assert new["val"] == 2894.3
    assert old["val"] not in (567.8, 601.2)
    assert new["val"] not in (567.8, 601.2)


# ─────────────────────────────────────────────────────────────────────────
# 3. Legacy single-line format — regression, must be completely unaffected
# ─────────────────────────────────────────────────────────────────────────

def test_legacy_single_line_pipe_format_still_works():
    evidence_list = [
        {
            "content": (
                "Company: TESTCO | Report: Income Statement | Period: 2023-2024 | "
                "Line Item: Revenue | 2023: 2,161.7 | 2024: 2,894.3"
            ),
            "company": "TESTCO",
        },
        {
            "content": (
                "Company: TESTCO | Report: Income Statement | Period: 2023-2024 | "
                "Line Item: Net Income | 2023: 567.8 | 2024: 601.2"
            ),
            "company": "TESTCO",
        },
    ]
    extracted = _extract_from_linearized_table(evidence_list)

    revenue_vals = {v["year"]: v["val"] for v in extracted.values() if v["canonical"] == "revenue"}
    assert revenue_vals == {"2023": 2161.7, "2024": 2894.3}

    old, new = _find_same_item_pair(list(extracted.values()), "revenue growth")
    assert old["val"] == 2161.7 and new["val"] == 2894.3


def test_mixed_legacy_and_markdown_evidence_in_same_call():
    """Old-format and new-format evidence items must coexist in one call
    and merge into a single extracted dict without clobbering each other."""
    legacy_item = {
        "content": (
            "Company: TESTCO | Report: Balance Sheet | Period: 2024 | "
            "Line Item: Total Assets | 2024: 5,000.0"
        ),
        "company": "TESTCO",
    }
    evidence_list = _testco_markdown_evidence() + [legacy_item]

    extracted = _extract_from_linearized_table(evidence_list)
    canonicals = {v["canonical"] for v in extracted.values()}
    assert "revenue" in canonicals
    assert "net_income" in canonicals
    assert "total_assets" in canonicals


# ─────────────────────────────────────────────────────────────────────────
# 4. llm_client row-based table truncation
# ─────────────────────────────────────────────────────────────────────────

def test_table_truncation_keeps_header_and_first_10_rows_with_marker():
    headers = ["Line Item", "2023", "2024"]
    rows = [[f"Line item {i}", str(i * 10), str(i * 20)] for i in range(15)]
    md = to_markdown_table(headers, rows)
    content = "Company: X | Report: Y | Period: Z\n\n" + md

    truncated = _truncate_evidence_content(content, max_chars=600, max_table_rows=10)
    lines = truncated.split("\n")

    assert lines[-1] == "...(more rows omitted)"
    table_lines = [l for l in lines if l.strip().startswith("|")]
    assert len(table_lines) == 12, "expected header + separator + 10 data rows"
    for l in table_lines:
        assert _is_table_line(l), f"truncated output line is not a valid table line: {l!r}"
    assert table_lines[0] == "| Line Item | 2023 | 2024 |"
    assert table_lines[1] == "|---|---|---|"
    assert "Line item 9" in table_lines[-1]
    assert "Line item 10" not in truncated


def test_table_truncation_no_marker_when_under_row_limit():
    headers = ["Line Item", "2023"]
    rows = [["Revenue", "100"], ["Net Income", "20"]]
    md = to_markdown_table(headers, rows)

    truncated = _truncate_evidence_content(md, max_chars=600, max_table_rows=10)
    assert "...(more rows omitted)" not in truncated
    assert truncated == md


def test_non_table_content_still_uses_character_truncation():
    long_prose = "This is ordinary narrative text. " * 50
    truncated = _truncate_evidence_content(long_prose, max_chars=100)
    assert truncated == long_prose[:100]
    assert len(truncated) == 100
