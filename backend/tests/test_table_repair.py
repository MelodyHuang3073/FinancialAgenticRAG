"""app/rag/table_repair.py: detects a PDF page whose table extraction
produced a malformed table_row (the page's own text-coordinate heuristics
mis-detected the column headers and a data VALUE ended up in the Line Item
label slot instead of the real text label -- confirmed on real filings,
e.g. Boeing's five-year financial-highlights table), and repairs just that
page via MinerU. MinerU itself is mocked throughout -- these are fast unit
tests of the detection/splicing logic, not a live-model integration test."""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag.table_repair import (
    find_malformed_pages,
    repair_malformed_tables,
    _html_tables_to_markdown,
    _ensure_header_row,
)


def _table_row(page_number, content, extra=None):
    row = {"type": "table_row", "page_number": page_number, "content": content}
    if extra:
        row.update(extra)
    return row


# ── find_malformed_pages ─────────────────────────────────────────────────

def test_detects_bare_number_as_line_item_label():
    passages = [_table_row(6, "Company: X | Report: X.pdf | Line Item: 89,463 | 2024: 66,517")]
    assert find_malformed_pages(passages) == {6}


def test_detects_bare_currency_symbol_as_line_item_label():
    passages = [_table_row(39, "Company: X | Line Item: $ | 2021: 79,557 | 2020: ")]
    assert find_malformed_pages(passages) == {39}


def test_real_text_label_is_not_flagged():
    passages = [_table_row(6, "Company: X | Line Item: Revenues | 2025: 89,463 | 2024: 66,517")]
    assert find_malformed_pages(passages) == set()


def test_bare_year_label_is_not_flagged():
    # A debt/benefit-payment maturity schedule genuinely has the year as
    # each row's own identity (Kraft Heinz "Future Benefit Payments") --
    # must not be treated as a mis-parsed value, unlike a dollar figure.
    passages = [_table_row(93, "Company: X | Line Item: 2027 | $: 64")]
    assert find_malformed_pages(passages) == set()


def test_bare_year_range_label_is_not_flagged():
    passages = [_table_row(93, "Company: X | Line Item: 2031-2035 | $: 228")]
    assert find_malformed_pages(passages) == set()


def test_non_table_row_passages_are_ignored():
    passages = [{"type": "text_note", "page_number": 6, "content": "Line Item: 89,463 |"}]
    assert find_malformed_pages(passages) == set()


def test_multiple_pages_all_collected():
    passages = [
        _table_row(6, "Line Item: 89,463 | 2024: 66,517"),
        _table_row(39, "Line Item: $ | 2021: 79,557"),
        _table_row(40, "Line Item: Revenues | 2025: 1,000"),  # clean, not flagged
    ]
    assert find_malformed_pages(passages) == {6, 39}


# ── repair_malformed_tables ──────────────────────────────────────────────

def _make_parser(linearize_return):
    parser = MagicMock()
    parser._linearize_markdown_tables.return_value = linearize_return
    return parser


def test_no_malformed_pages_returns_input_unchanged():
    passages = [_table_row(6, "Line Item: Revenues | 2025: 89,463")]
    parser = _make_parser(([], ""))
    result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")
    assert result is passages
    parser._linearize_markdown_tables.assert_not_called()


def test_mineru_not_installed_returns_input_unchanged():
    passages = [_table_row(6, "Line Item: 89,463 | 2024: 66,517")]
    parser = _make_parser(([], ""))
    with patch("app.rag.table_repair._mineru_available", return_value=False):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")
    assert result is passages


def test_successful_repair_drops_malformed_row_and_splices_in_new_ones():
    malformed = _table_row(6, "Company: X | Line Item: 89,463 | 2024: 66,517")
    other_page = _table_row(10, "Line Item: Net income | 2025: 100")  # untouched
    passages = [malformed, other_page]

    new_row = _table_row(6, "Company: X | Line Item: Revenues | 2025: 89,463 | 2024: 66,517")
    parser = _make_parser(([new_row], ""))

    with patch("app.rag.table_repair._mineru_available", return_value=True), \
         patch("app.rag.table_repair._extract_single_page_pdf", return_value=True), \
         patch("app.rag.table_repair._mineru_page_markdown", return_value="| Revenues | 89,463 | 66,517 |"):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")

    assert other_page in result  # untouched page kept as-is
    assert malformed not in result  # malformed row dropped
    repaired = [p for p in result if p.get("table_repair_source") == "mineru"]
    assert len(repaired) == 1
    assert repaired[0]["content"] == new_row["content"]


def test_page_extraction_failure_keeps_original_passage_for_that_page():
    malformed = _table_row(6, "Line Item: 89,463 | 2024: 66,517")
    passages = [malformed]
    parser = _make_parser(([], ""))

    with patch("app.rag.table_repair._mineru_available", return_value=True), \
         patch("app.rag.table_repair._extract_single_page_pdf", return_value=False):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")

    assert result == passages  # nothing repaired, nothing dropped either


def test_mineru_found_no_tables_keeps_original_passage():
    malformed = _table_row(6, "Line Item: 89,463 | 2024: 66,517")
    passages = [malformed]
    parser = _make_parser(([], ""))

    with patch("app.rag.table_repair._mineru_available", return_value=True), \
         patch("app.rag.table_repair._extract_single_page_pdf", return_value=True), \
         patch("app.rag.table_repair._mineru_page_markdown", return_value=None):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")

    assert result == passages


def test_linearize_raising_keeps_original_passage_for_that_page():
    malformed = _table_row(6, "Line Item: 89,463 | 2024: 66,517")
    passages = [malformed]
    parser = MagicMock()
    parser._linearize_markdown_tables.side_effect = RuntimeError("boom")

    with patch("app.rag.table_repair._mineru_available", return_value=True), \
         patch("app.rag.table_repair._extract_single_page_pdf", return_value=True), \
         patch("app.rag.table_repair._mineru_page_markdown", return_value="| A | B |"):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")

    assert result == passages


# ── _html_tables_to_markdown ─────────────────────────────────────────────

def test_html_table_without_spans_converts_cleanly():
    html = "<table><tr><td>Revenues</td><td>89,463</td><td>66,517</td></tr></table>"
    md = _html_tables_to_markdown(html)
    assert "| Revenues | 89,463 | 66,517 |" in md
    assert "| --- | --- | --- |" in md


def test_html_table_colspan_repeats_value_across_spanned_columns():
    # Real case: Kraft Heinz's benefit-payment schedule renders a spanned
    # single value as <td colspan="2">64</td> -- repeating it across both
    # columns keeps every row's OWN values aligned under the same column
    # positions, which is what the downstream header/value pairing needs.
    html = '<table><tr><td>2027</td><td colspan="2">64</td></tr></table>'
    md = _html_tables_to_markdown(html)
    assert "| 2027 | 64 | 64 |" in md


def test_html_with_no_table_returns_empty():
    assert _html_tables_to_markdown("<p>no table here</p>") == ""


def test_multiple_html_tables_each_converted_and_separated():
    html = "<table><tr><td>A</td><td>1</td></tr></table><table><tr><td>B</td><td>2</td></tr></table>"
    md = _html_tables_to_markdown(html)
    assert "| A | 1 |" in md
    assert "| B | 2 |" in md


# ── _ensure_header_row ───────────────────────────────────────────────────

def test_headerless_year_schedule_gets_synthetic_header_inserted():
    block = "| 2026 | $ | 69 |\n| --- | --- | --- |\n| 2027 | 64 | 64 |\n| 2028 | 61 | 61 |"
    fixed = _ensure_header_row(block)
    lines = fixed.split("\n")
    assert lines[0] == "| Year | Col1 | Col2 |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 2026 | $ | 69 |"  # original "header" row survives as real data
    assert "| 2027 | 64 | 64 |" in fixed


def test_real_text_header_row_is_left_untouched():
    block = "| Revenues | 2025 | 2024 |\n| --- | --- | --- |\n| Total | 89,463 | 66,517 |"
    assert _ensure_header_row(block) == block


def test_single_year_row_without_a_following_year_row_is_left_untouched():
    # Only the FIRST row looks like a year -- the second row (a real
    # header candidate check target) doesn't match, so this isn't treated
    # as a headerless schedule (avoids false-triggering on a table that
    # happens to have one row starting with a 4-digit number).
    block = "| 2025 | Revenue | Growth |\n| --- | --- | --- |\n| Total | 89,463 | 12% |"
    assert _ensure_header_row(block) == block


def test_too_few_rows_left_untouched():
    block = "| 2027 | 64 |"
    assert _ensure_header_row(block) == block


def test_one_page_repaired_another_page_repair_fails_independently():
    # Page 6 repairs successfully; page 39 (also malformed) fails to extract
    # -- each page's outcome must not affect the other's.
    page6 = _table_row(6, "Line Item: 89,463 | 2024: 66,517")
    page39 = _table_row(39, "Line Item: $ | 2021: 79,557")
    passages = [page6, page39]

    new_row = _table_row(6, "Line Item: Revenues | 2025: 89,463 | 2024: 66,517")
    parser = _make_parser(([new_row], ""))

    def fake_extract(content_bytes, page_num, out_path):
        return page_num == 6  # only page 6 "succeeds" at extraction

    with patch("app.rag.table_repair._mineru_available", return_value=True), \
         patch("app.rag.table_repair._extract_single_page_pdf", side_effect=fake_extract), \
         patch("app.rag.table_repair._mineru_page_markdown", return_value="| Revenues | 89,463 | 66,517 |"):
        result = repair_malformed_tables(parser, passages, b"pdf-bytes", "x.pdf", "X")

    assert page39 in result  # untouched, repair attempt failed
    assert page6 not in result  # dropped, replaced
    assert any(p.get("table_repair_source") == "mineru" for p in result)
