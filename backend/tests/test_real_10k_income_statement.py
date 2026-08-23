"""
Regression test built from a REAL 10-K: Adobe Systems Incorporated's fiscal
2017 Consolidated Statements of Income (values verbatim from the actual
filing). The fixture reproduces the real-world formatting quirks of that
page: '$' drawn as a separate glyph only on the first row of each section
and on subtotal/total rows (standard GAAP statement convention, NOT every
row), and the year-header row ("2017  2016  2015") separated from the first
data block by a section subheading ("Revenue:") with no numbers in it —
this is how almost every real income statement is laid out.

This specific structure exposed a real bug: the year-header candidate was
being overwritten by the very next non-matching line (the section
subheading), so every fragment on the page fell back to generic
Col1/Col2/Col3 headers instead of the real years. Fixed by only replacing
the header candidate when the new line itself contains extractable years.
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

reportlab = pytest.importorskip("reportlab")
pdfplumber = pytest.importorskip("pdfplumber")

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.rag.parser import FinancialFileParser
from app.rag.chunker import _is_table_line


# (label, [2017, 2016, 2015], has_dollar_glyph, kind) — values verbatim from
# Adobe's actual FY2017 10-K, Consolidated Statements of Income.
ROWS = [
    ("Revenue:", None, False, "section"),
    ("Subscription", ["6,133,869", "4,584,833", "3,223,904"], True, "data"),
    ("Product", ["706,767", "800,498", "1,125,146"], False, "data"),
    ("Services and support", ["460,869", "469,099", "446,461"], False, "data"),
    ("Total revenue", ["7,301,505", "5,854,430", "4,795,511"], False, "subtotal"),
    ("Cost of revenue:", None, False, "section"),
    ("Subscription", ["623,048", "461,860", "409,194"], False, "data"),
    ("Product", ["57,082", "68,917", "90,035"], False, "data"),
    ("Services and support", ["330,361", "289,131", "245,088"], False, "data"),
    ("Total cost of revenue", ["1,010,491", "819,908", "744,317"], False, "subtotal"),
    ("Gross profit", ["6,291,014", "5,034,522", "4,051,194"], False, "subtotal"),
    ("Operating expenses:", None, False, "section"),
    ("Research and development", ["1,224,059", "975,987", "862,730"], False, "data"),
    ("Sales and marketing", ["2,197,592", "1,910,197", "1,683,242"], False, "data"),
    ("General and administrative", ["624,706", "576,202", "533,478"], False, "data"),
    ("Amortization of purchased intangibles", ["76,562", "78,534", "68,649"], False, "data"),
    ("Total operating expenses", ["4,122,919", "3,540,920", "3,148,099"], False, "subtotal"),
    ("Operating income", ["2,168,095", "1,493,602", "903,095"], False, "subtotal"),
    ("Income before income taxes", ["2,137,641", "1,435,138", "873,781"], False, "subtotal"),
    ("Provision for income taxes", ["443,687", "266,356", "244,230"], False, "data"),
    ("Net income", ["1,693,954", "1,168,782", "629,551"], True, "total"),
]


def _build_adobe_income_statement_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width / 2, y, "ADOBE SYSTEMS INCORPORATED")
    y -= 12
    c.drawCentredString(width / 2, y, "CONSOLIDATED STATEMENTS OF INCOME")
    y -= 22

    col_x = [430, 510, 590]
    dollar_x = [x - 42 for x in col_x]

    c.setFont("Helvetica-Bold", 8)
    for ci, yr in enumerate(["2017", "2016", "2015"]):
        c.drawRightString(col_x[ci], y, yr)
    y -= 16

    row_h = 14
    label_x = 72
    for label, vals, has_dollar, kind in ROWS:
        if kind == "section":
            c.setFont("Helvetica", 8)
            c.drawString(label_x, y - 10, label)
            y -= row_h
            continue
        c.setFont("Helvetica-Bold" if kind in ("subtotal", "total") else "Helvetica", 8)
        indent = label_x + (10 if kind == "data" else 0)
        c.drawString(indent, y - 10, label)
        for ci, val in enumerate(vals):
            if has_dollar:
                c.drawString(dollar_x[ci], y - 10, "$")
            c.drawRightString(col_x[ci], y - 10, val)
        y -= row_h
        if kind in ("subtotal", "total"):
            y -= 3

    c.showPage()
    c.save()
    return buf.getvalue()


def test_all_income_statement_line_items_extracted_correctly():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_adobe_income_statement_pdf())) as pdf:
        md_tables, prose = parser._extract_page_tables_and_prose(pdf.pages[0])

    assert md_tables, "failed to detect any table on a real 10-K income statement layout"
    joined = "\n".join(md_tables)

    data_rows = [r for r in ROWS if r[3] != "section"]
    for label, vals, _, _ in data_rows:
        assert label in joined, f"missing line item: {label}"
        for v in vals:
            assert v in joined, f"missing value {v!r} for {label!r}"

    for md in md_tables:
        for line in md.split("\n"):
            assert _is_table_line(line)


def test_year_headers_persist_across_section_subheadings():
    """Regression: the year-header row is separated from the first data
    block by a section subheading with no numbers ('Revenue:') — every
    resulting table fragment on the page must still show real year
    headers, not fall back to generic Col1/Col2/Col3."""
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_adobe_income_statement_pdf())) as pdf:
        md_tables, _ = parser._extract_page_tables_and_prose(pdf.pages[0])

    assert len(md_tables) >= 2, "expected the page to fragment into multiple table blocks"
    for md in md_tables:
        header_line = md.split("\n")[0]
        assert "2017" in header_line and "2016" in header_line and "2015" in header_line, (
            f"table fragment fell back to generic headers: {header_line!r}"
        )
        assert "Col1" not in header_line


def test_net_income_row_values_correct_and_not_mixed_with_other_rows():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_adobe_income_statement_pdf())) as pdf:
        md_tables, _ = parser._extract_page_tables_and_prose(pdf.pages[0])

    net_income_line = next(
        line for md in md_tables for line in md.split("\n")
        if line.strip().startswith("| Net income ")
    )
    cells = [c.strip() for c in net_income_line.strip().strip("|").split("|")]
    assert cells == ["Net income", "$1,693,954", "$1,168,782", "$629,551"]
