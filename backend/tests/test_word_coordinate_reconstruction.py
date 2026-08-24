"""
Tests for Tier 2 table reconstruction (_reconstruct_table_from_word_positions):
row/column clustering from raw word bounding-box coordinates, used when a
real 10-K PDF defeats BOTH ruled-line detection (no vector lines) AND the
layout=True text-flow regex (each table cell ends up on its own line even
with layout preserved, so there's no shared line left for a same-line regex
to match). Fixture values are the real 3M FY2022 10-K Consolidated
Statement of Income figures (Net sales / Cost of sales, FY2022-2020).
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

reportlab = pytest.importorskip("reportlab")
pdfplumber = pytest.importorskip("pdfplumber")
fitz = pytest.importorskip("fitz")

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.rag.parser import FinancialFileParser
from app.rag.chunker import _is_table_line


ROWS = [
    ("Net sales", ["34,229", "35,355", "32,184"]),
    ("Cost of sales", ["19,232", "18,795", "16,605"]),
]


def _build_3m_income_statement_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width / 2, y, "3M COMPANY AND SUBSIDIARIES")
    y -= 12
    c.drawCentredString(width / 2, y, "Consolidated Statement of Income")
    y -= 22

    col_x = [430, 510, 590]
    c.setFont("Helvetica-Bold", 8)
    for ci, yr in enumerate(["2022", "2021", "2020"]):
        c.drawRightString(col_x[ci], y, yr)
    y -= 16

    row_h = 14
    label_x = 72
    c.setFont("Helvetica", 8)
    for label, vals in ROWS:
        c.drawString(label_x, y - 10, label)
        for ci, val in enumerate(vals):
            c.drawRightString(col_x[ci], y - 10, val)
        y -= row_h

    c.showPage()
    c.save()
    return buf.getvalue()


def _build_prose_only_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    t = c.beginText(72, 700)
    for line in [
        "Management's Discussion and Analysis. During fiscal year 2022, the Company",
        "continued to execute its strategic priorities, investing approximately 15",
        "percent of total revenue in research and development during the period.",
        "The Company expects fiscal 2023 growth of around 10 percent, driven by",
        "20 new product launches planned across calendar year 2023 and into 2024.",
    ]:
        t.textLine(line)
    c.drawText(t)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_pdfplumber_native_reconstruction_gets_correct_values():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_3m_income_statement_pdf())) as pdf:
        common = parser._pdfplumber_words_to_common(pdf.pages[0].extract_words())
        tables, _ = parser._reconstruct_table_from_word_positions(common)

    assert len(tables) == 1
    joined = tables[0]
    assert "Net sales" in joined and "34,229" in joined and "35,355" in joined and "32,184" in joined
    assert "Cost of sales" in joined and "19,232" in joined and "18,795" in joined and "16,605" in joined
    for line in joined.split("\n"):
        assert _is_table_line(line)


def test_fitz_native_reconstruction_gets_correct_values():
    """Same fixture, but through the fitz get_text("words") adapter —
    values must match exactly regardless of which engine's coordinate
    data is used."""
    parser = FinancialFileParser()
    doc = fitz.open(stream=_build_3m_income_statement_pdf(), filetype="pdf")
    common = parser._fitz_words_to_common(doc[0].get_text("words"))
    doc.close()
    tables, _ = parser._reconstruct_table_from_word_positions(common)

    assert len(tables) == 1
    joined = tables[0]
    assert "Net sales" in joined and "34,229" in joined and "35,355" in joined and "32,184" in joined
    assert "Cost of sales" in joined and "19,232" in joined and "18,795" in joined and "16,605" in joined


def test_net_sales_row_values_in_correct_order_same_row():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_3m_income_statement_pdf())) as pdf:
        common = parser._pdfplumber_words_to_common(pdf.pages[0].extract_words())
        tables, _ = parser._reconstruct_table_from_word_positions(common)

    net_sales_line = next(
        line for line in tables[0].split("\n") if line.strip().startswith("| Net sales ")
    )
    cells = [c.strip() for c in net_sales_line.strip().strip("|").split("|")]
    assert cells == ["Net sales", "34,229", "35,355", "32,184"]

    cost_of_sales_line = next(
        line for line in tables[0].split("\n") if line.strip().startswith("| Cost of sales ")
    )
    cells2 = [c.strip() for c in cost_of_sales_line.strip().strip("|").split("|")]
    assert cells2 == ["Cost of sales", "19,232", "18,795", "16,605"]


def test_full_pipeline_produces_correct_markdown_table():
    parser = FinancialFileParser()
    result = parser._parse_pdf("3m_2022_10k.pdf", _build_3m_income_statement_pdf(), "3M")

    table_passages = [p for p in result["passages"] if p["type"] == "table_row"]
    assert len(table_passages) == 2

    parent_content = result["passages"][0]["parent_content"]
    assert "| Net sales | 34,229 | 35,355 | 32,184 |" in parent_content
    assert "| Cost of sales | 19,232 | 18,795 | 16,605 |" in parent_content


TIGHT_ROWS = [
    ("Net sales", ["34,229", "35,355", "32,184"]),
    ("Cost of sales", ["19,232", "18,795", "16,605"]),
    ("Selling, general and administrative expenses", ["9,049", "8,543", "7,995"]),
    ("Research, development and related expenses", ["1,977", "1,878", "1,878"]),
    ("Goodwill impairment expense", ["—", "435", "—"]),
    ("Operating income", ["6,632", "8,317", "5,383"]),
]


def _build_tight_spacing_income_statement_pdf() -> bytes:
    """Real 10-Ks routinely space value columns far tighter than this
    module's other fixtures (e.g. 3M's actual FY2022 page 48 uses ~26.4pt
    between adjacent year columns — see the real x0/x1 measurements this
    module's adaptive-threshold algorithm was built from). This fixture
    uses a comparably tight 27pt column gap, plus line items whose values
    have genuinely different digit counts within the same column
    ("34,229" vs "9,049" vs "—"), which is exactly what makes x0 drift
    while x1 stays put — the scenario a fixed, larger hardcoded gap
    threshold (the old _WORD_COL_X_GAP = 30.0) would misdetect as having
    zero columns at all."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(width / 2, y, "TIGHTCO CORPORATION AND SUBSIDIARIES")
    y -= 11
    c.drawCentredString(width / 2, y, "Consolidated Statement of Income")
    y -= 20

    col_x = [430, 457, 484]  # 27pt gaps, tighter than every other fixture here
    c.setFont("Helvetica-Bold", 7)
    for ci, yr in enumerate(["2022", "2021", "2020"]):
        c.drawRightString(col_x[ci], y, yr)
    y -= 13

    row_h = 12
    label_x = 72
    c.setFont("Helvetica", 7)
    for label, vals in TIGHT_ROWS:
        c.drawString(label_x, y - 9, label)
        for ci, val in enumerate(vals):
            c.drawRightString(col_x[ci], y - 9, val)
        y -= row_h

    c.showPage()
    c.save()
    return buf.getvalue()


def test_tight_column_spacing_still_detected_as_table():
    """Stress test for the adaptive column-gap threshold: a 27pt real
    column gap must still be found even though it's tighter than the old
    hardcoded 30pt constant, and even with an em-dash placeholder value
    and rows of differing digit counts within the same column."""
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_tight_spacing_income_statement_pdf())) as pdf:
        common = parser._pdfplumber_words_to_common(pdf.pages[0].extract_words())
        tables, _ = parser._reconstruct_table_from_word_positions(common)

    assert len(tables) == 1, "failed to detect the table at tight (27pt) real-world column spacing"
    joined = tables[0]
    for label, vals in TIGHT_ROWS:
        assert label in joined, f"missing line item: {label}"
        for v in vals:
            assert v in joined, f"missing value {v!r} for {label!r}"

    sga_line = next(
        line for line in joined.split("\n") if line.strip().startswith("| Selling")
    )
    cells = [c.strip() for c in sga_line.strip().strip("|").split("|")]
    assert cells == [
        "Selling, general and administrative expenses", "9,049", "8,543", "7,995",
    ], f"SG&A row values wrong or out of order: {cells}"

    for line in joined.split("\n"):
        assert _is_table_line(line)


def test_pure_prose_page_not_misdetected_as_table():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_prose_only_pdf())) as pdf:
        common = parser._pdfplumber_words_to_common(pdf.pages[0].extract_words())
        tables, prose = parser._reconstruct_table_from_word_positions(common)
    assert tables == []
    assert "Management's Discussion" in prose

    result = parser._parse_pdf("prose.pdf", _build_prose_only_pdf(), "TESTCO")
    types = {p["type"] for p in result["passages"]}
    assert types == {"text_note"}
