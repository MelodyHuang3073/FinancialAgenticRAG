"""
Regression tests for the "borderless" financial table fallback in parser.py.

Most real 10-K filings render statement tables with NO vector grid lines at
all — columns are separated only by whitespace and/or alternating row
background shading (see e.g. Activision Blizzard's Consolidated Statements
of Cash Flows). pdfplumber's default find_tables() (the 'lines' strategy)
finds nothing on pages like that, since it requires actual ruled lines.

_layout_text_to_markdown_and_prose() is the fallback: it reconstructs table
rows from extract_text(layout=True), which DOES preserve column alignment as
literal whitespace even without ruled lines.
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
from reportlab.lib.colors import HexColor

from app.rag.parser import FinancialFileParser
from app.rag.chunker import chunk_text, _is_table_line


CASH_FLOW_ROWS = [
    # (label, [values...], kind)
    ("Cash flows from operating activities:", None, "section"),
    ("Net income", ["$1,503", "$1,848", "$273"], "data"),
    ("Deferred income taxes", ["(352)", "(35)", "(181)"], "data"),
    ("Provision for inventories", ["6", "6", "33"], "data"),
    ("Depreciation and amortization", ["328", "509", "888"], "data"),
    ("Net cash provided by operating activities", ["1,831", "1,790", "2,213"], "data"),
]

MDA_TEXT = "Management's Discussion and Analysis. During fiscal year 2023, the Company continued to execute its strategic priorities across all reporting segments."


def _build_borderless_pdf_bytes() -> bytes:
    """Build a one-page PDF whose table has NO ruled grid lines — only
    whitespace-aligned columns and alternating light-blue row shading,
    matching the standard real-10-K statement style (e.g. Activision
    Blizzard's Consolidated Statements of Cash Flows)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 60
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width / 2, y, "ACTIVISION BLIZZARD, INC. AND SUBSIDIARIES")
    y -= 12
    c.drawCentredString(width / 2, y, "CONSOLIDATED STATEMENTS OF CASH FLOWS")
    y -= 20

    col_x = [500, 570, 640]
    c.setFont("Helvetica-Bold", 8)
    for ci, yr in enumerate(["2019", "2018", "2017"]):
        c.drawRightString(col_x[ci], y, yr)
    y -= 14

    row_h = 14
    label_x = 72
    lightblue = HexColor("#dbeeff")
    band_idx = 0
    for label, vals, kind in CASH_FLOW_ROWS:
        if kind == "section":
            c.setFillColor(HexColor("#000000"))
            c.setFont("Helvetica-Bold", 8)
            c.drawString(label_x, y - 10, label)
            y -= row_h
            continue
        if band_idx % 2 == 0:
            c.setFillColor(lightblue)
            c.rect(60, y - row_h + 3, 620, row_h, fill=1, stroke=0)
        band_idx += 1
        c.setFillColor(HexColor("#000000"))
        c.setFont("Helvetica", 8)
        c.drawString(label_x + 10, y - 10, label)
        for ci, val in enumerate(vals):
            c.drawRightString(col_x[ci], y - 10, val)
        y -= row_h

    y -= 20
    c.setFont("Helvetica", 9)
    t = c.beginText(72, y)
    t.textLine(MDA_TEXT)
    c.drawText(t)

    c.showPage()
    c.save()
    return buf.getvalue()


def _build_pure_prose_pdf_bytes() -> bytes:
    """A page with no table at all — narrative text that happens to mention
    several years/percentages, to stress-test false-positive avoidance."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    t = c.beginText(72, 700)
    for line in [
        "Management's Discussion and Analysis. During fiscal year 2023, the Company",
        "continued to execute its strategic priorities, investing approximately 15",
        "percent of total revenue in research and development during the period.",
        "The Company expects fiscal 2024 growth of around 10 percent, driven by",
        "20 new product launches planned across calendar year 2024 and into 2025.",
    ]:
        t.textLine(line)
    c.drawText(t)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_borderless_table_detected_via_layout_fallback():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_borderless_pdf_bytes())) as pdf:
        page = pdf.pages[0]
        assert not page.find_tables(), "fixture must have NO ruled lines (sanity check)"
        md_tables, prose = parser._extract_page_tables_and_prose(page)

    assert md_tables, "layout fallback failed to detect the borderless table"
    joined = "\n".join(md_tables)
    for label, vals, kind in CASH_FLOW_ROWS:
        if kind != "data":
            continue
        assert label in joined, f"missing line item: {label}"
        for v in vals:
            assert v in joined, f"missing value {v!r} for {label!r}"

    for md in md_tables:
        for line in md.split("\n"):
            assert _is_table_line(line)

    assert "1,503" not in prose, "table values leaked into prose (duplicated)"


def test_borderless_table_survives_full_parse_pipeline():
    parser = FinancialFileParser()
    result = parser._parse_pdf("activision.pdf", _build_borderless_pdf_bytes(), "ACTIVISION")
    assert result["passages"], "no passages produced for borderless-table PDF"

    table_passages = [p for p in result["passages"] if p["type"] == "table_row"]
    data_rows = [r for r in CASH_FLOW_ROWS if r[2] == "data"]
    assert len(table_passages) == len(data_rows), (
        f"expected {len(data_rows)} table_row passages, got {len(table_passages)}"
    )

    page_text = result["passages"][0]["parent_content"]
    chunks = chunk_text(page_text, chunk_size=150)
    all_labels = [label for label, _, _ in data_rows]
    chunks_with_full_table = [
        ch for ch in chunks if all(label in ch for label in all_labels)
    ]
    assert len(chunks_with_full_table) == 1, "the reconstructed table must survive as one chunk"


def test_no_false_positive_table_on_pure_prose():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_pure_prose_pdf_bytes())) as pdf:
        page = pdf.pages[0]
        md_tables, prose = parser._extract_page_tables_and_prose(page)
    assert md_tables == [], "pure narrative text must never be misdetected as a table"
    assert "Management's Discussion" in prose

    result = parser._parse_pdf("prose.pdf", _build_pure_prose_pdf_bytes(), "TESTCO")
    types = {p["type"] for p in result["passages"]}
    assert types == {"text_note"}, f"expected only text_note passages, got {types}"


def _build_dollar_dash_pdf_bytes() -> bytes:
    """Build a borderless table where '$' is drawn as its own glyph right
    before each number (pdfplumber then extracts it as a separate text run
    with its own gap — "$  1,503", not "$1,503") and blank periods use the
    standard '—' placeholder. This is the exact real-world pattern that a
    naive digit-only value regex rejects outright."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 60
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width / 2, y, "ACTIVISION BLIZZARD, INC. AND SUBSIDIARIES")
    y -= 12
    c.drawCentredString(width / 2, y, "CONSOLIDATED STATEMENTS OF CASH FLOWS")
    y -= 20

    col_dollar_x = [460, 540, 610]
    col_num_x = [500, 570, 640]
    c.setFont("Helvetica-Bold", 8)
    for ci, yr in enumerate(["2019", "2018", "2017"]):
        c.drawRightString(col_num_x[ci], y, yr)
    y -= 14

    c.setFont("Helvetica", 8)
    row_h = 13

    def draw_row(label, vals, y):
        c.drawString(72, y - 10, label)
        for ci, v in enumerate(vals):
            if v.startswith("$"):
                c.drawString(col_dollar_x[ci], y - 10, "$")
                c.drawRightString(col_num_x[ci], y - 10, v[1:])
            else:
                c.drawRightString(col_num_x[ci], y - 10, v)

    rows = [
        ("Net income", ["$1,503", "$1,848", "$273"]),
        ("Non-cash operating lease cost", ["64", "—", "—"]),
        ("Depreciation and amortization", ["328", "509", "888"]),
    ]
    for label, vals in rows:
        draw_row(label, vals, y)
        y -= row_h

    c.showPage()
    c.save()
    return buf.getvalue()


def test_dollar_glyph_and_em_dash_placeholder_rows_detected():
    """Regression test: rows using a standalone '$' glyph before the number
    and '—' as a zero/blank placeholder must still be recognised — both are
    extremely common in real 10-K statements and previously caused the
    ENTIRE page's table detection to silently fail (falling all the way
    back to plain, unconverted extract_text())."""
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_dollar_dash_pdf_bytes())) as pdf:
        md_tables, prose = parser._extract_page_tables_and_prose(pdf.pages[0])

    assert md_tables, "failed to detect table with '$' glyph / '—' placeholder rows"
    joined = "\n".join(md_tables)
    assert "Net income" in joined
    assert "$1,503" in joined and "$1,848" in joined and "$273" in joined
    assert "Non-cash operating lease cost" in joined
    assert "—" in joined, "em-dash placeholder value must be preserved, not dropped"
    assert "Depreciation and amortization" in joined
    assert "328" in joined

    for md in md_tables:
        for line in md.split("\n"):
            assert _is_table_line(line)


def test_em_dash_punctuation_in_prose_not_misdetected_as_table_row():
    """An em-dash used as ordinary sentence punctuation (not a table
    placeholder) must never trigger a false-positive table row."""
    parser = FinancialFileParser()
    prose_lines = [
        "The Company — despite ongoing macroeconomic headwinds — reported growth.",
        "Revenue increased significantly — driven largely by strong demand — this year.",
    ]
    for line in prose_lines:
        assert not parser._LAYOUT_ROW_RE.match(line.strip()), f"false positive on: {line!r}"
