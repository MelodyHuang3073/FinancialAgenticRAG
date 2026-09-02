"""
Regression tests for the PDF table-extraction pipeline:

- parser.py's _parse_pdf() should detect real PDF tables (via pdfplumber's
  find_tables()) and convert them into Markdown pipe tables, merged into the
  page text alongside the surrounding prose.
- chunker.py's chunk_text() should then keep that Markdown table intact in a
  single chunk, never slicing through the middle of a row, while prose
  elsewhere on the page still uses the normal sliding-window chunking.

The fixture PDF is built at test time with reportlab so pdfplumber's default
'lines' table-detection strategy can find it (it requires actual vector
grid lines, not just visually-aligned text).
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
from app.rag.chunker import chunk_text, _is_table_line


TITLE = "AMCOR PLC Annual Report FY2023"

TABLE_ROWS = [
    ["Line Item", "FY2023", "FY2022"],
    ["Net sales", "14,694", "14,544"],
    ["Cost of sales", "(11,328)", "(11,229)"],
    ["Gross profit", "3,366", "3,315"],
    ["Operating income", "1,245", "1,198"],
    ["Net income", "902", "875"],
]

MDA_TEXT = (
    "Management's Discussion and Analysis. During fiscal year 2023, the "
    "Company continued to execute its strategic priorities, focusing on "
    "operational efficiency and disciplined capital allocation. Net sales "
    "grew modestly, driven by pricing actions that offset volume softness "
    "in several end markets. Gross margin expanded slightly as input cost "
    "inflation moderated versus the prior fiscal year, while the Company "
    "continued investing in sustainable packaging innovation across its "
    "global operations, notwithstanding ongoing macroeconomic uncertainty."
)


def _build_fixture_pdf_bytes() -> bytes:
    """Build a one-page PDF with a title, a real (grid-lined) financial
    table, and a following MD&A paragraph, using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, TITLE)
    y -= 40

    col_w = [200, 100, 100]
    row_h = 20
    top = y
    c.setFont("Helvetica", 10)
    for r, row in enumerate(TABLE_ROWS):
        x = 72
        band_top = top - r * row_h
        for ci, cell in enumerate(row):
            c.drawString(x + 4, band_top - 14, str(cell))
            x += col_w[ci]

    total_w = sum(col_w)
    total_h = row_h * len(TABLE_ROWS)
    c.setLineWidth(0.5)
    xx = 72
    for cw in [0] + col_w:
        xx += cw
        c.line(xx, top, xx, top - total_h)
    for r in range(len(TABLE_ROWS) + 1):
        yy = top - r * row_h
        c.line(72, yy, 72 + total_w, yy)

    y = top - total_h - 30
    c.setFont("Helvetica", 9)
    text_obj = c.beginText(72, y)
    text_obj.setFont("Helvetica", 9)
    words = MDA_TEXT.split(" ")
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if len(candidate) > 95:
            text_obj.textLine(line)
            line = w
        else:
            line = candidate
    if line:
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="module")
def parsed_page_text():
    parser = FinancialFileParser()
    result = parser._parse_pdf("amcor_fy2023.pdf", _build_fixture_pdf_bytes(), "AMCOR")
    assert result["passages"], "parser produced no passages at all"

    # Reconstruct the raw page_text the way _parse_pdf built it, by pulling
    # it back out of the parent_content of any returned passage (all child
    # passages on the page share the same parent_content prefix/content).
    page_text = result["passages"][0]["parent_content"]
    return result, page_text


def test_table_rows_are_markdown_pipe_format(parsed_page_text):
    result, page_text = parsed_page_text

    table_lines = [
        line for line in page_text.split("\n")
        if line.strip().startswith("|")
    ]
    assert table_lines, "no table lines found in parsed page text"
    # Every line item and numeric value must actually be present in the
    # extracted Markdown table, not just visually-aligned text.
    joined = "\n".join(table_lines)
    for row in TABLE_ROWS[1:]:
        for cell in row:
            assert cell in joined, f"expected cell {cell!r} missing from Markdown table"
    for line in table_lines:
        assert _is_table_line(line), f"table line not recognised as a table row: {line!r}"


def test_table_survives_small_chunk_size(parsed_page_text):
    _, page_text = parsed_page_text

    # Feed the composed page_text into chunk_text() with a deliberately tiny
    # chunk_size so it would clearly split anything not protected as a table
    # block.
    chunks = chunk_text(page_text, chunk_size=200)
    assert chunks, "chunk_text produced no chunks"

    line_items = [row[0] for row in TABLE_ROWS[1:]]
    chunks_containing_table = [
        i for i, ch in enumerate(chunks)
        if all(item in ch for item in line_items)
    ]
    assert chunks_containing_table, (
        "no single chunk contains all 5 table line items — the table was split"
    )
    assert len(chunks_containing_table) == 1


def test_mda_prose_uses_sliding_window_and_is_not_mixed_with_table():
    chunks = chunk_text(MDA_TEXT, chunk_size=200, overlap=40, min_chunk_size=50)
    assert len(chunks) > 1, "300-char MD&A prose should be split into multiple sliding-window chunks"
    for ch in chunks:
        assert "Net sales" not in ch or "14,694" not in ch, (
            "prose chunk unexpectedly contains table content"
        )
        first_line = next((l for l in ch.split("\n") if l.strip()), "")
        assert not _is_table_line(first_line), f"prose chunk misclassified as table: {ch!r}"


def test_page_text_full_pipeline_table_and_prose():
    """End-to-end: parse the fixture PDF, then re-chunk the composed page
    text at a small chunk_size and verify the table block and the MD&A
    prose land in different chunks, with the table intact in exactly one
    chunk and the prose split across the rest."""
    parser = FinancialFileParser()

    # Build page_text directly the same way _parse_pdf does internally, so
    # we test the full-length (untruncated) composed text rather than the
    # 1000-char-truncated parent_content.
    pdf_bytes = _build_fixture_pdf_bytes()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[0]
        md_tables, prose = parser._extract_page_tables_and_prose(page)
    assert md_tables, "pdfplumber failed to detect the fixture table"

    full_prose = parser._clean_text_content(prose)
    page_text = parser._compose_page_text(full_prose, md_tables)

    for line in md_tables[0].split("\n"):
        assert _is_table_line(line)

    chunks = chunk_text(page_text, chunk_size=200)
    line_items = [row[0] for row in TABLE_ROWS[1:]]
    table_chunks = [
        ch for ch in chunks if all(item in ch for item in line_items)
    ]
    assert len(table_chunks) == 1, "table must land in exactly one chunk"

    assert len(chunks) > 1, "expected multiple chunks (table + prose)"
    assert any("Management's Discussion" in ch for ch in chunks), "MD&A prose missing from chunks"
    assert not any("14,694" in ch for ch in chunks if ch not in table_chunks), (
        "table values leaked into a non-table chunk"
    )
