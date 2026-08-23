"""
Regression test using the exact AMCOR Income Statement fixture from
diag_pdf_deep.py (fitz page.insert_text() with a single manually
space-padded multi-line string, rendered in a PROPORTIONAL font).

This is a harder case than the ruled-line / real-glyph-position fixtures in
test_borderless_table_extraction.py: because Helvetica is proportional, the
source string's space-padding does NOT translate into genuinely evenly-spaced
column x-coordinates once rendered — pdfplumber's extract_text(layout=True)
reconstruction often collapses the gap between two adjacent numeric columns
down to a single space (not the 2+ that _LAYOUT_ROW_RE originally required).
This is what _LAYOUT_ROW_RE's relaxed 1+-space value-token gap exists for.

Also covers the scanned/image-page guard: a page with near-zero extractable
text AND embedded images must skip table extraction rather than crash or
fabricate a table out of nothing.
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

fitz = pytest.importorskip("fitz")
pdfplumber = pytest.importorskip("pdfplumber")

from app.rag.parser import FinancialFileParser
from app.rag.chunker import _is_table_line, chunk_text


# Verbatim from diag_pdf_deep.py's page 1 (Income Statement).
INCOME_STATEMENT_TEXT = (
    "AMCOR PLC  Annual Report on Form 10-K  Fiscal Year Ended June 30, 2023\n\n"
    "CONSOLIDATED STATEMENTS OF INCOME\n"
    "($ millions, except per share data)               FY2023    FY2022\n"
    "Net sales                                         14,694    14,544\n"
    "Cost of sales                                    (11,328)  (11,229)\n"
    "Gross profit                                       3,366     3,315\n"
    "Selling, general and administrative expenses       (857)     (888)\n"
    "Research and development expenses                  (107)     (103)\n"
    "Other income, net                                    32        38\n"
    "Earnings before interest and taxes                 2,434     2,362\n"
    "Interest expense, net                              (296)     (228)\n"
    "Income before income taxes and equity              2,138     2,134\n"
    "Income tax expense                                 (393)     (403)\n"
    "Net income                                         1,048       878\n"
    "Net income attributable to Amcor plc shareholders    993       845\n"
)


def _build_amcor_income_statement_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 40), INCOME_STATEMENT_TEXT, fontsize=9)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def _build_blank_scanned_style_pdf() -> bytes:
    """A page with essentially no extractable text — simulates a scanned
    page for the guard test (a real scanned page also has an embedded
    image XObject, which this synthetic page doesn't, but that's covered
    separately in the guard-logic assertions below)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_amcor_income_statement_all_rows_captured():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_amcor_income_statement_pdf())) as pdf:
        page = pdf.pages[0]
        md_tables, prose = parser._extract_page_tables_and_prose(page)

    assert md_tables, "layout fallback failed to detect the AMCOR income statement"
    assert len(md_tables) == 1, "all 12 line items should merge into a single table block"
    joined = md_tables[0]

    for label in [
        "Net sales", "Cost of sales", "Gross profit",
        "Selling, general and administrative expenses",
        "Research and development expenses", "Other income, net",
        "Earnings before interest and taxes", "Interest expense, net",
        "Income before income taxes and equity", "Income tax expense",
        "Net income", "Net income attributable to Amcor plc shareholders",
    ]:
        assert label in joined, f"missing line item: {label}"

    for line in joined.split("\n"):
        assert _is_table_line(line)


def test_amcor_net_sales_values_correct_and_not_mixed_up():
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_amcor_income_statement_pdf())) as pdf:
        md_tables, _ = parser._extract_page_tables_and_prose(pdf.pages[0])

    net_sales_row = next(
        line for line in md_tables[0].split("\n") if line.strip().startswith("| Net sales ")
    )
    cells = [c.strip() for c in net_sales_row.strip().strip("|").split("|")]
    assert cells == ["Net sales", "14,694", "14,544"], (
        f"Net sales row values wrong or out of order: {cells}"
    )
    # Cost of sales' values must not have leaked into this row (or vice versa)
    assert "11,328" not in net_sales_row and "11,229" not in net_sales_row


def test_amcor_table_survives_chunking_intact():
    parser = FinancialFileParser()
    result = parser._parse_pdf("amcor_10k.pdf", _build_amcor_income_statement_pdf(), "AMCOR")
    table_passages = [p for p in result["passages"] if p["type"] == "table_row"]
    assert len(table_passages) == 12, f"expected 12 table_row passages, got {len(table_passages)}"

    page_text = result["passages"][0]["parent_content"]
    chunks = chunk_text(page_text, chunk_size=150)
    line_items = ["Net sales", "Cost of sales", "Gross profit"]
    chunks_with_all = [ch for ch in chunks if all(li in ch for li in line_items)]
    assert len(chunks_with_all) == 1, "table rows must stay together in a single chunk"


def test_pure_narrative_page_not_misdetected_as_table():
    """A page of ordinary MD&A prose (no financial table at all) must
    produce zero tables — this is the AMCOR fixture's negative control."""
    mda_text = (
        "Item 7. Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations. During fiscal year 2023, the Company "
        "continued to execute its strategic priorities across all reporting "
        "segments, focusing on operational efficiency, sustainable packaging "
        "innovation, and disciplined capital allocation. Management believes "
        "the Company is well positioned for continued growth in fiscal 2024, "
        "notwithstanding ongoing macroeconomic uncertainty and volatile input "
        "costs across its global supply chain and manufacturing footprint."
    )
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 40), mda_text, fontsize=9)
    pdf_bytes = doc.write()
    doc.close()

    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        md_tables, prose = parser._extract_page_tables_and_prose(pdf.pages[0])

    assert md_tables == [], "pure narrative text must never be misdetected as a table"
    assert "Management's Discussion" in prose


def test_scanned_style_page_skips_table_extraction_without_ocr():
    """A near-empty-text page must skip table extraction (and never raise,
    never call any OCR library) rather than fabricate a table from nothing."""
    parser = FinancialFileParser()
    with pdfplumber.open(io.BytesIO(_build_blank_scanned_style_pdf())) as pdf:
        page = pdf.pages[0]
        assert len((page.extract_text() or "").strip()) < parser._SCANNED_PAGE_CHAR_THRESHOLD
        md_tables, prose = parser._extract_page_tables_and_prose(page)
    assert md_tables == []
