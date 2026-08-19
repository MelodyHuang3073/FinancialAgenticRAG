"""
Tests for FinancialTableAwareSplitter (app/rag/table_aware_splitter.py):
- strict Markdown tables (leading/trailing '|' + separator row) are kept intact
- loose/tolerant tables (edge '|' dropped, 3+ consecutive multi-pipe lines) are
  still detected and kept intact
- short runs (<3 lines) with no header+separator pair are NOT misdetected as
  a table — they fall through to ordinary prose splitting
- every chunk is prefixed with the "[Company: ... | Year: ... | Section: ... |
  Page: ...]" context header
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.table_aware_splitter import FinancialTableAwareSplitter, Document


def test_strict_markdown_table_kept_intact():
    page = Document(
        page_content=(
            "Some introductory prose about the company's fiscal year performance.\n\n"
            "| Line Item | 2023 | 2022 |\n"
            "|---|---|---|\n"
            "| Net sales | 14,694 | 14,544 |\n"
            "| Cost of sales | (11,328) | (11,229) |\n"
            "| Gross profit | 3,366 | 3,315 |\n\n"
            "Closing remarks about outlook for next fiscal year."
        ),
        metadata={"company": "AMCOR", "year": "2023", "section": "income_statement", "page": 5},
    )
    splitter = FinancialTableAwareSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([page])

    table_chunks = [c for c in chunks if c.metadata["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    for label in ["Net sales", "Cost of sales", "Gross profit"]:
        assert label in table_chunks[0].page_content
    assert "14,694" in table_chunks[0].page_content and "3,315" in table_chunks[0].page_content


def test_loose_borderless_table_with_dropped_edge_pipes_kept_intact():
    """Edge '|' characters dropped (common layout-analysis artifact) but
    interior pipes and 3+ consecutive rows survive — must still be caught."""
    page = Document(
        page_content=(
            "CONSOLIDATED STATEMENTS OF CASH FLOWS\n"
            "Net income | 1,503 | 1,848 | 273\n"
            "Deferred income taxes | (352) | (35) | (181)\n"
            "Depreciation and amortization | 328 | 509 | 888\n"
            "Net cash provided by operating activities | 1,831 | 1,790 | 2,213\n"
            "\n"
            "Management believes the Company is well positioned for fiscal 2024."
        ),
        metadata={"company": "ACTIVISION", "year": "2019", "section": "cash_flow", "page": 42},
    )
    splitter = FinancialTableAwareSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([page])

    table_chunks = [c for c in chunks if c.metadata["chunk_type"] == "table"]
    assert len(table_chunks) == 1, "loose fallback failed to detect the borderless table"
    for label in ["Net income", "Deferred income taxes",
                   "Depreciation and amortization",
                   "Net cash provided by operating activities"]:
        assert label in table_chunks[0].page_content


def test_short_multi_pipe_run_not_misdetected_as_table():
    """Only 2 consecutive multi-pipe lines, no header+separator pair —
    must NOT be treated as a table block (spec requires 3+ for the loose
    fallback)."""
    page = Document(
        page_content=(
            "A short note: values were reported as a | b | c and x | y | z "
            "in the appendix, but this is prose, not a table.\n"
            "Just a second line here | with | pipes | too, still prose overall."
        ),
        metadata={"company": "TESTCO", "year": "2023", "section": "notes", "page": 1},
    )
    splitter = FinancialTableAwareSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([page])
    assert all(c.metadata["chunk_type"] != "table" for c in chunks)


def test_pure_prose_produces_no_table_chunks_and_splits_normally():
    long_prose = " ".join([f"Sentence number {i} about the business." for i in range(60)])
    page = Document(
        page_content=long_prose,
        metadata={"company": "TESTCO", "year": "2023", "section": "mda", "page": 3},
    )
    splitter = FinancialTableAwareSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents([page])

    assert len(chunks) > 1, "long prose should be split into multiple chunks"
    assert all(c.metadata["chunk_type"] == "prose" for c in chunks)


def test_every_chunk_has_baked_context_header():
    page = Document(
        page_content="| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nSome trailing prose.",
        metadata={"company": "ACME", "year": "2022", "section": "notes_general", "page": 9},
    )
    splitter = FinancialTableAwareSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([page])

    assert chunks, "expected at least one chunk"
    expected_header = "[Company: ACME | Year: 2022 | Section: notes_general | Page: 9]"
    for c in chunks:
        assert c.page_content.startswith(expected_header)
