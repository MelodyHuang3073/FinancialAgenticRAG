"""
FinancialTableAwareSplitter — Borderless Table-Aware Chunking for 10-K PDFs.

Real 10-K statement tables are almost never drawn with ruled grid lines —
values are separated only by whitespace and/or alternating row shading.
A layout-analysis front end (pdfplumber's "stream"/text strategy, Docling,
LlamaParse, ...) converts that into a page of Markdown that mixes ordinary
prose with pipe-delimited table blocks. A generic length-based text splitter
(e.g. RecursiveCharacterTextSplitter) has no notion of "table row" and will
happily cut a table in half at the character-count boundary, separating a
line item's label from its value on either side of the cut.

This module detects those Markdown table blocks — tolerantly, since
layout-analysis tools frequently drop the leading/trailing '|' of a row even
though every interior column separator survives — and keeps each one intact
as exactly one chunk, while ordinary prose is still split normally.

This class is independent of this project's existing app.rag.chunker /
app.rag.parser pipeline (which solves the same table-preservation problem
for this app's own PDF ingestion path — see _is_table_line() /
_split_into_blocks() in chunker.py, and the two-tier pdfplumber table
detector in parser.py's _extract_page_tables_and_prose()). It is provided
as a self-contained, LangChain-shaped component per spec, for use with any
Document-list-in / Document-list-out ingestion pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# Document type: use the real LangChain Document if it's installed, else a
# minimal duck-typed stand-in with the same `.page_content` / `.metadata`
# shape, so this module has zero hard dependency on langchain being present.
# ──────────────────────────────────────────────────────────────────────────
try:
    from langchain_core.documents import Document  # type: ignore
except ImportError:
    try:
        from langchain.schema import Document  # type: ignore
    except ImportError:
        @dataclass
        class Document:  # type: ignore
            """Minimal LangChain-Document-compatible stand-in."""
            page_content: str
            metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# RecursiveCharacterTextSplitter: use the real LangChain implementation if
# available, else a small equivalent (paragraph → line → sentence → word →
# character fallback, with overlap) so prose chunking still behaves the way
# the spec describes even in an environment without langchain installed.
# ──────────────────────────────────────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError:
        class RecursiveCharacterTextSplitter:
            """
            Minimal stand-in for LangChain's RecursiveCharacterTextSplitter.
            Tries progressively finer separators ("\\n\\n" → "\\n" → ". " →
            " " → "") so cuts land on natural boundaries wherever possible,
            and stitches pieces back together up to chunk_size with overlap.
            """

            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                         separators: Optional[List[str]] = None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

            def _split_on(self, text: str, separators: List[str]) -> List[str]:
                if not separators:
                    return [text]
                sep, rest = separators[0], separators[1:]
                if sep == "":
                    return list(text)
                parts = text.split(sep)
                pieces: List[str] = []
                for i, part in enumerate(parts):
                    piece = part if i == len(parts) - 1 else part + sep
                    if len(piece) > self.chunk_size and rest:
                        pieces.extend(self._split_on(piece, rest))
                    elif piece:
                        pieces.append(piece)
                return pieces

            def split_text(self, text: str) -> List[str]:
                if len(text) <= self.chunk_size:
                    return [text] if text.strip() else []

                atoms = self._split_on(text, self.separators)
                chunks: List[str] = []
                current = ""
                for atom in atoms:
                    if len(current) + len(atom) <= self.chunk_size:
                        current += atom
                        continue
                    if current.strip():
                        chunks.append(current)
                    # start next chunk with overlap from the tail of the last one
                    overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                    current = overlap + atom
                    while len(current) > self.chunk_size:
                        chunks.append(current[:self.chunk_size])
                        current = current[self.chunk_size - self.chunk_overlap:]
                if current.strip():
                    chunks.append(current)
                return [c for c in chunks if c.strip()]


class FinancialTableAwareSplitter:
    """
    Splits Markdown-formatted 10-K page Documents into chunks that never cut
    a table row in half, and bakes page-level metadata into every chunk.

    Detection strategy (two tiers, in order of confidence):
      1. Strict Markdown table: a row line ``| a | b | c |`` (leading AND
         trailing '|'), typically paired with a ``|---|---|---|`` separator
         row directly below the header.
      2. Loose/tolerant fallback: layout-analysis tools converting a
         borderless 10-K table often drop the OUTER '|' at one or both edges
         of a row while every INTERIOR column separator survives — e.g.
         ``Net income | 1,503 | 1,848 | 273`` with no leading/trailing pipe.
         Any line containing 2+ '|' characters counts as "table-ish"; a run
         of 3 or more CONSECUTIVE such lines is treated as one table block
         even without a proper header/separator row.

    Non-table prose is split with RecursiveCharacterTextSplitter
    (chunk_size=1000, chunk_overlap=200 by default).

    Every resulting chunk (table or prose) is prefixed with:
        "[Company: {company} | Year: {year} | Section: {section} | Page: {page}]\\n\\n{chunk}"
    """

    #: A "strict" row: starts and ends with '|' after trimming whitespace.
    _STRICT_ROW_RE = re.compile(r"^\|.*\|$")
    #: A Markdown table separator row: |---|:--:|---| (dashes/colons only
    #: between pipes, at least one dash somewhere).
    _SEPARATOR_RE = re.compile(r"^\|?[\s:\-]*\|([\s:\-]*\|)*[\s:\-]*\|?$")

    #: Loose fallback: 2+ pipes anywhere on the line = "table-ish" (edge
    #: pipes may have been dropped by the layout-analysis conversion, but
    #: interior column separators are still intact).
    _MIN_LOOSE_PIPES = 2
    #: Minimum consecutive table-ish lines to count as a table block when
    #: NOT backed by a strict header+separator pair.
    _MIN_LOOSE_RUN = 3

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prose_splitter: Optional[RecursiveCharacterTextSplitter] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prose_splitter = prose_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Table-block detection
    # ------------------------------------------------------------------

    @classmethod
    def _is_strict_row(cls, line: str) -> bool:
        s = line.strip()
        return bool(s) and bool(cls._STRICT_ROW_RE.match(s))

    @classmethod
    def _is_separator_row(cls, line: str) -> bool:
        s = line.strip()
        return bool(s) and "-" in s and bool(cls._SEPARATOR_RE.match(s))

    @classmethod
    def _is_loose_table_line(cls, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # Must contain real column content, not just punctuation — avoids
        # matching e.g. a stray "-----|-----" style rule with no letters.
        has_content = bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s))
        return has_content and s.count("|") >= cls._MIN_LOOSE_PIPES

    def _find_table_blocks(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Scan `lines` and return a list of (start, end) index ranges
        (end exclusive) identifying contiguous table blocks, using a small
        state machine:

        - A "table-ish" line is either a strict `|...|` row, a separator
          row, or a loose 2+-pipe line.
        - Consecutive table-ish lines accumulate into one candidate run.
        - A candidate run is CONFIRMED as a table block if it contains a
          strict row immediately followed by a separator row (the reliable
          signal), OR if it has no such pair but is at least
          _MIN_LOOSE_RUN lines long (the fault-tolerant fallback for tables
          whose edge '|' characters were dropped during conversion).
        - Runs that are too short and have no strict header+separator pair
          are treated as ordinary prose (protects against false positives
          on a single stray line that happens to contain a couple of '|').
        """
        blocks: List[Tuple[int, int]] = []
        i, n = 0, len(lines)

        while i < n:
            if not (self._is_strict_row(lines[i]) or self._is_loose_table_line(lines[i])):
                i += 1
                continue

            start = i
            has_strict_header_pair = False
            j = i
            while j < n and (self._is_strict_row(lines[j])
                              or self._is_separator_row(lines[j])
                              or self._is_loose_table_line(lines[j])):
                if (j + 1 < n and self._is_strict_row(lines[j])
                        and self._is_separator_row(lines[j + 1])):
                    has_strict_header_pair = True
                j += 1
            end = j
            run_len = end - start

            if has_strict_header_pair or run_len >= self._MIN_LOOSE_RUN:
                blocks.append((start, end))
            # else: too short & no reliable header signal — leave as prose,
            # don't advance past it as a "confirmed" block.
            i = end if end > i else i + 1

        return blocks

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _bake_context(metadata: Dict[str, Any], content: str) -> str:
        """Prefix a chunk with its page-level metadata, per spec:
        "[Company: X | Year: Y | Section: Z | Page: N]\\n\\n{content}" """
        header = (
            f"[Company: {metadata.get('company', 'Unknown')} | "
            f"Year: {metadata.get('year', 'Unknown')} | "
            f"Section: {metadata.get('section', 'Unknown')} | "
            f"Page: {metadata.get('page', 'Unknown')}]"
        )
        return f"{header}\n\n{content}"

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of page-level Documents into table-safe, context-
        baked chunks. Table blocks are never split; prose is split with the
        configured RecursiveCharacterTextSplitter."""
        output: List[Document] = []
        for doc in documents:
            output.extend(self._split_one(doc))
        return output

    def _split_one(self, doc: Document) -> List[Document]:
        text = doc.page_content or ""
        lines = text.split("\n")
        table_ranges = self._find_table_blocks(lines)

        segments: List[Tuple[str, str]] = []  # (kind, text) where kind in {"table", "prose"}
        cursor = 0
        for start, end in table_ranges:
            if start > cursor:
                prose = "\n".join(lines[cursor:start]).strip("\n")
                if prose.strip():
                    segments.append(("prose", prose))
            table_text = "\n".join(lines[start:end])
            segments.append(("table", table_text))
            cursor = end
        if cursor < len(lines):
            prose = "\n".join(lines[cursor:]).strip("\n")
            if prose.strip():
                segments.append(("prose", prose))

        chunks: List[Document] = []
        for kind, seg_text in segments:
            if kind == "table":
                pieces = [seg_text]  # a table block is ALWAYS exactly one chunk
            else:
                pieces = self.prose_splitter.split_text(seg_text)

            for piece in pieces:
                if not piece.strip():
                    continue
                baked = self._bake_context(doc.metadata, piece)
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_type"] = "table" if kind == "table" else "prose"
                chunks.append(Document(page_content=baked, metadata=chunk_meta))

        return chunks


# ──────────────────────────────────────────────────────────────────────────
# Bonus: PDF → Markdown preprocessing for borderless tables (pdfplumber
# "stream"/text-strategy mode). This mirrors the two-tier detector already
# used in this project's own ingestion path — see
# app.rag.parser.FinancialFileParser._extract_page_tables_and_prose() /
# _table_to_markdown() / _layout_text_to_markdown_and_prose() — kept here as
# a minimal standalone illustration of how a page becomes the Markdown input
# this splitter expects.
# ──────────────────────────────────────────────────────────────────────────

def pdf_page_to_markdown(page) -> str:
    """
    Convert one pdfplumber Page into Markdown text, detecting borderless
    tables via the 'text' strategy (vertical_strategy="text",
    horizontal_strategy="text") — the standard fix for 10-K tables that have
    no ruled grid lines, since pdfplumber's default 'lines' strategy only
    finds tables drawn with actual vector rules.

    NOTE: this is a minimal illustration. Production code in this repo
    (app/rag/parser.py) additionally: tries the reliable 'lines' strategy
    first and only falls back to 'text'; excludes detected table bounding
    boxes from the prose extraction so values aren't duplicated; and adds a
    second, regex-based fallback (extract_text(layout=True) + tolerant row
    matching) for pages where even the 'text' strategy finds nothing.
    """
    tables = page.find_tables(table_settings={
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    })

    md_blocks = []
    for table in tables:
        rows = table.extract()
        if not rows or len(rows) < 2:
            continue
        header, *data_rows = rows

        def _cell(v):
            return "" if v is None else str(v).replace("\n", " ").strip()

        lines = ["| " + " | ".join(_cell(c) for c in header) + " |"]
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in data_rows:
            row = list(row) + [None] * (len(header) - len(row))
            lines.append("| " + " | ".join(_cell(c) for c in row[:len(header)]) + " |")
        md_blocks.append("\n".join(lines))

    prose = page.extract_text() or ""
    return "\n\n".join([prose] + md_blocks) if md_blocks else prose


# ──────────────────────────────────────────────────────────────────────────
# Test example
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # A page whose Markdown table has its OUTER pipes dropped on some rows
    # (the fault-tolerant case the loose detector exists for), preceded and
    # followed by ordinary prose.
    sample_page = Document(
        page_content=(
            "Management's Discussion and Analysis. During fiscal year 2023, "
            "the Company continued to execute its strategic priorities across "
            "all reporting segments, focusing on operational efficiency and "
            "disciplined capital allocation while navigating a challenging "
            "macroeconomic environment marked by persistent inflation and "
            "elevated interest rates that pressured consumer discretionary "
            "spending across several of its core end markets throughout the "
            "fiscal year, though management remains confident in the "
            "Company's long-term strategy.\n\n"
            "CONSOLIDATED STATEMENTS OF CASH FLOWS\n"
            "Line Item | 2019 | 2018 | 2017\n"
            "Net income | 1,503 | 1,848 | 273\n"
            "Deferred income taxes | (352) | (35) | (181)\n"
            "Depreciation and amortization | 328 | 509 | 888\n"
            "Net cash provided by operating activities | 1,831 | 1,790 | 2,213\n"
            "\n"
            "The Company expects continued investment in research and "
            "development in fiscal 2024, targeting approximately 15 percent "
            "of total revenue, as it launches an estimated 20 new products "
            "across its core franchises during the upcoming fiscal year."
        ),
        metadata={"company": "Activision Blizzard", "year": "2019",
                   "section": "cash_flow", "page": 42},
    )

    splitter = FinancialTableAwareSplitter(chunk_size=300, chunk_overlap=50)
    result = splitter.split_documents([sample_page])

    for i, doc in enumerate(result, 1):
        print(f"--- chunk {i} ({doc.metadata['chunk_type']}, {len(doc.page_content)} chars) ---")
        print(doc.page_content)
        print()

    table_chunks = [d for d in result if d.metadata["chunk_type"] == "table"]
    assert len(table_chunks) == 1, "the table must survive as exactly one chunk"
    assert "Net cash provided by operating activities" in table_chunks[0].page_content
    assert "1,831" in table_chunks[0].page_content
    print(f"OK: {len(result)} chunks total, table preserved intact as 1 chunk.")
