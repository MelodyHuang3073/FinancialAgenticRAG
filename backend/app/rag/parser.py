import os
import csv
import io
import json
import re
import html
from typing import List, Dict, Any, Tuple, Optional

from app.rag.chunker import chunk_text
from app.tools.table_parser import linearize_financial_table


class FinancialFileParser:
    """
    Parses PDF, CSV, TXT, MD, JSON financial files into structured passages for FinAgent-RAG.
    """

    def parse_file(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        company_name = os.path.splitext(filename)[0]

        if ext == '.pdf':
            return self._parse_pdf(filename, content_bytes, company_name)
        elif ext == '.csv':
            return self._parse_csv(filename, content_bytes, company_name)
        elif ext in ['.txt', '.md']:
            return self._parse_text(filename, content_bytes, company_name)
        elif ext == '.json':
            return self._parse_json(filename, content_bytes, company_name)
        else:
            return self._parse_text(filename, content_bytes, company_name)

    # ------------------------------------------------------------------
    # PDF parsing — 4 engines tried in order, early return on success
    # ------------------------------------------------------------------

    def _clean_text_content(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\x00", "")
        cleaned = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFEFF\uFFFD]", "", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"[ \t\r]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_readable(text: str, min_alnum: int = 5) -> bool:
        """Return True if the text contains at least min_alnum alphanumeric characters."""
        if not text or not text.strip():
            return False
        return sum(1 for c in text if c.isalnum()) >= min_alnum

    # ──────────────────────────────────────────────────────────────────────
    # PDF financial table detection & linearisation  (RC1 fix)
    # ──────────────────────────────────────────────────────────────────────

    # Regex: line that starts with a text label and has ≥2 space-separated
    # numeric values (possibly parenthesised for negatives like (1,234))
    _TABLE_ROW_RE = re.compile(
        r'^(.{2,55}?)\s{2,}([\(\-]?\d[\d,\.]*(?:\))?)\s{2,}([\(\-]?\d[\d,\.]*(?:\))?)',
        re.MULTILINE,
    )
    # Recognises year headers: FY2023, 2023, Dec 2023, December 31 2023, etc.
    _YEAR_HEADER_RE = re.compile(r'(?:FY\s*|fiscal\s+)?(20\d{2}|19\d{2})', re.IGNORECASE)
    # Known financial line-item keywords (triggers table detection)
    _FINANCIAL_KEYWORDS = {
        "revenue", "net sales", "net revenue", "total revenue",
        "gross profit", "gross margin",
        "operating income", "operating profit", "operating loss",
        "net income", "net loss", "net earnings",
        "cost of revenue", "cost of goods", "cost of sales",
        "ebitda", "ebit",
        "earnings per share", "eps", "diluted eps",
        "total assets", "total liabilities", "shareholders equity", "stockholders equity",
        "cash and cash equivalents", "long-term debt",
        "capital expenditure", "capex", "free cash flow",
        "depreciation", "amortization",
        "research and development", "r&d",
        # Chinese
        "營業收入", "營收", "毛利", "毛利率", "營業利益", "本期淨利", "淨利",
        "每股盈餘", "資本支出", "研發費用", "總資產", "股東權益",
    }

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: 10-K Section Anchoring — header patterns → section labels
    # ──────────────────────────────────────────────────────────────────────
    # Each tuple: (regex_pattern, section_label)
    # Ordered from most-specific to least-specific; first match wins.
    _SECTION_PATTERNS: List[tuple] = [
        # ── Income Statement ────────────────────────────────────────────
        (re.compile(
            r'consolidated\s+statements?\s+of\s+(?:operations|income|earnings|comprehensive)',
            re.IGNORECASE), "income_statement"),
        (re.compile(
            r'statements?\s+of\s+(?:operations|income|earnings)',
            re.IGNORECASE), "income_statement"),
        # ── Balance Sheet ────────────────────────────────────────────────
        (re.compile(
            r'consolidated\s+balance\s+sheets?',
            re.IGNORECASE), "balance_sheet"),
        (re.compile(
            r'balance\s+sheets?|financial\s+position',
            re.IGNORECASE), "balance_sheet"),
        # ── Cash Flow ────────────────────────────────────────────────────
        (re.compile(
            r'consolidated\s+statements?\s+of\s+cash\s+flows?',
            re.IGNORECASE), "cash_flow"),
        (re.compile(
            r'statements?\s+of\s+cash\s+flows?|cash\s+flow\s+statements?',
            re.IGNORECASE), "cash_flow"),
        # ── Stockholders' Equity ─────────────────────────────────────────
        (re.compile(
            r'statements?\s+of\s+(?:stockholders|shareholders|changes\s+in).*equity',
            re.IGNORECASE), "equity_statement"),
        # ── MD&A ─────────────────────────────────────────────────────────
        (re.compile(
            r"item\s+7[^a-z].*management.{0,20}discussion|management.{0,20}discussion"
            r".{0,40}analysis",
            re.IGNORECASE), "general_mda"),
        # ── Quantitative Market Risk ─────────────────────────────────────
        (re.compile(
            r'item\s+7a[^a-z].*quantitative.*market\s+risk',
            re.IGNORECASE), "notes_market_risk"),
        # ── Financial Statements & Notes header ──────────────────────────
        (re.compile(
            r'item\s+8[^a-z].*financial\s+statements',
            re.IGNORECASE), "income_statement"),  # starts financial statements section
        # ── Specific Notes ───────────────────────────────────────────────
        (re.compile(
            r'note\s+\d+[^\n]*(?:litigation|legal\s+proceed|contingenc)',
            re.IGNORECASE), "notes_litigation"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:income\s+tax|tax\s+provision)',
            re.IGNORECASE), "notes_income_tax"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:long.term\s+debt|debt|borrowing)',
            re.IGNORECASE), "notes_debt"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:segment|geographic)',
            re.IGNORECASE), "notes_segments"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:pension|retirement|benefit)',
            re.IGNORECASE), "notes_pension"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:lease|right.of.use)',
            re.IGNORECASE), "notes_leases"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:acquisit|business\s+combination)',
            re.IGNORECASE), "notes_acquisitions"),
        (re.compile(
            r'note\s+\d+[^\n]*(?:stock.based|share.based|equity\s+award)',
            re.IGNORECASE), "notes_stock_comp"),
        # Generic note catch-all
        (re.compile(
            r'notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?',
            re.IGNORECASE), "notes_general"),
        # ── Risk Factors ─────────────────────────────────────────────────
        (re.compile(
            r'item\s+1a[^a-z].*risk\s+factors?',
            re.IGNORECASE), "risk_factors"),
        # ── Business overview ────────────────────────────────────────────
        (re.compile(
            r'item\s+1[^a-z].*business|overview\s+of\s+(?:our\s+)?business',
            re.IGNORECASE), "business_overview"),
        # ── Selected Financial Data ──────────────────────────────────────
        (re.compile(
            r'selected\s+(?:financial|consolidated)\s+data',
            re.IGNORECASE), "selected_data"),
        # ── Cover page / general ─────────────────────────────────────────
        (re.compile(
            r'annual\s+report|form\s+10-?k|united\s+states.*securities',
            re.IGNORECASE), "cover_page"),
    ]

    def _detect_section(self, page_text: str) -> str:
        """
        Scan the first 600 characters of a page for known 10-K section headers.
        Returns the section label if a header is found, or empty string if not.
        A non-empty result means this page STARTS a new section.
        """
        scan_zone = page_text[:600]  # header zone only
        for pattern, label in self._SECTION_PATTERNS:
            if pattern.search(scan_zone):
                return label
        return ""  # no new section header on this page

    @staticmethod
    def _inject_section(passages: list, section: str) -> list:
        """Add 'section' key to all passage dicts in-place, return list."""
        for p in passages:
            p["section"] = section
        return passages

    @staticmethod
    def _to_num(raw: str) -> str:
        """Convert (1,234) → -1234; strip commas."""
        s = raw.strip()
        if s.startswith('(') and s.endswith(')'):
            return '-' + s[1:-1].replace(',', '')
        return s.replace(',', '')

    # ──────────────────────────────────────────────────────────────────────
    # pdfplumber table extraction → Markdown pipe tables
    # ──────────────────────────────────────────────────────────────────────

    _MD_SEPARATOR_RE = re.compile(r'^[\|\s\-:]+$')

    @staticmethod
    def _clean_table_cell(cell) -> str:
        """Normalise a single pdfplumber table cell into a Markdown-safe string."""
        if cell is None:
            return ""
        text = str(cell).replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.replace("|", "\\|")

    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """
        Convert a pdfplumber extract_table()/extract_tables() result
        (row x col list of str/None) into a Markdown pipe table.
        First row is treated as the header row. Returns "" if the table
        has no usable header or fewer than 2 rows.
        """
        if not table:
            return ""
        rows = [[self._clean_table_cell(c) for c in row] for row in table if row is not None]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) < 2:
            return ""

        n_cols = max(len(r) for r in rows)
        if n_cols == 0:
            return ""
        padded = [r + [""] * (n_cols - len(r)) for r in rows]

        header, data_rows = padded[0], padded[1:]
        if not any(c for c in header):
            return ""

        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * n_cols) + "|")
        for row in data_rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # A "layout row" is a label followed by 2+ whitespace-only numeric-looking
    # tokens — this is how pdfplumber's extract_text(layout=True) renders a
    # table COLUMN whose boundary is only whitespace/background-shading,
    # never a ruled line (the standard style for real 10-K statements, e.g.
    # Activision Blizzard's cash-flow statement: no grid lines, just
    # alternating row shading and right-aligned numbers).
    #
    # - Requires at least one letter/CJK char before the values, so a bare
    #   year-header line like "2019   2018   2017" (no real label) is never
    #   mistaken for a data row — it should stay a header candidate.
    # - The label-to-first-value gap only needs 1+ space (long line-item
    #   labels often butt right up against the first numeric column with
    #   little room to spare), but at least TWO numeric tokens are required
    #   (multi-space-separated) — real financial tables always show 2+
    #   comparison periods side by side, and requiring 2+ avoids matching an
    #   ordinary prose sentence that happens to trail off with one number.
    _LAYOUT_ROW_RE = re.compile(
        r'^(?=.*[A-Za-z一-鿿])(?P<label>\S.{0,80}?)\s+'
        r'(?P<values>[\(\-\$]?\d[\d,\.]*\)?%?(?:\s{2,}[\(\-\$]?\d[\d,\.]*\)?%?)+)\s*$'
    )

    def _layout_text_to_markdown_and_prose(self, layout_text: str) -> Tuple[List[str], str]:
        """
        Fallback table reconstruction for pages where find_tables() (ruled
        vector lines) found nothing. Many real 10-K filings render tables
        using only whitespace column alignment and/or alternating row
        background shading — no vector lines at all — which pdfplumber's
        line-based table detector cannot see. extract_text(layout=True)
        DOES preserve that column alignment as literal spacing even without
        ruled lines, so we regex-match contiguous "label <gap> num <gap>
        num..." lines with a consistent column count into Markdown tables.
        Returns (markdown_tables, prose) where prose is layout_text with
        the consumed table lines removed.
        """
        lines = layout_text.split('\n')

        tables: List[str] = []
        prose_lines: List[str] = []
        current_rows: List[Tuple[str, List[str]]] = []
        current_n_cols = [None]
        # Most recent non-blank, non-row line — the best candidate for a
        # column-header line (e.g. "2019   2018   2017") sitting directly
        # above the block, used instead of scanning the whole page (which
        # can be dominated by blank vertical padding above the table).
        header_candidate = [""]

        def _flush():
            if len(current_rows) >= 2:
                n_cols = current_n_cols[0]
                years = self._extract_year_headers(header_candidate[0]) if header_candidate[0] else []
                headers = ["Line Item"] + [
                    years[i] if i < len(years) else f"Col{i + 1}"
                    for i in range(n_cols)
                ]
                table = [headers] + [[label] + values for label, values in current_rows]
                md = self._table_to_markdown(table)
                if md:
                    tables.append(md)
            else:
                # Not enough consistent rows to count as a table — keep as
                # ordinary text instead of silently dropping it.
                for label, values in current_rows:
                    prose_lines.append((label + "  " + "  ".join(values)).strip())
            current_rows.clear()
            current_n_cols[0] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # A blank layout line is just vertical spacing (row gaps,
                # subtotal breathing room) — it must NOT break an
                # otherwise-contiguous table block into fragments.
                continue
            m = self._LAYOUT_ROW_RE.match(stripped)
            if m:
                label = m.group("label").strip()
                values = re.split(r'\s{2,}', m.group("values").strip())
                if current_n_cols[0] is not None and len(values) != current_n_cols[0]:
                    _flush()
                current_rows.append((label, values))
                current_n_cols[0] = len(values)
            else:
                _flush()
                prose_lines.append(stripped)
                header_candidate[0] = stripped
        _flush()

        return tables, "\n".join(prose_lines)

    def _extract_page_tables_and_prose(self, page) -> Tuple[List[str], str]:
        """
        Detect all tables on a pdfplumber page and convert each to a
        Markdown pipe table. Returns (markdown_tables, prose) where prose
        is the page text with the detected table regions excluded, so
        table content isn't duplicated as garbled space-aligned text
        alongside the clean Markdown version.

        Two detection tiers are tried in order:
        1. find_tables() — ruled vector-line tables (bbox-exclusion via
           outside_bbox() gives clean prose).
        2. _layout_text_to_markdown_and_prose() — whitespace/background-
           shading-only tables (no ruled lines), reconstructed from
           extract_text(layout=True). This is the common case for real
           10-K financial statements.

        TODO: tables that span two PDF pages are detected and linearised
        independently per page (no cross-page stitching). Revisit if
        multi-page financial tables need to be merged into one block.
        """
        try:
            found_tables = page.find_tables()
        except Exception:
            found_tables = []

        md_tables = []
        filtered_page = page
        for t in found_tables:
            try:
                md = self._table_to_markdown(t.extract())
            except Exception:
                md = ""
            if md:
                md_tables.append(md)
            try:
                filtered_page = filtered_page.outside_bbox(t.bbox)
            except Exception:
                pass

        if md_tables:
            try:
                prose = filtered_page.extract_text() or ""
            except Exception:
                try:
                    prose = page.extract_text() or ""
                except Exception:
                    prose = ""
            return md_tables, prose

        # ── Tier 2: no ruled-line tables — try whitespace/shading-only ──
        try:
            layout_text = page.extract_text(layout=True) or ""
        except Exception:
            layout_text = ""
        layout_tables, layout_prose = self._layout_text_to_markdown_and_prose(layout_text)
        if layout_tables:
            return layout_tables, layout_prose

        try:
            prose = page.extract_text() or ""
        except Exception:
            prose = ""
        return [], prose

    @staticmethod
    def _compose_page_text(prose: str, md_tables: List[str]) -> str:
        """Merge non-table prose with converted Markdown tables, one blank line apart."""
        if not md_tables:
            return prose
        parts = [prose.strip()] if prose and prose.strip() else []
        parts.extend(md_tables)
        return "\n\n".join(parts)

    def _parse_markdown_table_block(self, block: str) -> Tuple[List[str], List[List[str]]]:
        """Parse a Markdown pipe-table block (as produced by _table_to_markdown)
        into (headers, data_rows), skipping the |---|---| separator line."""
        rows = []
        for line in block.split('\n'):
            stripped = line.strip()
            if not stripped or self._MD_SEPARATOR_RE.match(stripped):
                continue
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            rows.append(cells)
        if not rows:
            return [], []
        return rows[0], rows[1:]

    def _find_markdown_table_blocks(self, text: str) -> List[str]:
        """Split text into prose/table blocks (reusing chunker's block
        splitter so detection stays consistent with chunk_text()) and
        return only the blocks recognised as Markdown pipe tables."""
        from app.rag.chunker import _split_into_blocks, _is_table_line
        blocks = _split_into_blocks(text)
        return [
            b for b in blocks
            if _is_table_line(next((l for l in b.split('\n') if l.strip()), ""))
        ]

    def _linearize_markdown_tables(
        self,
        company_name: str,
        filename: str,
        page_num: int,
        page_text: str,
    ) -> Tuple[list, str]:
        """
        Find Markdown pipe-table blocks in page_text (produced by
        _extract_page_tables_and_prose / _table_to_markdown) and convert
        each data row into its own 'table_row' passage, using the table's
        real column headers instead of the generic Col1/Col2 fallback.
        Returns (table_passages, prose_only_text) where prose_only_text is
        page_text with the table blocks removed, so callers can still chunk
        the remaining prose separately.
        """
        table_blocks = self._find_markdown_table_blocks(page_text)
        if not table_blocks:
            return [], page_text

        prose_only_text = page_text
        passages = []
        table_name = f"{filename} (Page {page_num} – Financial Table)"
        parent_id = f"parent_{company_name}_p{page_num}_tbl"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + page_text[:1000]
        )

        row_idx = 0
        for block in table_blocks:
            headers, data_rows = self._parse_markdown_table_block(block)
            if len(headers) < 2 or not data_rows:
                continue
            value_headers = headers[1:]

            for row in data_rows:
                if not row or not row[0].strip():
                    continue
                line_item = row[0].strip()
                values = row[1:]
                kv_parts = [
                    f"{value_headers[i] if i < len(value_headers) else f'Col{i + 1}'}: {values[i]}"
                    for i in range(len(values))
                ]
                row_idx += 1
                raw_data = {"line_item": line_item}
                for i, val in enumerate(values):
                    key = value_headers[i] if i < len(value_headers) else f"col{i + 1}"
                    raw_data[key] = val

                passages.append({
                    "id": f"pdf_tbl_{company_name}_p{page_num}r{row_idx}",
                    "company": company_name,
                    "table_name": table_name,
                    "period": "-".join(value_headers) if value_headers else "N/A",
                    "page_number": page_num,
                    "content": (
                        f"Company: {company_name} | Report: {table_name} | "
                        f"Line Item: {line_item} | " + " | ".join(kv_parts)
                    ),
                    "type": "table_row",
                    "raw_data": raw_data,
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                    "is_child": True,
                })

            prose_only_text = prose_only_text.replace(block, "", 1)

        return passages, prose_only_text

    def _is_financial_table_page(self, text: str) -> bool:
        """Return True if the page looks like a financial statement table."""
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in self._FINANCIAL_KEYWORDS if kw in text_lower)
        if keyword_hits < 2:
            return False
        # Must have at least 3 rows that match the table row pattern
        matches = self._TABLE_ROW_RE.findall(text)
        return len(matches) >= 3

    def _extract_year_headers(self, text: str) -> list:
        """
        Try to find year column headers from the first 400 chars of page text.
        Returns list of year strings e.g. ['2023', '2022'] in order of appearance.
        """
        header_zone = text[:400]
        years = []
        for m in self._YEAR_HEADER_RE.finditer(header_zone):
            yr = m.group(1)
            if yr not in years:
                years.append(yr)
        return years if years else []

    def _linearize_table_page(
        self,
        company_name: str,
        filename: str,
        page_num: int,
        page_text: str,
    ) -> list:
        """
        Parse a financial table page and return linearised passage dicts
        in 'Line Item: X | 2023: Y | 2022: Z' format.
        """
        year_headers = self._extract_year_headers(page_text)
        # Fallback header labels if no years detected
        if not year_headers:
            year_headers = ["Col1", "Col2"]

        parent_id = f"parent_{company_name}_p{page_num}_tbl"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + page_text[:1000]
        )

        passages = []
        table_name = f"{filename} (Page {page_num} – Financial Table)"

        for row_idx, m in enumerate(self._TABLE_ROW_RE.finditer(page_text)):
            line_item = m.group(1).strip()
            val1 = self._to_num(m.group(2))
            val2 = self._to_num(m.group(3))

            # Skip header-like rows or rows with obviously wrong item names
            if not line_item or line_item.replace(' ', '').isdigit():
                continue
            # Skip rows where "line item" is just a number (e.g. year itself)
            if re.match(r'^[\d\s\(\)\-\.,]+$', line_item):
                continue

            # Build linearised content
            h0 = year_headers[0] if len(year_headers) > 0 else "Col1"
            h1 = year_headers[1] if len(year_headers) > 1 else "Col2"
            linearized = (
                f"Company: {company_name} | Report: {table_name} | "
                f"Period: {'-'.join(year_headers)} | "
                f"Line Item: {line_item} | "
                f"{h0}: {val1} | {h1}: {val2}"
            )

            passages.append({
                "id": f"pdf_tbl_{company_name}_p{page_num}r{row_idx}",
                "company": company_name,
                "table_name": table_name,
                "period": '-'.join(year_headers) if year_headers else "N/A",
                "page_number": page_num,
                "content": linearized,
                "type": "table_row",
                "raw_data": {
                    "line_item": line_item,
                    year_headers[0] if year_headers else "col1": val1,
                    (year_headers[1] if len(year_headers) > 1 else "col2"): val2,
                },
                "parent_id": parent_id,
                "parent_content": parent_content,
                "is_child": True,
            })

        return passages

    def _chunk_text_to_passages(
        self,
        company_name: str,
        filename: str,
        page_num: int,
        text: str,
        section: str,
        parent_source_text: Optional[str] = None,
    ) -> list:
        """
        Chunk free text into 'text_note' passages. `parent_source_text` lets
        the parent record keep the full original page text (e.g. including a
        table that was linearised separately) even when only the prose part
        of the page is being chunked here.
        """
        source_text = parent_source_text if parent_source_text is not None else text
        chunks = chunk_text(text, chunk_size=800, overlap=120, min_chunk_size=100)
        if not chunks and text.strip():
            chunks = [text.strip()]

        parent_id = f"parent_{company_name}_p{page_num}"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + source_text[:1000]
        )

        passages = []
        for i, chunk in enumerate(chunks, 1):
            passages.append({
                "id": f"pdf_{company_name}_p{page_num}c{i}",
                "company": company_name,
                "table_name": f"{filename} (Page {page_num})",
                "period": "Uploaded PDF Document",
                "page_number": page_num,
                "content": (
                    f"Company: {company_name} | Document: {filename} "
                    f"| Page: {page_num} | Content: {chunk}"
                ),
                "type": "text_note",
                "section": section,
                "raw_data": {"paragraph": chunk, "page": page_num},
                # Parent-child fields
                "parent_id": parent_id,
                "parent_content": parent_content,
                "is_child": True,
            })
        return passages

    def _make_passages(
        self,
        company_name: str,
        filename: str,
        page_num: int,
        page_text: str,
        section: str = "unknown",
    ) -> list:
        """
        Entry point for a single page:
        - If the page contains Markdown pipe tables (from pdfplumber
          extract_tables/find_tables) → linearise each row individually,
          and chunk any remaining prose separately.
        - Else if the page looks like a space-aligned financial table
          (legacy RC1 heuristic — used when no Markdown tables were
          detected, e.g. on the pypdf/pdfminer fallback engines that have
          no table-detection API) → linearise it.
        - Otherwise → chunk as free text (original behaviour).
        Each passage is a child; the full page text is the parent.
        All passages receive the 'section' metadata tag for Step-3 anchored retrieval.
        """
        # ── Markdown tables (pdfplumber-detected) ──────────────────────────
        md_table_passages, prose_only_text = self._linearize_markdown_tables(
            company_name, filename, page_num, page_text
        )
        if md_table_passages:
            passages = self._inject_section(md_table_passages, section)
            prose_only_text = prose_only_text.strip()
            if prose_only_text and self._is_readable(prose_only_text):
                passages += self._chunk_text_to_passages(
                    company_name, filename, page_num, prose_only_text, section,
                    parent_source_text=page_text,
                )
            return passages

        # ── RC1: detect & linearise space-aligned financial tables ─────────
        if self._is_financial_table_page(page_text):
            table_passages = self._linearize_table_page(
                company_name, filename, page_num, page_text
            )
            if table_passages:
                return self._inject_section(table_passages, section)
            # If linearisation produced nothing useful, fall through to text path

        # ── Original path: chunk free text ────────────────────────────────
        return self._chunk_text_to_passages(company_name, filename, page_num, page_text, section)

    def _parse_pdf(self, filename: str, content_bytes: bytes, company_name: str) -> Dict[str, Any]:

        def _ret(passages):
            return {
                "company": company_name,
                "filename": filename,
                "passages": passages,
                "total_passages": len(passages),
                "warning": None if passages else (
                    f"Document '{filename}' was uploaded for {company_name}. "
                    "Notice: This PDF document contains scanned images or unextractable text. "
                    "For optimal financial analysis, please upload a searchable PDF, CSV, or TXT file."
                ),
            }

        # ── Pre-pass: pdfplumber table detection ───────────────────────
        # Runs regardless of which text-extraction engine below ends up
        # succeeding, so table structure survives even on the fitz path
        # (fitz has no table-detection API of its own).
        # TODO: tables spanning two PDF pages are detected & linearised
        # independently per page — no cross-page table stitching yet.
        md_tables_by_page: Dict[int, List[str]] = {}
        # For pages that DO have tables, also keep pdfplumber's bbox-excluded
        # prose so the fitz engine below can swap it in and avoid duplicating
        # the raw space-aligned table text alongside the clean Markdown table
        # (fitz has no bounding-box awareness of pdfplumber's detected tables).
        table_prose_by_page: Dict[int, str] = {}
        try:
            import pdfplumber as _pdfplumber_prepass
            with _pdfplumber_prepass.open(io.BytesIO(content_bytes)) as _pdf:
                for page_idx, page in enumerate(_pdf.pages):
                    try:
                        md_tables, filtered_prose = self._extract_page_tables_and_prose(page)
                    except Exception:
                        md_tables, filtered_prose = [], ""
                    if md_tables:
                        md_tables_by_page[page_idx] = md_tables
                        table_prose_by_page[page_idx] = filtered_prose
        except Exception as e:
            print(f"[FinancialFileParser] pdfplumber table pre-pass failed for '{filename}': {e}")

        # ── Engine 1: PyMuPDF (fitz) ──────────────────────────────────
        try:
            import fitz
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            passages = []
            current_section = "cover_page"   # Step 3: section state machine
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                best_text = ""

                # Try extraction modes in priority order; keep the richest result
                for mode in ("text", "blocks", "words"):
                    try:
                        if mode == "text":
                            raw = page.get_text("text") or ""
                        elif mode == "blocks":
                            blocks = page.get_text("blocks")
                            raw = "\n".join(
                                b[4].strip() for b in blocks if len(b) >= 5 and b[4].strip()
                            )
                        else:  # words
                            words = sorted(page.get_text("words"),
                                           key=lambda w: (round(w[1], 1), w[0]))
                            raw = " ".join(w[4] for w in words if w[4].strip())

                        cleaned = self._clean_text_content(raw)
                        if self._is_readable(cleaned) and len(cleaned) > len(best_text):
                            best_text = cleaned
                    except Exception:
                        continue

                if best_text:
                    # Merge in any Markdown tables pdfplumber detected on this
                    # page. On pages that have tables, swap fitz's own prose
                    # for pdfplumber's bbox-excluded prose (computed in the
                    # pre-pass above) so the raw space-aligned table text
                    # doesn't end up duplicated alongside the clean Markdown
                    # table below — fitz itself has no bounding-box awareness
                    # of pdfplumber's detected table regions.
                    page_md_tables = md_tables_by_page.get(page_idx, [])
                    if page_md_tables:
                        filtered_prose = self._clean_text_content(
                            table_prose_by_page.get(page_idx, "")
                        )
                        prose_for_page = (
                            filtered_prose
                            if self._is_readable(filtered_prose, min_alnum=1)
                            else best_text
                        )
                        best_text = self._compose_page_text(prose_for_page, page_md_tables)
                    # Step 3: update section state if this page has a new header
                    detected = self._detect_section(best_text)
                    if detected:
                        current_section = detected
                    passages.extend(
                        self._make_passages(company_name, filename, page_num,
                                            best_text, section=current_section)
                    )

            doc.close()
            if passages:
                return _ret(passages)
        except Exception as e:
            print(f"[FinancialFileParser] fitz failed for '{filename}': {e}")

        # ── Engine 2: pdfplumber ──────────────────────────────────────
        try:
            import pdfplumber
            passages = []
            current_section = "cover_page"   # Step 3: section state machine
            with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    best_text = ""

                    # Same engine detected the tables, so we can properly
                    # exclude their bounding boxes from the prose extraction
                    # (no duplicate garbled table text, unlike the fitz path).
                    md_tables, filtered_prose = self._extract_page_tables_and_prose(page)

                    if md_tables:
                        prose_cleaned = self._clean_text_content(filtered_prose)
                        if not self._is_readable(prose_cleaned, min_alnum=1):
                            # bbox exclusion left nothing usable; fall back to raw text
                            prose_cleaned = self._clean_text_content(page.extract_text() or "")
                        best_text = self._compose_page_text(prose_cleaned, md_tables)
                    else:
                        for extract_fn in (
                            lambda p: p.extract_text() or "",
                            lambda p: p.extract_text(layout=True) or "",
                            lambda p: " ".join(
                                w.get("text", "") for w in (p.extract_words() or [])
                                if w.get("text", "").strip()
                            ),
                        ):
                            try:
                                cleaned = self._clean_text_content(extract_fn(page))
                                if self._is_readable(cleaned) and len(cleaned) > len(best_text):
                                    best_text = cleaned
                            except Exception:
                                continue

                    if best_text:
                        detected = self._detect_section(best_text)
                        if detected:
                            current_section = detected
                        passages.extend(
                            self._make_passages(company_name, filename, page_num,
                                                best_text, section=current_section)
                        )

            if passages:
                return _ret(passages)
        except Exception as e:
            print(f"[FinancialFileParser] pdfplumber failed for '{filename}': {e}")

        # ── Engine 3: pypdf ───────────────────────────────────────────
        try:
            import pypdf
            passages = []
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                raw = page.extract_text() or ""
                cleaned = self._clean_text_content(raw)
                if self._is_readable(cleaned):
                    passages.extend(self._make_passages(company_name, filename, page_num, cleaned))

            if passages:
                return _ret(passages)
        except Exception as e:
            print(f"[FinancialFileParser] pypdf failed for '{filename}': {e}")

        # ── Engine 4: pdfminer.six (whole-doc fallback) ───────────────
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            miner_txt = pdfminer_extract_text(io.BytesIO(content_bytes)) or ""
            miner_clean = self._clean_text_content(miner_txt)
            if self._is_readable(miner_clean):
                passages = []
                miner_pages = [p.strip() for p in miner_clean.split("\f") if p.strip()]
                if not miner_pages:
                    miner_pages = [miner_clean.strip()]
                for page_num, page_body in enumerate(miner_pages, 1):
                    passages.extend(self._make_passages(company_name, filename, page_num, page_body))
                if passages:
                    return _ret(passages)
        except Exception as e:
            print(f"[FinancialFileParser] pdfminer failed for '{filename}': {e}")

        # All engines failed
        return _ret([])

    # ------------------------------------------------------------------
    # Helper: decode bytes with multi-encoding fallback
    # ------------------------------------------------------------------

    def _decode_bytes(self, content_bytes: bytes) -> str:
        """Robust multi-encoding decoder to prevent garbled text/mojibake."""
        if not content_bytes:
            return ""
        for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5', 'gbk', 'utf-16', 'latin1']:
            try:
                decoded = content_bytes.decode(enc)
                if decoded.count('\ufffd') < len(decoded) * 0.05:
                    return decoded
            except UnicodeDecodeError:
                continue
        return content_bytes.decode('utf-8', errors='ignore')

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    def _parse_csv(self, filename: str, content_bytes: bytes, company_name: str) -> Dict[str, Any]:
        text_str = self._clean_text_content(self._decode_bytes(content_bytes))
        reader = csv.reader(io.StringIO(text_str))
        rows = list(reader)

        passages = []
        if len(rows) > 1:
            headers = [h.strip() for h in rows[0]]
            table_rows = []
            for row in rows[1:]:
                if not row:
                    continue
                table_rows.append([str(cell).strip() for cell in row])

            if table_rows:
                # ── Build parent: full table as a Markdown pipe table, so the
                # frontend Source Evidence panel can render the complete
                # table for any 'table_row' hit, not just the one matched row ──
                full_table_md = self._table_to_markdown([headers] + table_rows)
                parent_id = f"parent_{company_name}_csv_table"
                parent_content = (
                    f"Company: {company_name} | Report: {filename} | Table: CSV Financial Table\n\n"
                    + full_table_md
                )

                # ── Build children: one passage per row (linearized) ──
                for row_idx, row in enumerate(table_rows, 1):
                    row_cells = [row[i] if i < len(row) else "" for i in range(len(headers))]
                    row_str = " | ".join(
                        f"{headers[i]}: {row_cells[i]}" for i in range(len(headers))
                    )
                    linearized = (
                        f"Company: {company_name} | Report: {filename} "
                        f"| Table: CSV Financial Table | Row {row_idx}: {row_str}"
                    )
                    passages.append({
                        "id": f"csv_{company_name}_row{row_idx}",
                        "company": company_name,
                        "table_name": f"{filename} (CSV Financial Table)",
                        "period": "Uploaded CSV",
                        "content": linearized,
                        "type": "table_row",
                        "raw_data": {"headers": headers, "row": row_cells},
                        # Parent-child fields
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "is_child": True,
                    })

        return {
            "company": company_name,
            "filename": filename,
            "passages": passages,
            "total_passages": len(passages)
        }

    # ------------------------------------------------------------------
    # TXT / MD parsing
    # ------------------------------------------------------------------

    def _parse_text(self, filename: str, content_bytes: bytes, company_name: str) -> Dict[str, Any]:
        text_str = self._clean_text_content(self._decode_bytes(content_bytes))
        chunks = chunk_text(text_str, chunk_size=800, overlap=120, min_chunk_size=100)
        if not chunks and text_str.strip():
            chunks = [text_str.strip()]

        # parent: first 1000 chars of the full document
        parent_id = f"parent_{company_name}_txt"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Content: "
            + text_str[:1000]
        )

        passages = []
        for idx, p in enumerate(chunks, 1):
            passages.append({
                "id": f"txt_{company_name}_{idx}",
                "company": company_name,
                "table_name": f"{filename} (Text Chunk {idx})",
                "period": "Uploaded File",
                "content": f"Company: {company_name} | Document: {filename} | Content: {p}",
                "type": "text_note",
                "raw_data": {"text": p},
                # Parent-child fields
                "parent_id": parent_id,
                "parent_content": parent_content,
                "is_child": True,
            })

        return {
            "company": company_name,
            "filename": filename,
            "passages": passages,
            "total_passages": len(passages)
        }

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_json(self, filename: str, content_bytes: bytes, company_name: str) -> Dict[str, Any]:
        try:
            data = json.loads(self._clean_text_content(self._decode_bytes(content_bytes)))
            linearized = f"Company: {company_name} | Data: " + json.dumps(data, ensure_ascii=False)
            chunks = chunk_text(linearized, chunk_size=800, overlap=120, min_chunk_size=100)
            passages = []
            for idx, chunk in enumerate(chunks, 1):
                passages.append({
                    "id": f"json_{company_name}_{idx}",
                    "company": company_name,
                    "table_name": f"{filename} (JSON Chunk {idx})",
                    "period": "Uploaded JSON",
                    "content": chunk,
                    "type": "table_row",
                    "raw_data": data
                })
            return {
                "company": company_name,
                "filename": filename,
                "passages": passages,
                "total_passages": len(passages)
            }
        except Exception:
            return self._parse_text(filename, content_bytes, company_name)
