import os
import csv
import io
import json
import re
import html
from typing import List, Dict, Any

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

    @staticmethod
    def _to_num(raw: str) -> str:
        """Convert (1,234) → -1234; strip commas."""
        s = raw.strip()
        if s.startswith('(') and s.endswith(')'):
            return '-' + s[1:-1].replace(',', '')
        return s.replace(',', '')

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

    def _make_passages(
        self,
        company_name: str,
        filename: str,
        page_num: int,
        page_text: str,
    ) -> list:
        """
        Entry point for a single page:
        - If page looks like a financial table → linearize each row individually
        - Otherwise → chunk as free text (original behaviour)
        Each passage is a child; the full page text is the parent.
        """
        # ── RC1: detect & linearise financial tables ──────────────────────
        if self._is_financial_table_page(page_text):
            table_passages = self._linearize_table_page(
                company_name, filename, page_num, page_text
            )
            if table_passages:
                return table_passages
            # If linearisation produced nothing useful, fall through to text path

        # ── Original path: chunk free text ────────────────────────────────
        chunks = chunk_text(page_text, chunk_size=800, overlap=120, min_chunk_size=100)
        if not chunks:
            chunks = [page_text.strip()]

        parent_id = f"parent_{company_name}_p{page_num}"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + page_text[:1000]
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
                "raw_data": {"paragraph": chunk, "page": page_num},
                # Parent-child fields
                "parent_id": parent_id,
                "parent_content": parent_content,
                "is_child": True,
            })
        return passages

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

        # ── Engine 1: PyMuPDF (fitz) ──────────────────────────────────
        try:
            import fitz
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            passages = []
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
                    passages.extend(self._make_passages(company_name, filename, page_num, best_text))

            doc.close()
            if passages:
                return _ret(passages)
        except Exception as e:
            print(f"[FinancialFileParser] fitz failed for '{filename}': {e}")

        # ── Engine 2: pdfplumber ──────────────────────────────────────
        try:
            import pdfplumber
            passages = []
            with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    best_text = ""

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
                        passages.extend(self._make_passages(company_name, filename, page_num, best_text))

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
                # ── Build parent: full table as a single string (≤1000 chars) ──
                full_table_lines = [" | ".join(headers)]
                for row in table_rows:
                    row_cells = [row[i] if i < len(row) else "" for i in range(len(headers))]
                    full_table_lines.append(" | ".join(row_cells))

                parent_id = f"parent_{company_name}_csv_table"
                raw_table_text = "\n".join(full_table_lines)
                parent_content = (
                    f"Company: {company_name} | Report: {filename} | Table: CSV Financial Table\n"
                    + raw_table_text[:1000]
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
