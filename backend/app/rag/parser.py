import os
import csv
import io
import json
import re
import html
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from app.rag.chunker import chunk_text
from app.tools.table_parser import linearize_financial_table, to_markdown_table, is_markdown_separator_row


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
    # Recognises year headers: formats like YYYY, Mon YYYY, Month DD YYYY, and similar
    # variants.
    _YEAR_HEADER_RE = re.compile(r'(?:FY\s*|fiscal\s+)?(20\d{2}|19\d{2})', re.IGNORECASE)
    #: Detects quarter-ordinal column headers like 'First Second Third Fourth' spread
    #: across lines.
    #: Allows for short gaps between ordinals to account for line-wrapped PDF text and
    #: avoid misattributing a nearby multi-year caption as the column year.
    _QUARTER_ORDINALS_RE = re.compile(
        r'\bfirst\b.{0,40}\bsecond\b.{0,40}\bthird\b.{0,40}\bfourth\b',
        re.IGNORECASE | re.DOTALL,
    )
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

    @staticmethod
    def _without_overlapping_spaces(page):
        """Return the pdfplumber page minus "space" characters that sit INSIDE
        another character's horizontal extent on the same line.

        Some generators (JnJ's earnings releases) emit right-aligned padding
        spaces whose boxes overlap the digits they pad, e.g. the text layer of
        "100.0" holds a space with x0 484.58 inside the "1" (484.22-489.37).
        pdfplumber then ends the word at that space and layout text reads
        "1 00.0"; the table builder turned that into two cells (`2022: 1 |
        Col3: 00.0`, and "67.0" into "6" + "7.0"), and the sandbox later
        picked "Sales to customers (2022): 1.0" as its answer. Real
        inter-word spaces sit BETWEEN glyphs and never overlap one, so they
        are kept; PyMuPDF already ignores these spaces.
        """
        try:
            from bisect import bisect_right
            rows: Dict[int, List[Any]] = {}
            for c in page.chars:
                if not c.get("text", " ").isspace():
                    rows.setdefault(round(c["top"]), []).append((c["x0"], c["x1"]))
            for lst in rows.values():
                lst.sort()
            xs_by_row = {k: [x0 for x0, _ in v] for k, v in rows.items()}

            def _stray(c) -> bool:
                if not c.get("text", "x").isspace():
                    return False
                key = round(c["top"])
                for k in (key - 1, key, key + 1):
                    lst = rows.get(k)
                    if not lst:
                        continue
                    i = bisect_right(xs_by_row[k], c["x0"] + 0.5)
                    # candidates: chars starting at or (within 0.5pt) after the
                    # space's own x0, whose box still extends past it
                    for j in range(max(0, i - 3), min(len(lst), i)):
                        x0, x1 = lst[j]
                        if x0 - 0.5 < c["x0"] and c["x0"] < x1 - 0.5:
                            return True
                return False

            if not any(_stray(c) for c in page.chars if c.get("text", "x").isspace()):
                return page
            return page.filter(
                lambda o: not (o.get("object_type") == "char" and _stray(o))
            )
        except Exception:
            return page

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

    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """
        Convert a pdfplumber extract_table()/extract_tables() result
        (row x col list of str/None) into a Markdown pipe table. First row
        is treated as the header row.

        This is a thin, pdfplumber-specific wrapper: raw pdfplumber tables
        sometimes include a spurious fully-blank leading row (e.g. from a
        table's border/padding), which would wrongly become the "header"
        if row 0 were used as-is — so blank rows are dropped from the WHOLE
        table first, and row 0 of what's left becomes the header. The
        actual Markdown formatting is delegated to the shared
        to_markdown_table() (app.tools.table_parser), which is also used
        by linearize_financial_table() for sample/CSV table data — one
        canonical formatter, not two divergent implementations.
        """
        if not table:
            return ""
        non_empty = [
            row for row in table
            if row is not None and any(c and str(c).strip() for c in row)
        ]
        if len(non_empty) < 2:
            return ""
        headers, rows = non_empty[0], non_empty[1:]
        return to_markdown_table(headers, rows)

    # A single table VALUE cell, as it appears in real 10-K statements:
    # - "1,503" / "(352)" / "-352" / "15%" — an ordinary number.
    # - "$ 1,503" / "$1,503" — many filings render the '$' as its own glyph,
    #   which pdfplumber then extracts as a separate text run with its own
    #   (sometimes wide) gap before the digits — the '\s{0,3}' absorbs that
    #   so "$  1,503" is captured as ONE value, not split into two.
    # - "—" / "–" / "-" alone — the standard placeholder for a zero/blank
    #   cell in a financial statement (e.g. "Non-cash operating lease cost
    #   64  —  —"). Without this alternative, ANY row containing a blank
    #   period fails to match at all, since the digit-based branch requires
    #   a literal digit.
    _VALUE_TOKEN = r'(?:\$\s{0,3})?[\(\-]?\d[\d,\.]*\)?%?|[—–-]'
    _LAYOUT_VALUE_RE = re.compile(_VALUE_TOKEN)

    # Digit-only value token matcher that excludes bare dash/placeholder tokens.
    # Used to identify real numeric column anchors without letting typographic dashes
    # act as anchors.
    _LAYOUT_DIGIT_VALUE_RE = re.compile(r'(?:\$\s{0,3})?[\(\-]?\d[\d,\.]*\)?%?')

    #: Words that mark a number as part of an inline narrative aside
    #: ("477 and 484", "years 1 to 3") rather than a real value-column
    #: entry — see _LAYOUT_DIGIT_VALUE_RE's docstring.
    _NUMERIC_CONNECTOR_WORDS = {"and", "to", "or", "through"}

    # Defines a 'layout row' as a label followed by 2+ whitespace-separated numeric-
    # looking tokens.
    # Requires at least one letter/CJK in the label and allows single-space gaps to
    # accommodate PDF text extraction quirks; only treats as table after 3+ consecutive
    # matches.
    _LAYOUT_ROW_RE = re.compile(
        r'^(?=.*[A-Za-z一-鿿])(?P<label>\S.{0,80}?)\s+'
        r'(?P<values>(?:' + _VALUE_TOKEN + r')(?:\s+(?:' + _VALUE_TOKEN + r'))+)\s*$'
    )

    @staticmethod
    def _normalize_layout_value(token: str) -> str:
        """Collapse the gap pdfplumber can leave between a standalone '$'
        glyph and its digits ("$  1,503" -> "$1,503")."""
        return re.sub(r'^\$\s+', '$', token.strip())

    @staticmethod
    def _split_text_header(line: str, n_cols: int) -> list:
        """Parse column names from a header line composed solely of words; each caption
        starts with a capital letter and a short leftmost row label is dropped. Accept
        only when exactly n_cols captions are found.
        """
        toks = (line or "").split()
        if not (3 <= n_cols <= 8) or len(toks) < n_cols or len(toks) > 24 or re.search(r"[0-9$%]", line or ""):
            return []
        caps = [i for i, t in enumerate(toks) if t[:1].isupper()]
        for k in range(len(caps)):
            starts = caps[k:]
            if len(starts) != n_cols or starts[0] > 5:
                continue
            names = [
                " ".join(toks[starts[j]: starts[j + 1] if j + 1 < n_cols else len(toks)])
                for j in range(n_cols)
            ]
            if all(len(nm.split()) <= 4 for nm in names):
                return names
        return []

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
        deferred_labels: List[str] = []
        prev_prose = [""]          # last prose line seen
        block_text_header = [""]   # the prose line directly above the current block

        def _flush():
            if deferred_labels:
                prose_lines.extend(deferred_labels)
                deferred_labels.clear()
            if len(current_rows) >= 3:
                n_cols = current_n_cols[0]
                years = self._extract_year_headers(header_candidate[0]) if header_candidate[0] else []
                if not years and header_candidate[0]:
                    # Normalises compact quarter column labels so the year is explicit,
                    # e.g. converts short forms like "1Q" with two-digit years into a
                    # clear "1Q YYYY" style.
                    _qs = re.findall(r"(?<![A-Za-z0-9])([1-4])Q(\d{2})(?![A-Za-z0-9])", header_candidate[0])
                    if len(_qs) == n_cols:
                        years = [f"{q}Q 20{yy}" for q, yy in _qs]
                if len(years) < n_cols:
                    years = self._split_text_header(block_text_header[0], n_cols) or years
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
            # "(Note 11)" style footnote references would otherwise be read as a
            # value column ("... charges (Note 11) (769) (817) (813)" = 4 values)
            _masked = re.sub(r"\((Notes?)\s+(\d+[A-Za-z]?)\)", r"(\1_\2)", stripped)
            # "1.0 %   — %   (4.2) %" : a percent sign printed as its own token
            _masked = re.sub(r"([0-9)])\s+%", r"\1%", _masked)
            _masked = re.sub(r"([—–])\s*%", r"\1", _masked)
            m = self._LAYOUT_ROW_RE.match(_masked)
            if m:
                label = m.group("label").strip().replace("(Note_", "(Note ").replace("(Notes_", "(Notes ")
                _vals_probe = self._LAYOUT_VALUE_RE.findall(m.group("values"))
                _yr_tokens = [v for v in _vals_probe if re.fullmatch(r"(?:19|20)\d{2}", v.strip())]
                if len(_yr_tokens) >= 2 and all(
                    re.fullmatch(r"(?:19|20)\d{2}|[0-3]?\d,?", v.strip()) for v in _vals_probe
                ):
                    # "Years Ended December 31,   2021  2020  2019" is the column
                    # header, not a data row of years
                    _flush()
                    header_candidate[0] = stripped
                    continue
                # findall (not a whitespace split) because a '$' and its
                # digits can legitimately be separated by the SAME width of
                # space as separates two different columns — the token
                # pattern itself is what decides where one value ends.
                values = [
                    self._normalize_layout_value(v)
                    for v in self._LAYOUT_VALUE_RE.findall(m.group("values"))
                ]
                if current_n_cols[0] is not None and len(values) != current_n_cols[0]:
                    _flush()
                if deferred_labels:
                    # sub-headings survive as prose; the table continues past them
                    prose_lines.extend(deferred_labels)
                    deferred_labels.clear()
                if not current_rows:
                    block_text_header[0] = prev_prose[0]
                current_rows.append((label, values))
                current_n_cols[0] = len(values)
            else:
                # Treats short label-only sub-headings printed inside a table as part of
                # the same table rather than as block terminators.
                # Prevents splitting a single table into separate blocks when an
                # internal row is a label.
                if (
                    current_rows
                    and len(stripped) <= 60
                    and not re.search(r"\d", stripped)
                    and not stripped.endswith(".")
                    and len(stripped.split()) <= 6
                ):
                    deferred_labels.append(stripped)
                    continue
                _flush()
                prose_lines.append(stripped)
                prev_prose[0] = stripped
                # Only overwrite the header candidate when the new line
                # itself looks like a year header, or none has been found
                # yet — a section subheading with no years (e.g. "Revenue:",
                # sitting between the year row and the first data row, which
                # is how almost every real income statement is laid out)
                # must not clobber a good candidate found earlier on the
                # same page.
                if self._extract_year_headers(stripped) or not header_candidate[0]:
                    header_candidate[0] = stripped
        _flush()

        return tables, "\n".join(prose_lines)

    # ──────────────────────────────────────────────────────────────────────
    # Word-coordinate table reconstruction (Tier 2)
    # ──────────────────────────────────────────────────────────────────────
    # Some real 10-K PDFs defeat BOTH ruled-line detection (no vector lines)
    # AND the layout=True text-flow regex above (each table cell — label,
    # each year's value, even a standalone '$' glyph — ends up on its OWN
    # line when serialised to plain text, so there is no shared "line" left
    # for a same-line regex to match, even with layout preserved). Working
    # directly from each word's (x0, top) bounding-box position sidesteps
    # the engine's own line-grouping heuristic entirely.

    #: Row-clustering tolerance: words whose vertical position is within
    #: this many points of the current row's running-average top are
    #: considered part of the same row. Compared against the row's average
    #: (not just the previous word) so the tolerance can't let a row's
    #: effective y-position drift across many words.
    _WORD_ROW_Y_TOLERANCE = 2.0

    #: '$' + immediately-following numeric token merge gap — measured as
    #: the actual whitespace between them (number's x0 minus '$'s x1), not
    #: x0-to-x0 (which would bake the '$' glyph's own width into the
    #: gap and unfairly penalize wider numbers, e.g. 5-digit vs 3-digit,
    #: even though the visual spacing is identical). A standalone '$'
    #: glyph must not survive into column clustering as its own word —
    #: its x1 sits well to the left of the actual value's right edge, and
    #: would either pull a value-column anchor off target or form a
    #: spurious anchor of its own.
    _WORD_DOLLAR_MERGE_GAP = 20.0

    #: Floor for the adaptive value-column tolerance (see
    #: _adaptive_x1_gap_threshold()), and the tolerance used as a fallback
    #: when a page doesn't have enough numeric tokens to find a clear
    #: same-column-vs-different-column split.
    _WORD_COL_MIN_TOLERANCE = 3.0

    #: Minimum consecutive data rows for a run to count as a real table.
    _WORD_TABLE_MIN_ROWS = 2

    #: Maximum numeric value columns a Tier 2-recovered row may have and
    #: still be labeled as a simple year/period comparison — see the
    #: real CVS Health segment-breakdown case in _flush()'s docstring
    #: comment for why a wider row must be left as prose instead.
    _WORD_TABLE_MAX_VALUE_COLS = 4

    #: Minimum fraction of numeric-token-bearing rows that must share the
    #: SAME anchor "shape" (which value-column anchors their numbers
    #: landed on) for the page's anchors to be trusted at all. Real
    #: right-aligned columns keep the same x1 for the whole table, so
    #: almost every data row shares one dominant shape; a page with no
    #: genuine column alignment (e.g. hand space-padded proportional-font
    #: text) scatters rows across many different shapes instead — below
    #: this fraction, Tier 2 abstains (returns no tables) so Tier 3's
    #: layout-text regex gets a chance instead of a fabricated table
    #: stitched from unrelated columns.
    _WORD_COL_MIN_SHAPE_CONFIDENCE = 0.5

    @staticmethod
    def _fitz_words_to_common(fitz_words) -> List[Dict[str, Any]]:
        """Convert PyMuPDF page.get_text("words") output — tuples of
        (x0, y0, x1, y1, text, block_no, line_no, word_no) — into the
        unified word format {"x0","top","x1","bottom","text"} shared with
        _pdfplumber_words_to_common(). PyMuPDF's coordinate origin is
        top-left with y increasing downward — the same convention as
        pdfplumber's top/bottom — so no axis flip is needed."""
        return [
            {"x0": w[0], "top": w[1], "x1": w[2], "bottom": w[3], "text": w[4]}
            for w in fitz_words if w[4].strip()
        ]

    @staticmethod
    def _pdfplumber_words_to_common(pdfplumber_words) -> List[Dict[str, Any]]:
        """Convert pdfplumber page.extract_words() output into the unified
        word format shared with _fitz_words_to_common()."""
        return [
            {"x0": w["x0"], "top": w["top"], "x1": w["x1"], "bottom": w["bottom"], "text": w["text"]}
            for w in pdfplumber_words if w.get("text", "").strip()
        ]

    def _cluster_words_into_rows(self, words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group words into rows by vertical position (see
        _WORD_ROW_Y_TOLERANCE), then sort each row left-to-right by x0."""
        if not words:
            return []
        words_sorted = sorted(words, key=lambda w: w["top"])
        rows: List[List[Dict[str, Any]]] = []
        current_row: List[Dict[str, Any]] = []
        current_top_sum = 0.0
        for w in words_sorted:
            if current_row:
                row_mean_top = current_top_sum / len(current_row)
                if abs(w["top"] - row_mean_top) > self._WORD_ROW_Y_TOLERANCE:
                    rows.append(current_row)
                    current_row = []
                    current_top_sum = 0.0
            current_row.append(w)
            current_top_sum += w["top"]
        if current_row:
            rows.append(current_row)
        for row in rows:
            row.sort(key=lambda w: w["x0"])
        return rows

    #: A candidate value-column anchor must be "voted for" by numeric
    #: tokens from at least this many DIFFERENT rows to count as a real
    #: column — filters out a one-off numeric-looking token (e.g. a page
    #: number, or a lone year in a caption) that would otherwise form its
    #: own phantom single-row column.
    _WORD_COL_MIN_ROW_SUPPORT = 2

    def _merge_dollar_sign_tokens(self, row_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge a standalone '$' word with the numeric word immediately
        to its right into one value token, when they're close enough
        (whitespace gap < _WORD_DOLLAR_MERGE_GAP) to clearly be the same
        currency value split into two words by the PDF's own text runs.
        Must run before column clustering — see _WORD_DOLLAR_MERGE_GAP."""
        merged: List[Dict[str, Any]] = []
        i = 0
        n = len(row_words)
        while i < n:
            w = row_words[i]
            if w["text"] == "$" and i + 1 < n:
                nxt = row_words[i + 1]
                if (
                    nxt["x0"] - w["x1"] < self._WORD_DOLLAR_MERGE_GAP
                    and self._LAYOUT_VALUE_RE.fullmatch(nxt["text"].strip())
                ):
                    merged.append({
                        "x0": w["x0"],
                        "top": min(w["top"], nxt["top"]),
                        "x1": nxt["x1"],
                        "bottom": max(w["bottom"], nxt["bottom"]),
                        "text": "$" + nxt["text"],
                    })
                    i += 2
                    continue
            merged.append(w)
            i += 1
        return merged

    def _adaptive_x1_gap_threshold(self, x1_values: List[float]) -> float:
        """
        Derive a column-separation tolerance directly from this page's own
        numeric-token right-edge (x1) positions, instead of a hardcoded
        gap in points. Right-aligned numbers in the SAME column line up
        almost exactly at their right edge across different rows
        (sub-point variance in practice, since x0 drifts with digit count
        but x1 doesn't); numbers in DIFFERENT columns are much farther
        apart — the real column-to-column spacing, which varies a lot by
        company/font/layout and can be smaller than any fixed hardcoded
        threshold (e.g. 26.4pt on a real 3M filing, which a fixed 30pt gap
        misses entirely).

        So the gaps between sorted x1 values (kept WITH duplicates — two
        tokens sharing (near-)identical x1 contribute a genuine ~0 gap,
        which is itself evidence of how tight "same column" alignment
        really is on this page) are bimodal: many tiny "same-column"
        gaps, a few large "different-column" gaps. This finds the
        boundary between the two groups by sorting the POSITIVE gap
        sizes and locating the pair of consecutive gap sizes with the
        largest RATIO jump between them — a simple, robust
        one-dimensional two-cluster split that works even with very few
        samples. The threshold is set halfway between that pair.

        When no clear ratio jump exists among the positive gaps (too few
        samples, or every positive gap is roughly the same size), the
        exact/near-zero gaps we already observed settle it: if any
        exist, they're direct proof within-column jitter is ~0 on this
        page, so every positive gap must be a real between-column gap —
        stay at the floor tolerance rather than inflating past it (which
        would wrongly merge those columns together). Only when there are
        NO zero gaps at all (e.g. exactly two numeric tokens on the
        whole page) do we fall back to half the smallest positive gap.
        """
        xs = sorted(x1_values)
        if len(xs) < 2:
            return self._WORD_COL_MIN_TOLERANCE
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        positive_gaps = sorted(g for g in gaps if g > 0.01)
        has_zero_gaps = len(positive_gaps) < len(gaps)

        if not positive_gaps:
            return self._WORD_COL_MIN_TOLERANCE
        if len(positive_gaps) == 1:
            if has_zero_gaps:
                return self._WORD_COL_MIN_TOLERANCE
            return max(self._WORD_COL_MIN_TOLERANCE, positive_gaps[0] * 0.5)

        best_ratio = 1.0
        best_idx = None
        for i in range(len(positive_gaps) - 1):
            ratio = positive_gaps[i + 1] / positive_gaps[i]
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_idx is not None and best_ratio >= 2.0:
            return max(
                self._WORD_COL_MIN_TOLERANCE,
                (positive_gaps[best_idx] + positive_gaps[best_idx + 1]) / 2,
            )
        if has_zero_gaps:
            return self._WORD_COL_MIN_TOLERANCE
        return max(self._WORD_COL_MIN_TOLERANCE, positive_gaps[0] * 0.5)

    def _cluster_value_column_anchors(
        self, rows: List[List[Dict[str, Any]]]
    ) -> Tuple[List[float], float]:
        """
        Find the page's value-column anchors: cluster the right edges
        (x1) of every numeric-looking token across ALL rows, using the
        adaptive tolerance from _adaptive_x1_gap_threshold(). Anchors
        supported by tokens from fewer than _WORD_COL_MIN_ROW_SUPPORT
        distinct rows are dropped as noise. Returns (sorted anchor x1
        positions, tolerance used) — the caller reuses the tolerance to
        assign individual words to their nearest anchor.

        Only rows with 2+ numeric-looking tokens contribute candidates —
        a row with exactly ONE numeric-looking token is far more likely a
        footnote reference marker ("(1)", "(2)") sitting near the left
        margin next to explanatory prose than a real table value; a
        genuine data row always carries 2+ values across year/period
        columns. Without this, two such markers at the same x1 (a real
        example: two footnote definitions "(1) Excludes ..." / "(2)
        Includes ..." stacked at a page's bottom) form their own
        spurious low-x1 "value column" anchor, close enough to a short
        label word's own x1 (e.g. "Net" in "Net income") to misclassify
        that label word as a value instead — scrambling label word order
        in the final row.
        """
        numeric_entries = []
        for row_idx, row in enumerate(rows):
            row_numeric = [
                w for i, w in enumerate(row)
                if self._LAYOUT_DIGIT_VALUE_RE.fullmatch(w["text"].strip())
                # Numeric tokens embedded in running prose with connector words (e.g.,
                # "... shares — 477 and 484", "years 1 to 3") represent asides or
                # ranges, not standalone column values.
                # Treating them as column anchors can misalign detected columns; exclude
                # such narrative-adjacent numbers from anchor voting.
                and not (i > 0 and row[i - 1]["text"].strip().lower() in self._NUMERIC_CONNECTOR_WORDS)
                and not (i + 1 < len(row) and row[i + 1]["text"].strip().lower() in self._NUMERIC_CONNECTOR_WORDS)
            ]
            if len(row_numeric) < 2:
                continue
            numeric_entries.extend((row_idx, w["x1"]) for w in row_numeric)
        if not numeric_entries:
            return [], self._WORD_COL_MIN_TOLERANCE

        tolerance = self._adaptive_x1_gap_threshold([x1 for _, x1 in numeric_entries])
        numeric_entries.sort(key=lambda e: e[1])

        # Each cluster's total span is capped at `tolerance`, measured
        # from its leftmost (first-added) member — NOT from the previous
        # member. A step-to-step-only check lets a "staircase" of
        # unrelated x1 values (each just barely within `tolerance` of the
        # last) chain into one cluster spanning several multiples of the
        # tolerance, which is exactly what happens on a page with no real
        # column alignment (e.g. hand space-padded proportional-font
        # text): individually-plausible small gaps accumulate into a
        # single bogus "column" stitched together from unrelated values.
        clusters: List[List[Tuple[int, float]]] = [[numeric_entries[0]]]
        for entry in numeric_entries[1:]:
            if entry[1] - clusters[-1][0][1] <= tolerance:
                clusters[-1].append(entry)
            else:
                clusters.append([entry])

        anchors = [
            sum(x1 for _, x1 in c) / len(c)
            for c in clusters
            if len({row_idx for row_idx, _ in c}) >= self._WORD_COL_MIN_ROW_SUPPORT
        ]
        return anchors, tolerance

    def _dominant_anchor_shape_fraction(
        self, rows: List[List[Dict[str, Any]]], anchors: List[float], tolerance: float
    ) -> float:
        """Compute, over rows with 2+ numeric tokens and at least one hitting an anchor,
        the fraction that share the single most common numeric "shape" (the sorted set
        of anchor indices). Use numeric tokens only; label words are excluded as noise.
        Exclude rows whose numbers match no anchors from both numerator and denominator
        so empty matches do not dominate the vote; this prevents unrelated numeric
        mentions from outvoting real table row shapes.
        """
        shapes = []
        for row in rows:
            numeric_words = [w for w in row if self._LAYOUT_VALUE_RE.fullmatch(w["text"].strip())]
            if len(numeric_words) < 2:
                continue
            # Repeated multi-category header rows whose numeric tokens are narrower than
            # the data below can produce scattered x1 anchors that do not reflect the
            # real data rows' shape.
            # Avoid letting such header-token clusters dominate anchor voting; they
            # often indicate header repetition rather than true column alignment.
            if self._is_bare_year_row(row):
                continue
            matches = [self._assign_to_value_anchor(w["x1"], anchors, tolerance) for w in numeric_words]
            shape = tuple(sorted(i for i in matches if i is not None))
            if shape:
                shapes.append(shape)
        if not shapes:
            return 0.0
        most_common_count = Counter(shapes).most_common(1)[0][1]
        return most_common_count / len(shapes)

    def _is_bare_year_row(self, row: List[Dict[str, Any]]) -> bool:
        """Detect rows whose value tokens are all bare 4-digit years (no currency, commas,
        decimals, or parens) and treat them as year-header lines rather than data. This
        prevents repeated-year header rows from diluting column-shape votes and ensures
        they're classified as headers/text, not data.
        """
        value_like_tokens = [
            w["text"].strip() for w in row
            if self._LAYOUT_VALUE_RE.fullmatch(w["text"].strip())
        ]
        return len(value_like_tokens) >= 2 and all(
            re.fullmatch(r'(?:19|20)\d{2}', t) for t in value_like_tokens
        )

    @staticmethod
    def _assign_to_value_anchor(x1: float, anchors: List[float], tolerance: float) -> Optional[int]:
        """Index of the value-column anchor closest to this word's x1, or
        None if it's farther than `tolerance` from every anchor — such a
        word belongs to the label column, not a value column."""
        best_idx = None
        best_dist = None
        for i, a in enumerate(anchors):
            dist = abs(x1 - a)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_idx = i
                best_dist = dist
        return best_idx

    def _cell_looks_numeric(self, cell_text: str) -> bool:
        """A cell counts as numeric if it fullmatches one value token
        outright, or — when column clustering wasn't precise enough and
        multiple numbers ended up glued into the same cell — if splitting
        on whitespace shows more than half of the individual tokens
        (ignoring bare '$' signs) look numeric. Defends is_data_row
        classification against imperfect clustering instead of silently
        discarding the whole row as prose."""
        cell_text = cell_text.strip()
        if not cell_text:
            return False
        if self._LAYOUT_VALUE_RE.fullmatch(cell_text):
            return True
        tokens = [t for t in cell_text.split() if t != "$"]
        if len(tokens) < 2:
            return False
        numeric_tokens = [t for t in tokens if self._LAYOUT_VALUE_RE.fullmatch(t)]
        return len(numeric_tokens) > len(tokens) / 2

    def _reconstruct_table_from_word_positions(
        self, words: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Reconstruct table rows purely from word bounding-box coordinates,
        ignoring however the source engine grouped words into "lines" in
        its own text stream (see the Tier 2 module comment above).

        Row clustering as described by _cluster_words_into_rows(); each
        row's '$' + number pairs are merged (_merge_dollar_sign_tokens()),
        value columns are found via _cluster_value_column_anchors() (x1 of
        numeric tokens only, adaptive tolerance), and every word is then
        assigned to its (row, column) cell — a value column if its x1 is
        close to an anchor, otherwise the label column — same-cell words
        are joined with a space, and each reconstructed row is classified
        as either:
          - a DATA row: first column is non-numeric text AND at least half
            of the remaining non-empty columns look numeric (reusing
            _LAYOUT_VALUE_RE's number/'—'-placeholder pattern), or
          - a TEXT-ONLY row: kept as prose (a section-header label like
            "Operating expenses", a year-header candidate, or genuine
            narrative text) rather than discarded — mirrors
            _layout_text_to_markdown_and_prose()'s header_candidate
            persistence (a section subheading between the year row and the
            first data row must not clobber a good year-header candidate).

        Returns (tables, prose) where prose is every text-only row's
        reconstructed text, in original top-to-bottom order, and `tables`
        is a list of (markdown_table, top_y) pairs — top_y is the page
        y-position of that specific table's own first row, for a caller
        that needs to search nearby page text for a header FOR THAT
        TABLE alone rather than reusing one shared reference point across
        every table this call recovers (see current_first_row_top's
        docstring for why that distinction matters).
        """
        rows = self._cluster_words_into_rows(words)
        if not rows:
            return [], ""
        rows = [self._merge_dollar_sign_tokens(row) for row in rows]

        def _rows_as_prose() -> str:
            return "\n".join(
                " ".join(w["text"] for w in row).strip() for row in rows
            ).strip()

        # If no strong value-column structure exists on the page, return every row as
        # prose rather than discarding content.
        # This ensures prose-only pages are preserved even when anchor candidates are
        # correctly excluded and no table anchors remain.
        anchors, tolerance = self._cluster_value_column_anchors(rows)
        if not anchors:
            return [], _rows_as_prose()
        if self._dominant_anchor_shape_fraction(rows, anchors, tolerance) < self._WORD_COL_MIN_SHAPE_CONFIDENCE:
            return [], _rows_as_prose()

        tables: List[str] = []
        # Each recovered table entry pairs with the same index in tables: the y-position
        # of that table's first accumulated row.
        # This lets callers resolve nearby headers using the specific table's actual
        # position instead of a shared reference.
        table_positions: List[float] = []
        prose_lines: List[str] = []
        current_rows: List[Tuple[str, List[str]]] = []
        current_n_cols = [None]
        # Track the y-position of the first row in the current accumulating run; capture
        # it once per run and reset after flush.
        # Without this, multiple tables recovered on one page could incorrectly share
        # the same header search origin and inherit another table's context.
        current_first_row_top = [None]
        # Which anchor INDICES were actually populated for the row run
        # currently being accumulated — not just how many. Two rows can
        # coincidentally produce the same value COUNT while their values
        # landed on entirely different anchors (e.g. on a page with no
        # real column alignment, where stray values drift onto whichever
        # anchor happens to be nearby); grouping them into one table would
        # silently splice unrelated columns together. Requiring the same
        # anchor-index set to continue a run catches this even when the
        # count alone wouldn't.
        current_col_shape = [None]
        header_candidate = [""]
        # The immediately-preceding text-only row's own line text — used
        # only to detect a standalone "After"/"Thereafter" row sitting
        # right above a header_candidate line, so the two physical rows
        # of a page-wrapped "After\n2023" column header can be stitched
        # back into one label. See the stitch site below.
        prev_line_text = [""]

        def _flush():
            n_cols = current_n_cols[0]
            # Rows with an unusually large number of numeric columns usually indicate a
            # single-period segment/category breakdown, not a multi-period comparison;
            # labeling them as years risks misinterpreting segment slices as whole-
            # period totals.
            # Treat implausibly-wide numeric rows as prose unless a nearby header shows
            # a clean repeating year pattern that matches the row width exactly.
            repeating_years = (
                self._extract_repeating_year_headers(header_candidate[0], n_cols)
                if header_candidate[0] and n_cols is not None else []
            )
            # A "Total | 2019 | ... | After 2023"-style header (see
            # _extract_period_headers) is the OTHER legitimate reason a
            # row can carry more value columns than _WORD_TABLE_MAX_VALUE_COLS
            # expects -- a standard 10-K "Contractual Obligations" table,
            # not a single-period segment breakdown like CVS's. Same exact-
            # count-match safety property as repeating_years.
            period_headers = (
                self._extract_period_headers(header_candidate[0], n_cols)
                if header_candidate[0] and n_cols is not None else []
            )
            if (
                n_cols is not None and n_cols > self._WORD_TABLE_MAX_VALUE_COLS
                and not repeating_years and not period_headers
            ):
                for label, values in current_rows:
                    prose_lines.append((label + "  " + "  ".join(values)).strip())
            elif len(current_rows) >= self._WORD_TABLE_MIN_ROWS:
                years = repeating_years or period_headers or (
                    self._extract_year_headers(header_candidate[0]) if header_candidate[0] else []
                )
                headers = ["Line Item"] + [
                    years[i] if i < len(years) else f"Col{i + 1}"
                    for i in range(n_cols)
                ]
                table = [headers] + [[label] + values for label, values in current_rows]
                md = self._table_to_markdown(table)
                if md:
                    tables.append(md)
                    table_positions.append(
                        current_first_row_top[0] if current_first_row_top[0] is not None else 0.0
                    )
            else:
                for label, values in current_rows:
                    prose_lines.append((label + "  " + "  ".join(values)).strip())
            current_rows.clear()
            current_n_cols[0] = None
            current_col_shape[0] = None
            current_first_row_top[0] = None

        for row_words in rows:
            cells: List[List[str]] = [[] for _ in range(len(anchors) + 1)]
            for w in row_words:
                w_text = w["text"].strip()
                # Only tokens that clearly resemble numeric values (or a bare '$',
                # handled specially) are eligible for value-anchor assignment.
                # Ordinary label words must not be moved into a value column solely
                # because their x1 happens to fall within tolerance of an unrelated
                # numeric anchor elsewhere on the page.
                # This prevents label words from being misassigned as numeric cells.
                is_value_like = bool(self._LAYOUT_VALUE_RE.fullmatch(w_text)) or w_text == "$"
                col = self._assign_to_value_anchor(w["x1"], anchors, tolerance) if is_value_like else None
                # A standalone '$' that missed matching its own number can still lie
                # within tolerance of some value anchor and attach to that column,
                # producing non-numeric cell text and breaking numeric parsing.
                # Since a bare '$' conveys no numeric magnitude once a numeric column
                # exists, drop such standalone dollar glyphs rather than carrying them
                # into cells.
                # Also drop any lone '$' that would otherwise end up in the label to
                # avoid littering labels with stray symbols.
                if w_text == "$":
                    continue
                cells[0 if col is None else col + 1].append(w["text"])
            cell_texts = [" ".join(parts).strip() for parts in cells]

            first_col = cell_texts[0]
            rest_cols = [c for c in cell_texts[1:] if c]
            has_label = bool(first_col) and not self._LAYOUT_VALUE_RE.fullmatch(first_col)
            numeric_rest = [c for c in rest_cols if self._cell_looks_numeric(c)]

            is_data_row = (
                has_label
                and len(rest_cols) >= 2
                and len(numeric_rest) >= max(1, len(rest_cols) // 2)
            )

            # If all value-like tokens in a row are bare 4-digit years (no currency
            # symbols, commas, decimals, or parens), treat the row as a year-header
            # line, not data.
            # This captures repeated multi-category year header rows that might
            # otherwise be mis-parsed as short data rows and discarded, preserving
            # header text for downstream column assignment.
            if is_data_row and self._is_bare_year_row(row_words):
                is_data_row = False

            if is_data_row:
                # Drop empty cells (matching rest_cols above) — a column
                # bucket that's genuinely unused by every data row (e.g.
                # one created by a single stray word from the page title,
                # surviving the row-support filter at exactly the minimum
                # vote count) must not show up as a spurious blank leading
                # value in every row.
                values = [
                    self._normalize_layout_value(c) if self._LAYOUT_VALUE_RE.fullmatch(c) else c
                    for c in cell_texts[1:] if c
                ]
                col_shape = tuple(i for i, c in enumerate(cell_texts[1:]) if c)
                if current_col_shape[0] is not None and col_shape != current_col_shape[0]:
                    _flush()
                if current_first_row_top[0] is None:
                    current_first_row_top[0] = row_words[0]["top"] if row_words else 0.0
                current_rows.append((first_col, values))
                current_n_cols[0] = len(values)
                current_col_shape[0] = col_shape
            else:
                _flush()
                # Reconstruct rows from row_words' original left-to-right ordering
                # (sorted by x0), not from anchor-bucketed cell_texts.
                # Anchor bucketing adapts to numeric spacing and can reorder
                # header/prose words relative to data; preserving original word order
                # avoids scrambling header sequences like year lists.
                line_text = " ".join(w["text"] for w in row_words).strip()
                if line_text:
                    prose_lines.append(line_text)
                    if self._extract_year_headers(line_text) or not header_candidate[0]:
                        # Handle cases where a table's last column header is split
                        # across two physical rows (e.g., a leading word on the prior
                        # line over a trailing year).
                        # If a standalone trailing label word sits immediately above a
                        # year-like cell, stitch that word onto the year so the logical
                        # column header is unified.
                        if prev_line_text[0].strip().lower() == "after":
                            m = re.search(r'(?:20|19)\d{2}$', line_text)
                            if m:
                                line_text = line_text[:m.start()] + f"After {m.group(0)}"
                        header_candidate[0] = line_text
                    prev_line_text[0] = line_text
        _flush()

        return list(zip(tables, table_positions)), "\n".join(prose_lines)

    @staticmethod
    def _compact_row_cells(row: list) -> list:
        """Collapse a raw per-row cell grid into [label, value1, value2, ...] by dropping
        blank and lone-symbol cells (e.g., '$') while keeping meaningful placeholders
        (e.g., em dash). Also rejoin split tokens like a trailing ')' or '%' back onto
        their digits so single logical values occupy one cell. This stabilizes column
        indices across rows so later index-based header matching remains correct.
        """
        if not row:
            return row
        label = row[0]
        rest = [c for c in row[1:] if c and c.strip() not in ("", "$")]
        _NUMERIC_NO_PCT_RE = re.compile(r'^\(?-?\$?\s*\d[\d,]*\.?\d*\)?$')
        merged: list = []
        i = 0
        while i < len(rest):
            cell = rest[i].strip()
            if (
                i + 1 < len(rest)
                and re.match(r'^\(\s*[\d,.]+$', cell)
                and rest[i + 1].strip() in (')', '%)')
            ):
                merged.append(cell + rest[i + 1].strip())
                i += 2
            elif (
                i + 1 < len(rest)
                and _NUMERIC_NO_PCT_RE.match(cell)
                and rest[i + 1].strip() == '%'
            ):
                merged.append(cell + '%')
                i += 2
            else:
                merged.append(rest[i])
                i += 1

        # If a label wraps across two PDF lines and a fragment from the prior line was
        # pushed into the first value column as a non-numeric cell, reclaim at most one
        # such leading non-numeric cell and prepend it back to the label.
        # Limit to a single reclaim to avoid destructively popping entire multi-column
        # header rows that legitimately contain non-numeric period strings.
        _numeric_cell_re = re.compile(r'^\(?-?\$?\s*\d[\d,]*\.?\d*\)?%?$')
        if (
            merged
            and merged[0].strip() not in ("—", "-", "–")
            and not _numeric_cell_re.match(merged[0].strip())
        ):
            label = f"{merged.pop(0).strip()} {label}".strip()

        return [label] + merged

    def _inject_missing_year_header(self, rows: list, page, table_top: float) -> list:
        """When the top detected table row lacks a year, search the page text above the
        table bbox for the nearest date line and synthesize a header from it, keeping
        the detected top row as a data row. Also handle headers split across detected
        rows by checking the next rows for a year-containing row and consuming it as the
        header if found. Expects rows compacted so value-column counts align.
        """
        if not rows:
            return rows
        header_text = " ".join(c or "" for c in rows[0])
        if self._extract_year_headers(header_text):
            # Same override as _reinject_year_header_if_missing's own
            # early-exit — an existing year-looking header is normally
            # trusted, but not when it's actually a Tier-2-reconstructed
            # table's own header_candidate guess (an unrelated multi-year
            # sentence scanned in passing) being routed back through here
            # via _reinject_year_header_if_missing, with a quarter-
            # ordinal header genuinely sitting closer to this table.
            try:
                above_probe = (
                    page.within_bbox((0, 0, page.width, max(0, table_top)), relative=False)
                    .extract_text() or ""
                )
            except Exception:
                above_probe = ""
            if not self._QUARTER_ORDINALS_RE.search(above_probe):
                return rows  # already has a real year header — nothing to fix

        # Comparative multi-year tables almost always show at least two year columns.
        # Treat a single isolated year above a table as likely prose, not a column
        # header.
        n_value_cols = max((len(r) - 1 for r in rows[1:]), default=0)
        min_years_needed = min(2, n_value_cols) if n_value_cols else 2

        try:
            above = page.within_bbox((0, 0, page.width, max(0, table_top)), relative=False)
            above_text = above.extract_text() or ""
        except Exception:
            above_text = ""

        # Detect quarter-ordinal headers (single-year quarterly breakdowns) before
        # multi-year checks.
        # Use the closest match above the table, not the first regex hit in the page
        # span.
        quarter_matches = list(self._QUARTER_ORDINALS_RE.finditer(above_text))
        quarter_match = quarter_matches[-1] if quarter_matches else None
        if quarter_match:
            year_matches = list(self._YEAR_HEADER_RE.finditer(above_text[:quarter_match.start()]))
            if year_matches:
                yr = year_matches[-1].group(1)
                n_value_cols_q = max((len(r) - 1 for r in rows[1:]), default=4)
                n_value_cols_q = max(n_value_cols_q, 4)
                quarter_labels = ["Q1", "Q2", "Q3", "Q4"]
                synthesized_q = ["Line Item"] + [
                    f"{quarter_labels[i]} {yr}" if i < 4 else f"Col{i + 1}"
                    for i in range(n_value_cols_q)
                ]
                return [synthesized_q, rows[0]] + rows[1:]

        years: list = []
        above_lines = above_text.split("\n")
        for idx in range(len(above_lines) - 1, -1, -1):
            line = above_lines[idx]
            # A repeating year sequence (e.g. "2018 2017 2016 2018 2017
            # 2016" for a table with two metric categories side by side,
            # each spanning the same 3 years) exactly fills every value
            # column when it matches — preferred over the plain deduped
            # extraction below, which would collapse it to 3 years and
            # leave the extra columns as generic Col4/Col5/Col6 fallbacks.
            # See _extract_repeating_year_headers's docstring.
            repeating = (
                self._extract_repeating_year_headers(line, n_value_cols)
                if n_value_cols else []
            )
            if repeating:
                years = repeating
                break
            # A "Total | 2019 | ... | After 2023"-style header (see
            # _extract_period_headers) also exactly fills every value
            # column — tried before the plain year-dedup fallback below,
            # which would otherwise drop "Total"/"After 2023" entirely and
            # shift every year one column to the left.
            period_headers = (
                self._extract_period_headers(line, n_value_cols)
                if n_value_cols else []
            )
            if period_headers:
                # Handle headers that wrap across physical lines where the last header's
                # top line is separate.
                # Avoid treating the wrapped bottom line alone as a distinct year label.
                if idx > 0 and period_headers[-1].isdigit():
                    prev_line = above_lines[idx - 1].strip()
                    if prev_line.lower() == "after":
                        period_headers[-1] = f"After {period_headers[-1]}"
                years = period_headers
                break
            candidate = self._extract_year_headers(line)
            if len(candidate) >= min_years_needed:
                years = candidate
                break

        if years:
            n_value_cols = max((len(r) - 1 for r in rows[1:]), default=len(years))
            n_value_cols = max(n_value_cols, len(years))
            # Synthesized header first cell must be the generic Line Item label, not the
            # first data row's value.
            # Keep rows[0] as a data row to avoid duplicating its text into the header.
            synthesized = ["Line Item"] + [
                years[i] if i < len(years) else f"Col{i + 1}" for i in range(n_value_cols)
            ]
            return [synthesized, rows[0]] + rows[1:]

        # Nothing above the table either — check whether one of the next
        # few rows is itself the real (split-out) header.
        for i in range(1, min(4, len(rows))):
            row_i = rows[i]
            candidate_years = self._extract_year_headers(" ".join(c or "" for c in row_i))
            if not candidate_years:
                continue
            # A row with a single cell is more likely a section/divider label than a
            # multi-column header.
            # Require either multiple header cells or multiple years in that single cell
            # before accepting it.
            if len(row_i) <= 1 and len(candidate_years) < 2:
                continue
            n_value_cols = max((len(r) - 1 for r in rows[i + 1:]), default=len(candidate_years))
            n_value_cols = max(n_value_cols, len(candidate_years))
            synthesized = ["Line Item"] + [
                candidate_years[j] if j < len(candidate_years) else f"Col{j + 1}"
                for j in range(n_value_cols)
            ]
            return [synthesized] + rows[:i] + rows[i + 1:]

        # If no header is found above a table fragment, try a small search window
        # starting at the table top.
        # This captures page-top tables whose header text sits slightly outside the
        # initial 'above' span.
        try:
            nearby = page.within_bbox(
                (0, max(0, table_top), page.width, table_top + 60), relative=False
            )
            nearby_text = nearby.extract_text() or ""
        except Exception:
            nearby_text = ""
        for line in nearby_text.split("\n"):
            years = self._extract_year_headers(line)
            # Enforce the same multi-year header guard used earlier: require a genuine
            # multi-row year header, not a single in-table year divider. Without this, a
            # lone year cell inside the table can be mistaken for the actual header;
            # keep the minimum-years check to avoid that mislabeling.
            if len(years) < 2:
                continue
            n_value_cols = max((len(r) - 1 for r in rows[1:]), default=len(years))
            n_value_cols = max(n_value_cols, len(years))
            synthesized = ["Line Item"] + [
                years[j] if j < len(years) else f"Col{j + 1}" for j in range(n_value_cols)
            ]
            return [synthesized] + rows

        # If no period signal is near the table, rows[0] may be data not a header.
        # Detect when rows[0]’s non-label cells look like numeric data matching other
        # rows; in that case synthesize a generic "Line Item | Col1 | Col2 | ..." header
        # and treat rows[0] as data to avoid using numbers as column labels.
        value_cells = [c for c in rows[0][1:] if c and str(c).strip()]
        if value_cells:
            value_like = sum(
                1 for c in value_cells if self._LAYOUT_VALUE_RE.fullmatch(str(c).strip())
            )
            if value_like >= max(1, len(value_cells) // 2 + len(value_cells) % 2):
                n_value_cols = max((len(r) - 1 for r in rows), default=0)
                synthesized = ["Line Item"] + [f"Col{i + 1}" for i in range(n_value_cols)]
                return [synthesized] + rows
        return rows  # rows[0] already looks like a real text header — leave as-is

    def _reinject_year_header_if_missing(self, md: str, page, table_top: float) -> str:
        """Post-process a Markdown table string to inject a missing year header by looking
        at nearby page text.
        Operates on Markdown text so it is format-agnostic; used when an independently-
        built table lacks the correct header.
        Avoids trusting a nearby multi-year sentence when a closer quarter-ordinal
        header is present.
        """
        lines = md.split("\n")
        if len(lines) < 2:
            return md
        if self._extract_year_headers(lines[0]):
            try:
                above_text = (
                    page.within_bbox((0, 0, page.width, max(0, table_top)), relative=False)
                    .extract_text() or ""
                ) if table_top else ""
            except Exception:
                above_text = ""
            if not self._QUARTER_ORDINALS_RE.search(above_text):
                return md  # already has a real year header — nothing to fix

        rows = [
            [c.strip() for c in line.strip().strip("|").split("|")]
            for line in lines
            if not is_markdown_separator_row(line)
        ]
        rows = self._inject_missing_year_header(rows, page, table_top)
        # _inject_missing_year_header() keeps the old header guess as an
        # ordinary data row on the assumption it's usually a real section
        # label (Tier 1's case, e.g. "Assets"). Tier 2's own header
        # fallback is a generic "Line Item" placeholder instead, which
        # carries no information and must not linger as a bogus row now
        # that a real year header has been injected in front of it.
        if len(rows) >= 2 and str(rows[1][0] or "").strip().lower() == "line item":
            rows = [rows[0]] + rows[2:]
        return self._table_to_markdown(rows)

    #: Maximum vertical gap (points) between one table's bottom edge and
    #: the next table's top edge for them to be merged into one table —
    #: see _merge_adjacent_tables(). Small enough that it won't bridge a
    #: real gap between two genuinely different tables/sections on the
    #: same page, but generous enough to absorb ordinary row spacing.
    _TABLE_MERGE_MAX_GAP = 15.0

    #: Maximum horizontal bbox-edge difference (points) for two tables to
    #: be considered "the same columns" when merging — see
    #: _merge_adjacent_tables().
    _TABLE_MERGE_MAX_X_DRIFT = 5.0

    def _merge_adjacent_tables(self, found_tables: list) -> List[list]:
        """
        pdfplumber's find_tables() sometimes fragments ONE cohesive
        financial statement into many separate single/few-row "tables" —
        some real 10-Ks draw a thin ruled line between EVERY row, not
        just around the table as a whole, so each rule boundary gets
        detected as its own table region. Left alone, that means a whole
        "Consolidated Statement of Cash Flows" ends up as 20+ disconnected
        one-row fragments instead of one coherent chunk — bad for both
        the Source Evidence panel (which can only show one tiny fragment
        at a time) and retrieval (a search for one line item never
        surfaces the surrounding statement it belongs to).

        Groups tables that are vertically adjacent (the next one starts
        at or just below where the previous one ends — no real content
        gap) AND share the same horizontal extent (same left/right edges,
        i.e. genuinely the same columns) into one combined table. Returns
        a list of groups, each group a list of pdfplumber Table objects
        in top-to-bottom order — a page with no fragmentation just gets
        back the same tables as singleton groups.
        """
        if not found_tables:
            return []
        ordered = sorted(found_tables, key=lambda t: t.bbox[1])
        groups: List[list] = [[ordered[0]]]
        for t in ordered[1:]:
            prev = groups[-1][-1]
            vertical_gap = t.bbox[1] - prev.bbox[3]
            same_columns = (
                abs(t.bbox[0] - prev.bbox[0]) < self._TABLE_MERGE_MAX_X_DRIFT
                and abs(t.bbox[2] - prev.bbox[2]) < self._TABLE_MERGE_MAX_X_DRIFT
            )
            if same_columns and vertical_gap < self._TABLE_MERGE_MAX_GAP:
                groups[-1].append(t)
            else:
                groups.append([t])
        return groups

    #: A raw pdfplumber cell is treated as a genuine VALUE cell (for
    #: building this table's column-position clusters below) only when
    #: its own extracted text already looks numeric — reusing
    #: _LAYOUT_VALUE_RE-shaped intent without importing Tier 2's own
    #: pattern, since a label cell's text ("Cost of sales") must never
    #: contribute a column anchor.
    _RULED_CELL_VALUE_RE = re.compile(r'^\(?-?\$?\s*\d[\d,]*\.?\d*%?\)?$')

    def _recover_ruled_row_cells(
        self, page, table
    ) -> List[List[Optional[str]]]:
        """Recover missing cells in ruled-line tables when many rows lack bounded cells but
        their text exists on the page.
        Cluster numeric cells across the whole table by right-edge x coordinate and crop
        label/value text from page coordinates rather than relying on per-row cell
        indices.
        This prevents misplacing values when pdfplumber returns inconsistent column
        counts; use coordinate-based clusters only for numeric-aligned columns.
        """
        pdf_rows = table.rows
        extracted = table.extract()
        if not pdf_rows or not extracted:
            return extracted

        numeric_x1s: List[float] = []
        for r, vals in zip(pdf_rows, extracted):
            for ci, c in enumerate(r.cells):
                if c and ci < len(vals) and vals[ci] and self._RULED_CELL_VALUE_RE.match(str(vals[ci]).strip()):
                    numeric_x1s.append(c[2])
        if len(numeric_x1s) < 2:
            return extracted

        tolerance = self._adaptive_x1_gap_threshold(numeric_x1s)
        xs = sorted(numeric_x1s)
        clusters: List[List[float]] = [[xs[0]]]
        for x in xs[1:]:
            if x - clusters[-1][-1] <= tolerance:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        # (left_x, right_x1) per value column, left-to-right — left edge
        # is the previous cluster's own right edge (or the table's own
        # left edge for the first), so cropping a cluster never bleeds
        # into its neighbor.
        cluster_x1s = [sum(c) / len(c) for c in clusters]

        # The label/value0 boundary is NOT the table's own left edge —
        # it's wherever the label column's real bboxes (from whichever
        # rows DO have one) actually end. Falls back to the table's own
        # left edge only when literally no row has a bounded label cell
        # at all (extremely unusual — every "total"/header row on a
        # real financial statement has one).
        label_bboxes = [r.cells[0] for r in pdf_rows if r.cells and r.cells[0]]
        label_x1 = max(c[2] for c in label_bboxes) if label_bboxes else table.bbox[0]

        col_ranges: List[Tuple[float, float]] = []
        prev_x1 = label_x1
        for x1 in cluster_x1s:
            col_ranges.append((prev_x1, x1))
            prev_x1 = x1
        if not col_ranges:
            return extracted

        # A row only NEEDS this reconstruction when pdfplumber bounded
        # fewer real cells than this table has value columns — a fully-
        # ruled row (every column already populated) is left untouched,
        # so this can only ever ADD recovered data, never override a
        # cell pdfplumber genuinely got right.
        min_expected_cells = 1 + len(col_ranges)  # label + every value column
        out_rows: List[List[Optional[str]]] = []
        for pdf_row, vals in zip(pdf_rows, extracted):
            present = [c for c in pdf_row.cells if c]
            if len(present) >= min_expected_cells or not present:
                out_rows.append(list(vals))
                continue
            row_top = min(c[1] for c in present)
            row_bottom = max(c[3] for c in present)

            def _crop_text(x0: float, x1: float) -> Optional[str]:
                if x1 <= x0:
                    return None
                try:
                    text = page.crop((x0, row_top, x1, row_bottom)).extract_text() or ""
                except Exception:
                    return None
                text = " ".join(text.split())
                return text or None

            label = _crop_text(table.bbox[0], label_x1)
            new_row: List[Optional[str]] = [label]
            for x0, x1 in col_ranges:
                new_row.append(_crop_text(x0, x1))
            out_rows.append(new_row)
        return out_rows

    _STRIP_YEAR_CELL_RE = re.compile(r'^(?:19|20)\d{2}$')
    _STRIP_NUMBER_CELL_RE = re.compile(r'^\(?-?\$?\s*[\d,]+\.?\d*\)?\s*%?$')

    def _is_wide_header_strip(self, rows: list) -> bool:
        """Detect ruled-line groups that contain only multi-group column headers and no
        actual data rows.
        Such groups typically have a few short rows and multiple bare-year cells but no
        real numeric values, indicating they are header bands rather than data.
        Used to avoid treating header-only ruled bands as tables and losing those labels
        from nearby prose.
        """
        if not rows or len(rows) > 3:
            return False
        year_cells = 0
        for row in rows:
            for cell in row:
                c = (cell or "").strip()
                if not c:
                    continue
                if self._STRIP_YEAR_CELL_RE.match(c):
                    year_cells += 1
                elif self._STRIP_NUMBER_CELL_RE.match(c):
                    return False
        return year_cells >= 4

    def _ruled_line_tables_and_prose(self, page) -> Tuple[List[str], str]:
        """
        Tier 1: ruled vector-line tables via pdfplumber's find_tables(),
        with the detected table regions excluded from the prose via
        outside_bbox() so table content isn't duplicated as garbled
        space-aligned text alongside the clean Markdown version.

        A ruled-line table somewhere on the page does NOT mean every
        table row on that page has ruled lines around it — real 10-Ks
        commonly have a leading row or two (e.g. "Net income" at the top
        of a cash-flow statement) with no rule at all before the first
        one appears. Those rows carry real numeric data but would
        otherwise be lost to plain, un-tabulated prose: once THIS
        function returns any table, the caller never falls through to
        Tier 2/3, since as far as it knows Tier 1 already succeeded on
        this page. So after excluding every ruled-line table's own
        region, this ALSO tries Tier 2 (word-coordinate reconstruction,
        which needs no ruled lines at all) on whatever words are left,
        and folds in anything it finds — Tier 1 and Tier 2 combine
        instead of being mutually exclusive. Returns ([], "") only when
        NEITHER tier found anything on this page.

        Factored out from _extract_page_tables_and_prose() so the
        _parse_pdf() pre-pass (which needs ONLY this combined tier, ahead
        of the fitz-native word-coordinate pass below) doesn't have to
        duplicate this logic.
        """
        try:
            found_tables = page.find_tables()
        except Exception:
            found_tables = []

        md_tables = []
        skipped_header_strip = False
        self._last_strip_prose = None
        filtered_page = page
        merged_groups = self._merge_adjacent_tables(found_tables)
        for group in merged_groups:
            try:
                rows: list = []
                for t in group:
                    try:
                        recovered_rows = self._recover_ruled_row_cells(page, t)
                    except Exception:
                        recovered_rows = t.extract()
                    rows.extend(self._compact_row_cells(r) for r in recovered_rows)
                # pdf layout tools can yield rows merged into a single cell when
                # internal vertical separators are missing, producing label-only cells.
                # Require most rows to have a real second cell before synthesizing a
                # period header; otherwise defer to a reconstruction method that doesn't
                # rely on the tool's column detection.
                real_rows = [r for r in rows if r and r[0] and str(r[0]).strip()]
                split_rows = [r for r in real_rows if len(r) >= 2]
                well_split = bool(real_rows) and len(split_rows) >= max(1, len(real_rows) // 2)
                if not well_split:
                    continue
                # A ruled-line "table" that is ONLY a multi-group header strip
                # (segment names over repeated "2021 2020 Change" sub-headers)
                # with its data rows unruled below it: leave its words in the
                # page so they stay in the text above the rows Tier 2 recovers,
                # instead of turning them into a junk table and REMOVING them.
                if self._is_wide_header_strip(rows):
                    skipped_header_strip = True
                    continue
                rows = self._inject_missing_year_header(rows, page, group[0].bbox[1])
                md = self._table_to_markdown(rows)
            except Exception:
                md = ""
            if md:
                md_tables.append(md)
            for t in group:
                try:
                    # A strict outside_bbox test can exclude words whose bboxes slightly
                    # bleed into an adjacent box due to line-height/descender spacing.
                    # Shrink the exclusion box inward by a small margin so nearby
                    # genuine content isn't dropped while still excluding truly
                    # overlapping items.
                    x0, top, x1, bottom = t.bbox
                    pad = 1.0
                    shrunk_bbox = (x0, min(top + pad, bottom), x1, max(bottom - pad, top))
                    filtered_page = filtered_page.outside_bbox(shrunk_bbox)
                except Exception:
                    pass

        # ── Recover rows with no ruled lines at all, via Tier 2 ──────────
        try:
            leftover_words = self._pdfplumber_words_to_common(filtered_page.extract_words() or [])
            recovered_tables, recovered_prose = self._reconstruct_table_from_word_positions(leftover_words)
        except Exception:
            recovered_tables, recovered_prose = [], None

        if recovered_tables:
            # Each recovered table's OWN top position (see
            # _reconstruct_table_from_word_positions's docstring) is used
            # as the reference point for "text just above THIS table" —
            # NOT one shared position for every table recovered on the
            # page (the first ruled-line group's own top), which used to
            # make a SECOND (or later) recovered table search from the
            # FIRST one's position instead of its own. Falls back to the
            # first ruled-line group's top only when a table's own
            # position wasn't captured for some reason (top == 0.0, e.g.
            # an empty `rows` edge case upstream).
            fallback_ref_top = merged_groups[0][0].bbox[1] if merged_groups else 0
            recovered_tables = [
                self._reinject_year_header_if_missing(
                    md, page, top if top else fallback_ref_top
                )
                for md, top in recovered_tables
            ]
            md_tables.extend(recovered_tables)
            return md_tables, recovered_prose

        if not md_tables:
            if skipped_header_strip:
                # One row per line, header lines included -- far easier to read
                # than the one-cell-per-line text the fitz engine produces for
                # a wide multi-segment table (9 value columns per row).
                try:
                    self._last_strip_prose = page.extract_text() or None
                except Exception:
                    self._last_strip_prose = None
            return [], ""
        try:
            prose = filtered_page.extract_text() or ""
        except Exception:
            try:
                prose = page.extract_text() or ""
            except Exception:
                prose = ""
        return md_tables, prose

    #: Below this many extracted characters, a page with images on it is
    #: treated as scanned/image-based rather than text-native.
    _SCANNED_PAGE_CHAR_THRESHOLD = 20

    def _extract_page_tables_and_prose(self, page) -> Tuple[List[str], str]:
        """
        Detect all tables on a pdfplumber page and convert each to a
        Markdown pipe table. Returns (markdown_tables, prose) where prose
        is the page text with the detected table regions excluded, so
        table content isn't duplicated as garbled space-aligned text
        alongside the clean Markdown version.

        Four detection tiers are tried in order:
        0. Scanned-page guard — if extract_text() returns almost nothing
           AND the page actually has embedded images, this is very likely
           a scanned/image-based page rather than a text-native one. Table
           extraction is skipped entirely (a warning is logged) rather than
           attempting OCR — this project's input is 10-K EDGAR filings,
           which are text-native, so OCR would be unnecessary complexity
           for a case that isn't expected to occur in practice; this guard
           exists purely so a scanned page fails safely/visibly instead of
           silently producing garbage from near-empty text.
        1. _ruled_line_tables_and_prose() — ruled vector-line tables.
        2. _reconstruct_table_from_word_positions() — word-coordinate
           reconstruction (pdfplumber-native, via extract_words()). Fixes
           real 10-Ks where each table cell ends up on its own line even
           with layout=True, so Tier 3's same-line regex has nothing to
           match.
        3. _layout_text_to_markdown_and_prose() — whitespace/background-
           shading-only tables reconstructed from extract_text(layout=True)
           text-flow regex. Common case for real 10-K financial statements
           whose cells DO share a line once layout is preserved.

        TODO: tables that span two PDF pages are detected and linearised
        independently per page (no cross-page stitching). Revisit if
        multi-page financial tables need to be merged into one block.
        """
        try:
            probe_text = page.extract_text() or ""
        except Exception:
            probe_text = ""
        try:
            has_images = bool(page.images)
        except Exception:
            has_images = False
        if len(probe_text.strip()) < self._SCANNED_PAGE_CHAR_THRESHOLD and has_images:
            page_num = getattr(page, "page_number", "?")
            print(f"[FinancialFileParser] page {page_num} appears to be "
                  f"scanned/image-based, table extraction skipped")
            return [], probe_text

        # ── Tier 1: ruled-line tables ──
        md_tables, prose = self._ruled_line_tables_and_prose(page)
        if md_tables:
            return md_tables, prose

        # ── Tier 2: word-coordinate reconstruction (pdfplumber-native) ──
        try:
            common_words = self._pdfplumber_words_to_common(page.extract_words() or [])
            word_tables_with_pos, word_prose = self._reconstruct_table_from_word_positions(common_words)
        except Exception:
            word_tables_with_pos, word_prose = [], ""
        # This call site doesn't need per-table position (no header
        # re-injection happens here) — plain markdown strings only.
        word_tables = [md for md, _top in word_tables_with_pos]
        if word_tables:
            return word_tables, word_prose

        # ── Tier 3: layout-text regex fallback ──
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

    _DATE_CELL_RE = re.compile(
        r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}$",
        re.IGNORECASE,
    )
    _GROUP_PHRASE_RE = re.compile(
        r"(three|six|nine|twelve)\s+months?\s+ended|(?:fiscal\s+)?year\s+ended",
        re.IGNORECASE,
    )
    _GROUP_CODE = {"three": "3M", "six": "6M", "nine": "9M", "twelve": "12M"}

    def _merge_tail_artifact_column(self, headers: list, data_rows: list):
        """Merge a trailing one-cell column that only contains the tail fragment of the
        prior column's text back into that prior column.
        This addresses extraction artifacts where a stray fragment (e.g., closing
        punctuation or percent sign) becomes its own column.
        Apply when the last column consistently appears to be a suffix to the preceding
        value.
        """
        if len(headers) < 5 or not data_rows:
            return headers, data_rows
        last = [r[-1].strip() for r in data_rows if len(r) == len(headers) and r[-1].strip()]
        if len(last) < 1:
            return headers, data_rows
        if not all(re.fullmatch(r"\)?%?\)?", c) for c in last):
            return headers, data_rows
        new_rows = []
        for r in data_rows:
            if len(r) == len(headers):
                r = list(r)
                tail = r.pop().strip()
                if tail:
                    r[-1] = (r[-1].strip() + tail)
            new_rows.append(r)
        return headers[:-1], new_rows

    def _drop_garbled_caption_rows(self, data_rows: list) -> list:
        """Word-split caption fragments printed as an early data row ("Three M None
        | Mon th s Ended | Six Mo | o nt hs E nded") are noise in a numeric table."""
        if len(data_rows) < 4:
            return data_rows
        numeric_rows = sum(1 for r in data_rows if any(re.search(r"\d", c or "") for c in r[1:]))
        if numeric_rows < 0.6 * len(data_rows):
            return data_rows
        kept = []
        for idx, r in enumerate(data_rows):
            cells = [c.strip() for c in r if c and c.strip()]
            if (idx < 2 and len(cells) >= 3 and not any(re.search(r"\d", c) for c in cells)
                    and any(len(t) <= 2 and t.lower() not in {"of", "to", "at", "in", "on", "by", "vs", "us", "as", "or", "&", "%", "a"}
                            for c in cells for t in c.split())):
                continue
            kept.append(r)
        return kept

    def _is_text_header_row(self, row: list) -> bool:
        """Every filled cell (label included) is words with no digits at all --
        a caption row, not data."""
        cells = [c.strip() for c in row if c and c.strip()]
        if not (len(cells) >= 3 and all(re.search(r"[A-Za-z]", c) and not re.search(r"\d", c) for c in cells)):
            return False
        # reject garbled captions whose words were split by the extractor
        # ("Three M None", "Mon th s Ended"): every token must be a real word
        short_ok = {"of", "to", "at", "in", "on", "by", "vs", "us", "as", "or", "&", "%", "a", "per"}
        for c in cells:
            for tok in c.split():
                if len(tok) <= 2 and tok.lower() not in short_ok:
                    return False
        return True

    def _is_date_header_row(self, row: list) -> bool:
        """Identify a row where the label and every filled cell are calendar dates as
        actually being the column header for the following table.
        Such rows typically represent period labels (dates) rather than data and should
        be promoted to header status.
        Promote only when all cells are date-like to avoid misclassifying genuine data
        rows.
        """
        cells = [c.strip() for c in row if c and c.strip()]
        return len(cells) >= 2 and all(self._DATE_CELL_RE.match(c) for c in cells)

    def _refine_value_headers(self, value_headers: list, header_prose: str) -> list:
        """Disambiguate generic or duplicated value-column labels (e.g., "Col3" or repeated
        year) using header lines printed above the table.
        Handle common shapes like amount/percent column pairs and period-grouped year
        labels by composing meaningful column labels from header context.
        Leave labels unchanged if they do not match the expected header patterns.
        """
        n = len(value_headers)
        if n < 3:
            return value_headers
        year_like = [h for h in value_headers if re.fullmatch(r"(?:19|20)\d{2}", h)]
        ambiguous = any(re.fullmatch(r"Col\d+", h) for h in value_headers) or (
            len(year_like) != len(set(year_like))
        )
        if not ambiguous:
            return value_headers
        lines = [l for l in header_prose.splitlines() if l.strip()]
        zone = "\n".join(lines[:40])
        raw_years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", zone)

        # (a) Amount / percent-to-sales pairs
        if (n in (4, 5) and re.search(r"(?:^|\s)amount(?:\s|$)", zone, re.IGNORECASE)
                and re.search(r"to\s+sales", zone, re.IGNORECASE)):
            years = list(dict.fromkeys(raw_years))
            if len(years) == 2:
                y1, y2 = years
                labels = [y1, "% to Sales '" + y1[-2:], y2, "% to Sales '" + y2[-2:]]
                if n == 5:
                    labels.append("% Increase (Decrease)")
                return labels

        # (b) period groups x years: read the FIRST header block only (group
        # phrases up to the first year, then the years that follow), so
        # footnotes further down the page that repeat "Twelve months ended
        # ... 2021" cannot pollute it.
        tokens = sorted(
            [(m.start(), "g", self._GROUP_CODE.get((m.group(1) or "").lower(), "FY"))
             for m in self._GROUP_PHRASE_RE.finditer(zone)]
            + [(m.start(1), "y", m.group(1))
               for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", zone)]
        )
        i = 0
        while i < len(tokens) and tokens[i][1] != "g":
            i += 1
        groups: list = []
        while i < len(tokens) and tokens[i][1] == "g":
            groups.append(tokens[i][2])
            i += 1
        years_block: list = []
        while i < len(tokens) and tokens[i][1] == "y" and len(years_block) < n:
            years_block.append(tokens[i][2])
            i += 1
        if len(groups) >= 2 and len(years_block) == n and n % len(groups) == 0:
            per_group = n // len(groups)
            block = years_block[:per_group]
            if years_block == block * len(groups) and len(set(block)) == per_group:
                return [f"{y} ({g})" for g in groups for y in block]

        # (c) quarter headers "1Q21 4Q20 3Q20 2Q20 1Q20" (JPM's financial highlights):
        # label the columns "1Q 2021" ... so the sandbox sees the year
        if all(re.fullmatch(r"Col\d+", h) for h in value_headers):
            qs = re.findall(r"(?<![A-Za-z0-9])([1-4])Q(\d{2})(?![A-Za-z0-9])", zone)
            if len(qs) >= n:
                first = qs[:n]
                if len(set(first)) == n:
                    return [f"{q}Q 20{yy}" for q, yy in first]
        return value_headers

    # Handle grouped-column tables made of repeated column groups (same sub-columns
    # repeated side-by-side). Detect the repeated sub-header token sequence (the sub-
    # header repeats as a unit and contains words) and split the block into group
    # columns so downstream logic can interpret each segment independently.
    _GC_BAD_UNIT_RE = re.compile(
        r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?$|.*[,:]$|^\d{1,2}$|^(?:of|to|and|the|in|at|for|by|on|per|as)$",
        re.IGNORECASE,
    )
    _GC_NUM_RE = re.compile(r"^\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?$|^NM$|^[—–-]$")
    _GC_PERIOD_VOCAB = {
        "three", "six", "nine", "twelve", "months", "month", "ended", "year", "fiscal",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    }

    @staticmethod
    def _multi_number_cell_fraction(tables: List[str]) -> float:
        """Share of non-empty value cells holding 2+ separate numbers."""
        total = multi = 0
        for t in tables:
            for line in t.splitlines()[2:]:
                if not line.startswith("|"):
                    continue
                for cell in line.strip("|").split("|")[1:]:
                    c = cell.strip()
                    if not c:
                        continue
                    total += 1
                    if len(re.findall(r"(?<![\w.])\(?\$?\d[\d,]*\.?\d*\)?%?(?![\w.])", c)) >= 2:
                        multi += 1
        return multi / total if total >= 5 else 0.0

    @staticmethod
    def _md_data_rows(tables: List[str]) -> int:
        """Data rows (excluding header + separator) across markdown tables."""
        return sum(max(0, len([l for l in t.splitlines() if l.startswith("|")]) - 2) for t in tables)

    def _periodic_subheader(self, words: list):
        """(unit, groups, words) when the line ENDS with one short unit
        (2-4 tokens, at least one alphabetic) repeated 2+ whole times; any
        tokens before that run are the row-label caption, e.g. "(in
        millions, except ratios)". `words` in the result is only the
        periodic run."""
        toks = [w["text"] for w in words]
        n = len(toks)
        best = None
        for m in (2, 3, 4):
            for g in range(n // m, 1, -1):
                run = toks[n - m * g:]
                unit = run[:m]
                if run != unit * g:
                    continue
                if not any(re.search(r"[A-Za-z]", t) for t in unit):
                    continue
                # Do not treat runs of dates, connector words, or bare day numbers as
                # column captions. Such sequences can mimic headers and cause
                # mislabeling; require stronger header signals before accepting them as
                # column labels.
                if any(self._GC_BAD_UNIT_RE.match(t) for t in unit):
                    continue
                if best is None or m * g > best[0]:
                    best = (m * g, unit, g, words[n - m * g:])
                break
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _stacked_subheader(self, lines: list, i: int, run_words: list):
        """Reconstruct multi-line sub-column captions that were printed above numeric
        columns by assigning each non-numeric caption word to the nearest column anchor
        derived from the first data rows.
        Returns (unit, groups, words) like _periodic_subheader, or None when captions do
        not form a repeating pattern.
        """
        data = None
        for k in range(i + 1, min(len(lines), i + 6)):
            if sum(1 for w in lines[k]["words"] if self._GC_NUM_RE.match(w["text"])) >= 3:
                data = k
                break
        if data is None:
            return None
        # numbers left of the caption run are part of the row label ("Net sales fiscal
        # year 2023"), not a value column
        min_x = min(w["x0"] for w in run_words) - 60
        # numbers are right-aligned, so their RIGHT edges line up per column
        edges = sorted(
            w["x1"]
            for k in range(data, min(len(lines), data + 8))
            for w in lines[k]["words"]
            if self._GC_NUM_RE.match(w["text"]) and (w["x0"] + w["x1"]) / 2 >= min_x
        )
        anchors: List[List[float]] = []
        for c in edges:
            if anchors and c - anchors[-1][-1] <= 6:
                anchors[-1].append(c)
            else:
                anchors.append([c])
        # a typical figure is ~20pt wide: the caption is centred over its digits
        anchors_x = [sum(a) / len(a) - 10 for a in anchors if len(a) >= 2]
        n = len(anchors_x)
        if n < 4 or n > 16:
            return None
        cols: List[List[Dict[str, Any]]] = [[] for _ in anchors_x]
        for k in range(max(0, i - 1), data):
            for w in lines[k]["words"]:
                if self._GC_NUM_RE.match(w["text"]):
                    continue
                cx = (w["x0"] + w["x1"]) / 2
                a = min(range(n), key=lambda q: abs(anchors_x[q] - cx))
                if abs(anchors_x[a] - cx) <= 45:
                    cols[a].append(w)
        labels = []
        for ws in cols:
            ws.sort(key=lambda w: (round(w["top"]), w["x0"]))
            labels.append(" ".join(w["text"] for w in ws).strip())
        if any(not lab for lab in labels):
            return None
        for m in (2, 3, 4):
            if (n % m == 0 and n // m >= 2 and labels == labels[:m] * (n // m)
                    and any(re.search(r"[A-Za-z]", lab) for lab in labels[:m])
                    and not any(self._GC_BAD_UNIT_RE.match(t) for lab in labels[:m] for t in lab.split())):
                words = [
                    {"text": labels[q], "x0": anchors_x[q] - 10, "x1": anchors_x[q] + 10, "top": lines[i]["top"]}
                    for q in range(n)
                ]
                return labels[:m], n // m, words
        return None

    def _realign_sparse_rows(self, page, md_tables: List[str]) -> List[str]:
        """Put the values of rows that have FEWER numbers than the table has value
        columns back under the columns they were printed under.

        Ruled-line cell compaction (`_compact_row_cells`) drops blank cells so
        every row has the same shape, which left-shifts a sparse row: MGM's
        "Adjusted EBITDAR $957,307 ... $3,497,254" total row (printed under the
        three-month-2022 and twelve-month-2022 columns) came out under the first
        TWO columns, i.e. the full-year figure labelled as the 2021 quarter. The
        column positions come from the table's own full rows (right edges of
        their numbers); a sparse row's numbers are assigned to the nearest one."""
        # cheap pre-check first: pdfplumber word extraction is the expensive part, so
        # only pages that actually contain a table with a sparse row pay for it
        def _has_sparse(md: str) -> bool:
            rows = [
                [c.strip() for c in l.strip().strip("|").split("|")]
                for l in md.splitlines() if l.startswith("|")
            ]
            if len(rows) < 5:
                return False
            n = len(rows[0]) - 1
            body = rows[2:]
            return n >= 3 and any(len(r) == n + 1 and 0 < sum(1 for c in r[1:] if c) < n for r in body)

        if not any(_has_sparse(md) for md in md_tables):
            return md_tables
        try:
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
        except Exception:
            return md_tables
        lines: Dict[int, list] = {}
        for w in words:
            lines.setdefault(round(w["top"] / 3), []).append(w)

        def _line_for(label: str):
            toks = label.split()[:2]
            if not toks:
                return None
            for key in sorted(lines):
                ws = sorted(lines[key] + lines.get(key + 1, []), key=lambda w: w["x0"])
                texts = [w["text"] for w in ws]
                for k in range(len(texts) - len(toks) + 1):
                    if texts[k:k + len(toks)] == toks:
                        return [w for w in ws if self._GC_NUM_RE.match(w["text"])]
            return None

        out: List[str] = []
        for md in md_tables:
            rows_raw = [l for l in md.splitlines() if l.startswith("|")]
            if len(rows_raw) < 5:
                out.append(md)
                continue
            cells = [[c.strip() for c in l.strip().strip("|").split("|")] for l in rows_raw]
            header, body = cells[0], cells[2:]
            n = len(header) - 1
            if n < 3:
                out.append(md)
                continue
            # a last column holding only the tail of the previous cell (")%") is an
            # extraction artifact, not a real column: leave such a table to
            # _merge_tail_artifact_column
            _tail = [r[-1] for r in body if len(r) == n + 1 and r[-1]]
            if _tail and all(re.fullmatch(r"\)?%?\)?", c) for c in _tail):
                out.append(md)
                continue
            full = [r for r in body if len(r) == n + 1 and all(c for c in r[1:])]
            sparse = [r for r in body if len(r) == n + 1 and 0 < sum(1 for c in r[1:] if c) < n]
            if len(full) < 2 or not sparse:
                out.append(md)
                continue
            anchors: List[List[float]] = [[] for _ in range(n)]
            for r in full[:8]:
                nums = _line_for(r[0])
                if nums and len(nums) == n:
                    for q, w in enumerate(sorted(nums, key=lambda w: w["x0"])):
                        anchors[q].append(w["x1"])
            if any(not a for a in anchors):
                out.append(md)
                continue
            anchor_x = [sum(a) / len(a) for a in anchors]
            changed = False
            for r in sparse:
                filled = [c for c in r[1:] if c]
                nums = _line_for(r[0])
                if not nums or len(nums) != len(filled):
                    continue
                new_vals = [""] * n
                ok = True
                for w, val in zip(sorted(nums, key=lambda w: w["x0"]), filled):
                    q = min(range(n), key=lambda q: abs(anchor_x[q] - w["x1"]))
                    if abs(anchor_x[q] - w["x1"]) > 12 or new_vals[q]:
                        ok = False
                        break
                    new_vals[q] = val
                if ok and new_vals != r[1:]:
                    r[1:] = new_vals
                    changed = True
            if not changed:
                out.append(md)
                continue
            rebuilt = self._table_to_markdown([header] + body)
            out.append(rebuilt if rebuilt else md)
        return out

    def _grouped_column_tables(self, page) -> List[str]:
        try:
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
        except Exception:
            return []
        if len(words) < 30:
            return []
        lines: List[Dict[str, Any]] = []
        for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
            if lines and abs(lines[-1]["top"] - w["top"]) <= 2.5:
                lines[-1]["words"].append(w)
            else:
                lines.append({"top": w["top"], "words": [w]})
        for ln in lines:
            ln["words"].sort(key=lambda w: w["x0"])

        tables: List[str] = []
        i = 1
        while i < len(lines):
            sub = self._periodic_subheader(lines[i]["words"])
            if sub is None:
                i += 1
                continue
            unit, ngroups, sub_words = sub
            # more numbers in the first data row than sub-columns: the captions are
            # stacked over several lines -- rebuild them from the numeric columns
            _first_data = next(
                (lines[k] for k in range(i + 1, min(len(lines), i + 6))
                 if sum(1 for w in lines[k]["words"] if self._GC_NUM_RE.match(w["text"])) >= 3),
                None,
            )
            if _first_data is not None and sum(
                1 for w in _first_data["words"] if self._GC_NUM_RE.match(w["text"])
            ) > len(sub_words):
                stacked = self._stacked_subheader(lines, i, sub_words)
                if stacked is None:
                    i += 1
                    continue
                unit, ngroups, sub_words = stacked
            m = len(unit)
            groups_x = [(sub_words[g * m]["x0"], sub_words[g * m + m - 1]["x1"]) for g in range(ngroups)]
            bounds = [(groups_x[g][1] + groups_x[g + 1][0]) / 2 for g in range(ngroups - 1)]
            sub_centers = [(w["x0"] + w["x1"]) / 2 for w in sub_words]

            # group names: taken from the nearest line above the sub-header whose
            # words split into DIFFERENT names per group (stacked captions such as
            # "Net US | Net US" or "EPS | EPS" repeat the same text in every group
            # and are skipped; a caption carrying a year or period wording wins)
            def _is_vocab(t: str) -> bool:
                tl = t.lower().strip(",:")
                return tl in self._GC_PERIOD_VOCAB or bool(re.fullmatch(r"\d{1,2}|(?:19|20)\d{2}", tl))

            def _names_from(line_words: list):
                prev = list(line_words)
                if not all(_is_vocab(w["text"]) for w in prev):
                    k = 0
                    while k < len(prev) and _is_vocab(prev[k]["text"]):
                        k += 1
                    prev = prev[k:]
                parts: List[List[str]] = [[] for _ in range(ngroups)]
                for w in prev:
                    cx = (w["x0"] + w["x1"]) / 2
                    parts[sum(1 for b in bounds if cx > b)].append(w["text"])
                return [" ".join(pp).strip() for pp in parts]

            names = None
            names_idx = i - 1
            for back in range(1, 5):
                if i - back < 0:
                    break
                cand = _names_from(lines[i - back]["words"])
                if any(not c for c in cand) or len(set(cand)) < ngroups:
                    continue
                # Define a group name as a short caption that labels a table section.
                # Do not treat a full prose line above a table split by x-position as a
                # group name.
                if any(len(c) > 60 or len(c.split()) > 8 for c in cand):
                    continue
                # Do not treat a data row from the table above (split by x-position) as
                # a group name.
                # Exclude standalone dates or bare years from being considered numeric
                # figures.
                if any(
                    sum(
                        1 for t in c.split()
                        if re.fullmatch(r"\(?\$?\d[\d,.]*\)?%?", t)
                        and not re.fullmatch(r"(?:19|20)\d{2},?|\d{1,2},", t)
                    ) >= 2
                    for c in cand
                ):
                    continue
                names = cand
                names_idx = i - back
                if any(re.search(r"(?:19|20)\d{2}|months", c, re.IGNORECASE) for c in cand):
                    break
            if names is None:
                i += 1
                continue
            # Keep group labels short to avoid harming retrieval ranking and chunking.
            # Long verbose labels may push important rows out of model context.
            def _short(nm: str) -> str:
                nm = re.sub(
                    r"(?i)\b(three|six|nine|twelve)\s+months?\s+ended\b",
                    lambda mm: self._GROUP_CODE[mm.group(1).lower()], nm)
                nm = re.sub(r"(?i)\b(?:fiscal\s+)?years?\s+ended\b", "FY", nm)
                return nm.strip()
            names = [_short(nm) for nm in names]

            # A standalone period heading above the names line distinguishes stacked
            # tables.
            # If missing, different period tables can appear identical and be conflated.
            _lead_period = ""
            for k in range(names_idx, max(-1, names_idx - 3), -1):
                mp = re.match(
                    r"(?i)^\s*(three|six|nine|twelve)\s+months?\s+ended\b",
                    " ".join(w["text"] for w in lines[k]["words"]),
                )
                if mp:
                    _lead_period = self._GROUP_CODE[mp.group(1).lower()]
                    break
            if _lead_period and not any(re.search(r"\b\d{1,2}M\b", nm) for nm in names):
                names = [f"{nm} ({_lead_period})" for nm in names]

            def _sub_label(g: int, s: int) -> str:
                sub_tok = unit[s]
                name = names[g]
                if re.fullmatch(r"(?:19|20)\d{2}", sub_tok):
                    if re.search(r"(?:19|20)\d{2}", name):
                        return f"{name} {sub_tok}"
                    return f"{name} '{sub_tok[-2:]}"  # year-less: never read as that year's figure by the sandbox
                return f"{name} {sub_tok}"

            # Define caption as the nearest sentence or heading above the group-name
            # line.
            # Store it in the first header cell and copy into each row so queries on
            # context (e.g., region) can find relevant rows.
            caption = ""
            for k in range(names_idx - 1, max(-1, names_idx - 7), -1):
                ws_k = lines[k]["words"]
                txt_k = " ".join(w["text"] for w in ws_k).strip()
                n_num = sum(1 for w in ws_k if self._GC_NUM_RE.match(w["text"]))
                if len(txt_k.split()) >= 4 and len(txt_k) <= 160 and n_num <= 3:
                    caption = txt_k[:100]
                    break
            headers = [("Table: " + caption) if caption else "Line Item"] + [
                _sub_label(g, s) for g in range(ngroups) for s in range(m)
            ]
            first_x = groups_x[0][0]
            rows: List[List[str]] = []
            bad_rows = 0
            pending_label = ""
            j = i + 1
            label_only_run = 0
            while j < len(lines):
                ws = lines[j]["words"]
                if self._periodic_subheader(ws) is not None:
                    break
                label_words, value_words = [], []
                for w in ws:
                    t = w["text"]
                    is_num = bool(self._GC_NUM_RE.match(t)) or t in ("$", "%")
                    if not value_words and (not is_num or w["x1"] < first_x - 40):
                        label_words.append(t)
                    elif is_num:
                        value_words.append(w)
                    else:
                        # footnote marker such as "(d)" sitting between values
                        continue
                nums = [w for w in value_words if w["text"] not in ("$", "%")]
                if len(nums) < 2 and label_words and re.search(r"[A-Za-z]", " ".join(label_words)):
                    pending_label = " ".join(label_words).strip()  # a wrapped row label
                if len(nums) < 2:
                    label_only_run += 1
                    if label_only_run >= 2 or (rows and not nums and re.match(r"^\(?[a-z]\)", " ".join(label_words))):
                        break
                    j += 1
                    continue
                label_only_run = 0
                if len(nums) > len(sub_centers):
                    # more values than sub-columns: the sub-header line is
                    # incomplete (a stacked caption such as "Rigid / Packaging"
                    # printed on two lines) -- never mislabel, count it as bad
                    bad_rows += 1
                    j += 1
                    continue
                cells = [""] * (ngroups * m)
                collided = False
                for w in nums:
                    cx = (w["x0"] + w["x1"]) / 2
                    idx = min(range(len(sub_centers)), key=lambda q: abs(sub_centers[q] - cx))
                    if cells[idx]:
                        collided = True
                    else:
                        cells[idx] = w["text"]
                if collided:
                    bad_rows += 1
                    j += 1
                    continue
                # a "%" printed as its own token right after a value belongs to it
                for w in value_words:
                    if w["text"] == "%":
                        cx = (w["x0"] + w["x1"]) / 2
                        idx = min(range(len(sub_centers)), key=lambda q: abs(sub_centers[q] - cx))
                        for back in (idx, idx - 1):
                            if 0 <= back < len(cells) and cells[back] and not cells[back].endswith("%"):
                                cells[back] += "%"
                                break
                label = " ".join(label_words).strip()
                if not label and pending_label:
                    label = pending_label  # numbers printed on the line below their wrapped label
                pending_label = ""
                if label and re.search(r"[A-Za-z]", label) and not re.match(r"^\(\d\)", label):
                    rows.append([label] + cells)
                j += 1
            fill = (
                sum(sum(1 for c in r[1:] if c) / max(1, len(r) - 1) for r in rows) / len(rows)
                if rows else 0.0
            )
            if len(rows) >= 3 and bad_rows <= max(1, len(rows) // 5) and fill >= 0.35:
                md = self._table_to_markdown([headers] + rows)
                if md:
                    tables.append(md)
            i = max(j, i + 1)
        return tables

    _UNIT_MENTION_RE = re.compile(
        r"(?i)(?:\$\s*|usd\s*|dollars\s+in\s+|\bin\s+)(millions?|billions?|thousands?)\b"
    )

    def _page_unit(self, page_text: str) -> str:
        """Extract the single monetary unit declared on a page's captions (e.g. unit
        indicator phrases), or "" when none or multiple units appear.
        This is necessary because table rows lack unit markers and answers built from
        rows alone would have an unknown unit.
        """
        units = {m.group(1).lower().rstrip("s") for m in self._UNIT_MENTION_RE.finditer(page_text or "")}
        if len(units) != 1:
            return ""
        return "USD " + next(iter(units)) + "s"

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
        unit_note = self._page_unit(page_text)
        report_label = f"{table_name} [{unit_note}]" if unit_note else table_name

        # Capture the section heading/title immediately before the first table block as
        # parent context.
        # Keep only the last few lines immediately above the table, since they typically
        # contain the table's actual title and distinguish supplementary schedules from
        # primary statements.
        first_block_pos = page_text.find(table_blocks[0])
        leading_context = page_text[:first_block_pos].strip() if first_block_pos > 0 else ""
        if leading_context:
            leading_lines = [l for l in leading_context.split("\n") if l.strip()]
            leading_context = "\n".join(leading_lines[-6:])

        # The full CLEAN Markdown table(s) — not a raw page_text[:1000]
        # slice, which can cut off before ever reaching the table (real
        # pages usually lead with boilerplate/title text) or capture it
        # only partially. Both the frontend's Source Evidence panel and
        # pot_reasoner's row-scoped formula extraction rely on
        # parent_content actually containing a genuine '|---|' table
        # block to render/parse it as a table instead of falling back to
        # plain text.
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + (leading_context + "\n\n" if leading_context else "")
            + "\n\n".join(table_blocks)
        )

        row_idx = 0
        header_prose = page_text
        for _blk in table_blocks:
            header_prose = header_prose.replace(_blk, "", 1)
        for block in table_blocks:
            headers, data_rows = self._parse_markdown_table_block(block)
            if len(headers) < 2 or not data_rows:
                continue
            headers, data_rows = self._merge_tail_artifact_column(headers, data_rows)
            data_rows = self._drop_garbled_caption_rows(data_rows)
            value_headers = self._refine_value_headers(headers[1:], header_prose)
            block_headers_raw = [c.strip() for c in block.split("\n")[0].strip().strip("|").split("|")]
            md_value_headers = list(value_headers)
            md_header_rows: list = []

            for row_no, row in enumerate(data_rows):
                if not row or not row[0].strip():
                    continue
                if self._is_text_header_row(row):
                    # a wrapped column-header line printed as a data row
                    # ("Total Stores at Beginning of Second Quarter | Stores Opened |
                    # ..."): use it as the header of the numeric rows that follow
                    # when its name count equals their value count
                    names = [c.strip() for c in row if c and c.strip()]
                    nxt = next((r for r in data_rows[row_no + 1:] if r and r[0].strip()), None)
                    if nxt is not None:
                        nvals = len([c for c in nxt[1:] if c and c.strip()])
                        if nvals == len(names) and nvals >= 4:
                            groups = list(dict.fromkeys(
                                re.findall(r"(?:fiscal|fy)\s*((?:19|20)\d{2})", header_prose, re.IGNORECASE)
                            ))
                            if len(groups) >= 2 and nvals % len(groups) == 0:
                                per = nvals // len(groups)
                                names = [f"Fiscal {groups[i // per]} {nm}" for i, nm in enumerate(names)]
                            value_headers = names
                            md_value_headers = list(names)
                            md_header_rows.append([c.strip() for c in row])
                    continue
                if self._is_date_header_row(row):
                    # the table's real column header, printed as a row: label
                    # the rows that follow with these dates instead
                    value_headers = [c.strip() for c in row if c and c.strip()]
                    md_value_headers = list(value_headers)
                    md_header_rows.append([c.strip() for c in row])
                    continue
                line_item = row[0].strip()
                values = row[1:]
                table_caption = headers[0].strip() if headers and headers[0].startswith("Table:") else ""
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
                    # Retrieval weight for a unit was applied using the display name
                    # only; internal content altered BM25 length normalisation per unit-
                    # annotated row and changed retrieval rankings. Use caution: this
                    # can demote rows that should rank highly if their unit label
                    # differs between display and content.
                    "table_name": report_label,
                    "period": "-".join(value_headers) if value_headers else "N/A",
                    "page_number": page_num,
                    "content": (
                        f"Company: {company_name} | Report: {table_name} | "
                        f"Line Item: {line_item} | "
                        + (f"{table_caption} | " if table_caption else "")
                        + " | ".join(kv_parts)
                    ),
                    "type": "table_row",
                    "raw_data": raw_data,
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                    "is_child": True,
                })

            # Parent markdown header shown to the LLM retained generic column headers
            # while per-row chunks used refined labels; header block should be rewritten
            # to use consistent, clear column labels so parent context matches per-row
            # labels.
            if len(md_value_headers) == len(block_headers_raw) - 1 and (
                md_value_headers != block_headers_raw[1:] or md_header_rows
            ):
                blines = block.split("\n")
                drop = {tuple(r) for r in md_header_rows}
                kept = blines[:2]
                for ln in blines[2:]:
                    cells = tuple(c.strip() for c in ln.strip().strip("|").split("|"))
                    if cells in drop:
                        continue
                    kept.append(ln)
                kept[0] = "| " + " | ".join([block_headers_raw[0]] + md_value_headers) + " |"
                parent_content = parent_content.replace(block, "\n".join(kept), 1)

            prose_only_text = prose_only_text.replace(block, "", 1)

        for _p in passages:
            _p["parent_content"] = parent_content

        return passages, prose_only_text

    #: A document Table of Contents often appears as short labels followed by a small
    #: integer, which can be mistaken for a two-column financial table. The reliable
    #: signal for a real TOC page is the presence of a section marker like PART I or PART
    #: II near the start; do not treat the phrase Table of Contents alone as sufficient
    #: evidence. Use this combined check to avoid misclassifying content pages as TOC and
    #: losing structured table rows.
    _TABLE_OF_CONTENTS_RE = re.compile(r'\btable\s+of\s+contents\b', re.IGNORECASE)
    _TOC_PART_MARKER_RE = re.compile(r'\bpart\s+(?:i|ii|iii|iv)\b', re.IGNORECASE)

    def _is_table_of_contents_page(self, page_text: str) -> bool:
        zone = page_text[:600]
        return bool(
            self._TABLE_OF_CONTENTS_RE.search(zone)
            and self._TOC_PART_MARKER_RE.search(zone)
        )

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

    def _extract_repeating_year_headers(self, text: str, n_cols: int) -> list:
        """
        Detect a header whose year tokens repeat as a clean multiple across
        n_cols columns -- e.g. "(Millions) 2018 2017 2016 2018 2017 2016"
        for a 6-value-column row that's really TWO metric categories (Net
        Sales, Operating Income) side by side, each spanning the SAME
        3-year span, not a genuine 6-year comparison. Returns exactly
        n_cols year strings (the unique year sequence tiled to fill every
        column) when the header shows this repeating structure, else [].

        Distinct from _extract_year_headers's plain dedup, and deliberately
        narrow: only fires when the RAW (non-deduplicated) year count in
        the header line exactly equals n_cols and is a clean whole-number
        repetition of its own unique prefix. A wide row whose header does
        NOT show this (e.g. a single "2018" next to a genuinely one-period,
        multi-segment breakdown -- CVS Health's segment table, where 6
        value columns are 6 different segments for ONE year, not repeated
        year groups) must fall through unrecognised, so it still gets
        caught by _WORD_TABLE_MAX_VALUE_COLS's width cap in the caller
        rather than being mislabeled as a multi-year table.
        """
        if n_cols <= 0:
            return []
        header_zone = text[:400]
        raw_years = [m.group(1) for m in self._YEAR_HEADER_RE.finditer(header_zone)]
        if len(raw_years) != n_cols:
            return []
        unique = list(dict.fromkeys(raw_years))
        k = len(unique)
        if k < 2 or n_cols % k != 0 or n_cols // k < 2:
            return []
        if raw_years != unique * (n_cols // k):
            return []
        return raw_years

    #: A standard 10-K "Contractual Obligations" table headers its columns
    #: "Total | 2019 | 2020 | 2021 | 2022 | 2023 | After 2023" — a lifetime
    #: TOTAL column and a catch-all THEREAFTER/AFTER-<year> bucket mixed in
    #: with the bare years. Ordered so "After 2023" / "Thereafter" match as
    #: ONE token before the bare-year alternative can independently match
    #: just the trailing "2023".
    _PERIOD_TOKEN_RE = re.compile(
        r'\bAfter\s+(?:20|19)\d{2}\b|\bThereafter\b|\bTotal\b|\b(?:20|19)\d{2}\b',
        re.IGNORECASE,
    )

    def _extract_period_headers(self, text: str, n_cols: int) -> list:
        """Detect headers that mix bare years with summary period labels such as "Total"
        and "After <year>/Thereafter" so column labels align with values.
        Returns exactly n_cols period-label strings when the header's token count
        matches n_cols, else [].
        The exact-count requirement prevents misapplying this to unrelated lines
        containing the word "Total".
        """
        if n_cols <= 0:
            return []
        header_zone = text[:400]
        tokens = []
        for m in self._PERIOD_TOKEN_RE.finditer(header_zone):
            t = re.sub(r'\s+', ' ', m.group(0)).strip()
            tokens.append("Total" if t.lower() == "total" else t.title())
        if len(tokens) != n_cols:
            return []
        # Must actually contain a real bare year somewhere (never JUST
        # "Total"/"Thereafter" repeated) — otherwise this is indistinguishable
        # from any other short line that happens to say "Total" n_cols times.
        if not any(re.fullmatch(r'(?:20|19)\d{2}', t) for t in tokens):
            return []
        return tokens

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
        # Full page_text, not a [:1000] preview — see _chunk_text_to_passages'
        # docstring for why a hardcoded character cap on parent_content
        # silently defeats its own purpose.
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + page_text
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
        """Chunk free text into prose passages while preserving the full page text in
        parent_content so downstream consumers can access the complete source.
        Args: parent_source_text carries the full source page (not a truncated preview).
        Returns: created text_note chunks; parent_content must not be truncated here
        because consumer-specific limits belong at the consumer layer.
        """
        source_text = parent_source_text if parent_source_text is not None else text
        # Use a chunk size of 3000 characters to keep date/year lines and related data
        # together; per-chunk retrieval boosts examine only the candidate chunk's text,
        # so too-small splits can separate date context from answer text and misdirect
        # boosts. parent_content still provides the full page to the LLM, so this change
        # affects which chunk is retrieved, not what the model ultimately sees.
        chunks = chunk_text(text, chunk_size=3000, overlap=120, min_chunk_size=100)
        if not chunks and text.strip():
            chunks = [text.strip()]

        parent_id = f"parent_{company_name}_p{page_num}"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Page: {page_num} | "
            + source_text
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
        - If the page is the filing's own front-matter Table of Contents
          → always chunk as free text, regardless of what table-detection
          below would otherwise find (see _TABLE_OF_CONTENTS_RE's
          docstring).
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
        # ── Table of Contents page: never table-detected ────────────────────
        if self._is_table_of_contents_page(page_text):
            return self._chunk_text_to_passages(company_name, filename, page_num, page_text, section)

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

        # ── Pre-pass: pdfplumber ruled-line table detection (Tier 1 only) ──
        # Runs regardless of which text-extraction engine below ends up
        # succeeding, since ruled-line detection has no fitz equivalent —
        # both engines rely on pdfplumber's find_tables() for this tier.
        # Tiers 2 (word-coordinate) and 3 (layout regex) are deliberately
        # NOT computed here: Tier 2 needs to run against whichever engine's
        # OWN page object ends up handling the page (fitz's page.get_text
        # ("words") vs pdfplumber's page.extract_words() can see different
        # word/coordinate data for the same PDF), so it's tried per-engine
        # below instead of being pre-computed once. Tier 3 only needs the
        # raw extract_text(layout=True) string, which IS cheap to
        # pre-compute once here for the fitz path to reuse.
        # TODO: tables spanning two PDF pages are detected & linearised
        # independently per page — no cross-page table stitching yet.
        ruled_tables_by_page: Dict[int, List[str]] = {}
        # For pages that DO have ruled tables, also keep pdfplumber's
        # bbox-excluded prose so the fitz engine below can swap it in and
        # avoid duplicating the raw space-aligned table text alongside the
        # clean Markdown table (fitz has no bounding-box awareness of
        # pdfplumber's detected table regions).
        ruled_prose_by_page: Dict[int, str] = {}
        strip_prose_by_page: Dict[int, str] = {}
        layout_text_by_page: Dict[int, str] = {}
        try:
            import pdfplumber as _pdfplumber_prepass
            with _pdfplumber_prepass.open(io.BytesIO(content_bytes)) as _pdf:
                for page_idx, page in enumerate(_pdf.pages):
                    page = self._without_overlapping_spaces(page)
                    try:
                        md_tables, prose = self._ruled_line_tables_and_prose(page)
                    except Exception:
                        md_tables, prose = [], ""
                    if md_tables:
                        md_tables = self._realign_sparse_rows(page, md_tables)
                    grouped_tables: List[str] = []
                    try:
                        grouped_tables = self._grouped_column_tables(page)
                    except Exception:
                        grouped_tables = []
                    if grouped_tables and md_tables:
                        # A ruled-line detection that only caught a caption/header
                        # strip (JPM's VaR page: "Three months ended | 2023: | 2022:")
                        # must not hide the real grouped table; keep the ruled
                        # tables only when they carry more data rows.
                        def _md_rows(tabs):
                            return sum(max(0, len([l for l in t.splitlines() if l.startswith("|")]) - 2) for t in tabs)
                        if _md_rows(grouped_tables) > _md_rows(md_tables):
                            md_tables = []
                        else:
                            grouped_tables = []
                    if md_tables and not grouped_tables and self._multi_number_cell_fraction(md_tables) >= 0.2:
                        # If consecutive numeric values appear in a single ruled cell
                        # (e.g., "66.56 66.11 63.93"), the column split likely failed —
                        # reparse the page using a different layout interpretation.
                        md_tables = []
                    if md_tables and not grouped_tables:
                        # Ruled-line detection may capture only part of a table. If the
                        # plain layout-text extraction yields clearly more rows, discard
                        # the ruled fragment and escalate the page to higher-tier
                        # processing.
                        try:
                            _lt = page.extract_text(layout=True) or ""
                            _l_tabs, _ = self._layout_text_to_markdown_and_prose(_lt) if _lt else ([], "")
                            _l_rows = self._md_data_rows(_l_tabs)
                            _r_rows = self._md_data_rows(md_tables)
                            if _l_rows >= _r_rows + 3 and _l_rows >= 1.5 * _r_rows:
                                md_tables = []
                        except Exception:
                            pass
                    if md_tables:
                        ruled_tables_by_page[page_idx] = md_tables
                        ruled_prose_by_page[page_idx] = prose
                    elif grouped_tables:
                        ruled_tables_by_page[page_idx] = grouped_tables
                        ruled_prose_by_page[page_idx] = page.extract_text() or ""
                    elif getattr(self, "_last_strip_prose", None):
                        strip_prose_by_page[page_idx] = self._last_strip_prose
                    try:
                        layout_text_by_page[page_idx] = page.extract_text(layout=True) or ""
                    except Exception:
                        layout_text_by_page[page_idx] = ""
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
                    # ── Tier 1: ruled-line tables (from the pre-pass) ──
                    # On pages that have them, swap fitz's own prose for
                    # pdfplumber's bbox-excluded prose (computed in the
                    # pre-pass above) so the raw space-aligned table text
                    # doesn't end up duplicated alongside the clean Markdown
                    # table below — fitz itself has no bounding-box awareness
                    # of pdfplumber's detected table regions.
                    page_md_tables = ruled_tables_by_page.get(page_idx, [])
                    if page_md_tables:
                        filtered_prose = self._clean_text_content(
                            ruled_prose_by_page.get(page_idx, "")
                        )
                        prose_for_page = (
                            filtered_prose
                            if self._is_readable(filtered_prose, min_alnum=1)
                            else best_text
                        )
                        best_text = self._compose_page_text(prose_for_page, page_md_tables)
                    else:
                        # ── Tier 2: fitz-native word-coordinate reconstruction ──
                        try:
                            common_words = self._fitz_words_to_common(page.get_text("words"))
                            word_tables_with_pos, word_prose = self._reconstruct_table_from_word_positions(common_words)
                        except Exception:
                            word_tables_with_pos, word_prose = [], ""
                        # This call site doesn't need per-table position
                        # (no header re-injection happens here) — plain
                        # markdown strings only.
                        word_tables = [md for md, _top in word_tables_with_pos]
                        if word_tables and self._multi_number_cell_fraction(word_tables) >= 0.2:
                            # column clustering failed (several values packed into one
                            # cell, e.g. "Col1: 66.56 66.11 63.93"); let Tier 3 read the page
                            word_tables = []
                        if word_tables:
                            # Tier 2 extraction can recover only a subset of a table's
                            # rows while a more layout-aware Tier 3 read recovers more;
                            # prefer the more complete Tier 3 result when it clearly
                            # yields more data rows.
                            _lt = layout_text_by_page.get(page_idx, "")
                            if _lt:
                                _l_tabs, _ = self._layout_text_to_markdown_and_prose(_lt)
                                _l_rows = self._md_data_rows(_l_tabs)
                                _w_rows = self._md_data_rows(word_tables)
                                if _l_rows >= _w_rows + 3 and _l_rows >= 1.5 * _w_rows:
                                    word_tables = []
                        if word_tables:
                            prose_for_page = self._clean_text_content(word_prose)
                            if not self._is_readable(prose_for_page, min_alnum=1):
                                prose_for_page = best_text
                            best_text = self._compose_page_text(prose_for_page, word_tables)
                        else:
                            # ── Tier 3: layout-text regex fallback (pre-computed) ──
                            layout_text = layout_text_by_page.get(page_idx, "")
                            layout_tables, layout_prose = (
                                self._layout_text_to_markdown_and_prose(layout_text)
                                if layout_text else ([], "")
                            )
                            if layout_tables:
                                prose_for_page = self._clean_text_content(layout_prose)
                                if not self._is_readable(prose_for_page, min_alnum=1):
                                    prose_for_page = best_text
                                best_text = self._compose_page_text(prose_for_page, layout_tables)
                            elif page_idx in strip_prose_by_page and self._is_readable(
                                self._clean_text_content(strip_prose_by_page[page_idx]), min_alnum=1
                            ):
                                # Wide multi-segment table whose ruled part was only a
                                # header strip: pdfplumber's one-row-per-line text (header
                                # lines included) beats fitz's one-cell-per-line text.
                                best_text = self._clean_text_content(strip_prose_by_page[page_idx])
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
                    page = self._without_overlapping_spaces(page)
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

        # parent: first 5000 chars of the full document. Unlike a single PDF
        # page (naturally bounded to a few thousand characters, so left
        # uncapped elsewhere — see _chunk_text_to_passages' docstring), an
        # uploaded .txt/.md file has no such bound, so this keeps a generous
        # but finite preview rather than the [:1000] cap used before (too
        # short for a multi-paragraph note to survive intact).
        parent_id = f"parent_{company_name}_txt"
        parent_content = (
            f"Company: {company_name} | Document: {filename} | Content: "
            + text_str[:5000]
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
