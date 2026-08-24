import os
import csv
import io
import json
import re
import html
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from app.rag.chunker import chunk_text
from app.tools.table_parser import linearize_financial_table, to_markdown_table


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
    # - The label-to-first-value gap, AND the gap between subsequent value
    #   tokens, only need 1+ space. A 2+-space requirement between values
    #   looks safer on paper, but real-world layout=True output frequently
    #   collapses adjacent numeric columns down to a single space when the
    #   source PDF wasn't built with a wide, rule-aligned column grid (e.g.
    #   a proportional-font page where columns are only whitespace-padded
    #   in the source text, not laid out on a true fixed grid) — with a 2+
    #   requirement, EVERY row on such a page fails to match and the whole
    #   table silently falls back to unconverted plain text. At least TWO
    #   value tokens are still required — real financial tables always show
    #   2+ comparison periods side by side — which is what actually keeps
    #   this from matching an ordinary prose sentence that trails off with
    #   one number; an isolated single-row false match (e.g. a "Month DD,
    #   YYYY" date landing on two token-shaped fragments) is harmless on its
    #   own since _flush() below additionally requires 3+ CONSECUTIVE
    #   matching rows before anything is treated as a real table.
    _LAYOUT_ROW_RE = re.compile(
        r'^(?=.*[A-Za-z一-鿿])(?P<label>\S.{0,80}?)\s+'
        r'(?P<values>(?:' + _VALUE_TOKEN + r')(?:\s+(?:' + _VALUE_TOKEN + r'))+)\s*$'
    )

    @staticmethod
    def _normalize_layout_value(token: str) -> str:
        """Collapse the gap pdfplumber can leave between a standalone '$'
        glyph and its digits ("$  1,503" -> "$1,503")."""
        return re.sub(r'^\$\s+', '$', token.strip())

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
            if len(current_rows) >= 3:
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
                current_rows.append((label, values))
                current_n_cols[0] = len(values)
            else:
                _flush()
                prose_lines.append(stripped)
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
            row_numeric = [w for w in row if self._LAYOUT_VALUE_RE.fullmatch(w["text"].strip())]
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
        """
        Of every row with 2+ numeric-looking tokens, what fraction share
        the single most common "shape" (the sorted set of anchor indices
        those numbers matched)? Computed from numeric tokens ONLY — label
        words are excluded, since a label word coincidentally landing
        near an anchor by chance is noise, not evidence of real column
        structure. See _WORD_COL_MIN_SHAPE_CONFIDENCE for how this is
        used.
        """
        shapes = []
        for row in rows:
            numeric_words = [w for w in row if self._LAYOUT_VALUE_RE.fullmatch(w["text"].strip())]
            if len(numeric_words) < 2:
                continue
            matches = [self._assign_to_value_anchor(w["x1"], anchors, tolerance) for w in numeric_words]
            shapes.append(tuple(sorted(i for i in matches if i is not None)))
        if not shapes:
            return 0.0
        most_common_count = Counter(shapes).most_common(1)[0][1]
        return most_common_count / len(shapes)

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

    def _reconstruct_table_from_word_positions(self, words: List[Dict[str, Any]]) -> Tuple[List[str], str]:
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

        Returns (markdown_tables, prose) where prose is every text-only
        row's reconstructed text, in original top-to-bottom order.
        """
        rows = self._cluster_words_into_rows(words)
        if not rows:
            return [], ""
        rows = [self._merge_dollar_sign_tokens(row) for row in rows]
        anchors, tolerance = self._cluster_value_column_anchors(rows)
        if not anchors:
            return [], ""
        if self._dominant_anchor_shape_fraction(rows, anchors, tolerance) < self._WORD_COL_MIN_SHAPE_CONFIDENCE:
            return [], ""

        tables: List[str] = []
        prose_lines: List[str] = []
        current_rows: List[Tuple[str, List[str]]] = []
        current_n_cols = [None]
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

        def _flush():
            if len(current_rows) >= self._WORD_TABLE_MIN_ROWS:
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
                for label, values in current_rows:
                    prose_lines.append((label + "  " + "  ".join(values)).strip())
            current_rows.clear()
            current_n_cols[0] = None
            current_col_shape[0] = None

        for row_words in rows:
            cells: List[List[str]] = [[] for _ in range(len(anchors) + 1)]
            for w in row_words:
                col = self._assign_to_value_anchor(w["x1"], anchors, tolerance)
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
                current_rows.append((first_col, values))
                current_n_cols[0] = len(values)
                current_col_shape[0] = col_shape
            else:
                _flush()
                line_text = " ".join(c for c in cell_texts if c).strip()
                if line_text:
                    prose_lines.append(line_text)
                    if self._extract_year_headers(line_text) or not header_candidate[0]:
                        header_candidate[0] = line_text
        _flush()

        return tables, "\n".join(prose_lines)

    @staticmethod
    def _compact_row_cells(row: list) -> list:
        """
        Collapse pdfplumber's raw per-row cell grid down to [label, value1,
        value2, ...] by dropping blank cells and lone '$' cells. A real
        10-K commonly shows the currency symbol only on the FIRST row of a
        section and on subtotal/total rows... except pdfplumber's ruled-
        line cell detector does the opposite just as often — a data row
        gets a dedicated '$' cell but the very next subtotal row doesn't
        (or vice versa) — so the SAME logical column lands at a different
        index depending on which row you're looking at. Since every row
        is later matched to the header purely by column INDEX (see
        _extract_from_markdown_table_block in pot_reasoner.py), that
        positional drift silently drops values from any row whose shape
        doesn't match the header's. Compacting away the blank/'$' noise
        makes every row's shape the same: label, then exactly its real
        values in order — immune to whichever row happened to get the
        '$' cell. A real placeholder value like '—' (em dash) is kept,
        since it's a meaningful zero/blank for that period, not noise.
        Also re-merges a negative number's closing parenthesis back onto
        its digits when pdfplumber's cell-splitting has put them in two
        adjacent cells ('(116' + ')' -> '(116)') — otherwise that single
        value eats two column slots instead of one, throwing off every
        later column's index-based year match on that row.
        """
        if not row:
            return row
        label = row[0]
        rest = [c for c in row[1:] if c and c.strip() not in ("", "$")]
        merged: list = []
        i = 0
        while i < len(rest):
            cell = rest[i].strip()
            if (
                i + 1 < len(rest)
                and re.match(r'^\(\s*[\d,.]+$', cell)
                and rest[i + 1].strip() == ')'
            ):
                merged.append(cell + ')')
                i += 2
            else:
                merged.append(rest[i])
                i += 1
        return [label] + merged

    def _inject_missing_year_header(self, rows: list, page, table_top: float) -> list:
        """
        pdfplumber's find_tables() draws its table's bounding box from the
        ruled lines alone — on a real 10-K, a horizontal rule often sits
        between the true "As of .../For the years ended ..." date line and
        the section-label row below it (e.g. "Assets", "Net sales"), so
        the date line falls OUTSIDE the detected box and the section-label
        row is mistaken for the table's own header. Every value on the
        table then silently fails downstream extraction, which requires a
        recognisable year in the header to accept a row's value at all.

        If rows[0] doesn't contain a year, look at the page text just
        above this table's bounding box for the nearest line that does,
        and synthesize a proper header from it — keeping rows[0] itself
        as an ordinary data row (its label is usually still real, e.g.
        "Assets"; it simply has no values of its own) rather than
        discarding it. Expects rows already compacted by
        _compact_row_cells() so "how many years" and "how many value
        columns the widest data row has" agree without extra bookkeeping.
        """
        if not rows:
            return rows
        header_text = " ".join(c or "" for c in rows[0])
        if self._extract_year_headers(header_text):
            return rows  # already has a real year header — nothing to fix

        try:
            above = page.within_bbox((0, 0, page.width, max(0, table_top)), relative=False)
            above_text = above.extract_text() or ""
        except Exception:
            above_text = ""
        years: list = []
        for line in reversed(above_text.split("\n")):
            years = self._extract_year_headers(line)
            if years:
                break
        if not years:
            return rows  # no candidate found anywhere above — leave as-is

        n_value_cols = max((len(r) - 1 for r in rows[1:]), default=len(years))
        n_value_cols = max(n_value_cols, len(years))
        synthesized = [rows[0][0] or "Line Item"] + [
            years[i] if i < len(years) else f"Col{i + 1}" for i in range(n_value_cols)
        ]
        return [synthesized, rows[0]] + rows[1:]

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
        filtered_page = page
        for group in self._merge_adjacent_tables(found_tables):
            try:
                rows: list = []
                for t in group:
                    rows.extend(self._compact_row_cells(r) for r in t.extract())
                rows = self._inject_missing_year_header(rows, page, group[0].bbox[1])
                md = self._table_to_markdown(rows)
            except Exception:
                md = ""
            if md:
                md_tables.append(md)
            for t in group:
                try:
                    filtered_page = filtered_page.outside_bbox(t.bbox)
                except Exception:
                    pass

        # ── Recover rows with no ruled lines at all, via Tier 2 ──────────
        try:
            leftover_words = self._pdfplumber_words_to_common(filtered_page.extract_words() or [])
            recovered_tables, recovered_prose = self._reconstruct_table_from_word_positions(leftover_words)
        except Exception:
            recovered_tables, recovered_prose = [], None

        if recovered_tables:
            md_tables.extend(recovered_tables)
            return md_tables, recovered_prose

        if not md_tables:
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
            word_tables, word_prose = self._reconstruct_table_from_word_positions(common_words)
        except Exception:
            word_tables, word_prose = [], ""
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
            + "\n\n".join(table_blocks)
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
        layout_text_by_page: Dict[int, str] = {}
        try:
            import pdfplumber as _pdfplumber_prepass
            with _pdfplumber_prepass.open(io.BytesIO(content_bytes)) as _pdf:
                for page_idx, page in enumerate(_pdf.pages):
                    try:
                        md_tables, prose = self._ruled_line_tables_and_prose(page)
                    except Exception:
                        md_tables, prose = [], ""
                    if md_tables:
                        ruled_tables_by_page[page_idx] = md_tables
                        ruled_prose_by_page[page_idx] = prose
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
                            word_tables, word_prose = self._reconstruct_table_from_word_positions(common_words)
                        except Exception:
                            word_tables, word_prose = [], ""
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
