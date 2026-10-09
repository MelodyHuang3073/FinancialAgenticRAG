"""
Detects PDF pages whose table extraction produced a malformed table_row
passage (the page's OWN text-coordinate heuristics mis-detected the column
headers, most commonly on five-year/multi-period "financial highlights"
summary tables and multi-level-header tables) and repairs just those pages
using MinerU's trained layout + table-structure-recognition model, which
reconstructs the table from the page's visual structure instead of
column-position heuristics on extracted text.

General, structural detection (not tied to any one company/filing): a
`table_row` passage is "malformed" when its own Line Item label IS itself a
bare number/currency symbol -- a real line item label is never a bare
number, so this only ever fires on a genuine extraction failure. Confirmed
against the project's two real corpora (2026-10-09): 183/19,922 table_row
passages in the out-of-sample set, 963/54,880 in the original 150-question
set, both well below the threshold where every table on a page could
plausibly be legitimately numeric-only.

Repair is strictly local (MinerU's "basic" managed tier -- layout + OCR +
table-structure-recognition ONNX models, no VLM, no network call) and
best-effort: any failure (MinerU not installed, its local service not
running, a page it also can't parse, a timeout) is swallowed and the
original (possibly still-malformed) passages are kept, so an upload never
fails or slows down meaningfully because of this step.
"""
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Set

#: A real financial-statement line item label is never a bare number,
#: currency symbol, or parenthesized/signed number on its own -- this
#: fires only when the page's own column-header detection failed and a
#: data VALUE got written into the Line Item slot instead of its real
#: text label (see this module's docstring for confirmed prevalence).
#: Excludes a bare year/year-range ("2027", "2031-2035"): confirmed real
#: case (Kraft Heinz's "Future Benefit Payments" schedule) where that IS
#: the row's genuine identity, not a mis-parsed value -- see
#: _ensure_header_row's docstring. A genuinely wrong dollar/currency value
#: in the Line Item slot (Boeing's "89,463", Pfizer's "$") never looks
#: like a bare year, so excluding years costs no real detections.
_LINE_ITEM_RE = re.compile(r"Line Item:\s*([^|]+?)\s*\|")
_BARE_NUMERIC_LABEL_RE = re.compile(r"^[\$\d][\d,\.\(\)\-\$%]*$")
_BARE_YEAR_RE = re.compile(r"^\d{4}(?:-\d{4})?$")


def _is_malformed_line_item_label(label: str) -> bool:
    label = label.strip()
    if _BARE_YEAR_RE.match(label):
        return False
    return bool(_BARE_NUMERIC_LABEL_RE.match(label))

#: Per-repair-attempt timeout; a page MinerU can't finish within this is
#: treated as a failed repair (original passages kept), not a hung upload.
MINERU_REPAIR_TIMEOUT_SECONDS = 90


def _passage_is_malformed(passage: Dict[str, Any]) -> bool:
    if passage.get("type") != "table_row":
        return False
    m = _LINE_ITEM_RE.search(passage.get("content") or "")
    return bool(m) and _is_malformed_line_item_label(m.group(1))


def find_malformed_pages(passages: List[Dict[str, Any]]) -> Set[int]:
    """Page numbers (1-indexed, matching passage['page_number']) that have
    at least one table_row passage with the malformed-label signature."""
    pages: Set[int] = set()
    for p in passages:
        if _passage_is_malformed(p):
            page_num = p.get("page_number")
            if page_num is not None:
                pages.add(page_num)
    return pages


def _mineru_available() -> bool:
    try:
        import mineru  # noqa: F401
    except Exception:
        return False
    return True


def _extract_single_page_pdf(content_bytes: bytes, page_num: int, out_path: str) -> bool:
    """Writes just `page_num` (1-indexed) of the PDF to out_path as its own
    single-page PDF, so MinerU only has to look at the one page being
    repaired. Returns False (nothing written) if the page index is out of
    range or extraction fails for any reason."""
    try:
        import fitz
        src = fitz.open(stream=content_bytes, filetype="pdf")
        if not (1 <= page_num <= len(src)):
            return False
        mini = fitz.open()
        mini.insert_pdf(src, from_page=page_num - 1, to_page=page_num - 1)
        mini.save(out_path)
        return True
    except Exception:
        return False


def _html_tables_to_markdown(html: str) -> str:
    """Converts every <table> in `html` to a Markdown pipe table (the shape
    parser._linearize_markdown_tables expects); any non-table content is
    dropped since only table blocks are ever fed in here.

    MinerU renders a table as HTML instead of a Markdown pipe table
    whenever it has merged cells (colspan/rowspan) -- confirmed real case:
    a pension-benefits table with Pension/Postretirement/Total column
    groups each spanning two sub-columns. Markdown pipe tables can't
    express a span, so a spanned cell's text is repeated across the
    columns it covers -- an approximation that keeps every row's OWN
    values aligned under the right columns (what
    _linearize_markdown_tables's row-label/value-header pairing actually
    needs) at the cost of the spanning column's group label appearing
    once per sub-column instead of once overall.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    for table in soup.find_all("table"):
        md_rows = []
        max_cols = 0
        parsed_rows = []
        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["td", "th"]):
                text = cell.get_text(strip=True)
                span = int(cell.get("colspan", 1) or 1)
                cells.extend([text] * max(span, 1))
            if cells:
                parsed_rows.append(cells)
                max_cols = max(max_cols, len(cells))
        if not parsed_rows:
            continue
        for row in parsed_rows:
            row = row + [""] * (max_cols - len(row))
            md_rows.append("| " + " | ".join(c.replace("|", "/") or " " for c in row) + " |")
        header_sep = "| " + " | ".join(["---"] * max_cols) + " |"
        blocks.append(md_rows[0] + "\n" + header_sep + "\n" + "\n".join(md_rows[1:]))
    return "\n\n".join(blocks)


def _ensure_header_row(block: str) -> str:
    """A table with NO real header row -- every row is a same-shaped data
    tuple led by a year (a debt/benefit-payment maturity schedule: "2026 |
    $ | 69", "2027 | | 64", ...) -- gets its first row wrongly treated as
    the header by parser._parse_markdown_table_block (which unconditionally
    takes row 0 as headers), silently discarding that row's real data value
    as if it were a column name instead. Confirmed real case: Kraft
    Heinz's "Future Benefit Payments" schedule.

    Detected structurally: the row that would become the header has the
    SAME shape (a bare year/year-range in its own first cell) as the row
    right after it -- a real header never does. When detected, a generic
    "Year | Col1 | Col2..." header is inserted ahead of both, so the
    original first row survives as a genuine data row instead of being
    discarded. Left untouched for the overwhelmingly more common case of a
    real text-label header row (e.g. Boeing's "U.S. dollars in millions...
    | 2025 | 2024 | ...").
    """
    lines = [l for l in block.split("\n") if l.strip()]
    if len(lines) < 3:
        return block

    def _cells(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header_cells = _cells(lines[0])
    first_data_cells = _cells(lines[2])  # lines[1] is the "---" separator
    if not header_cells or not first_data_cells:
        return block
    if not (_BARE_YEAR_RE.match(header_cells[0]) and _BARE_YEAR_RE.match(first_data_cells[0])):
        return block

    n_cols = len(header_cells)
    synthetic_header = "| " + " | ".join(["Year"] + [f"Col{i}" for i in range(1, n_cols)]) + " |"
    return "\n".join([synthetic_header, lines[1], lines[0]] + lines[2:])


def _mineru_page_markdown(pdf_path: str) -> Optional[str]:
    """The markdown for every table block MinerU found on the (single-page)
    PDF at pdf_path, concatenated, or None if MinerU found no tables there
    or the call failed/timed out.

    Enforced via a worker thread + hard `.result(timeout=...)`, not just a
    try/except around a plain call: a single-page MinerU inference normally
    takes a few seconds, but under system load (e.g. other MinerU workers
    already running) it was observed to occasionally take 100+ seconds --
    confirmed real variance, not a hypothetical -- an upload must never
    hang on that.
    """
    try:
        import mineru
    except Exception:
        return None

    import concurrent.futures

    def _call():
        result = mineru.parse(pdf_path, tier="basic")
        return result.structured_content()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            content = pool.submit(_call).result(timeout=MINERU_REPAIR_TIMEOUT_SECONDS)
    except Exception:
        return None
    raw_table_blocks = [
        block.get("content", "")
        for page in (content.get("pages") or [])
        for block in (page.get("blocks") or [])
        if block.get("type") == "table" and block.get("content")
    ]
    # A table block comes back as HTML instead of a Markdown pipe table
    # whenever it has merged cells (colspan/rowspan) -- see
    # _html_tables_to_markdown's docstring. Each block is independently
    # HTML or Markdown; check per block rather than assuming one shape for
    # the whole page.
    table_blocks = [
        _ensure_header_row(_html_tables_to_markdown(b) if "<table" in b else b)
        for b in raw_table_blocks
    ]
    table_blocks = [b for b in table_blocks if b.strip()]
    if not table_blocks:
        return None
    return "\n\n".join(table_blocks)


def repair_malformed_tables(
    parser: Any,
    passages: List[Dict[str, Any]],
    content_bytes: bytes,
    filename: str,
    company_name: str,
) -> List[Dict[str, Any]]:
    """Best-effort: detects malformed table_row passages, re-parses just
    their pages via MinerU, and replaces them with freshly-linearized
    passages built from MinerU's clean table output (reusing
    parser._linearize_markdown_tables -- the SAME passage-construction code
    the normal parse path uses, so a repaired passage is indistinguishable
    in shape from a normally-parsed one).

    `passages` is returned unchanged (same list, same order for every
    untouched passage) whenever there is nothing to repair, MinerU isn't
    available, or every repair attempt fails -- this function never raises
    and never makes the result worse than the input.
    """
    malformed_pages = find_malformed_pages(passages)
    if not malformed_pages or not _mineru_available():
        return passages

    repaired_by_page: Dict[int, List[Dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="table_repair_") as tmpdir:
        for page_num in sorted(malformed_pages):
            single_page_path = os.path.join(tmpdir, f"page_{page_num}.pdf")
            if not _extract_single_page_pdf(content_bytes, page_num, single_page_path):
                continue
            mineru_markdown = _mineru_page_markdown(single_page_path)
            if not mineru_markdown:
                continue
            try:
                new_table_passages, _prose = parser._linearize_markdown_tables(
                    company_name, filename, page_num, mineru_markdown
                )
            except Exception:
                continue
            if new_table_passages:
                repaired_by_page[page_num] = new_table_passages

    if not repaired_by_page:
        return passages

    result: List[Dict[str, Any]] = []
    for p in passages:
        if p.get("page_number") in repaired_by_page and _passage_is_malformed(p):
            continue  # dropped -- its page's repaired passages are spliced in below instead
        result.append(p)
    for page_num, new_passages in repaired_by_page.items():
        for p in new_passages:
            p = dict(p)
            p["table_repair_source"] = "mineru"
            result.append(p)
    return result
