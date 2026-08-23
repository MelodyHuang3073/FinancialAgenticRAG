import re
from typing import List, Dict, Any


def _cell_to_markdown(value: Any) -> str:
    """Normalise a single cell into a Markdown-table-safe string.
    Whitespace/newlines are collapsed and '|' is escaped because either one
    would otherwise corrupt the table's own pipe syntax — this is NOT a
    numeric/formatting cleanup step; the cell's original characters
    (commas, parens, decimals, currency symbols, ...) are left untouched."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace("|", "\\|")


def to_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Build a standard Markdown pipe table from headers + rows:
        | Line Item | FY2023 | FY2022 |
        |---|---|---|
        | Net sales | 14,694 | 14,544 |
        | Cost of sales | (11,328) | (11,229) |

    This is the single canonical table formatter for the project — called
    both by app.rag.parser (converting pdfplumber-detected PDF tables) and
    by linearize_financial_table() below (converting sample/CSV table
    data), so there is exactly one place that defines what a serialised
    table cell looks like.

    Cell values are NOT reformatted or rounded — whatever string comes in
    goes out unchanged (aside from whitespace/newline collapsing and '|'
    escaping, which is purely about not breaking the table syntax itself).
    Numeric cleanup belongs to the extraction stage, not here.

    Rows that are entirely empty (all cells None/blank) are dropped —
    pdfplumber's raw table extraction frequently includes a spurious blank
    row (e.g. from a table's border/padding). Returns "" if there's no
    usable header or no data rows left after that filtering.
    """
    if not headers or not rows:
        return ""
    header_cells = [_cell_to_markdown(h) for h in headers]
    if not any(header_cells):
        return ""

    data_rows = [[_cell_to_markdown(c) for c in row] for row in rows]
    data_rows = [r for r in data_rows if any(r)]
    if not data_rows:
        return ""

    n_cols = max(len(header_cells), max(len(r) for r in data_rows))
    header_cells = header_cells + [""] * (n_cols - len(header_cells))

    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("|" + "|".join(["---"] * n_cols) + "|")
    for row in data_rows:
        row = row + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(row[:n_cols]) + " |")
    return "\n".join(lines)


_MD_SEPARATOR_RE = re.compile(r'^\|?[\s:\-]*\|([\s:\-]*\|)*[\s:\-]*\|?$')


def is_markdown_separator_row(line: str) -> bool:
    """True if `line` is a Markdown table separator row, e.g. '|---|---|'
    or '| :--- | ---: |'. Shared by pot_reasoner.py (to find and parse a
    Markdown table embedded in evidence) and llm_client.py (to avoid
    character-truncating a table mid-row)."""
    s = line.strip()
    return bool(s) and "-" in s and bool(_MD_SEPARATOR_RE.match(s))


def linearize_financial_table(
    company: str,
    year_context: str,
    table_title: str,
    headers: List[str],
    rows: List[List[Any]],
    statement_type: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Linearizes financial tables using Header-Prepended Row strategy (FinAgent-RAG Stage 2).
    Example Output passage:
    "Company: TSMC | Statement: Income Statement | Year: 2024 | Line Item: Revenue | 2023: 2,161.7B | 2024: 2,894.3B | Unit: TWD"

    Args:
        company        : Company name
        year_context   : Period / fiscal year string
        table_title    : Table / report name
        headers        : Column headers (first column is the line item label)
        rows           : Data rows
        statement_type : One of income_statement | balance_sheet | cash_flow | notes | unknown
    """
    passages = []

    full_table_md = to_markdown_table(headers, rows)
    parent_id = f"parent_{company}_{table_title}"
    parent_content = (
        f"Company: {company} | Report: {table_title} | Period: {year_context}\n\n" + full_table_md
        if full_table_md else ""
    )

    for row_idx, row in enumerate(rows):
        if not row:
            continue
        line_item = str(row[0]).strip()
        cell_pairs = []

        for col_idx in range(1, len(row)):
            header_name = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx}"
            val = str(row[col_idx]).strip()
            cell_pairs.append(f"{header_name}: {val}")

        linearized_text = f"Company: {company} | Report: {table_title} | Period: {year_context} | Line Item: {line_item} | " + " | ".join(cell_pairs)

        passages.append({
            "id": f"{table_title}_{row_idx}",
            "company": company,
            "table_name": table_title,
            "period": year_context,
            "line_item": line_item,
            "content": linearized_text,
            "type": "table_row",
            "statement_type": statement_type,
            "raw_data": dict(zip(headers, row)),
            "parent_id": parent_id,
            "parent_content": parent_content,
            "is_child": True,
        })

    return passages
