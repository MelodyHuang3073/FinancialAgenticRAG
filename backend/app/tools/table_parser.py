from typing import List, Dict, Any


def _cell_to_markdown(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text.replace("|", "\\|")


def _rows_to_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Build a Markdown pipe table from headers + rows, used as the shared
    parent_content for every row-passage of one table (so the frontend's
    Source Evidence panel can show the complete table, not just the single
    matched row)."""
    if not headers or not rows:
        return ""
    header_cells = [_cell_to_markdown(h) for h in headers]
    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for row in rows:
        row_cells = [_cell_to_markdown(c) for c in row]
        if len(row_cells) < len(header_cells):
            row_cells += [""] * (len(header_cells) - len(row_cells))
        lines.append("| " + " | ".join(row_cells[:len(header_cells)]) + " |")
    return "\n".join(lines)


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

    full_table_md = _rows_to_markdown_table(headers, rows)
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
