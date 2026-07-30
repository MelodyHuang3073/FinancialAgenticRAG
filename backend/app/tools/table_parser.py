from typing import List, Dict, Any

def linearize_financial_table(company: str, year_context: str, table_title: str, headers: List[str], rows: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Linearizes financial tables using Header-Prepended Row strategy (FinAgent-RAG Stage 2).
    Example Output passage:
    "Company: TSMC | Statement: Income Statement | Year: 2024 | Line Item: Revenue | 2023: 2,161.7B | 2024: 2,894.3B | Unit: TWD"
    """
    passages = []
    
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
            "raw_data": dict(zip(headers, row))
        })
        
    return passages
