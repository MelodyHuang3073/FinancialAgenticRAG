import json
import os
from typing import List, Dict, Any
from app.tools.table_parser import linearize_financial_table
from app.tools.hybrid_retriever import HybridFinancialRetriever
from app.rag.chunker import chunk_text

class FinancialVectorStoreManager:
    def __init__(self, sample_json_path: str = None, load_sample_data: bool = False):
        self.corpus: List[Dict[str, Any]] = []
        self.retriever: HybridFinancialRetriever = None
        self.uploaded_files: List[Dict[str, Any]] = []
        # parent_id -> parent_content (≤1000 chars context)
        self.parent_map: Dict[str, str] = {}

        if load_sample_data and sample_json_path and os.path.exists(sample_json_path):
            self.load_from_json(sample_json_path)

    def load_from_json(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for company_data in data:
            company = company_data.get("company", "Unknown")
            period = company_data.get("period", "Current")
            
            # Linearize statements
            for stmt in company_data.get("statements", []):
                table_name = stmt.get("table_name", "Financial Table")
                headers = stmt.get("headers", [])
                rows = stmt.get("rows", [])
                
                passages = linearize_financial_table(company, period, table_name, headers, rows)
                self.corpus.extend(passages)

            # Ingest text notes & MD&A
            for idx, note in enumerate(company_data.get("notes", [])):
                note_text = f"Company: {company} | Note: {note.get('title')} | Period: {period} | Text: {note.get('content')}"
                chunks = chunk_text(note_text, chunk_size=800, overlap=120, min_chunk_size=500)
                for chunk_idx, chunk in enumerate(chunks, 1):
                    self.corpus.append({
                        "id": note.get("id", f"{company}_note_{idx}") if len(chunks) == 1 else f"{note.get('id', f'{company}_note_{idx}')}_chunk_{chunk_idx}",
                        "company": company,
                        "table_name": f"{note.get('title', 'MD&A Note')} (Chunk {chunk_idx})",
                        "period": period,
                        "content": chunk,
                        "type": "text_note",
                        "raw_data": note
                    })

        self.retriever = HybridFinancialRetriever(self.corpus)

    def add_parsed_passages(self, filename: str, company: str, passages: List[Dict[str, Any]]):
        for p in passages:
            pid = p.get("parent_id")
            pc = p.get("parent_content")
            if pid and pc and pid not in self.parent_map:
                self.parent_map[pid] = pc
        self.corpus.extend(passages)
        self.uploaded_files.append({
            "filename": filename,
            "company": company,
            "passage_count": len(passages)
        })
        self.retriever = HybridFinancialRetriever(self.corpus)

    def get_parent_content(self, parent_id: str) -> str:
        """Return stored parent_content for a given parent_id, or empty string."""
        return self.parent_map.get(parent_id, "")

    def add_custom_document(self, company: str, title: str, text: str):
        doc_id = f"custom_{len(self.corpus) + 1}"
        chunks = chunk_text(f"Company: {company} | Document: {title} | Text: {text}", chunk_size=800, overlap=120, min_chunk_size=500)
        for chunk_idx, chunk in enumerate(chunks, 1):
            entry_id = doc_id if len(chunks) == 1 else f"{doc_id}_chunk_{chunk_idx}"
            entry = {
                "id": entry_id,
                "company": company,
                "table_name": f"{title} (Chunk {chunk_idx})" if len(chunks) > 1 else title,
                "period": "Uploaded Document",
                "content": chunk,
                "type": "custom_upload",
                "raw_data": {"title": title, "text": text}
            }
            self.corpus.append(entry)
        self.uploaded_files.append({
            "filename": title,
            "company": company,
            "passage_count": len(chunks)
        })
        self.retriever = HybridFinancialRetriever(self.corpus)
        return doc_id

    def search(self, query: str, top_k: int = 5, exclude_ids: List[str] = None,
               entity: str = None) -> List[Dict[str, Any]]:
        if not self.retriever:
            return []
        return self.retriever.search(query, top_k=top_k, exclude_ids=exclude_ids, entity=entity)
