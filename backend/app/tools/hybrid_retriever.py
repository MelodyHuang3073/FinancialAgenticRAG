import math
import re
from typing import List, Dict, Any


class HybridFinancialRetriever:
    FINANCIAL_TERMS = [
        "營業收入", "營收", "毛利", "毛利率", "營業利益", "營業利益率", "營業費用",
        "研發費用", "本期淨利", "淨利", "每股盈餘", "資本支出", "銷貨成本", "營業成本",
        "revenue", "gross profit", "gross margin", "operating income", "operating margin",
        "net income", "eps", "capital expenditure", "capex", "r&d"
    ]

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus

    def _tokenize(self, text: str) -> List[str]:
        text_lower = text.lower()
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+(?:\.\d+)?|\d+[,.]?\d*', text_lower)
        for term in self.FINANCIAL_TERMS:
            if term in text_lower:
                tokens.append(term)
        return [t for t in tokens if len(t) > 0]

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        score = 0.0
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        doc_set = set(doc_tokens)
        for token in query_tokens:
            if token in doc_set:
                tf = doc_tokens.count(token)
                score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / 50.0)))
        return score

    def _line_item_match_score(self, query: str, doc_content: str) -> float:
        q_lower = query.lower()
        content_lower = doc_content.lower()
        for item in self.FINANCIAL_TERMS:
            if item in q_lower and item in content_lower:
                return 1.5
        return 1.0

    def search(self, query: str, top_k: int = 5, exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
        exclude_ids = set(exclude_ids or [])
        query_tokens = self._tokenize(query)
        query_years = set(re.findall(r'20\d\d', query))
        query_quarters = set(re.findall(r'q[1-4]', query.lower()))

        scored_results = []
        for doc in self.corpus:
            if doc['id'] in exclude_ids:
                continue
            content = doc['content']
            doc_tokens = self._tokenize(content)
            bm25 = self._bm25_score(query_tokens, doc_tokens)
            overlap_count = sum(1 for q in query_tokens if q in content.lower())
            multiplier = 1.0
            multiplier *= self._line_item_match_score(query, content)

            doc_period = str(doc.get('period', '')) + " " + content
            if query_years:
                has_year_match = any(y in doc_period for y in query_years)
                if has_year_match:
                    multiplier *= 1.4
                else:
                    if any(y in doc_period for y in ['2021', '2022', '2023', '2024', '2025']):
                        multiplier *= 0.6
            if query_quarters:
                has_q_match = any(q in doc_period.lower() for q in query_quarters)
                if has_q_match:
                    multiplier *= 1.3
            final_score = (bm25 * 0.7 + overlap_count * 0.3) * multiplier
            if final_score > 0.01:
                scored_results.append((final_score, doc))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_results[:top_k]:
            doc_copy = dict(doc)
            doc_copy['relevance_score'] = round(float(score), 4)
            # Parent-Child expansion: if this is a child chunk,
            # attach parent_content so PoT reasoner gets richer context
            if doc_copy.get('is_child') and not doc_copy.get('parent_content'):
                # parent_content is already embedded in the passage dict by parser.py
                pass  # already present; nothing to do
            results.append(doc_copy)
        return results
