"""A page whose text layer is garbled by a broken font encoding is recognised, ordinary pages are not,
and a page MinerU cannot read keeps its passages. Texts are synthetic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag import table_repair
from app.rag.table_repair import find_garbled_pages, repair_garbled_pages


def _note(page, text):
    return {"type": "text_note", "page_number": page, "content": f"Company: X | Page: {page} | Content: {text}"}


GARBLED = " ".join(["(;J ?D9EC; 9;J A7<H@ 5C?AB8 =E9H ;I@@ G;DE"] * 8)
NORMAL = " ".join(["Net income for the year was higher than the prior year because revenue grew"] * 6)


def test_garbled_page_is_found_and_a_normal_page_is_not():
    passages = [_note(1, GARBLED), _note(2, NORMAL)]
    assert find_garbled_pages(passages) == {1}


def test_a_short_page_is_never_called_garbled():
    assert find_garbled_pages([_note(1, "(;J ?D9EC; 9;J")]) == set()


def test_numbers_and_ordinary_symbols_do_not_make_a_page_garbled():
    text = " ".join(["Revenue (in millions) $4,684 $5,140 2025 2024 Q3 10-K see Note 5; also (a) total"] * 8)
    assert find_garbled_pages([_note(1, text)]) == set()


def test_a_page_mineru_cannot_read_keeps_its_passages(monkeypatch):
    passages = [_note(1, GARBLED)]
    monkeypatch.setattr(table_repair, "_mineru_available", lambda: True)
    monkeypatch.setattr(table_repair, "_extract_single_page_pdf", lambda *a, **k: True)
    monkeypatch.setattr(table_repair, "_mineru_page_text", lambda path: None)
    assert repair_garbled_pages(object(), passages, b"", "f.pdf", "X") is passages
