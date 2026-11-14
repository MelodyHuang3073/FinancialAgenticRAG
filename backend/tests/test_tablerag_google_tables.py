"""Offline tests of the long-table builder for the Google TableRAG baseline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines.tablerag_google.build_tables import COLUMNS, company_for, long_table, page_unit, parse_number


def test_parse_number():
    assert parse_number("$5,363") == 5363.0
    assert parse_number("(370)") == -370.0
    assert parse_number("$(1,577)") == -1577.0
    assert parse_number("0.375%") == 0.375
    assert parse_number("n/a") is None


def test_page_unit():
    assert page_unit("Dollars in millions, except per share") == "millions"
    assert page_unit("(in thousands)") == "thousands"
    assert page_unit("no unit here") == ""


def _row(company, page, item, cells, parent=""):
    raw = {"line_item": item}
    raw.update(cells)
    return {"company": company, "type": "table_row", "page_number": str(page), "raw_data": raw, "parent_content": parent, "content": ""}


def test_long_table_one_row_per_cell_with_page_unit_and_default():
    corpus = [
        _row("A", 3, "Revenue", {"2020": "$1,000", "2019": "900"}, "Dollars in millions"),
        _row("A", 4, "Cost", {"2020": "(400)", "2019": ""}),                      # empty cell dropped; unit from default
        _row("B", 1, "Other company", {"2020": "5"}),
        {"company": "A", "type": "text_note", "page_number": "9", "content": "x", "parent_content": ""},
    ]
    rows = long_table(corpus, "A")
    assert rows[0] == COLUMNS
    assert rows[1:] == [[3, "Revenue", "2020", "$1,000", 1000.0, "millions"],
                        [3, "Revenue", "2019", "900", 900.0, "millions"],
                        [4, "Cost", "2020", "(400)", -400.0, "millions"]]


def test_company_for_tolerates_date_decoration():
    assert company_for("PEPSICO_2023_8K_dated-2023-05-30", ["PEPSICO_2023_8K_2023-05-30"]) == "PEPSICO_2023_8K_2023-05-30"
