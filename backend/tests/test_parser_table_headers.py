"""FinancialFileParser table-header helpers (app/rag/parser.py): the single-
unit-per-page detector, splitting a word-only header line into column names,
and Tier 3 (layout-text) keeping a table's internal sub-headings inside one
block while still finding its year header."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag.parser import FinancialFileParser


def test_page_unit_only_when_single_unit():
    p = FinancialFileParser()
    assert p._page_unit("(in millions, except per share) 2022 2021") == "USD millions"
    assert p._page_unit("$ million 2022 $ million 2023") == "USD millions"
    assert p._page_unit("no unit stated anywhere") == ""
    assert p._page_unit("(in millions) and also in billions") == ""


def test_text_header_split_into_columns():
    names = FinancialFileParser._split_text_header(
        "By Business Segment Organic sales Acquisitions Divestitures Translation Total sales change", 5
    )
    assert names == ["Organic sales", "Acquisitions", "Divestitures", "Translation", "Total sales change"]
    assert FinancialFileParser._split_text_header("Net sales for 2022 were higher", 3) == []


def test_tier3_keeps_subheadings_inside_one_table_with_year_header():
    layout = "\n".join([
        "A reconciliation of operating income is as follows:",
        "   (dollars in millions)",
        "   Years Ended December 31,      2021     2020     2019",
        "   Operating Income",
        "   Total reportable segments     $ 33,392 $ 32,629 $ 32,723",
        "   Corporate and other              (449)  (1,472)  (1,403)",
        "   Reconciling items:",
        "     Severance charges              (209)    (221)    (204)",
        "     Other components of pension charges (Note 11) (769) (817) (813)",
        "   Consolidated operating income  32,448   28,798   30,378",
    ])
    tables, _ = FinancialFileParser()._layout_text_to_markdown_and_prose(layout)
    assert len(tables) == 1
    assert tables[0].splitlines()[0] == "| Line Item | 2021 | 2020 | 2019 |"
    assert "Other components of pension charges (Note 11)" in tables[0]
