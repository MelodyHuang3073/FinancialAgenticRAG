"""
End-to-end PDF parse diagnostic.
Uploads a realistic searchable financial PDF to /api/debug-pdf and /api/upload-file.
Also downloads a real AMCOR-style 10-K PDF from SEC for testing if available.
"""
import sys, io, re, json, time, requests
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SERVER = "http://localhost:8000"

def count_alnum(text):
    return len(re.findall(r'[A-Za-z0-9\u4e00-\u9fff]', text or ''))

# ---- Build a multi-page PDF with real financial content ----
import fitz

doc = fitz.open()
pages_content = [
    # Page 1: Income Statement
    (
        "AMCOR PLC  Annual Report on Form 10-K  Fiscal Year Ended June 30, 2023\n\n"
        "CONSOLIDATED STATEMENTS OF INCOME\n"
        "($ millions, except per share data)               FY2023    FY2022\n"
        "Net sales                                         14,694    14,544\n"
        "Cost of sales                                    (11,328)  (11,229)\n"
        "Gross profit                                       3,366     3,315\n"
        "Selling, general and administrative expenses       (857)     (888)\n"
        "Research and development expenses                  (107)     (103)\n"
        "Other income, net                                    32        38\n"
        "Earnings before interest and taxes                 2,434     2,362\n"
        "Interest expense, net                              (296)     (228)\n"
        "Income before income taxes and equity              2,138     2,134\n"
        "Income tax expense                                 (393)     (403)\n"
        "Net income                                         1,048       878\n"
        "Net income attributable to Amcor plc shareholders    993       845\n"
    ),
    # Page 2: Balance Sheet
    (
        "AMCOR PLC  CONSOLIDATED BALANCE SHEETS\n"
        "($ millions)                                      FY2023    FY2022\n"
        "ASSETS\n"
        "Current assets:\n"
        "  Cash and cash equivalents                          627       701\n"
        "  Trade receivables, net                           1,521     1,402\n"
        "  Inventories                                        789       750\n"
        "  Prepaid expenses and other current assets          308       247\n"
        "Total current assets                               3,245     3,100\n"
        "Non-current assets:\n"
        "  Property, plant and equipment, net               3,874     3,876\n"
        "  Goodwill                                         5,528     5,597\n"
        "  Other intangible assets, net                     1,342     1,416\n"
        "  Other non-current assets                         2,258     2,196\n"
        "Total assets                                      16,247    16,185\n"
        "LIABILITIES\n"
        "Current liabilities:\n"
        "  Short-term debt                                    263       303\n"
        "  Trade payables                                   1,538     1,497\n"
        "  Other current liabilities                          299       250\n"
        "Total current liabilities                          2,100     2,050\n"
        "Total liabilities                                  9,783     9,942\n"
        "Total equity                                       6,464     6,243\n"
    ),
    # Page 3: Cash Flow Statement
    (
        "AMCOR PLC  CONSOLIDATED STATEMENTS OF CASH FLOWS\n"
        "($ millions)                                      FY2023    FY2022\n"
        "Cash flows from operating activities:\n"
        "  Net income                                       1,048       878\n"
        "  Depreciation and amortization                      626       626\n"
        "  Share-based compensation expense                    59        58\n"
        "  Changes in working capital                        (273)      (86)\n"
        "Net cash from operating activities                 1,460     1,476\n"
        "Cash flows from investing activities:\n"
        "  Capital expenditure                               (591)     (510)\n"
        "  Acquisitions, net of cash                         (95)     (236)\n"
        "Net cash used in investing activities               (686)     (746)\n"
        "Cash flows from financing activities:\n"
        "  Dividends paid                                    (742)     (744)\n"
        "  Share buybacks                                    (150)     (200)\n"
        "  Net repayment of debt                              (89)     (148)\n"
        "Net cash used in financing activities               (981)   (1,092)\n"
        "Net change in cash                                  (207)     (362)\n"
        "Cash at end of period                                627       701\n"
    ),
]

for page_text in pages_content:
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 40), page_text, fontsize=9)

pdf_bytes = doc.write()
doc.close()

print(f"Built test PDF: {len(pdf_bytes)} bytes, {len(pages_content)} pages")
print()

# ---- Check if server is running ----
print("=" * 60)
print("Checking server...")
try:
    r = requests.get(f"{SERVER}/api/health", timeout=5)
    print(f"Server health: {r.json()}")
    server_up = True
except Exception as e:
    print(f"Server NOT running: {e}")
    server_up = False

# ---- Run /api/debug-pdf ----
if server_up:
    print()
    print("=" * 60)
    print("Uploading to /api/debug-pdf ...")
    try:
        r = requests.post(
            f"{SERVER}/api/debug-pdf",
            files={'file': ('AMCOR_debug_test.pdf', pdf_bytes, 'application/pdf')},
            timeout=60
        )
        data = r.json()
        print(f"Status: {r.status_code}")
        for engine_name, engine_data in data.get("engines", {}).items():
            if "error" in engine_data:
                print(f"  {engine_name}: ERROR - {engine_data['error']}")
            else:
                total_alnum = engine_data.get("total_alnum", "N/A")
                num_pages = engine_data.get("num_pages", engine_data.get("chars", "?"))
                print(f"  {engine_name}: pages={num_pages}, total_alnum={total_alnum}")
                if engine_name == "fitz":
                    for pg in engine_data.get("pages", [])[:3]:
                        print(f"    Page {pg['page']}: alnum={pg['alnum']}, preview={repr(pg['preview'][:50])}")

        pr = data.get("parse_result", {})
        print(f"\nparse_result: total_passages={pr.get('total_passages')}, warning={pr.get('warning')}")
        if pr.get('passages_preview'):
            for i, prev in enumerate(pr['passages_preview'][:2], 1):
                print(f"  chunk {i}: {prev[:80]}")
    except Exception as e:
        print(f"debug-pdf FAILED: {e}")

    # ---- Run /api/upload-file ----
    print()
    print("=" * 60)
    print("Uploading to /api/upload-file ...")
    try:
        r = requests.post(
            f"{SERVER}/api/upload-file",
            files={'file': ('AMCOR_debug_test.pdf', pdf_bytes, 'application/pdf')},
            data={'company': 'AMCOR_debug_test'},
            timeout=60
        )
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(data, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"upload-file FAILED: {e}")

# ---- Direct Python parser test (no server) ----
print()
print("=" * 60)
print("Direct parser test (no server)...")
from app.rag.parser import FinancialFileParser
parser = FinancialFileParser()
result = parser.parse_file("AMCOR_direct_test.pdf", pdf_bytes)
print(f"total_passages: {result['total_passages']}")
print(f"warning: {result.get('warning')}")
for i, p in enumerate(result['passages'][:5], 1):
    print(f"  [{i}] id={p['id']}, page={p.get('page_number')}")
    print(f"       content[:100]: {p['content'][:100]}")

print()
print("DONE.")
