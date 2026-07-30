"""
Complete end-to-end upload diagnostic.
Tests: parse_file → add_parsed_passages → HTTP upload endpoint
Prints every step with detailed error information.
"""
import sys, os, io, traceback
sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("=" * 60)
print("PHASE 1: Direct Python import tests")
print("=" * 60)

# 1. Import parser
try:
    from app.rag.parser import FinancialFileParser
    print("[OK] FinancialFileParser imported")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] parser import: {e}")
    sys.exit(1)

# 2. Import vector store
try:
    from app.rag.vector_store import FinancialVectorStoreManager
    print("[OK] FinancialVectorStoreManager imported")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] vector_store import: {e}")
    sys.exit(1)

# 3. Import chunker
try:
    from app.rag.chunker import chunk_text
    print("[OK] chunk_text imported")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] chunker import: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("PHASE 2: Parsing tests")
print("=" * 60)

parser = FinancialFileParser()

# Test TXT
txt_content = b"AMCOR FY2023 Annual Report\nCurrent Assets: 3,245 million\nInventory: 789 million\nCurrent Liabilities: 2,100 million\nQuick Ratio FY2023: (3245-789)/2100 = 1.17\nFY2022: Current Assets 3,100M Inventory 750M Current Liabilities 2,050M\nQuick Ratio FY2022: (3100-750)/2050 = 1.15\n"

try:
    result = parser.parse_file("AMCOR_FY2023.txt", txt_content)
    print(f"[OK] TXT parse: total_passages={result['total_passages']}")
    for p in result['passages']:
        print(f"     id={p['id']}, content[:80]={p['content'][:80]}")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] TXT parse: {e}")

# Test CSV
csv_content = b"Metric,FY2023,FY2022\nCurrent Assets,3245,3100\nInventory,789,750\nCurrent Liabilities,2100,2050\nQuick Ratio,1.17,1.15\n"

try:
    result = parser.parse_file("AMCOR_FY2023.csv", csv_content)
    print(f"[OK] CSV parse: total_passages={result['total_passages']}")
    for p in result['passages']:
        print(f"     id={p['id']}, content[:100]={p['content'][:100]}")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] CSV parse: {e}")

# Test PDF
try:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "AMCOR PLC Annual Report FY2023\n"
        "Balance Sheet Summary\n"
        "Current Assets: $3,245 million (FY2023) | $3,100 million (FY2022)\n"
        "Inventory: $789 million (FY2023) | $750 million (FY2022)\n"
        "Current Liabilities: $2,100 million (FY2023) | $2,050 million (FY2022)\n"
        "Quick Ratio: 1.17 (FY2023) | 1.15 (FY2022)\n"
    )
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()

    result = parser.parse_file("AMCOR_FY2023.pdf", pdf_bytes)
    print(f"[OK] PDF parse: total_passages={result['total_passages']}")
    for p in result['passages']:
        snippet = p['content'][:100]
        if 'scanned' in p['content'].lower() or 'unextractable' in p['content'].lower():
            print(f"     [WARN] Fallback scanned notice! content={snippet}")
        else:
            print(f"     id={p['id']}, content[:100]={snippet}")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] PDF parse: {e}")

print()
print("=" * 60)
print("PHASE 3: Vector store add_parsed_passages test")
print("=" * 60)

try:
    vs = FinancialVectorStoreManager(load_sample_data=False)
    print(f"[OK] VectorStore created. corpus size before: {len(vs.corpus)}")

    txt_content = b"AMCOR FY2023 Balance Sheet: Current Assets 3245M, Inventory 789M, Current Liabilities 2100M."
    res = parser.parse_file("AMCOR_FY2023.txt", txt_content)
    print(f"[OK] parse_file returned {res['total_passages']} passages")

    vs.add_parsed_passages("AMCOR_FY2023.txt", "AMCOR_FY2023", res['passages'])
    print(f"[OK] corpus size after: {len(vs.corpus)}")
    print(f"[OK] uploaded_files: {vs.uploaded_files}")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] VectorStore: {e}")

print()
print("=" * 60)
print("PHASE 4: HTTP endpoint test (live server on :8000)")
print("=" * 60)

try:
    import requests

    # Health check first
    r = requests.get("http://localhost:8000/api/health", timeout=5)
    h = r.json()
    print(f"[OK] Health: indexed_documents={h['indexed_documents']}, uploaded_files_count={h['uploaded_files_count']}")

    # Upload a TXT file
    txt_bytes = b"AMCOR FY2023 Annual Report\nCurrent Assets: 3245M\nInventory: 789M\nCurrent Liabilities: 2100M\nQuick Ratio FY2023: 1.17\nQuick Ratio FY2022: 1.15\n"
    files = {'file': ('AMCOR_FY2023.txt', txt_bytes, 'text/plain')}
    data = {'company': 'AMCOR_FY2023'}
    r2 = requests.post("http://localhost:8000/api/upload-file", files=files, data=data, timeout=15)
    resp = r2.json()
    print(f"[OK] HTTP upload status={r2.status_code}, response={resp}")

    if resp.get('passages_added', 0) == 0:
        print("[FAIL] passages_added is 0! Upload did not create chunks.")
    else:
        print(f"[OK] {resp['passages_added']} chunks added successfully!")

    # Health after upload
    r3 = requests.get("http://localhost:8000/api/health", timeout=5)
    h2 = r3.json()
    print(f"[OK] Health after upload: indexed_documents={h2['indexed_documents']}, uploaded_files_count={h2['uploaded_files_count']}")

except requests.exceptions.ConnectionError:
    print("[WARN] Backend is not running on :8000 — skipping HTTP test")
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] HTTP test: {e}")

print()
print("DONE.")
