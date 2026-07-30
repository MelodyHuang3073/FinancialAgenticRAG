"""
End-to-end upload pipeline diagnostic script.
Simulates exactly what happens when a user uploads a file via the frontend.
"""
import sys
import io

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("STEP 1: Import Check")
print("=" * 60)

try:
    from app.rag.parser import FinancialFileParser
    print("[OK] FinancialFileParser imported")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] parser import: {e}")
    sys.exit(1)

try:
    from app.rag.vector_store import FinancialVectorStoreManager
    print("[OK] FinancialVectorStoreManager imported")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] vector_store import: {e}")
    sys.exit(1)

try:
    from app.rag.chunker import chunk_text
    print("[OK] chunk_text imported")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] chunker import: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: TXT file parse test")
print("=" * 60)

txt_bytes = b"Net Revenue 2022: 34,229 million. Net Revenue 2021: 35,355 million. Operating Income 2022: 4,792 million. Total Assets: 46,006 million."
parser = FinancialFileParser()
try:
    result = parser.parse_file("3M_2022_10K.txt", txt_bytes)
    print(f"[OK] total_passages = {result['total_passages']}")
    for i, p in enumerate(result['passages']):
        print(f"     passage[{i}] content[:100]: {p['content'][:100]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] TXT parse: {e}")

print()
print("=" * 60)
print("STEP 3: CSV file parse test")
print("=" * 60)

csv_bytes = b"Year,Net Revenue,Operating Income\n2022,34229,4792\n2021,35355,6893\n"
try:
    result = parser.parse_file("3M_2022_10K.csv", csv_bytes)
    print(f"[OK] total_passages = {result['total_passages']}")
    for i, p in enumerate(result['passages']):
        print(f"     passage[{i}] content[:120]: {p['content'][:120]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] CSV parse: {e}")

print()
print("=" * 60)
print("STEP 4: PDF file parse test (using fitz)")
print("=" * 60)

try:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "3M Company 2022 Annual Report\nNet Revenue 2022: $34,229 million\nNet Revenue 2021: $35,355 million\nOperating Income: $4,792 million")
    pdf_bytes = doc.write()
    doc.close()
    result = parser.parse_file("3M_2022_10K.pdf", pdf_bytes)
    print(f"[OK] total_passages = {result['total_passages']}")
    for i, p in enumerate(result['passages']):
        print(f"     passage[{i}] content[:120]: {p['content'][:120]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] PDF parse: {e}")

print()
print("=" * 60)
print("STEP 5: VectorStore add_parsed_passages test")
print("=" * 60)

try:
    vs = FinancialVectorStoreManager(load_sample_data=False)
    print(f"[OK] VectorStore created, corpus size BEFORE: {len(vs.corpus)}")
    
    txt_bytes = b"Net Revenue 2022: 34,229 million. Operating Income 2022: 4,792 million."
    res = parser.parse_file("3M_2022_10K.txt", txt_bytes)
    print(f"[OK] parse_file returned passages: {res['total_passages']}")
    
    vs.add_parsed_passages("3M_2022_10K.txt", "3M_2022_10K", res['passages'])
    print(f"[OK] corpus size AFTER: {len(vs.corpus)}")
    print(f"[OK] uploaded_files: {vs.uploaded_files}")
    print(f"[OK] retriever is set: {vs.retriever is not None}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] VectorStore: {e}")

print()
print("=" * 60)
print("STEP 6: chunk_text behavior test")
print("=" * 60)

sample = "Net Revenue 2022: 34229 million."
result = chunk_text(sample, chunk_size=800, overlap=120, min_chunk_size=300)
print(f"[OK] chunk_text(short text): {len(result)} chunks")

long_text = "A" * 3000
result = chunk_text(long_text, chunk_size=800, overlap=120, min_chunk_size=300)
print(f"[OK] chunk_text(3000 chars): {len(result)} chunks")

# The critical test - what chunk_size is used for PDF?
# parser.py line 164: chunks = chunk_text(text_content, chunk_size=800, overlap=120, min_chunk_size=300)
# Does min_chunk_size=300 cause the entire passage to be dropped if text < 300 chars?
short_pdf_text = "Net Revenue 2022: 34229"
result = chunk_text(short_pdf_text, chunk_size=800, overlap=120, min_chunk_size=300)
print(f"[OK] chunk_text(short <300 chars text): {len(result)} chunks -> {result}")

print()
print("DONE - Diagnostics Complete")
