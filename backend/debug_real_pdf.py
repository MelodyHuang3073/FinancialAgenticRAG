"""
Test parsing the actual 3M_2022_10K.pdf that the user uploads.
This finds the PDF wherever it is on the system, or uses the one from Downloads.
"""
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')

# Search for the 3M PDF on the system
search_roots = [
    r'C:\Users\y9205\Downloads',
    r'C:\Users\y9205\Desktop',
    r'C:\Users\y9205\financial_agentic_rag',
    r'C:\Users\y9205\Documents',
]

found_pdf = None
for root in search_roots:
    if os.path.exists(root):
        matches = glob.glob(os.path.join(root, '**', '3M*.pdf'), recursive=True)
        if matches:
            found_pdf = matches[0]
            break
        matches = glob.glob(os.path.join(root, '*.pdf'))
        if matches:
            found_pdf = matches[0]
            break

if not found_pdf:
    print("[WARN] Could not find a 3M_2022_10K.pdf on disk.")
    print("       Using a synthetic PDF for testing...")
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "3M Company Annual Report 2022\nNet Revenues: $34,229M\nOperating Income: $4,792M")
    pdf_bytes = doc.write()
    doc.close()
    source = "synthetic"
else:
    print(f"[OK] Found PDF: {found_pdf} ({os.path.getsize(found_pdf):,} bytes)")
    with open(found_pdf, 'rb') as f:
        pdf_bytes = f.read()
    source = found_pdf

print(f"PDF size: {len(pdf_bytes):,} bytes")

# Parse it
from app.rag.parser import FinancialFileParser
parser = FinancialFileParser()

result = parser.parse_file('3M_2022_10K.pdf', pdf_bytes)
print(f"total_passages: {result['total_passages']}")

if result['total_passages'] == 0:
    print("[FAIL] No passages generated from PDF!")
    print("       This is the root cause of '0 chunks' issue.")
elif result['total_passages'] == 1 and 'scanned' in result['passages'][0]['content']:
    print("[FAIL] Got only the scanned-PDF fallback message!")
    print("       fitz/pdfplumber/pypdf all failed to extract text.")
    print("       PDF may be truly image-only (scanned).")
else:
    print("[OK] Passages generated successfully!")
    for i, p in enumerate(result['passages'][:3]):
        print(f"  passage[{i}] id={p['id']} content[:100]={p['content'][:100]}")
