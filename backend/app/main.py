import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.rag.vector_store import FinancialVectorStoreManager
from app.rag.parser import FinancialFileParser
from app.agent.orchestrator import FinAgentRAGOrchestrator

app = FastAPI(title="FinAgent-RAG API (財報問答 Agentic RAG 系統)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = FinancialVectorStoreManager(load_sample_data=False)
file_parser = FinancialFileParser()
orchestrator = FinAgentRAGOrchestrator(vector_store=vector_store)

class ChatRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 3

class UploadDocumentRequest(BaseModel):
    company: str
    title: str
    content: str

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "indexed_documents": len(vector_store.corpus),
        "uploaded_files_count": len(vector_store.uploaded_files)
    }

@app.get("/api/sample-prompts")
def get_sample_prompts():
    return []

@app.get("/api/sample-data")
def get_sample_data():
    return []

@app.get("/api/uploaded-files")
def get_uploaded_files():
    return vector_store.uploaded_files

@app.post("/api/debug-pdf")
async def debug_pdf_endpoint(file: UploadFile = File(...)):
    """Debug endpoint: shows exactly what each PDF engine extracts, step by step."""
    import io, re
    try:
        content_bytes = await file.read()
        file_size = len(content_bytes)
        report = {"filename": file.filename, "file_size_bytes": file_size, "engines": {}}

        def count_alnum(text: str) -> int:
            return len(re.findall(r'[A-Za-z0-9\u4e00-\u9fff]', text))

        # Engine 1: fitz
        try:
            import fitz
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            pages = []
            for i, page in enumerate(doc, 1):
                t = page.get_text("text") or ""
                pages.append({"page": i, "chars": len(t), "alnum": count_alnum(t), "preview": t[:80]})
            report["engines"]["fitz"] = {"num_pages": len(doc), "pages": pages[:5], "total_alnum": sum(p["alnum"] for p in pages)}
            doc.close()
        except Exception as e:
            report["engines"]["fitz"] = {"error": str(e)}

        # Engine 2: pdfplumber
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    t = page.extract_text() or ""
                    pages.append({"page": i, "chars": len(t), "alnum": count_alnum(t), "preview": t[:80]})
            report["engines"]["pdfplumber"] = {"num_pages": len(pages), "pages": pages[:5], "total_alnum": sum(p["alnum"] for p in pages)}
        except Exception as e:
            report["engines"]["pdfplumber"] = {"error": str(e)}

        # Engine 3: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages = []
            for i, page in enumerate(reader.pages, 1):
                t = page.extract_text() or ""
                pages.append({"page": i, "chars": len(t), "alnum": count_alnum(t), "preview": t[:80]})
            report["engines"]["pypdf"] = {"num_pages": len(reader.pages), "pages": pages[:5], "total_alnum": sum(p["alnum"] for p in pages)}
        except Exception as e:
            report["engines"]["pypdf"] = {"error": str(e)}

        # Engine 4: pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            t = pdfminer_extract_text(io.BytesIO(content_bytes)) or ""
            report["engines"]["pdfminer"] = {"chars": len(t), "alnum": count_alnum(t), "preview": t[:200]}
        except Exception as e:
            report["engines"]["pdfminer"] = {"error": str(e)}

        # Full parse result
        try:
            result = file_parser.parse_file(file.filename, content_bytes)
            report["parse_result"] = {
                "total_passages": result["total_passages"],
                "warning": result.get("warning"),
                "passages_preview": [p["content"][:100] for p in result["passages"][:3]]
            }
        except Exception as e:
            report["parse_result"] = {"error": str(e)}

        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...), company: Optional[str] = Form(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")

    try:
        content_bytes = await file.read()
        c_name = company or os.path.splitext(file.filename)[0]

        parsed_res = file_parser.parse_file(file.filename, content_bytes)

        warning = parsed_res.get("warning")

        # If the PDF is scanned/unreadable, do NOT add the fallback notice to the corpus.
        # Return a clear 422 so the frontend shows a proper warning message.
        if parsed_res["total_passages"] == 0:
            return {
                "status": "warning",
                "filename": file.filename,
                "company": c_name,
                "passages_added": 0,
                "total_indexed": len(vector_store.corpus),
                "warning": warning or (
                    f"'{file.filename}' 無法提取有效文字內容，新增 0 個 Chunk。"
                    "請確認檔案格式：可搜尋的 PDF、CSV 或 TXT 效果最佳。"
                ),
            }

        vector_store.add_parsed_passages(
            filename=file.filename,
            company=c_name,
            passages=parsed_res["passages"]
        )

        resp = {
            "status": "success",
            "filename": file.filename,
            "company": c_name,
            "passages_added": parsed_res["total_passages"],
            "total_indexed": len(vector_store.corpus),
        }
        if warning:
            resp["warning"] = warning
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        result = orchestrator.process_query(query=req.query, max_iterations=req.max_iterations)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
def upload_document(req: UploadDocumentRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    doc_id = vector_store.add_custom_document(company=req.company, title=req.title, text=req.content)
    return {"status": "success", "doc_id": doc_id, "total_indexed": len(vector_store.corpus)}
