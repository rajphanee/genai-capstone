"""
api.py - REST API (Task 2) built with FastAPI.

Run:
    uvicorn src.api:app --reload --port 8000
Then open http://localhost:8000/docs for the interactive Swagger page.

Endpoints:
    GET    /health                  is the service up? which LLM / embeddings?
    POST   /documents               upload one or more files (multipart form, field "files")
    GET    /documents               list indexed documents
    DELETE /documents/{doc_id}      remove a document and its chunks
    POST   /ask                     {"question": "...", "doc_ids": ["optional", ...]}

Error contract (every error has the same JSON shape):
    {"error": {"code": "UNSUPPORTED_FILE", "message": "..."}}
    4xx = the caller's problem (bad file, bad question), 5xx = our problem.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets "src" import without pip install

from src.errors import DocQAError  # noqa: E402
from src.service import DocQAService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Enterprise Document Q&A (RAG + Agents)", version="1.0.0")

_service = None


def service() -> DocQAService:
    """Create the service on first use (so importing this file never fails on a bad .env)."""
    global _service
    if _service is None:
        _service = DocQAService.create()
    return _service


# ------------------------------------------------------------------ error handlers
@app.exception_handler(DocQAError)
async def platform_error(_request: Request, exc: DocQAError):
    return JSONResponse(status_code=exc.http_status, content={"error": {"code": exc.code, "message": str(exc)}})


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception):
    log.exception("unexpected error")       # full traceback goes to the server log only
    return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR",
                                                            "message": "Something went wrong. Check the server log."}})


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question", examples=["How many PTO days do I get?"])
    doc_ids: Optional[List[str]] = Field(None, description="Only search these documents (optional)")


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {"status": "ok", **service().info()}


@app.post("/documents")
async def upload(files: List[UploadFile] = File(...)):
    """Upload several files at once. One bad file does not stop the others."""
    results = []
    for f in files:
        try:
            data = await f.read()
            results.append({"filename": f.filename, "ok": True, **service().ingest(f.filename, data)})
        except DocQAError as exc:
            results.append({"filename": f.filename, "ok": False, "error": {"code": exc.code, "message": str(exc)}})
    status = 200 if any(r["ok"] for r in results) else 422
    return JSONResponse(status_code=status, content={"results": results})


@app.get("/documents")
def list_documents():
    return {"documents": service().list_documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    service().delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/ask")
def ask(body: AskRequest):
    return service().ask(body.question, doc_ids=body.doc_ids)
