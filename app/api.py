"""
Tender Intelligence API
FastAPI REST backend — wraps the RAG pipeline with HTTP endpoints.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_pipeline import (
    ingest_and_index,
    ingest_directory,
    load_vector_store,
    build_qa_chain,
    extract_tender_intelligence,
    ask_question,
    FAISS_INDEX_PATH,
)


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tender Intelligence API",
    description=(
        "LLM-powered document intelligence for tunnelling tender analysis. "
        "Upload tender documents, run structured extraction, and ask free-form questions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state — in production this would be a database
_ingested_files: list[dict] = []
_vector_store = None
_qa_chain = None


def _get_chain():
    """Load or return cached QA chain."""
    global _vector_store, _qa_chain

    if _qa_chain is None:
        vs = load_vector_store()
        if vs is None:
            raise HTTPException(
                status_code=400,
                detail="No documents ingested yet. Upload a tender document first.",
            )
        _vector_store = vs
        _qa_chain = build_qa_chain(_vector_store)

    return _qa_chain


def _reset_chain():
    """Force chain rebuild after new document ingestion."""
    global _qa_chain
    _qa_chain = None


# ── Request / Response models ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    filename: str
    ingested_at: str
    extraction: dict
    summary: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    source_chunks: list
    model: str = "gpt-3.5-turbo"

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    index_exists: bool
    version: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check — confirms API is running and index status."""
    index_exists = Path(FAISS_INDEX_PATH).exists()
    return HealthResponse(
        status="ok",
        documents_loaded=len(_ingested_files),
        index_exists=index_exists,
        version="1.0.0",
    )


@app.post("/ingest", tags=["Documents"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and ingest a tender document (.txt or .pdf).

    Pipeline:
    1. Save uploaded file to temp location
    2. Load and parse document
    3. Chunk into 512-token segments
    4. Embed with OpenAI text-embedding-3-small
    5. Store in FAISS vector index

    Returns chunk count and metadata.
    """
    allowed = {".txt", ".pdf"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Accepted: {allowed}",
        )

    # Save to uploads dir
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)

    try:
        vs = ingest_and_index(str(dest), rebuild=False)
        _reset_chain()

        record = {
            "filename": file.filename,
            "ingested_at": datetime.utcnow().isoformat(),
            "size_bytes": len(content),
        }
        _ingested_files.append(record)

        return {
            "status": "ingested",
            "filename": file.filename,
            "ingested_at": record["ingested_at"],
            "message": "Document embedded and added to FAISS index.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/sample", tags=["Documents"])
def ingest_sample_documents():
    """
    Ingest the included sample tunnelling tender documents.
    Use this to quickly demo the system without uploading your own files.
    """
    sample_dir = "data/sample_tenders"

    if not Path(sample_dir).exists():
        raise HTTPException(status_code=404, detail="Sample tender directory not found.")

    try:
        vs = ingest_directory(sample_dir, rebuild=True)
        _reset_chain()

        files = list(Path(sample_dir).glob("*.txt")) + list(Path(sample_dir).glob("*.pdf"))
        for f in files:
            _ingested_files.append({
                "filename": f.name,
                "ingested_at": datetime.utcnow().isoformat(),
                "size_bytes": f.stat().st_size,
            })

        return {
            "status": "ingested",
            "documents": [f.name for f in files],
            "message": f"Successfully ingested {len(files)} sample tender documents.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all ingested documents."""
    return {
        "count": len(_ingested_files),
        "documents": _ingested_files,
    }


@app.post("/analyse", tags=["Analysis"])
def analyse_document():
    """
    Run structured extraction across all ingested tender documents.

    Extracts 14 key fields including:
    - Project name, client, contract reference
    - Contract value, duration, start date
    - Scope summary, key requirements
    - Evaluation criteria and weightings
    - Key risks and BoQ total

    Returns a structured JSON report.
    """
    chain = _get_chain()

    try:
        extraction = extract_tender_intelligence(chain)

        # Build a plain-English summary
        name = extraction.get("project_name", {}).get("value", "Unknown project")
        client = extraction.get("client", {}).get("value", "Unknown client")
        value = extraction.get("contract_value", {}).get("value", "Not specified")
        duration = extraction.get("contract_duration", {}).get("value", "Not specified")

        summary = (
            f"{name} for {client}. "
            f"Estimated contract value: {value}. "
            f"Duration: {duration}."
        )

        return {
            "status": "complete",
            "analysed_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "extraction": extraction,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse, tags=["Analysis"])
def ask(request: QuestionRequest):
    """
    Ask a free-form question about the ingested tender documents.

    The RAG pipeline:
    1. Embeds your question
    2. Retrieves the 6 most semantically similar chunks
    3. Passes them to GPT-3.5 with a structured prompt
    4. Returns the answer with source references

    Example questions:
    - "What are the ground conditions for this project?"
    - "What is the programme for tunnel breakthrough?"
    - "What insurance requirements are specified?"
    - "Summarise the key risks."
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    chain = _get_chain()

    try:
        result = ask_question(chain, request.question)
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            source_chunks=result["source_chunks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["System"])
def clear_index():
    """
    Clear the FAISS vector index and all ingested document records.
    Use to start fresh with new documents.
    """
    global _ingested_files, _qa_chain, _vector_store

    index_path = Path(FAISS_INDEX_PATH)
    if index_path.exists():
        shutil.rmtree(index_path)

    uploads_path = Path("data/uploads")
    if uploads_path.exists():
        shutil.rmtree(uploads_path)

    _ingested_files = []
    _qa_chain = None
    _vector_store = None

    return {"status": "cleared", "message": "FAISS index and uploaded documents removed."}
