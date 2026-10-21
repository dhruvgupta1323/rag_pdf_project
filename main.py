import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from schemas import (
    DocumentStatus,
    EvaluationMetricsResponse,
    HealthResponse,
    ProcessLocalPdfRequest,
    ProcessPdfResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from services import REPORT_PATH, rag_service

app = FastAPI(
    title="Custom RAG Knowledge Assistant API",
    description=(
        "FastAPI backend for document-grounded Question Answering using "
        "PyMuPDF, Sentence Transformers, FAISS, and Google Gemini."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend applications (Streamlit, React, Vue, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SYSTEM & HEALTH ENDPOINTS
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="Root Welcome & API Overview",
)
def root():
    return {
        "message": "Welcome to Custom RAG Assistant API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
)
def health_check():
    status_info = rag_service.get_status()
    return HealthResponse(
        status="healthy",
        service="rag-fastapi-backend",
        version="1.0.0",
        document_indexed=status_info["is_indexed"],
        active_document=status_info["document_name"],
    )


# ---------------------------------------------------------------------------
# DOCUMENT MANAGEMENT ENDPOINTS
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/documents/status",
    response_model=DocumentStatus,
    tags=["Documents"],
    summary="Get Active Document Status",
)
def get_document_status():
    """
    Returns the indexing status of the currently loaded PDF document.
    """
    status_info = rag_service.get_status()
    return DocumentStatus(**status_info)


@app.post(
    "/api/v1/documents/upload",
    response_model=ProcessPdfResponse,
    tags=["Documents"],
    summary="Upload and Index PDF Document",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to be indexed"),
    chunk_size: int = Form(400, ge=50, le=2000, description="Word count per chunk"),
    overlap: int = Form(50, ge=0, description="Word overlap between chunks"),
):
    """
    Uploads a PDF, extracts text, chunks it, generates embeddings,
    and builds an in-memory FAISS index.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .pdf extension.",
        )

    if overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Overlap ({overlap}) must be less than chunk_size ({chunk_size}).",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty.",
            )

        result = rag_service.process_pdf(
            pdf_input=content,
            filename=file.filename,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return ProcessPdfResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the PDF: {str(exc)}",
        )


@app.post(
    "/api/v1/documents/process-local",
    response_model=ProcessPdfResponse,
    tags=["Documents"],
    summary="Index an existing local PDF file",
)
def process_local_pdf(request: ProcessLocalPdfRequest):
    """
    Indexes a local PDF file already present on the server (e.g. Finance_Bill.pdf).
    """
    pdf_path = Path(request.file_path)
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Local PDF file not found at path: {request.file_path}",
        )

    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Overlap ({request.overlap}) must be less than chunk_size ({request.chunk_size}).",
        )

    try:
        result = rag_service.process_pdf(
            pdf_input=pdf_path,
            filename=pdf_path.name,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
        return ProcessPdfResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process local PDF: {str(exc)}",
        )


@app.delete(
    "/api/v1/documents",
    tags=["Documents"],
    summary="Clear/Unload Active Document",
)
def clear_active_document():
    """
    Unloads the current active document and its FAISS index from memory.
    """
    rag_service.clear()
    return {"message": "Active document has been cleared from memory."}


# ---------------------------------------------------------------------------
# SEARCH & QUERY ENDPOINTS
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search & Retrieval"],
    summary="Retrieve Relevant Chunks without LLM Generation",
)
def search_document(request: SearchRequest):
    """
    Directly performs FAISS semantic search against indexed chunks and
    returns relevant excerpts with cosine similarity scores.
    """
    if not rag_service.is_indexed():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been indexed yet. Please process a PDF document first.",
        )

    try:
        raw_results = rag_service.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
        )

        formatted_results = []
        for r in raw_results:
            chunk = r.get("chunk", {})
            formatted_results.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", "Unknown"),
                "score": float(r.get("score", 0.0)),
                "text": chunk.get("text", ""),
                "word_count": chunk.get("word_count"),
            })

        return SearchResponse(
            query=request.query,
            total_results=len(formatted_results),
            results=formatted_results,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(exc)}",
        )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Search & Retrieval"],
    summary="Ask Question & Generate Grounded Answer",
)
def query_document(request: QueryRequest):
    """
    Retrieves the most relevant passages from the active document and passes
    them to Google Gemini to produce a citation-grounded response.
    """
    if not rag_service.is_indexed():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been indexed yet. Please process a PDF document first.",
        )

    start_time = time.time()

    try:
        history_dicts = (
            [msg.model_dump() for msg in request.conversation_history]
            if request.conversation_history
            else []
        )

        answer, sources, chunks = rag_service.answer(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            conversation_history=history_dicts,
        )

        elapsed = round(time.time() - start_time, 3)

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            execution_time_seconds=elapsed,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query answering failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# EVALUATION & METRICS ENDPOINTS
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/evaluation",
    response_model=EvaluationMetricsResponse,
    tags=["Evaluation"],
    summary="Get Evaluation Report & Performance Metrics",
)
def get_evaluation():
    """
    Returns aggregated RAG evaluation metrics (hit rate, precision, recall,
    correct rejection rate, faithfulness score) and per-question results.
    """
    metrics = rag_service.get_evaluation_metrics()
    return EvaluationMetricsResponse(**metrics)


@app.get(
    "/api/v1/evaluation/dataset",
    tags=["Evaluation"],
    summary="Get Evaluation Dataset Test Cases",
)
def get_evaluation_dataset():
    """
    Returns the test question dataset used for evaluation benchmarking.
    """
    dataset_file = Path("evaluation_dataset.json")
    if not dataset_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation dataset file (evaluation_dataset.json) not found.",
        )
    import json
    with open(dataset_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get(
    "/api/v1/evaluation/download",
    tags=["Evaluation"],
    summary="Download Evaluation CSV Report",
)
def download_evaluation_report():
    """
    Downloads the rag_evaluation_report.csv file directly.
    """
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation report file (rag_evaluation_report.csv) was not found.",
        )

    return FileResponse(
        path=REPORT_PATH,
        filename="rag_evaluation_report.csv",
        media_type="text/csv",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
