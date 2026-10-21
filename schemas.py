from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: user, assistant, or system")
    content: str = Field(..., description="Message text content")
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional source citations for assistant messages"
    )


class SourceCitation(BaseModel):
    source: str = Field(..., description="Filename or identifier of the source document")
    page: Union[int, str] = Field(..., description="Page number where evidence was found")


class RetrievedChunk(BaseModel):
    chunk_id: Optional[int] = Field(None, description="Index of the chunk in the document")
    document_id: Optional[str] = Field(None, description="Hash ID of the document")
    source: str = Field(..., description="Source document name")
    page: Union[int, str] = Field(..., description="Page number of the chunk")
    score: float = Field(..., description="Cosine similarity score")
    text: str = Field(..., description="Text content of the retrieved chunk")
    word_count: Optional[int] = Field(None, description="Number of words in this chunk")


class DocumentStatus(BaseModel):
    is_indexed: bool = Field(..., description="Whether a document is currently indexed and ready")
    document_id: Optional[str] = Field(None, description="SHA256 hash of active document")
    document_name: Optional[str] = Field(None, description="Filename of active document")
    pages_count: int = Field(0, description="Total pages extracted")
    chunks_count: int = Field(0, description="Total text chunks created")
    vectors_count: int = Field(0, description="Total vectors indexed in FAISS")
    chunk_size: Optional[int] = Field(None, description="Configured chunk size in words")
    overlap: Optional[int] = Field(None, description="Configured overlap in words")


class ProcessPdfResponse(BaseModel):
    message: str
    document_id: str
    document_name: str
    pages_count: int
    chunks_count: int
    indexed_vectors: int


class ProcessLocalPdfRequest(BaseModel):
    file_path: str = Field("Finance_Bill.pdf", description="Local relative or absolute path to the PDF file")
    chunk_size: int = Field(400, ge=50, le=2000, description="Chunk size in words")
    overlap: int = Field(50, ge=0, description="Chunk overlap in words (must be less than chunk_size)")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using the indexed document")
    top_k: int = Field(3, ge=1, le=10, description="Maximum number of relevant chunks to retrieve")
    min_score: float = Field(0.30, ge=-1.0, le=1.0, description="Minimum similarity score threshold")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Previous conversation turns for contextual answering"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunks: List[RetrievedChunk]
    execution_time_seconds: Optional[float] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(3, ge=1, le=20, description="Maximum number of chunks to return")
    min_score: float = Field(0.30, ge=-1.0, le=1.0, description="Minimum similarity score threshold")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[RetrievedChunk]


class EvaluationMetricsResponse(BaseModel):
    report_available: bool
    total_questions: int = 0
    hit_rate: Optional[float] = None
    mean_precision: Optional[float] = None
    mean_recall: Optional[float] = None
    rejection_rate: Optional[float] = None
    exact_match_rate: Optional[float] = None
    mean_faithfulness: Optional[float] = None
    questions: List[Dict[str, Any]] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    document_indexed: bool
    active_document: Optional[str] = None
