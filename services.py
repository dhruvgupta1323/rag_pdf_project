import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import pandas as pd

from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import build_vector_store, search_chunks
from rag.generator import generate_answer

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "rag_evaluation_report.csv"


class RAGService:
    """
    Manages active document state, indexing, chunk retrieval,
    and Gemini question answering.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.document_id: Optional[str] = None
        self.document_name: Optional[str] = None
        self.pages: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.index = None
        self.chunk_size: Optional[int] = None
        self.overlap: Optional[int] = None

    def is_indexed(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_indexed": self.is_indexed(),
                "document_id": self.document_id,
                "document_name": self.document_name,
                "pages_count": len(self.pages),
                "chunks_count": len(self.chunks),
                "vectors_count": self.index.ntotal if self.index is not None else 0,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            }

    def clear(self):
        with self._lock:
            self.document_id = None
            self.document_name = None
            self.pages = []
            self.chunks = []
            self.index = None
            self.chunk_size = None
            self.overlap = None

    def process_pdf(
        self,
        pdf_input: Any,
        filename: Optional[str] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> Dict[str, Any]:
        """
        Extracts, chunks, embeds, and indexes a PDF.
        Supports file paths, byte streams, and UploadFiles.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and less than chunk_size.")

        extracted_pages = extract_pdf(pdf_input, filename=filename)
        if not extracted_pages:
            raise ValueError("No pages were extracted from the PDF.")

        new_chunks = chunk_pages(
            extracted_pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not new_chunks:
            raise ValueError(
                "No readable text chunks could be extracted. The PDF might be scanned or image-based."
            )

        new_embeddings = embed_chunks(new_chunks)
        new_index = build_vector_store(new_chunks, new_embeddings)

        doc_name = filename or extracted_pages[0].get("source", "document.pdf")
        doc_id = extracted_pages[0]["document_id"]

        with self._lock:
            self.document_id = doc_id
            self.document_name = doc_name
            self.pages = extracted_pages
            self.chunks = new_chunks
            self.index = new_index
            self.chunk_size = chunk_size
            self.overlap = overlap

        return {
            "message": f"Successfully indexed '{doc_name}' with {len(new_chunks)} chunks from {len(extracted_pages)} pages.",
            "document_id": doc_id,
            "document_name": doc_name,
            "pages_count": len(extracted_pages),
            "chunks_count": len(new_chunks),
            "indexed_vectors": new_index.ntotal,
        }

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching chunks for a search query.
        """
        if not self.is_indexed():
            raise ValueError("No document has been indexed yet. Please process a PDF first.")

        return search_chunks(
            question=query,
            index=self.index,
            chunks=self.chunks,
            top_k=top_k,
            min_score=min_score,
        )

    def answer(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.30,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs retrieval and generates a grounded answer with citations.
        Returns (answer, sources, retrieved_chunks).
        """
        if not self.is_indexed():
            raise ValueError("No document has been indexed yet. Please process a PDF first.")

        history = conversation_history or []

        # Find previous user question for query contextualization if needed
        previous_question = next(
            (
                msg.get("content", "")
                for msg in reversed(history)
                if msg.get("role") == "user"
            ),
            "",
        )

        retrieval_query = (
            f"{previous_question}\n{question}"
            if previous_question
            else question
        )

        results = self.search(retrieval_query, top_k=top_k, min_score=min_score)

        # Generate answer from Gemini
        answer = generate_answer(
            question=question,
            results=results,
            conversation_history=history,
        )

        # Compile unique source citations
        sources: List[Dict[str, Any]] = []
        for result in results:
            chunk = result.get("chunk", {})
            source_item = {
                "source": chunk.get("source", "Unknown source"),
                "page": chunk.get("page", "Unknown"),
            }
            if source_item not in sources:
                sources.append(source_item)

        # Format retrieved chunks for response
        formatted_chunks: List[Dict[str, Any]] = []
        for r in results:
            chunk = r.get("chunk", {})
            formatted_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", "Unknown"),
                "score": float(r.get("score", 0.0)),
                "text": chunk.get("text", ""),
                "word_count": chunk.get("word_count"),
            })

        return answer, sources, formatted_chunks

    @staticmethod
    def get_evaluation_metrics() -> Dict[str, Any]:
        """
        Parses rag_evaluation_report.csv and returns aggregated performance metrics.
        """
        if not REPORT_PATH.exists():
            return {
                "report_available": False,
                "total_questions": 0,
                "error": "No evaluation report found on disk.",
            }

        try:
            df = pd.read_csv(REPORT_PATH)
        except Exception as exc:
            return {
                "report_available": False,
                "total_questions": 0,
                "error": f"Failed to read evaluation report: {str(exc)}",
            }

        if df.empty:
            return {
                "report_available": True,
                "total_questions": 0,
                "questions": [],
            }

        def to_bool(value):
            if isinstance(value, bool):
                return value
            if pd.isna(value):
                return False
            return str(value).strip().lower() in {"true", "1", "yes"}

        # Numeric conversions
        for col in ["retrieval_precision", "retrieval_recall", "faithfulness_score", "exact_match"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Boolean conversions
        for col in ["answerable", "retrieval_hit", "answer_abstained", "abstention_correct"]:
            if col in df.columns:
                df[col] = df[col].map(to_bool)

        answerable_df = df[df["answerable"]] if "answerable" in df.columns else df.iloc[0:0]
        unanswerable_df = df[~df["answerable"]] if "answerable" in df.columns else df.iloc[0:0]

        hit_rate = (
            float(answerable_df["retrieval_hit"].mean())
            if not answerable_df.empty and "retrieval_hit" in answerable_df.columns
            else None
        )
        mean_precision = (
            float(answerable_df["retrieval_precision"].mean())
            if not answerable_df.empty and "retrieval_precision" in answerable_df.columns
            else None
        )
        mean_recall = (
            float(answerable_df["retrieval_recall"].mean())
            if not answerable_df.empty and "retrieval_recall" in answerable_df.columns
            else None
        )
        rejection_rate = (
            float(unanswerable_df["abstention_correct"].mean())
            if not unanswerable_df.empty and "abstention_correct" in unanswerable_df.columns
            else None
        )
        exact_match_rate = (
            float(answerable_df["exact_match"].mean())
            if not answerable_df.empty and "exact_match" in answerable_df.columns and answerable_df["exact_match"].notna().any()
            else None
        )
        faithfulness_rate = (
            float(df["faithfulness_score"].mean())
            if "faithfulness_score" in df.columns and df["faithfulness_score"].notna().any()
            else None
        )

        records = df.where(pd.notnull(df), None).to_dict(orient="records")

        return {
            "report_available": True,
            "total_questions": len(df),
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "mean_precision": round(mean_precision, 4) if mean_precision is not None else None,
            "mean_recall": round(mean_recall, 4) if mean_recall is not None else None,
            "rejection_rate": round(rejection_rate, 4) if rejection_rate is not None else None,
            "exact_match_rate": round(exact_match_rate, 4) if exact_match_rate is not None else None,
            "mean_faithfulness": round(faithfulness_rate, 4) if faithfulness_rate is not None else None,
            "questions": records,
        }


# Singleton service instance
rag_service = RAGService()
