# FastAPI Backend for Custom RAG Knowledge Assistant

This backend provides a high-performance REST API built with **FastAPI**, **Sentence Transformers**, **FAISS**, and **Google Gemini** for document ingestion, semantic chunk search, grounded question answering with citations, and evaluation metrics.

---

## 🚀 Quick Start

### 1. Start the FastAPI Server
Run with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Or directly with Python:
```bash
python main.py
```

### 2. Interactive API Documentation
- **Live Render Swagger UI:** [https://rag-pdf-project-4xkb.onrender.com/docs](https://rag-pdf-project-4xkb.onrender.com/docs)
- **Live Render ReDoc:** [https://rag-pdf-project-4xkb.onrender.com/redoc](https://rag-pdf-project-4xkb.onrender.com/redoc)
- **Local Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Local ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📌 Available Endpoints

### 🩺 System & Health
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API welcome, version, and links |
| `GET` | `/health` | Healthcheck and active document indexing status |

### 📄 Document Ingestion & Management
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/documents/status` | Current status (total pages, chunks, vectors, parameters) |
| `POST` | `/api/v1/documents/upload` | Multipart form upload of any PDF file with customizable `chunk_size` and `overlap` |
| `POST` | `/api/v1/documents/process-local` | Process an existing local PDF (e.g. `Finance_Bill.pdf`) |
| `DELETE` | `/api/v1/documents` | Unload / clear the active document and FAISS index from memory |

### 🔍 Search & Question Answering
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/search` | Fast FAISS semantic search returning top-k matching excerpts and similarity scores (no LLM call) |
| `POST` | `/api/v1/query` | Complete RAG pipeline: retrieves relevant chunks, queries Gemini, and returns grounded answer with page citations |

### 📊 Evaluation & Metrics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/evaluation` | Returns benchmark performance metrics (hit rate, precision, recall, rejection rate, exact match, faithfulness) |
| `GET` | `/api/v1/evaluation/dataset` | Returns the test dataset cases from `evaluation_dataset.json` |
| `GET` | `/api/v1/evaluation/download` | Download `rag_evaluation_report.csv` directly |

---

## 💡 Example Requests

### 1. Upload & Index a PDF
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@Finance_Bill.pdf" \
  -F "chunk_size=400" \
  -F "overlap=50"
```

### 2. Process a Local PDF
```bash
curl -X POST "http://localhost:8000/api/v1/documents/process-local" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "Finance_Bill.pdf", "chunk_size": 400, "overlap": 50}'
```

### 3. Semantic Search
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "financial proposals 2026-2027", "top_k": 3, "min_score": 0.30}'
```

### 4. Ask a Question (RAG with Citations)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What financial year are the Central Government proposals intended for?",
    "top_k": 3,
    "min_score": 0.30,
    "conversation_history": []
  }'
```

### 5. Check Evaluation Metrics
```bash
curl -X GET "http://localhost:8000/api/v1/evaluation"
```

---

## 🧪 Running Tests

Execute the API test suite:
```bash
python test_api.py
```
