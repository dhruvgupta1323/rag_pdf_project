from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-fastapi-backend"


def test_document_status_initial():
    client.delete("/api/v1/documents")
    response = client.get("/api/v1/documents/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_indexed"] is False


def test_search_without_document_fails():
    client.delete("/api/v1/documents")
    response = client.post("/api/v1/search", json={"query": "test query"})
    assert response.status_code == 400
    assert "No document has been indexed" in response.json()["detail"]


def test_query_without_document_fails():
    client.delete("/api/v1/documents")
    response = client.post("/api/v1/query", json={"question": "What is this?"})
    assert response.status_code == 400
    assert "No document has been indexed" in response.json()["detail"]


def test_evaluation_endpoints():
    response = client.get("/api/v1/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "report_available" in data
    assert "total_questions" in data

    download_resp = client.get("/api/v1/evaluation/download")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"].startswith("text/csv")


def test_process_local_pdf_and_search():
    # Process Finance_Bill.pdf
    process_resp = client.post(
        "/api/v1/documents/process-local",
        json={
            "file_path": "Finance_Bill.pdf",
            "chunk_size": 400,
            "overlap": 50,
        },
    )
    assert process_resp.status_code == 200
    process_data = process_resp.json()
    assert process_data["chunks_count"] > 0
    assert process_data["pages_count"] > 0

    # Verify status
    status_resp = client.get("/api/v1/documents/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["is_indexed"] is True
    assert status_data["chunks_count"] == process_data["chunks_count"]

    # Test search
    search_resp = client.post(
        "/api/v1/search",
        json={"query": "financial proposals 2026-2027", "top_k": 2},
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total_results"] > 0
    assert len(search_data["results"]) <= 2
    assert "text" in search_data["results"][0]
    assert "score" in search_data["results"][0]


def test_clear_document():
    del_resp = client.delete("/api/v1/documents")
    assert del_resp.status_code == 200

    status_resp = client.get("/api/v1/documents/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_indexed"] is False


if __name__ == "__main__":
    test_root()
    print("test_root passed")
    test_health()
    print("test_health passed")
    test_document_status_initial()
    print("test_document_status_initial passed")
    test_search_without_document_fails()
    print("test_search_without_document_fails passed")
    test_query_without_document_fails()
    print("test_query_without_document_fails passed")
    test_evaluation_endpoints()
    print("test_evaluation_endpoints passed")
    test_process_local_pdf_and_search()
    print("test_process_local_pdf_and_search passed")
    test_clear_document()
    print("test_clear_document passed")
    print("ALL TESTS PASSED SUCCESSFULLY!")
