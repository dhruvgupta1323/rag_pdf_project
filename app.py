import io
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests
import streamlit as st

# Load environment variables (.env)
load_dotenv()

# Conditional import of local RAG pipeline for offline / standalone fallback
try:
    from rag.pdf_handle import extract_pdf
    from rag.chunker import chunk_pages
    from rag.embeddings import embed_chunks
    from rag.vector_store import build_vector_store, search_chunks
    from rag.generator import generate_answer
    LOCAL_RAG_AVAILABLE = True
except ImportError:
    LOCAL_RAG_AVAILABLE = False


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPORT_PATH = Path(__file__).resolve().parent / "rag_evaluation_report.csv"
MIN_SCORE = 0.30
DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://rag-pdf-project-4xkb.onrender.com",
).strip().rstrip("/")

st.set_page_config(
    page_title="Custom RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("📚 Custom RAG Knowledge Assistant")
st.write(
    "Upload a PDF, ask questions about its content, "
    "and receive answers grounded in the document."
)


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

defaults = {
    "chunks": [],
    "index": None,
    "document_name": None,
    "messages": [],
    "document_id": None,
    "chunks_count": 0,
    "vectors_count": 0,
    "backend_url": DEFAULT_BACKEND_URL,
    "backend_connected": False,
    "backend_mode": True,
    "synced_backend": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# BACKEND API CLIENT HELPERS
# --------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def ping_backend(url: str):
    """Check if the backend FastAPI service is reachable and healthy."""
    if not url or not url.strip():
        return False, "No URL provided"
    clean_url = url.strip().rstrip("/")
    try:
        resp = requests.get(f"{clean_url}/health", timeout=6)
        if resp.status_code == 200:
            return True, resp.json()
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as exc:
        return False, str(exc)


def get_backend_document_status(url: str):
    """Retrieve indexing status of the document currently active on the backend."""
    if not url:
        return None
    clean_url = url.strip().rstrip("/")
    try:
        resp = requests.get(f"{clean_url}/api/v1/documents/status", timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def upload_pdf_to_backend(url: str, file_bytes: bytes, filename: str, chunk_size: int, overlap: int):
    """Upload and process PDF using the FastAPI backend API."""
    clean_url = url.strip().rstrip("/")
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {"chunk_size": int(chunk_size), "overlap": int(overlap)}
    resp = requests.post(
        f"{clean_url}/api/v1/documents/upload",
        files=files,
        data=data,
        timeout=180,
    )
    if resp.status_code != 200:
        detail = "Failed to process PDF on backend"
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")
    return resp.json()


def query_backend_api(url: str, question: str, top_k: int, min_score: float, history: list):
    """Query backend RAG API for answer with evidence."""
    clean_url = url.strip().rstrip("/")
    history_payload = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ["user", "assistant"]
    ]
    payload = {
        "question": question,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "conversation_history": history_payload,
    }
    resp = requests.post(
        f"{clean_url}/api/v1/query",
        json=payload,
        timeout=90,
    )
    if resp.status_code != 200:
        detail = "Query answering failed"
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"Backend query error ({resp.status_code}): {detail}")
    return resp.json()


def clear_backend_document(url: str):
    """Clear active document in backend memory."""
    if not url:
        return
    clean_url = url.strip().rstrip("/")
    try:
        requests.delete(f"{clean_url}/api/v1/documents", timeout=5)
    except Exception:
        pass


def get_backend_csv_report(url: str):
    """Download evaluation report CSV from backend if available."""
    if not url:
        return None
    clean_url = url.strip().rstrip("/")
    try:
        resp = requests.get(f"{clean_url}/api/v1/evaluation/download", timeout=6)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


# --------------------------------------------------
# EVALUATION DASHBOARD
# --------------------------------------------------

def to_bool(value):
    """Convert CSV values to booleans safely."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def show_metric(label, value, help_text):
    if value is None or pd.isna(value):
        st.metric(label, "N/A", help=help_text)
    else:
        st.metric(label, f"{value:.1%}", help=help_text)


def render_evaluation_dashboard():
    st.header("📊 RAG Metrics & Evaluation")
    st.caption(
        "Review retrieval quality, answer behavior, and faithfulness "
        "from your evaluation report."
    )

    df = None
    report_bytes = None

    # Try reading from local report path first
    if REPORT_PATH.exists():
        try:
            df = pd.read_csv(REPORT_PATH)
            report_bytes = REPORT_PATH.read_bytes()
        except Exception as exc:
            st.error(f"Could not read local evaluation report: {exc}")

    # If local not found, try fetching from connected backend
    if (df is None or df.empty) and st.session_state.backend_connected:
        raw_csv = get_backend_csv_report(st.session_state.backend_url)
        if raw_csv:
            try:
                df = pd.read_csv(io.BytesIO(raw_csv))
                report_bytes = raw_csv
            except Exception as exc:
                st.error(f"Could not parse backend evaluation report: {exc}")

    if df is None or df.empty:
        st.warning(
            "No evaluation report found locally or on the backend. Run "
            "`python evaluate_rag.py` from your project folder, "
            "then refresh the dashboard."
        )
        return

    # Convert metric columns to numbers.
    numeric_columns = [
        "retrieval_precision",
        "retrieval_recall",
        "faithfulness_score",
        "exact_match",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Normalize Boolean columns.
    for column in [
        "answerable",
        "retrieval_hit",
        "answer_abstained",
        "abstention_correct",
    ]:
        if column in df.columns:
            df[column] = df[column].map(to_bool)

    answerable_df = (
        df[df["answerable"]]
        if "answerable" in df.columns
        else df.iloc[0:0]
    )

    unanswerable_df = (
        df[~df["answerable"]]
        if "answerable" in df.columns
        else df.iloc[0:0]
    )

    # Calculate metrics.
    hit_rate = (
        answerable_df["retrieval_hit"].mean()
        if not answerable_df.empty and "retrieval_hit" in answerable_df.columns
        else None
    )

    mean_precision = (
        answerable_df["retrieval_precision"].mean()
        if not answerable_df.empty and "retrieval_precision" in answerable_df.columns
        else None
    )

    mean_recall = (
        answerable_df["retrieval_recall"].mean()
        if not answerable_df.empty and "retrieval_recall" in answerable_df.columns
        else None
    )

    rejection_rate = (
        unanswerable_df["abstention_correct"].mean()
        if not unanswerable_df.empty and "abstention_correct" in unanswerable_df.columns
        else None
    )

    exact_match_rate = (
        answerable_df["exact_match"].mean()
        if not answerable_df.empty
        and "exact_match" in answerable_df.columns
        and answerable_df["exact_match"].notna().any()
        else None
    )

    faithfulness_rate = (
        df["faithfulness_score"].mean()
        if "faithfulness_score" in df.columns
        and df["faithfulness_score"].notna().any()
        else None
    )

    # Metric cards.
    st.subheader("Overall performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        show_metric(
            "Retrieval hit rate",
            hit_rate,
            "Percentage of answerable questions where a labelled relevant page was retrieved.",
        )
    with col2:
        show_metric(
            "Mean retrieval precision",
            mean_precision,
            "Average fraction of retrieved chunks from labelled relevant pages for answerable questions.",
        )
    with col3:
        show_metric(
            "Mean retrieval recall",
            mean_recall,
            "Average fraction of labelled relevant pages retrieved for answerable questions.",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        show_metric(
            "Correct abstention rate",
            rejection_rate,
            "Percentage of unanswerable questions where a refusal was recognized.",
        )
    with col5:
        show_metric(
            "Reference exact-match rate",
            exact_match_rate,
            "Percentage of answerable responses containing complete reference wording.",
        )
    with col6:
        show_metric(
            "Mean faithfulness score",
            faithfulness_rate,
            "Average score assigned by faithfulness evaluator.",
        )

    st.divider()

    # Individual test results.
    st.subheader("Question-level results")
    filter_options = ["All questions"]
    if "answerable" in df.columns:
        filter_options += ["Answerable questions", "Unanswerable questions"]

    selected_filter = st.selectbox("Filter test questions", filter_options)

    filtered_df = df.copy()
    if selected_filter == "Answerable questions":
        filtered_df = filtered_df[filtered_df["answerable"]]
    elif selected_filter == "Unanswerable questions":
        filtered_df = filtered_df[~filtered_df["answerable"]]

    display_columns = [
        column
        for column in [
            "id",
            "question",
            "answerable",
            "retrieved_pages",
            "retrieval_hit",
            "retrieval_precision",
            "retrieval_recall",
            "generated_answer",
            "exact_match",
            "faithfulness_score",
            "unsupported_claims",
            "generation_error",
            "faithfulness_error",
        ]
        if column in filtered_df.columns
    ]

    if display_columns:
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    # Faithfulness chart.
    if (
        "question" in df.columns
        and "faithfulness_score" in df.columns
        and df["faithfulness_score"].notna().any()
    ):
        st.subheader("Faithfulness by question")
        chart_df = df[["question", "faithfulness_score"]].dropna().copy()
        chart_df["question"] = chart_df["question"].astype(str).str.slice(0, 65)
        st.bar_chart(chart_df.set_index("question")["faithfulness_score"])

    # Display evaluation errors.
    error_columns = [
        col for col in ["generation_error", "faithfulness_error"] if col in df.columns
    ]
    if error_columns:
        errors_mask = df[error_columns].fillna("").apply(
            lambda col: col.astype(str).str.strip().ne("")
        ).any(axis=1)
        errors_df = df.loc[errors_mask]
        if not errors_df.empty:
            with st.expander("View evaluation errors"):
                st.dataframe(errors_df, use_container_width=True, hide_index=True)

    # Download the CSV report.
    if report_bytes:
        st.download_button(
            label="Download evaluation report CSV",
            data=report_bytes,
            file_name="rag_evaluation_report.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("🌐 Backend API Connection")
    backend_url_input = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="FastAPI Backend API URL on Render or localhost.",
    )

    if backend_url_input.strip() != st.session_state.backend_url:
        st.session_state.backend_url = backend_url_input.strip().rstrip("/")
        st.session_state.synced_backend = False
        st.cache_data.clear()

    # Ping backend
    is_healthy, health_payload = ping_backend(st.session_state.backend_url)
    st.session_state.backend_connected = is_healthy

    if is_healthy:
        service_name = health_payload.get("service", "FastAPI Backend") if isinstance(health_payload, dict) else "FastAPI Backend"
        version_name = health_payload.get("version", "1.0.0") if isinstance(health_payload, dict) else "1.0.0"
        st.success(f"🟢 **Connected:** `{service_name}` (v{version_name})")
        route_backend = st.toggle("Route queries through Backend API", value=True)
    else:
        err_hint = health_payload if isinstance(health_payload, str) else "offline"
        st.warning(f"🔴 **Backend Unreachable** ({err_hint})")
        if LOCAL_RAG_AVAILABLE:
            st.info("Falling back to local in-memory RAG pipeline.")
        else:
            st.error("No local ML dependencies installed. Start the backend service.")
        route_backend = False

    st.session_state.backend_mode = route_backend

    # Auto-sync active document status from backend on initial load
    if is_healthy and route_backend and not st.session_state.synced_backend:
        doc_stat = get_backend_document_status(st.session_state.backend_url)
        if doc_stat and doc_stat.get("is_indexed"):
            st.session_state.document_name = doc_stat.get("document_name")
            st.session_state.document_id = doc_stat.get("document_id")
            st.session_state.chunks_count = doc_stat.get("chunks_count", 0)
            st.session_state.vectors_count = doc_stat.get("vectors_count", 0)
        st.session_state.synced_backend = True

    st.divider()

    st.header("Document settings")
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
    )

    chunk_size = st.number_input(
        "Chunk size (words)",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
    )

    overlap = st.number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(50, int(chunk_size) - 1),
        step=10,
    )

    process_button = st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=5,
        value=3,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.button("Unload doc", use_container_width=True):
            if st.session_state.backend_mode and st.session_state.backend_connected:
                clear_backend_document(st.session_state.backend_url)
            st.session_state.document_name = None
            st.session_state.document_id = None
            st.session_state.chunks_count = 0
            st.session_state.vectors_count = 0
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    show_dashboard = st.checkbox(
        "Show Metrics & Evaluation Dashboard",
        value=False,
    )


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if process_button:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        file_bytes = uploaded_file.getvalue()

        # Path 1: Backend API Mode
        if st.session_state.backend_mode and st.session_state.backend_connected:
            try:
                with st.spinner("Uploading and indexing PDF on FastAPI backend..."):
                    res = upload_pdf_to_backend(
                        url=st.session_state.backend_url,
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                    )
                    st.session_state.document_name = res.get("document_name", uploaded_file.name)
                    st.session_state.document_id = res.get("document_id")
                    st.session_state.chunks_count = res.get("chunks_count", 0)
                    st.session_state.vectors_count = res.get("indexed_vectors", 0)
                    st.session_state.messages = []
                st.success(
                    f"Successfully indexed {res.get('chunks_count')} chunks "
                    f"from {res.get('pages_count')} pages on backend in "
                    f"{res.get('processing_time_seconds', 0):.2f}s."
                )
            except Exception as exc:
                st.error(f"Could not process PDF on backend: {exc}")

        # Path 2: Local RAG Pipeline Fallback
        elif LOCAL_RAG_AVAILABLE:
            try:
                with st.spinner("Extracting text, creating chunks and indexing locally..."):
                    pages = extract_pdf(uploaded_file)
                    chunks = chunk_pages(
                        pages,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                    )
                    if not chunks:
                        raise ValueError(
                            "No readable text was found in this PDF. "
                            "Scanned PDFs may require OCR."
                        )
                    embeddings = embed_chunks(chunks)
                    index = build_vector_store(chunks, embeddings)

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.document_id = pages[0]["document_id"]
                    st.session_state.chunks_count = len(chunks)
                    st.session_state.vectors_count = index.ntotal
                    st.session_state.messages = []

                st.success(
                    f"Successfully indexed {len(chunks)} chunks "
                    f"from {len(pages)} pages locally."
                )
            except Exception as exc:
                st.error(f"Could not process the PDF locally: {exc}")

        else:
            st.error("Backend is unreachable and local RAG dependencies are not installed.")


# --------------------------------------------------
# SHOW DASHBOARD
# --------------------------------------------------

if show_dashboard:
    render_evaluation_dashboard()
    st.divider()


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

is_ready = bool(st.session_state.document_name) and (
    st.session_state.vectors_count > 0 or st.session_state.index is not None
)

if is_ready:
    mode_badge = (
        "⚡ FastAPI Backend API"
        if (st.session_state.backend_mode and st.session_state.backend_connected)
        else "💻 Local Engine"
    )
    st.success(f"Active document: **{st.session_state.document_name}** ({mode_badge})")

    col1, col2 = st.columns(2)
    col1.metric("Text chunks", st.session_state.chunks_count or len(st.session_state.chunks))
    col2.metric(
        "Indexed vectors",
        st.session_state.vectors_count or (st.session_state.index.ntotal if st.session_state.index else 0),
    )
else:
    st.info("Upload a PDF in the sidebar and click Process PDF to begin.")


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander("View source pages"):
                    for source in sources:
                        st.markdown(f"- **{source['source']}**, page {source['page']}")


# --------------------------------------------------
# QUESTION INPUT AND ANSWER GENERATION
# --------------------------------------------------

question = st.chat_input("Ask a question about your document...")

if question:
    if not is_ready:
        st.warning("Please upload and process a PDF first.")
    else:
        conversation_history = st.session_state.messages[-6:]
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                # Backend API execution
                if st.session_state.backend_mode and st.session_state.backend_connected:
                    with st.spinner("Searching document & generating answer via backend..."):
                        api_res = query_backend_api(
                            url=st.session_state.backend_url,
                            question=question,
                            top_k=top_k,
                            min_score=MIN_SCORE,
                            history=conversation_history,
                        )
                        answer = api_res.get("answer", "")
                        sources = api_res.get("sources", [])
                        results = [
                            {
                                "chunk": {
                                    "source": c.get("source"),
                                    "page": c.get("page"),
                                    "text": c.get("text"),
                                },
                                "score": c.get("score", 0.0),
                            }
                            for c in api_res.get("retrieved_chunks", [])
                        ]
                        exec_time = api_res.get("execution_time_seconds")

                # Local RAG execution
                elif LOCAL_RAG_AVAILABLE and st.session_state.index is not None:
                    with st.spinner("Searching document & generating answer locally..."):
                        previous_question = next(
                            (
                                message["content"]
                                for message in reversed(conversation_history)
                                if message.get("role") == "user"
                            ),
                            "",
                        )
                        retrieval_question = (
                            f"{previous_question}\n{question}"
                            if previous_question
                            else question
                        )

                        results = search_chunks(
                            retrieval_question,
                            st.session_state.index,
                            st.session_state.chunks,
                            top_k=top_k,
                            min_score=MIN_SCORE,
                        )

                        answer = generate_answer(
                            question,
                            results,
                            conversation_history=conversation_history,
                        )

                        sources = []
                        for result in results:
                            chunk = result["chunk"]
                            source = {
                                "source": chunk.get("source", "Unknown source"),
                                "page": chunk.get("page", "Unknown"),
                            }
                            if source not in sources:
                                sources.append(source)
                        exec_time = None

                else:
                    raise RuntimeError("Neither backend API nor local index is available.")

                st.markdown(answer)
                if exec_time is not None:
                    st.caption(f"⏱️ Response generated in {exec_time:.2f}s via FastAPI backend")

                # Display retrieved chunks for debugging/transparency
                with st.expander("View retrieved chunks"):
                    if results:
                        for i, result in enumerate(results, start=1):
                            chunk = result["chunk"]
                            st.markdown(f"### Retrieved chunk {i}")
                            st.write(f"Source: {chunk.get('source', 'Unknown')}")
                            st.write(f"Page: {chunk.get('page', 'Unknown')}")
                            st.write(f"Similarity score: {result['score']:.4f}")
                            st.write(chunk.get("text", ""))
                            st.divider()
                    else:
                        st.info("No relevant chunks were retrieved.")

                # Display source pages
                if sources:
                    with st.expander("View source pages"):
                        for source in sources:
                            st.markdown(f"- **{source['source']}**, page {source['page']}")

                # Save conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                error_message = "I couldn't complete that answer. Please try again."
                st.error(f"Something went wrong while answering: {exc}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
