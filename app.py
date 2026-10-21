
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import build_vector_store, search_chunks
from rag.generator import generate_answer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPORT_PATH = Path(__file__).resolve().parent / "rag_evaluation_report.csv"
MIN_SCORE = 0.30

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
# SESSION STATE
# --------------------------------------------------

defaults = {
    "chunks": [],
    "index": None,
    "document_name": None,
    "messages": [],
    "document_id": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


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
        "from your custom evaluation report."
    )

    if not REPORT_PATH.exists():
        st.warning(
            "No evaluation report found. Run "
            "`python evaluate_rag.py` from your project folder, "
            "then refresh the dashboard."
        )
        return

    try:
        df = pd.read_csv(REPORT_PATH)
    except Exception as exc:
        st.error(f"Could not read the evaluation report: {exc}")
        return

    if df.empty:
        st.info("The evaluation report contains no test results.")
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
        if not answerable_df.empty
        and "retrieval_hit" in answerable_df.columns
        else None
    )

    mean_precision = (
        answerable_df["retrieval_precision"].mean()
        if not answerable_df.empty
        and "retrieval_precision" in answerable_df.columns
        else None
    )

    mean_recall = (
        answerable_df["retrieval_recall"].mean()
        if not answerable_df.empty
        and "retrieval_recall" in answerable_df.columns
        else None
    )

    rejection_rate = (
        unanswerable_df["abstention_correct"].mean()
        if not unanswerable_df.empty
        and "abstention_correct" in unanswerable_df.columns
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
            "Percentage of answerable questions where a labelled "
            "relevant page was retrieved.",
        )

    with col2:
        show_metric(
            "Mean retrieval precision",
            mean_precision,
            "Average fraction of retrieved chunks from labelled "
            "relevant pages for answerable questions.",
        )

    with col3:
        show_metric(
            "Mean retrieval recall",
            mean_recall,
            "Average fraction of labelled relevant pages retrieved "
            "for answerable questions.",
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        show_metric(
            "Correct abstention rate",
            rejection_rate,
            "Percentage of unanswerable questions where a refusal "
            "was recognized by the answer evaluator.",
        )

    with col5:
        show_metric(
            "Reference exact-match rate",
            exact_match_rate,
            "Percentage of answerable responses containing the complete "
            "reference wording. Correct paraphrases may not match.",
        )

    with col6:
        show_metric(
            "Mean faithfulness score",
            faithfulness_rate,
            "Average score assigned by your faithfulness evaluator. "
            "This is an estimate, not a guarantee of truth.",
        )

    st.divider()

    # Individual test results.
    st.subheader("Question-level results")

    filter_options = ["All questions"]

    if "answerable" in df.columns:
        filter_options += [
            "Answerable questions",
            "Unanswerable questions",
        ]

    selected_filter = st.selectbox(
        "Filter test questions",
        filter_options,
    )

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
    else:
        st.info("No recognized evaluation columns were found.")

    # Faithfulness chart.
    if (
        "question" in df.columns
        and "faithfulness_score" in df.columns
        and df["faithfulness_score"].notna().any()
    ):
        st.subheader("Faithfulness by question")

        chart_df = df[
            ["question", "faithfulness_score"]
        ].dropna().copy()

        chart_df["question"] = (
            chart_df["question"].astype(str).str.slice(0, 65)
        )

        st.bar_chart(
            chart_df.set_index("question")["faithfulness_score"]
        )

    # Display evaluation errors.
    error_columns = [
        column
        for column in ["generation_error", "faithfulness_error"]
        if column in df.columns
    ]

    if error_columns:
        errors_mask = df[error_columns].fillna("").apply(
            lambda col: col.astype(str).str.strip().ne("")
        ).any(axis=1)

        errors_df = df.loc[errors_mask]

        if not errors_df.empty:
            with st.expander("View evaluation errors"):
                st.dataframe(
                    errors_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # Download the CSV report.
    st.download_button(
        label="Download evaluation report CSV",
        data=REPORT_PATH.read_bytes(),
        file_name="rag_evaluation_report.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# SIDEBAR: PDF SETTINGS
# --------------------------------------------------

with st.sidebar:
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

    if st.button("Clear chat", use_container_width=True):
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
        try:
            with st.spinner(
                "Extracting text, creating chunks and indexing..."
            ):
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

                # Update active document only after indexing succeeds.
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_id = pages[0]["document_id"]
                st.session_state.messages = []

            st.success(
                f"Successfully indexed {len(chunks)} chunks "
                f"from {len(pages)} pages."
            )

        except Exception as exc:
            st.error(f"Could not process the PDF: {exc}")


# --------------------------------------------------
# SHOW DASHBOARD
# --------------------------------------------------

if show_dashboard:
    render_evaluation_dashboard()
    st.divider()


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

if st.session_state.index is not None:
    st.success(
        f"Active document: {st.session_state.document_name}"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Text chunks",
        len(st.session_state.chunks),
    )

    col2.metric(
        "Indexed vectors",
        st.session_state.index.ntotal,
    )

else:
    st.info(
        "Upload a PDF in the sidebar and click Process PDF to begin."
    )


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
                        st.markdown(
                            f"- **{source['source']}**, "
                            f"page {source['page']}"
                        )


# --------------------------------------------------
# QUESTION INPUT AND ANSWER GENERATION
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your document..."
)

if question:
    if st.session_state.index is None:
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
                with st.spinner(
                    "Searching the document and generating an answer..."
                ):
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

                    # Retrieve relevant chunks.
                    results = search_chunks(
                        retrieval_question,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=top_k,
                        min_score=MIN_SCORE,
                    )

                    # Generate an answer from the retrieved evidence.
                    answer = generate_answer(
                        question,
                        results,
                        conversation_history=conversation_history,
                    )

                st.markdown(answer)

                # Build a unique list of source pages.
                sources = []

                for result in results:
                    chunk = result["chunk"]

                    source = {
                        "source": chunk.get("source", "Unknown source"),
                        "page": chunk.get("page", "Unknown"),
                    }

                    if source not in sources:
                        sources.append(source)

                # Display retrieved chunks for debugging.
                with st.expander("View retrieved chunks"):
                    if results:
                        for i, result in enumerate(results, start=1):
                            chunk = result["chunk"]

                            st.markdown(f"### Retrieved chunk {i}")
                            st.write(
                                f"Source: {chunk.get('source', 'Unknown')}"
                            )
                            st.write(
                                f"Page: {chunk.get('page', 'Unknown')}"
                            )
                            st.write(
                                f"Similarity score: {result['score']:.4f}"
                            )
                            st.write(chunk.get("text", ""))
                            st.divider()
                    else:
                        st.info("No relevant chunks were retrieved.")

                # Display source pages.
                if sources:
                    with st.expander("View source pages"):
                        for source in sources:
                            st.markdown(
                                f"- **{source['source']}**, "
                                f"page {source['page']}"
                            )

                # Save conversation.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                error_message = (
                    "I couldn't complete that answer. Please try again."
                )
                st.error(
                    f"Something went wrong while answering: {exc}"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
