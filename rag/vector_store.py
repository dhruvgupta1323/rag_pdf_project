import faiss
import numpy as np

from rag.embeddings import embed_query


def build_vector_store(chunks, embeddings):
    """
    Build a FAISS index from chunk embeddings.

    Args:
        chunks: List of chunk dictionaries.
        embeddings: NumPy array or list of embedding vectors.

    Returns:
        A FAISS index.
    """

    if not chunks:
        raise ValueError("No chunks available to index.")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(vectors) != len(chunks):
        raise ValueError(
            "Number of embeddings must match number of chunks."
        )

    if vectors.shape[1] == 0:
        raise ValueError("Embedding vectors cannot be empty.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    vectors = np.ascontiguousarray(vectors)

    # The embedding model normalizes vectors.
    # Inner product therefore gives cosine similarity.
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


def search_chunks(question, index, chunks, top_k=3, min_score=0.30):
    """
    Retrieve relevant chunks using FAISS and filter out
    chunks whose similarity score is below min_score.

    Args:
        question: User's question.
        index: Existing FAISS index.
        chunks: List of chunk dictionaries.
        top_k: Maximum number of chunks to retrieve.
        min_score: Minimum cosine similarity score to accept.

    Returns:
        A list of dictionaries containing scores and chunks.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1 and 1.")

    if not chunks or index.ntotal == 0:
        return []

    # Convert the question into an embedding.
    query_vector = embed_query(question)

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    ).reshape(1, -1)

    query_vector = np.ascontiguousarray(query_vector)

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Question embedding dimension does not match FAISS index."
        )

    # Retrieve the top matching vectors.
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        score = float(score)

        # Ignore chunks below the relevance threshold.
        if score < min_score:
            continue

        results.append({
            "score": score,
            "chunk": chunks[int(idx)]
        })

    return results