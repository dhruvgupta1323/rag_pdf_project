
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def embed_chunks(chunks):
    """
    Convert text chunks into numerical vectors.

    Args:
        chunks: List of dictionaries from chunk_pages().

    Returns:
        A list of 384-dimensional embedding vectors.
    """

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def embed_query(question):
    """Convert a user's question into an embedding vector."""

    return model.encode(
        question,
        normalize_embeddings=True
    )
