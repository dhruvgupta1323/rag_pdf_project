
from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import build_vector_store, search_chunks


PDF_PATH = "Finance_Bill.pdf"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 3


def prepare_document():
    """Extract, chunk, embed and index the PDF."""

    print("\n[1/4] Extracting PDF text...")
    pages = extract_pdf(PDF_PATH)
    print(f"SUCCESS: Extracted {len(pages)} pages.")

    print("\n[2/4] Creating chunks...")
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    print(f"SUCCESS: Created {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No text chunks were created.")

    print("\n[3/4] Generating embeddings...")
    embeddings = embed_chunks(chunks)
    print(f"SUCCESS: Generated {len(embeddings)} embeddings.")

    print("\n[4/4] Building FAISS index...")
    index = build_vector_store(chunks, embeddings)
    print(f"SUCCESS: Indexed {index.ntotal} vectors.")

    return index, chunks


def evaluate_question(index, chunks, question):
    """Display the passages retrieved for one question."""

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    results = search_chunks(
        question,
        index,
        chunks,
        top_k=TOP_K,
    )

    if not results:
        print("NO RESULTS: FAISS returned no chunks.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n--- Result #{rank} ---")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print("Retrieved text:")
        print(chunk["text"])
        print("-" * 70)

    print("\nEvaluate the results:")
    print("1. Does a retrieved passage contain the answer?")
    print("2. Is the source page correct?")
    print("3. Are the retrieved passages relevant?")
    print("4. If the answer is missing, does another chunk contain it?")


def main():
    index, chunks = prepare_document()

    print("\nDocument ready for retrieval evaluation.")
    print("Enter a question whose answer you can verify in the PDF.")
    print("Type 'exit' to finish.")

    while True:
        question = input("\nEnter your test question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Retrieval evaluation finished.")
            break

        if not question:
            print("Please enter a non-empty question.")
            continue

        try:
            evaluate_question(index, chunks, question)
        except Exception as exc:
            print(f"RETRIEVAL TEST FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
