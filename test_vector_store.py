from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import build_vector_store, search_chunks


def main():
    # Step 1: Extract text from the PDF.
    pages = extract_pdf("Finance_Bill.pdf")
    print("Total pages:", len(pages))

    # Step 2: Split text into chunks.
    chunks = chunk_pages(
        pages,
        chunk_size=400,
        overlap=50
    )
    print("Total chunks:", len(chunks))

    if not chunks:
        print("No text found in the PDF.")
        return

    # Step 3: Generate embeddings.
    embeddings = embed_chunks(chunks)
    print("Total embeddings:", len(embeddings))

    # Step 4: Build the FAISS index.
    index = build_vector_store(chunks, embeddings)
    print("Vectors stored in FAISS:", index.ntotal)

    # Step 5: Ask a question.
    question = input(
        "\nAsk a question about your PDF: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    # Step 6: Retrieve the most relevant chunks.
    results = search_chunks(
        question,
        index,
        chunks,
        top_k=3
    )

    # Step 7: Display retrieved content.
    print("\nQUESTION:", question)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nRESULT {rank}")
        print("-" * 40)
        print("Similarity:", round(result["score"], 4))
        print("Document:", chunk["source"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])

    print("\nRetrieval completed.")


if __name__ == "__main__":
    main()