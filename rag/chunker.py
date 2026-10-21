def chunk_pages(pages,chunk_size=400,overlap=50):
    """
    Split extracted PDF pages into overlapping chunks.

    Args:
        pages: List of dictionaries returned by extract_pdf().
        chunk_size: Maximum words per chunk.
        overlap: Number of words shared between chunks.

    Returns:
        A list of chunks with source metadata.
    """
    if chunk_size<=0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap<0 or overlap>=chunk_size:
        raise ValueError("overlap must be a non-negative integer.")


    chunks = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text=page["text"].strip()

        if not text:
            continue  # Skip empty pages

        words = text.split()

        for start in range(0, len(words),step):
            end = min(start + chunk_size, len(words))
            chunk_words=words[start:end]

            if not chunk_words:
                break

            chunks.append({ 
                "chunk_id": chunk_id,
                "document_id": page["document_id"],
                "source": page["source"],
                "page": page["page"],
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words)
            })


            chunk_id += 1

            if end == len(words):
                break  # No more words to process

    return chunks
             

   

    
