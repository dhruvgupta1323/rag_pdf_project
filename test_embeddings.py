from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks, embed_query

# Step 1: Extract text from the PDF
pages = extract_pdf("Finance_Bill.pdf")
print("Total pages:", len(pages))

# Step 2: Split the pages into chunks
chunks = chunk_pages(
    pages,
    chunk_size=400,
    overlap=50
)
print("Total chunks:", len(chunks))

# Step 3: Generate embeddings for the chunks
embeddings = embed_chunks(chunks)
print("Total embeddings:", len(embeddings))

# Step 4: Inspect the first embedding
if len(embeddings) > 0:
    print("Embedding shape:", embeddings[0].shape)
    print("First 10 values:", embeddings[0][:10])

# Step 5: Generate an embedding for a question
question = "What changes does the Finance Bill introduce?"
query_vector = embed_query(question)

print("Question:", question)
print("Query vector shape:", query_vector.shape)