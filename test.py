from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages

import pymupdf
import hashlib
from pathlib import Path


def extract_pdf(uploaded_file):
    """
    Extract PDF text from either:
    1. A local file path (str or Path)
    2. A Streamlit UploadedFile object
    """

    # Handle a local file path
    if isinstance(uploaded_file, (str, Path)):
        file_path = Path(uploaded_file)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Please provide a PDF file.")

        pdf_bytes = file_path.read_bytes()
        filename = file_path.name

    # Handle a Streamlit UploadedFile
    else:
        filename = uploaded_file.name

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file.")

        pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    document_id = hashlib.sha256(pdf_bytes).hexdigest()
    pages = []

    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:

            if pdf.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                pages.append({
                    "document_id": document_id,
                    "source": filename,
                    "page": page_number,
                    "text": text
                })

    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(
            "Could not read the PDF. Check that it is valid."
        ) from exc

    return pages


# Step 1: Extract text from your PDF
pages = extract_pdf("Finance_Bill.pdf")

# Step 2: Create chunks
chunks = chunk_pages(
    pages,
    chunk_size=400,
    overlap=50
)

# Step 3: Print summary
print("=" * 60)
print("PDF CHUNKING REPORT")
print("=" * 60)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

# Step 4: Manually inspect every chunk
for chunk in chunks:
    print("\n" + "=" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Document ID:", chunk["document_id"])
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Word count:", chunk["word_count"])
    print("-" * 60)
    print(chunk["text"])

# Step 5: Validate chunk sizes
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

for chunk in chunks:
    actual_words = len(chunk["text"].split())

    assert actual_words == chunk["word_count"]
    assert actual_words <= 400

print("All chunks passed validation!")
