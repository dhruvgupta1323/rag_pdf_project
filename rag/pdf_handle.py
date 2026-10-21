import pymupdf
import hashlib
from pathlib import Path


def extract_pdf(uploaded_file, filename: str = None):
    """
    Extract text from a local PDF path, Streamlit UploadedFile, FastAPI UploadFile, or raw bytes.
    """

    # Case 1: Raw bytes
    if isinstance(uploaded_file, (bytes, bytearray)):
        pdf_bytes = bytes(uploaded_file)
        filename = filename or "document.pdf"

    # Case 2: A local file path, such as "Finance_Bill.pdf"
    elif isinstance(uploaded_file, (str, Path)):
        file_path = Path(uploaded_file)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {file_path.resolve()}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Please provide a PDF file.")

        pdf_bytes = file_path.read_bytes()
        filename = filename or file_path.name

    # Case 3: FastAPI UploadFile or object with .filename
    elif hasattr(uploaded_file, "filename"):
        filename = filename or uploaded_file.filename
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file.")
        if hasattr(uploaded_file, "file"):
            pdf_bytes = uploaded_file.file.read()
        elif hasattr(uploaded_file, "read"):
            pdf_bytes = uploaded_file.read()
        else:
            raise ValueError("Unable to read content from file object.")

    # Case 4: Streamlit UploadedFile or file-like object with .getvalue()
    elif hasattr(uploaded_file, "getvalue"):
        filename = filename or getattr(uploaded_file, "name", "document.pdf")

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file.")

        pdf_bytes = uploaded_file.getvalue()

    else:
        raise ValueError(f"Unsupported file type: {type(uploaded_file)}")

    if not pdf_bytes:
        raise ValueError("The PDF file is empty.")

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
            "Could not read the PDF. Check that it is a valid PDF."
        ) from exc
    print(f"Extracted {len(pages)} pages from {filename}.")
    return pages