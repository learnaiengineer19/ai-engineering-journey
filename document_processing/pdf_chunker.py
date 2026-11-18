
"""
LESSON 12 - PART 7: REUSABLE PDF PROCESSOR
=========================================

OBJECTIVE:
Extract text from a PDF, clean it, split it into
overlapping chunks and preserve metadata.

CONCEPTS COVERED:
1. Reusable Python functions
2. PDF extraction with PyMuPDF
3. Text cleaning
4. Character-based chunking
5. Nested for loops
6. List comprehensions and dictionaries
7. Unique chunk IDs
8. Page-level metadata
9. Input validation

AI ENGINEERING CONNECTION:
This function forms the document processing stage
of our RAG ingestion pipeline.

PIPELINE:
PDF -> Extract -> Clean -> Chunk -> Metadata

NEXT STEP:
Generate embeddings with Ollama and store
the chunks in ChromaDB.

RUN FROM PROJECT ROOT:
python document_processing/pdf_chunker.py
"""

from pathlib import Path

import pymupdf

from text_cleaner import clean_text
from text_chunker import chunk_text


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

PDF_PATH = Path(
    "document_processing/documents/insurance_policy.pdf"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --------------------------------------------------
# STEP 2: REUSABLE PDF PROCESSING FUNCTION
# --------------------------------------------------

def process_pdf(
    pdf_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Extract, clean and chunk a PDF document.

    Args:
        pdf_path:
            Path to the PDF file.

        chunk_size:
            Maximum number of characters per chunk.

        chunk_overlap:
            Number of overlapping characters
            between consecutive chunks.

    Returns:
        List of dictionaries containing:
        - chunk_id
        - chunk_index
        - page_number
        - source
        - text

    Raises:
        FileNotFoundError:
            If the PDF does not exist.

        ValueError:
            If chunk settings are invalid or
            the PDF contains no extractable text.
    """

    # ----------------------------------------------
    # STEP 2.1: VALIDATE FILE PATH
    # ----------------------------------------------

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    # ----------------------------------------------
    # STEP 2.2: INITIALIZE CHUNKS LIST
    # ----------------------------------------------

    chunks_data = []

    # ----------------------------------------------
    # STEP 2.3: OPEN PDF
    # ----------------------------------------------

    with pymupdf.open(pdf_path) as document:

        print("PDF filename:", pdf_path.name)
        print("Total pages:", len(document))

        # ------------------------------------------
        # STEP 2.4: PROCESS EACH PAGE
        # ------------------------------------------

        for page_number, page in enumerate(
            document,
            start=1
        ):

            # Extract text from the current page.
            page_text = page.get_text(
                "text",
                sort=True
            )

            # Normalize whitespace.
            cleaned_text = clean_text(page_text)

            # Skip empty or scanned pages that have
            # no extractable text.
            if not cleaned_text:
                print(
                    f"Page {page_number}: "
                    "No extractable text found"
                )
                continue

            # --------------------------------------
            # STEP 2.5: SPLIT TEXT INTO CHUNKS
            # --------------------------------------

            page_chunks = chunk_text(
                text=cleaned_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            print(
                f"Page {page_number}: "
                f"{len(page_chunks)} chunks"
            )

            # --------------------------------------
            # STEP 2.6: PROCESS EVERY CHUNK
            # --------------------------------------

            for chunk_index, chunk in enumerate(
                page_chunks,
                start=1
            ):

                # Generate a unique ID for this
                # chunk within our sample PDF.

                chunk_id = (
                    f"{pdf_path.stem}"
                    f"_page_{page_number}"
                    f"_chunk_{chunk_index}"
                )

                # Store the chunk and its metadata.

                chunk_dict = {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "source": pdf_path.name,
                    "text": chunk
                }

                chunks_data.append(chunk_dict)

    # ----------------------------------------------
    # STEP 2.7: VALIDATE CHUNK IDS
    # ----------------------------------------------

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks_data
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs detected"
        )

    if not chunks_data:
        raise ValueError(
            "No text chunks generated from PDF"
        )

    # ----------------------------------------------
    # STEP 2.8: RETURN CHUNKS
    # ----------------------------------------------

    return chunks_data


# --------------------------------------------------
# STEP 3: RUN THE PDF PROCESSOR
# --------------------------------------------------

if __name__ == "__main__":

    chunks_data = process_pdf(
        pdf_path=PDF_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("\n" + "=" * 50)
    print("PDF PROCESSING SUMMARY")
    print("=" * 50)

    print("Total chunks:", len(chunks_data))

    unique_ids = {
        chunk["chunk_id"]
        for chunk in chunks_data
    }

    print("Unique chunk IDs:", len(unique_ids))

    # ----------------------------------------------
    # STEP 4: DISPLAY FIRST THREE CHUNKS
    # ----------------------------------------------

    for chunk in chunks_data[:3]:

        print("\nChunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("Chunk index:", chunk["chunk_index"])
        print("Source:", chunk["source"])
        print("Characters:", len(chunk["text"]))
        print("Text preview:", chunk["text"][:150])
        print("-" * 50)
