
"""
LESSON 12: PDF TEXT EXTRACTION WITH PYMUPDF
==========================================

OBJECTIVE:
Extract text from a PDF document using Python.

CONCEPTS COVERED:
1. Opening PDF documents
2. Reading document metadata
3. Iterating through PDF pages
4. Extracting text from each page
5. Handling empty pages
6. Preparing document text for RAG

AI ENGINEERING CONNECTION:
PDF extraction is the first step in a document
ingestion pipeline. The extracted text will
later be cleaned, chunked, embedded and
stored in ChromaDB.

LIBRARY:
PyMuPDF

RUN FROM PROJECT ROOT:
python document_processing/pdf_extractor.py
"""


from pathlib import Path
import pymupdf
from text_cleaner import clean_text

PDF_PATH = Path(
    "document_processing/documents/insurance_policy.pdf"
)

if not PDF_PATH.is_file():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )

pages_data = []

with pymupdf.open(PDF_PATH) as document:

    print("PDF filename:", PDF_PATH.name)
    print("Total pages:", len(document))

    for page_number, page in enumerate(
        document, start=1
    ):

        page_text = page.get_text("text", sort=True)

        cleaned_text = clean_text(page_text)

        pages_data.append({
            "page_number": page_number,
            "source": PDF_PATH.name,
            "text": cleaned_text
        })

        removed_characters = len(page_text) - len(cleaned_text)

        print(f"\nPage: {page_number}")
        print("Original characters:", len(page_text))
        print("Cleaned characters:", len(cleaned_text))
        print("Removed characters:", removed_characters)
        print("Cleaned preview:", cleaned_text[:300])

print("\nTotal page records:", len(pages_data))

