
"""
LESSON 12 - PART 7: PDF INGESTION INTO CHROMADB
==============================================

OBJECTIVE:
Process a PDF document, generate embeddings for
its chunks and store them in ChromaDB.

PIPELINE:
PDF
  |
  v
Extract Text (PyMuPDF)
  |
  v
Clean Text
  |
  v
Split Text into Overlapping Chunks
  |
  v
Generate Embeddings (Ollama)
  |
  v
Store IDs + Text + Embeddings + Metadata
  |
  v
ChromaDB Persistent Collection

CONCEPTS COVERED:
1. Reusing Python modules
2. List comprehensions
3. Extracting data from dictionaries
4. Batch embedding generation
5. ChromaDB persistent collections
6. upsert() for repeatable ingestion
7. Validating record counts
8. Verifying stored documents

MODELS:
- nomic-embed-text (768-dimensional embeddings)

RUN FROM PROJECT ROOT:
python document_processing/ingest_pdf.py

PREREQUISITES:
- Ollama running locally
- nomic-embed-text model downloaded
- PDF available in document_processing/documents/
- process_pdf() implemented in pdf_chunker.py
"""

from pathlib import Path

import chromadb
from ollama import embed

from pdf_chunker import process_pdf


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

PDF_PATH = Path(
    "document_processing/documents/insurance_policy.pdf"
)

CHROMA_PATH = "vector_databases/chroma_data"

COLLECTION_NAME = "insurance_policy_chunks"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --------------------------------------------------
# STEP 2: PROCESS PDF
# --------------------------------------------------

# The process_pdf() function:
# 1. Opens the PDF
# 2. Extracts text from each page
# 3. Cleans the text
# 4. Creates overlapping chunks
# 5. Adds chunk IDs and page metadata
# 6. Returns a list of dictionaries

chunks_data = process_pdf(
    pdf_path=PDF_PATH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print("\nTotal chunks generated:", len(chunks_data))

if not chunks_data:
    raise ValueError("No text chunks were generated.")


# --------------------------------------------------
# STEP 3: PREPARE CHROMADB FIELDS
# --------------------------------------------------

# Extract unique IDs from each chunk dictionary.

chunk_ids = [
    chunk["chunk_id"]
    for chunk in chunks_data
]


# Extract the text content from each chunk.

documents = [
    chunk["text"]
    for chunk in chunks_data
]


# Extract metadata from each chunk.
#
# Metadata is stored separately from the actual
# document text.
#
# Later, we can use page_number and source
# for PDF citations.

metadatas = [
    {
        "source": chunk["source"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"]
    }
    for chunk in chunks_data
]


# --------------------------------------------------
# STEP 4: GENERATE EMBEDDINGS
# --------------------------------------------------

# Ollama supports batch embedding generation.
# We send all chunk texts as a list.

print("\nGenerating embeddings using Ollama...")

response = embed(
    model=EMBEDDING_MODEL,
    input=documents
)

# Each item in response.embeddings represents
# the embedding of the corresponding chunk.
#
# Convert each vector into a Python list.

embeddings = [
    list(vector)
    for vector in response.embeddings
]


# --------------------------------------------------
# STEP 5: VALIDATE INGESTION DATA
# --------------------------------------------------

print("\nINGESTION VALIDATION")
print("-" * 50)

print("Total chunk IDs:", len(chunk_ids))
print("Total documents:", len(documents))
print("Total metadata records:", len(metadatas))
print("Total embeddings:", len(embeddings))

# Ensure each chunk has:
# - one ID
# - one document
# - one metadata dictionary
# - one embedding

if not (
    len(chunk_ids)
    == len(documents)
    == len(metadatas)
    == len(embeddings)
):
    raise ValueError(
        "Ingestion record counts do not match."
    )

# Validate unique chunk IDs.

if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError(
        "Duplicate chunk IDs detected."
    )

# Validate embedding dimensions.

embedding_dimensions = len(embeddings[0])

print("Embedding dimensions:", embedding_dimensions)

if any(
    len(vector) != embedding_dimensions
    for vector in embeddings
):
    raise ValueError(
        "Embedding dimensions are inconsistent."
    )


# --------------------------------------------------
# STEP 6: CONNECT TO PERSISTENT CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# Create a separate collection for PDF chunks.
#
# embedding_function=None means we supply
# embeddings ourselves through Ollama.

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    metadata={
        "embedding_model": EMBEDDING_MODEL
    }
)


# --------------------------------------------------
# STEP 7: STORE CHUNKS IN CHROMADB
# --------------------------------------------------

# upsert() inserts new records when IDs are new.
#
# If the ID already exists, upsert() updates
# that record rather than creating a duplicate.

collection.upsert(
    ids=chunk_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\nPDF chunks stored successfully!")


# --------------------------------------------------
# STEP 8: VERIFY STORED RECORDS
# --------------------------------------------------

print("\nCHROMADB VERIFICATION")
print("-" * 50)

print("Collection:", collection.name)
print("Total stored records:", collection.count())

# Retrieve a specific chunk by ID.
#
# This avoids assuming Chroma returns records
# in the same order they were inserted.

first_chunk_id = chunk_ids[0]

stored_data = collection.get(
    ids=[first_chunk_id],
    include=["documents", "metadatas"]
)

print("\nSample stored record:")

if stored_data["ids"]:
    print("Chunk ID:", stored_data["ids"][0])
    print("Text:", stored_data["documents"][0][:200])
    print("Metadata:", stored_data["metadatas"][0])


# --------------------------------------------------
# STEP 9: INGESTION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("PDF INGESTION COMPLETED")
print("=" * 50)

print("PDF:", PDF_PATH.name)
print("Chunks generated:", len(chunks_data))
print("Collection:", COLLECTION_NAME)
print("Embedding model:", EMBEDDING_MODEL)
print("Embedding dimensions:", embedding_dimensions)
print("Records in collection:", collection.count())
