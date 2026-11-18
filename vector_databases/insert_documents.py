
"""
LESSON 11: INSERTING DOCUMENTS INTO CHROMADB
===========================================

OBJECTIVE:
Store insurance documents, embeddings and
metadata in a persistent ChromaDB collection.

CONCEPTS:
1. Vector database ingestion
2. Document IDs
3. Embedding generation
4. Metadata
5. ChromaDB collections
6. Persistent storage

AI ENGINEERING CONNECTION:
In production RAG applications, ingestion is
usually performed separately from querying.

EMBEDDING MODEL:
nomic-embed-text via local Ollama.
"""

import chromadb
from ollama import embed


# --------------------------------------------------
# STEP 1: DEFINE DOCUMENTS
# --------------------------------------------------

documents = [
    "Vehicle accident claims require a police report.",
    "Car insurance premiums are renewed annually.",
    "Travel insurance covers certain medical emergencies.",
    "Home insurance protects against fire and flood damage.",
    "Vehicle collision claims require photographs of the damage."
]


# --------------------------------------------------
# STEP 2: DEFINE UNIQUE DOCUMENT IDS
# --------------------------------------------------

document_ids = [
    "motor_001",
    "motor_002",
    "travel_001",
    "home_001",
    "motor_003"
]


# --------------------------------------------------
# STEP 3: DEFINE DOCUMENT METADATA
# --------------------------------------------------

metadatas = [
    {"category": "motor", "source": "claims_guide"},
    {"category": "motor", "source": "renewal_guide"},
    {"category": "travel", "source": "travel_guide"},
    {"category": "home", "source": "home_guide"},
    {"category": "motor", "source": "claims_guide"}
]


# --------------------------------------------------
# STEP 4: CONNECT TO CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

collection = client.get_or_create_collection(
    name="insurance_documents",
    embedding_function=None
)


# --------------------------------------------------
# STEP 5: GENERATE EMBEDDINGS USING OLLAMA
# --------------------------------------------------

response = embed(
    model="nomic-embed-text",
    input=documents
)

embeddings = [
    list(vector)
    for vector in response.embeddings
]


# --------------------------------------------------
# STEP 6: YOUR IMPLEMENTATION
# --------------------------------------------------

collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# --------------------------------------------------
# STEP 7: VERIFY THE INSERTION
# --------------------------------------------------

print("Collection:", collection.name)
print("Total records:", collection.count())

stored_data = collection.get(
    include=["documents", "metadatas"]
)

print(stored_data)
print("Stored IDs:", stored_data["ids"])
print("Stored documents:", stored_data["documents"])
print("Stored metadata:", stored_data["metadatas"])
