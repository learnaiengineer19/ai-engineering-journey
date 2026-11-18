
"""
LESSON 11: VECTOR DATABASES WITH CHROMADB
========================================

OBJECTIVE:
Understand vector databases and create a
persistent local ChromaDB database.

CONCEPTS COVERED:
1. Vector databases
2. Persistent storage
3. ChromaDB clients
4. Collections
5. Document IDs and metadata
6. Vector search (upcoming)

AI ENGINEERING CONNECTION:
A vector database provides persistent storage
and retrieval capabilities for RAG applications.

MODELS:
- nomic-embed-text: Embeddings (Ollama)
- qwen2:7b: Text generation (Ollama)
"""

import chromadb


# --------------------------------------------------
# 1. CREATE PERSISTENT CHROMA CLIENT
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)


# --------------------------------------------------
# 2. CREATE OR LOAD A COLLECTION
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="insurance_documents",
    embedding_function=None,
    metadata={"embedding_model": "nomic-embed-text"}
)


# --------------------------------------------------
# 3. INSPECT THE COLLECTION
# --------------------------------------------------

print("Collection name:", collection.name)
print("Total documents:", collection.count())

print("ChromaDB initialized successfully!")
