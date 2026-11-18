
"""
LESSON 11: METADATA FILTERING WITH CHROMADB
==========================================

OBJECTIVE:
Combine semantic vector search with structured
metadata filters.

CONCEPTS COVERED:
1. Metadata filtering
2. ChromaDB where parameter
3. Filtered vector similarity search
4. Query embeddings using Ollama
5. Ranked results and distances

AI ENGINEERING CONNECTION:
Production RAG applications often contain
documents belonging to different customers,
products, categories or jurisdictions.

Metadata filtering lets us limit retrieval
to the relevant subset of documents.
"""

import chromadb
from ollama import embed


# --------------------------------------------------
# STEP 1: CONNECT TO PERSISTENT CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

collection = client.get_collection(
    name="insurance_documents",
    embedding_function=None
)


# --------------------------------------------------
# STEP 2: DEFINE QUESTION AND CATEGORY
# --------------------------------------------------

query = "What documents do I need after a car crash?"

category = "motor"


# --------------------------------------------------
# STEP 3: GENERATE QUERY EMBEDDING
# --------------------------------------------------

response = embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = response.embeddings[0]


# --------------------------------------------------
# STEP 4: SEARCH USING METADATA FILTER
# --------------------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"category": category},
    include=["documents", "metadatas", "distances"]
)


# --------------------------------------------------
# STEP 5: DISPLAY SEARCH RESULTS
# --------------------------------------------------

documents = results["documents"][0]
document_ids = results["ids"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for index, doc in enumerate(documents):
    print(f"\nRank: {index + 1}")
    print(f"Document ID: {document_ids[index]}")
    print(f"Document: {doc}")
    print(f"Category: {metadatas[index]['category']}")
    print(f"Distance: {distances[index]:.4f}")
    print("-" * 50)

# TODO:
# 1. Extract IDs, documents, metadata and distances
# 2. Loop through the retrieved documents
# 3. Print each document's rank, text and category
# 4. Print its distance
