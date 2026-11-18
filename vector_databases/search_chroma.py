
"""
LESSON 11: VECTOR SIMILARITY SEARCH WITH CHROMADB
=================================================

OBJECTIVE:
Search stored document embeddings using ChromaDB
and retrieve the most relevant documents.

CONCEPTS COVERED:
1. Persistent vector database
2. Query embeddings
3. Nearest-neighbour similarity search
4. Document IDs and metadata
5. Vector distances
6. Ranking search results

AI ENGINEERING CONNECTION:
Vector search is the retrieval component of RAG.
It identifies relevant documents that can later
be passed to an LLM for grounded answer generation.

LOCAL MODELS:
- nomic-embed-text: Generates embeddings
- qwen2:7b: Generates answers (used in later steps)
"""

import chromadb
from ollama import embed


# --------------------------------------------------
# STEP 1: CONNECT TO PERSISTENT CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

# Access the collection created in Part 3.
collection = client.get_collection(
    name="insurance_documents",
    embedding_function=None
)


# --------------------------------------------------
# STEP 2: DEFINE USER QUESTION
# --------------------------------------------------

query = "What documents do I need after a car crash?"


# --------------------------------------------------
# STEP 3: GENERATE QUERY EMBEDDING
# --------------------------------------------------

# Convert the user's question into a numerical vector.
# Use the same embedding model used during ingestion.

response = embed(
    model="nomic-embed-text",
    input=query
)

# Extract the first (and only) query vector.
query_embedding = response.embeddings[0]


# --------------------------------------------------
# STEP 4: SEARCH THE VECTOR DATABASE
# --------------------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)


# --------------------------------------------------
# STEP 5: EXTRACT SEARCH RESULTS
# --------------------------------------------------

# Chroma returns nested lists because it supports
# searching for multiple queries at once.

document_ids = results["ids"][0]
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# --------------------------------------------------
# STEP 6: DISPLAY SEARCH RESULTS
# --------------------------------------------------

print("\nUser Question:", query)
print("Total results:", len(documents))

for index, document in enumerate(documents):

    print(f"\nRank {index + 1}")

    print("ID:", document_ids[index])
    print("Document:", document)
    print("Metadata:", metadatas[index])
    print("Distance:", round(distances[index], 4))


# --------------------------------------------------
# STEP 7: DISPLAY THE CLOSEST DOCUMENT
# --------------------------------------------------

if documents:

    print("\nMost Relevant Document:")
    print(documents[0])

    print("Distance:", round(distances[0], 4))
