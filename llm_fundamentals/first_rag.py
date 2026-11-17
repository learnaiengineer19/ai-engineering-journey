
"""
LESSON 10: FIRST END-TO-END RAG APPLICATION
==========================================

OBJECTIVE:
Combine semantic search with a generative LLM
to answer questions using retrieved documents.

COMPONENTS:
1. Embedding model: nomic-embed-text
2. Retrieval: cosine similarity
3. Generative LLM: qwen2:7b
4. Context construction
5. Grounded answer generation

AI ENGINEERING CONNECTION:
This is the core architecture of a basic RAG system.

LIMITATIONS:
- Small in-memory document collection
- Document embeddings recalculated for each search
- No production vector database
- No automated relevance evaluation
- No source verification
"""

from ollama import chat

# Reuse your previously implemented retrieval function.
# Ensure semantic_search.py does not execute demo code on import.
from semantic_search import search_documents


documents = [
    "Vehicle accident claims require a police report.",
    "Car insurance premiums are renewed annually.",
    "Travel insurance covers certain medical emergencies.",
    "Home insurance protects against fire and flood damage.",
    "Vehicle collision claims require photographs of the damage."
]

query = "Does my car insurance cover engine replacement??"


# --------------------------------------------------
# STEP 1: RETRIEVE THE TOP TWO DOCUMENTS
# --------------------------------------------------

retrieved_documents = search_documents(
    query=query,
    documents=documents,
    top_k=2
)


# --------------------------------------------------
# STEP 2: BUILD THE CONTEXT
# --------------------------------------------------

context = ""

for index, (document, score) in enumerate(retrieved_documents, start=1):
    context += "[ Document " + str(index) + " ] " + document + "\n"


# --------------------------------------------------
# STEP 3: PREPARE THE LLM MESSAGES
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance information assistant. "
            "Answer the user's question using only the "
            "provided context. "
            "If the context does not contain the answer, "
            "say that you do not have enough information. "
            "Do not invent insurance requirements."
            "Also cite the document number for your answers in your response"
        )
    },
    {
        "role": "user",
        "content": (
            f"Context:\n{context}\n"
            f"Question: {query}\n"
            "Answer briefly and clearly."
        )
    }
]


# --------------------------------------------------
# STEP 4: CALL QWEN2
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages
)


# --------------------------------------------------
# STEP 5: DISPLAY THE ANSWER
# --------------------------------------------------

print("\nRetrieved Context:")
print(context)

print("\nRAG Answer:")
print(response.message.content)
