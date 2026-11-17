
"""
LESSON 10: FIRST SEMANTIC SEARCH ENGINE
======================================

OBJECTIVE:
Retrieve the most relevant document using embeddings
and cosine similarity.

CONCEPTS:
1. Document embeddings
2. Query embeddings
3. Cosine similarity
4. Ranking documents
5. Semantic retrieval

AI ENGINEERING CONNECTION:
This is the retrieval component of a RAG pipeline.
"""

from ollama import embed
from sklearn.metrics.pairwise import cosine_similarity


documents = [
    "Vehicle accident claims require a police report.",
    "Car insurance premiums are renewed annually.",
    "Travel insurance covers certain medical emergencies.",
    "Home insurance protects against fire and flood damage.",
    "Vehicle collision claims require photographs of the damage."
]

query = "What documents do I need after a car crash?"

def generate_embeddings(
    sentences: list[str]
) -> list[list[float]]:

    # Generate embeddings for all sentences
    # in a single Ollama request.
    response = embed(
        model="nomic-embed-text",
        input=sentences
    )

    # Convert each embedding into a Python list.
    result_list = []

    for embedding in response.embeddings:
        result_list.append(list(embedding))

    return result_list

document_embeddings = generate_embeddings(documents)
query_embedding = generate_embeddings(query)


scores = cosine_similarity(
    query_embedding,
    document_embeddings
)

# print(
#     f"Query score against the document "
#     f"{scores[0]}" 
#       )

# print("highest score", max(scores[0]))

best_index = scores[0].argmax()
# print(best_index)
best_document = documents[best_index]

# print("User question:", query)








def search_documents(
    query: str,
    documents: list[str],
    top_k: int = 3
) -> list[tuple[str, float]]:

    # Step 1: Validate inputs
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not documents:
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Step 2: Generate document embeddings
    document_embeddings = generate_embeddings(documents)

    # Step 3: Generate one embedding for the whole query
    query_embedding = generate_embeddings([query])

    # Step 4: Calculate similarity scores
    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    # Step 5: Pair each score with its document index
    document_scores = []

    for index, score in enumerate(scores[0]):
        document_scores.append((index, float(score)))

    # Step 6: Sort by similarity, highest first
    ranked_scores = sorted(
        document_scores,
        key=lambda item: item[1],
        reverse=True
    )

    # Step 7: Select top-k results
    top_k_values = ranked_scores[:top_k]

    # Step 8: Map indices back to document text
    results = []

    for index, score in top_k_values:
        results.append((documents[index], score))

    return results



if __name__ == "__main__":
    results = search_documents(
        query=query,
        documents=documents,
        top_k=3
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Document: {document}")
        
    print("Most relevant document:", best_document)