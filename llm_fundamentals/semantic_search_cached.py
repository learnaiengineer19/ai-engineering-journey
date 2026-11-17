
"""
LESSON 10: SEMANTIC SEARCH WITH SAVED EMBEDDINGS
================================================

OBJECTIVE:
Reuse saved document embeddings to perform
semantic search without regenerating document
embeddings for every user question.

CONCEPTS:
1. Loading persisted embeddings
2. Query embedding generation
3. Cosine similarity
4. Sorting and ranking
5. Top-K document retrieval

AI ENGINEERING CONNECTION:
Separating ingestion and retrieval improves
performance and reduces repeated computation.
"""

from ollama import embed
from sklearn.metrics.pairwise import cosine_similarity

from load_embeddings import load_embeddings


def search_saved_documents(
    query: str,
    filename: str,
    top_k: int = 3
) -> list[tuple[str, float]]:

    # STEP 1: Validate inputs
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # STEP 2: Load saved document data
    data = load_embeddings(filename)

    documents = data["documents"]
    document_embeddings = data["embeddings"]

    if not documents:
        return []

    # STEP 3: Generate ONLY the query embedding
    response = embed(
        model=data["model"],
        input=query.strip()
    )

    query_embedding = response.embeddings

    # STEP 4: Calculate similarity scores
    similarity = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    # STEP 5: Pair document indices with scores
    document_scores = []

    for index, score in enumerate(similarity[0]):
        document_scores.append((index, float(score)))

    # STEP 6: Sort by similarity (highest first)
    ranked_scores = sorted(
        document_scores,
        key=lambda item: item[1],
        reverse=True
    )

    # STEP 7: Select top-K results
    top_k_values = ranked_scores[:top_k]

    # STEP 8: Map indices back to original documents
    results = []

    for index, score in top_k_values:
        results.append((documents[index], score))

    return results


# --------------------------------------------------
# TEST THE SEMANTIC SEARCH
# --------------------------------------------------

if __name__ == "__main__":

    query = "What documents do I need after a car crash?"

    results = search_saved_documents(
        query=query,
        filename="llm_fundamentals/embeddings.json",
        top_k=2
    )

    print("\nUser question:", query)

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Document: {document}")
