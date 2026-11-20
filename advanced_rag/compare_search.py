from advanced_rag.chroma_bm25_search import search_bm25

import chromadb
from ollama import embed

CHROMA_PATH = "vector_databases/chroma_data"
COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_MODEL = "nomic-embed-text"


def search_semantic(query: str, top_k: int = 3) -> list[dict]:
    """
    Search existing PDF chunks using vector embeddings.

    Args:
        query: User question.
        top_k: Maximum results to retrieve.

    Returns:
        Ranked chunks with IDs, text, metadata
        and ChromaDB distances.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Convert the question into a 768-dimensional
    # embedding using our existing Ollama model.
    response = embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = list(response.embeddings[0])

    # Search the existing stored embeddings.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns nested lists because it
    # supports multiple queries in one request.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    ranked_results = []

    for index in range(len(ids)):

        ranked_results.append({
            "chunk_id": ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "distance": float(distances[index])
        })

    return ranked_results


if __name__ == "__main__":

    question = "Is mechanical engine breakdown covered?"

    bm25_results = search_bm25(
        query=question,
        top_k=3
    )

    semantic_results = search_semantic(
        query=question,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("USER QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("BM25 KEYWORD SEARCH")
    print("=" * 60)

    for rank, result in enumerate(bm25_results, start=1):
        print(
            f"{rank}. {result['chunk_id']} "
            f"| Score: {result['score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("SEMANTIC VECTOR SEARCH")
    print("=" * 60)

    for rank, result in enumerate(semantic_results, start=1):
        print(
            f"{rank}. {result['chunk_id']} "
            f"| Distance: {result['distance']:.4f}"
        )