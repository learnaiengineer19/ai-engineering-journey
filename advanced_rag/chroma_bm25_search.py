import chromadb

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

collection = client.get_collection(
    name="insurance_policy_chunks"
)

data = collection.get(
    include=["documents", "metadatas"]
)

chunk_ids = data["ids"]
documents = data["documents"]
metadatas = data["metadatas"]

print("Total chunks:", len(documents))


import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Convert document text into lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


# Convert all 16 chunks into lists of tokens.
tokenized_documents = [
    tokenize(document)
    for document in documents
]

# Build the BM25 index from the tokenized chunks.
bm25 = BM25Okapi(tokenized_documents)

print("BM25 index built for:", len(tokenized_documents), "chunks")

def search_bm25(query: str, top_k: int = 3) -> list[dict]:
    """
    Search our insurance-policy chunks using BM25.

    Args:
        query: User's question.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        Ranked list of dictionaries containing:
        - chunk_id
        - text
        - metadata
        - score
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    # Tokenize the user's question.
    query_tokens = tokenize(query)

    # Score all 16 chunks using the existing BM25 index.
    scores = bm25.get_scores(query_tokens)

    # Associate each score with the correct chunk.
    results = []

    for index, score in enumerate(scores):
        results.append({
            "chunk_id": chunk_ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "score": float(score)
        })

    # Sort from highest score to lowest.
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    # Exclude chunks with no positive BM25 relevance.
    relevant_results = [
        result for result in results
        if result["score"] > 0
    ]

    return relevant_results[:top_k]

if __name__ == "__main__":

    question = "Is mechanical engine breakdown covered?"

    results = search_bm25(
        query=question,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("USER QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("BM25 PDF SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"] or {}

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page_number')}")
        print(f"BM25 Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}")
        print("-" * 50)