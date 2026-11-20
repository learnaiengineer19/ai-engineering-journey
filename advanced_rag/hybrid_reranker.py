from sentence_transformers import CrossEncoder

from advanced_rag.chroma_bm25_search import search_bm25
from advanced_rag.compare_search import search_semantic
from advanced_rag.rrf_fusion import reciprocal_rank_fusion

def rerank_documents(
    question: str,
    documents: list[dict],
    model: CrossEncoder,
    top_k: int = 3
) -> list[dict]:
    """
    Rerank hybrid search results using a cross-encoder.

    Each document dictionary must contain:
        chunk_id, text, metadata, rrf_score

    Returns:
        Top-K documents sorted by reranker score.
    """

    if not documents:
        return []

    # STEP 1: Build question-document pairs.
    pairs = [
        (question, document["text"])
        for document in documents
    ]

    # STEP 2: Calculate relevance scores.
    scores = model.predict(pairs)

    # STEP 3: Attach a reranker score to each document.
    reranked_results = []

    for index, score in enumerate(scores):
        reranked_results.append({
            **documents[index],
            "reranker_score": float(score)
        })

    # STEP 4: Sort from highest to lowest relevance.
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]

def filter_by_relevance(
    reranked_results: list[dict],
    min_score: float = 0.0
) -> list[dict]:
    """
    Filter retrieved chunks based on reranker score.

    Args:
        reranked_results:
            Documents sorted by reranker relevance.

        min_score:
            Experimental minimum relevance score.

    Returns:
        Documents with reranker_score >= min_score.

    Note:
        The threshold must be evaluated using
        representative test questions before
        production use.
    """

    filtered_results = [
        result
        for result in reranked_results
        if result["reranker_score"] >= min_score
    ]

    return filtered_results


if __name__ == "__main__":

    question = "What is the capital of France?"

    # Stage 1: Keyword retrieval
    bm25_results = search_bm25(question, top_k=5)

    # Stage 2: Semantic retrieval
    semantic_results = search_semantic(question, top_k=5)

    # Stage 3: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results
    )

    # Stage 4: Cross-encoder reranking
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    reranked_results = rerank_documents(
        question=question,
        documents=fused_results,
        model=model,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("HYBRID SEARCH + RERANKING RESULTS")
    print("=" * 60)

    for rank, result in enumerate(reranked_results, start=1):
        metadata = result["metadata"] or {}

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page_number')}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")
        print(f"Retrieved By: {result['retrieved_by']}")
        print(f"Text Preview: {result['text'][:200]}")


    final_results = reranked_results[:3]

    print("\nFINAL RESULTS WITHOUT THRESHOLD")

    for rank, result in enumerate(final_results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")

    filtered_results = filter_by_relevance(
        reranked_results,
        min_score=0.0
    )

    print("\n" + "=" * 60)
    print("RELEVANCE FILTERING RESULTS")
    print("=" * 60)

    print("Before filtering:", len(reranked_results))
    print("After filtering:", len(filtered_results))

    for rank, result in enumerate(filtered_results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")

    