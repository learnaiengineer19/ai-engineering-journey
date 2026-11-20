
"""
LESSON 13 - PART 5: RECIPROCAL RANK FUSION (RRF)
================================================

OBJECTIVE:
Combine BM25 and semantic search rankings into
a single ranked result list.

WHY RRF?
BM25 scores and ChromaDB distances have different
scales and meanings. RRF uses rank positions
rather than comparing raw values.

FORMULA:
RRF score = sum(1 / (k + rank))

WHERE:
- k is a smoothing constant (default: 60)
- rank starts at 1
- each retrieval method contributes to the score

RUN FROM PROJECT ROOT:
python -m advanced_rag.rrf_fusion
"""

from advanced_rag.chroma_bm25_search import search_bm25
from advanced_rag.compare_search import search_semantic


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    semantic_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked result lists using RRF.

    Args:
        bm25_results: Ranked BM25 documents.
        semantic_results: Ranked semantic documents.
        k: RRF smoothing constant.

    Returns:
        Unique documents sorted by combined RRF score.
    """

    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")

    fused_results = {}

    # Process BM25 and semantic rankings.
    for search_method, results in [
        ("bm25", bm25_results),
        ("semantic", semantic_results)
    ]:

        # enumerate starts ranking at 1.
        for rank, result in enumerate(results, start=1):

            chunk_id = result["chunk_id"]

            # Create the result entry the first time
            # this chunk appears in either list.
            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "retrieved_by": []
                }

            # Add this method's contribution to
            # the chunk's combined RRF score.
            fused_results[chunk_id]["rrf_score"] += (
                1 / (k + rank)
            )

            # Track which retrieval methods found it.
            fused_results[chunk_id]["retrieved_by"].append(
                search_method
            )

    # Convert the dictionary values into a list.
    ranked_results = list(fused_results.values())

    # Sort by descending RRF score.
    # chunk_id provides deterministic tie-breaking.
    ranked_results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["chunk_id"]
        )
    )

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

    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        semantic_results=semantic_results,
        k=60
    )

    print("\n" + "=" * 60)
    print("USER QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("HYBRID SEARCH - RRF RESULTS")
    print("=" * 60)

    for rank, result in enumerate(fused_results, start=1):

        metadata = result["metadata"] or {}

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page_number')}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Retrieved by: {result['retrieved_by']}")
        print(f"Text: {result['text'][:200]}")
        print("-" * 50)
