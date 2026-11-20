"""
LESSON 13 - PART 8.1
QUERY REWRITING + HYBRID RETRIEVAL
==================================

OBJECTIVE:
Compare retrieval using the original question
and an LLM-generated rewritten question.

IMPORTANT:
- Never discard the original question.
- Use rewriting only to improve retrieval.
- Deduplicate results using stable chunk IDs.
- Preserve metadata for citations.

RUN:
python -m advanced_rag.query_rewrite_hybrid
"""

from advanced_rag.query_rewriter import rewrite_query
from advanced_rag.chroma_bm25_search import search_bm25
from advanced_rag.compare_search import search_semantic
from advanced_rag.rrf_fusion import reciprocal_rank_fusion


def retrieve_with_rewriting(
    question: str,
    top_k: int = 5
) -> dict:
    """
    Retrieve insurance chunks using both the
    original and rewritten questions.

    Returns:
        {
            "original_question": str,
            "rewritten_query": str,
            "original_results": list,
            "rewritten_results": list
        }
    """

    rewritten_query = rewrite_query(question)

    # STEP 1: Search with original question.
    original_bm25 = search_bm25(question, top_k)
    original_semantic = search_semantic(question, top_k)

    original_results = reciprocal_rank_fusion(
        original_bm25,
        original_semantic
    )

    # STEP 2: Search with rewritten question.
    rewritten_bm25 = search_bm25(rewritten_query, top_k)
    rewritten_semantic = search_semantic(rewritten_query, top_k)

    rewritten_results = reciprocal_rank_fusion(
        rewritten_bm25,
        rewritten_semantic
    )

    return {
        "original_question": question,
        "rewritten_query": rewritten_query,
        "original_results": original_results,
        "rewritten_results": rewritten_results
    }


if __name__ == "__main__":

    question = (
        "Does normal wear and tear qualify "
        "for insurance coverage?"
    )

    result = retrieve_with_rewriting(question)

    print("\nORIGINAL QUESTION:")
    print(result["original_question"])

    print("\nREWRITTEN QUERY:")
    print(result["rewritten_query"])

    print("\nTOP 3 ORIGINAL RETRIEVAL RESULTS:")

    for rank, chunk in enumerate(
        result["original_results"][:3],
        start=1
    ):
        print(
            f"{rank}. {chunk['chunk_id']} "
            f"| RRF: {chunk['rrf_score']:.6f}"
        )

    print("\nTOP 3 REWRITTEN RETRIEVAL RESULTS:")

    for rank, chunk in enumerate(
        result["rewritten_results"][:3],
        start=1
    ):
        print(
            f"{rank}. {chunk['chunk_id']} "
            f"| RRF: {chunk['rrf_score']:.6f}"
        )