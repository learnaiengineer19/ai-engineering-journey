
"""
LESSON 13 - PART 10.4: COMPARE RETRIEVAL METHODS
================================================

OBJECTIVE:
Evaluate three retrieval techniques using the
same insurance-policy questions.

RETRIEVERS:
1. BM25 keyword search
2. ChromaDB semantic search
3. Hybrid search using RRF

METRICS:
- Hit Rate@3
- Mean Reciprocal Rank@3

IMPORTANT:
We compare retrieved chunk IDs against
our manually identified relevant chunk IDs.

RUN FROM PROJECT ROOT:
python -m advanced_rag.evaluation.compare_retrievers
"""

from advanced_rag.chroma_bm25_search import search_bm25
from advanced_rag.compare_search import search_semantic
from advanced_rag.rrf_fusion import reciprocal_rank_fusion

from advanced_rag.evaluation.test_cases import EVALUATION_CASES

from advanced_rag.evaluation.evaluate_retrieval import (
    hit_rate_at_k,
    reciprocal_rank_at_k,
    mean_reciprocal_rank
)


TOP_K = 3


def evaluate_retriever(
    retriever_name: str,
    search_function,
    evaluation_cases: list[dict],
    top_k: int = 3
) -> dict:
    """
    Evaluate a retrieval method across test questions.

    Args:
        retriever_name: Descriptive retriever name.
        search_function: Callable returning ranked chunks.
        evaluation_cases: Questions and relevant chunk IDs.
        top_k: Number of retrieved results to evaluate.

    Returns:
        Dictionary containing metric averages and
        individual question results.
    """

    hit_scores = []
    reciprocal_ranks = []
    question_results = []

    for test_case in evaluation_cases:

        question = test_case["question"]
        relevant_ids = test_case["relevant_chunk_ids"]

        # Execute the supplied retrieval method.
        retrieved_results = search_function(
            question,
            top_k=top_k
        )

        # Extract stable chunk IDs.
        retrieved_ids = [
            result["chunk_id"]
            for result in retrieved_results
        ]

        # Calculate metrics for this question.
        hit = hit_rate_at_k(
            retrieved_ids,
            relevant_ids,
            k=top_k
        )

        rr = reciprocal_rank_at_k(
            retrieved_ids,
            relevant_ids,
            k=top_k
        )

        hit_scores.append(hit)
        reciprocal_ranks.append(rr)

        question_results.append({
            "question": question,
            "retrieved_ids": retrieved_ids,
            "hit": hit,
            "reciprocal_rank": rr
        })

    return {
        "retriever": retriever_name,
        "hit_rate": sum(hit_scores) / len(hit_scores)
            if hit_scores else 0.0,
        "mrr": mean_reciprocal_rank(reciprocal_ranks),
        "question_results": question_results
    }


def search_hybrid(
    question: str,
    top_k: int = 3
) -> list[dict]:
    """
    Combine BM25 and semantic retrieval using RRF.

    Retrieve more candidates than the final top_k
    so both methods have a useful candidate pool.
    """

    candidate_k = max(5, top_k)

    bm25_results = search_bm25(
        query=question,
        top_k=candidate_k
    )

    semantic_results = search_semantic(
        query=question,
        top_k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results
    )

    return fused_results[:top_k]


if __name__ == "__main__":

    retrievers = [
        ("BM25", search_bm25),
        ("Semantic", search_semantic),
        ("Hybrid RRF", search_hybrid)
    ]

    print("\n" + "=" * 65)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 65)

    evaluation_results = []

    for name, search_function in retrievers:

        result = evaluate_retriever(
            retriever_name=name,
            search_function=search_function,
            evaluation_cases=EVALUATION_CASES,
            top_k=TOP_K
        )

        evaluation_results.append(result)

        print(f"\nRetriever: {name}")
        print(f"Hit Rate@{TOP_K}: {result['hit_rate']:.4f}")
        print(f"MRR@{TOP_K}: {result['mrr']:.4f}")

        for question_result in result["question_results"]:

            print(
                f"\n  Question: {question_result['question']}"
            )

            print(
                f"  Hit: {question_result['hit']}"
            )

            print(
                "  Reciprocal Rank: "
                f"{question_result['reciprocal_rank']:.4f}"
            )

            print(
                f"  Retrieved IDs: "
                f"{question_result['retrieved_ids']}"
            )

        print("-" * 65)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Retriever':<15} "
        f"{'Hit Rate@3':<15} "
        f"{'MRR@3':<15}"
    )

    for result in evaluation_results:

        print(
            f"{result['retriever']:<15} "
            f"{result['hit_rate']:<15.4f} "
            f"{result['mrr']:<15.4f}"
        )
