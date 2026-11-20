"""
LESSON 13 - PART 10.2: RETRIEVAL EVALUATION
===========================================

OBJECTIVE:
Evaluate retrieval quality using Hit Rate@K.

Hit Rate@K measures whether at least one
known relevant chunk is present in the
top-K search results. It doesn't care whether it appeared at Rank 1 or Rank 3.

RETURN VALUES:
1 = Relevant evidence retrieved.
0 = No relevant evidence retrieved.
"""


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 3
) -> int:
    """
    Calculate Hit Rate@K for one question.

    Args:
        retrieved_ids:
            Chunk IDs returned by the retriever,
            ordered from most to least relevant.

        relevant_ids:
            Known relevant chunk IDs.

        k:
            Number of top results to evaluate.

    Returns:
        1 if any relevant chunk is retrieved,
        otherwise 0.
    """

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    # Consider only the top-K retrieved chunks.
    top_k_ids = retrieved_ids[:k]

    # Convert expected relevant IDs into a set.
    # Set membership checks are efficient.
    relevant_set = set(relevant_ids)

    for chunk_id in top_k_ids:

        if chunk_id in relevant_set:
            return 1

    return 0

def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 3
) -> float:
    """
    Calculate reciprocal rank for one question.

    The score is based on the position of the
    FIRST relevant chunk in the top-K results.

    Examples:
        Rank 1 -> 1.0
        Rank 2 -> 0.5
        Rank 3 -> 0.3333
        Not found -> 0.0
    """

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0

def mean_reciprocal_rank(
    reciprocal_ranks: list[float]
) -> float:
    """
    Calculate the average reciprocal rank
    across multiple evaluation questions.
    """

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

if __name__ == "__main__":

    relevant_ids = [
        "insurance_policy_page_3_chunk_1"
    ]

    # Test A: Correct chunk at Rank 1.
    retrieved_a = [
        "insurance_policy_page_3_chunk_1",
        "insurance_policy_page_3_chunk_2",
        "insurance_policy_page_4_chunk_3"
    ]

    # Test B: Correct chunk at Rank 3.
    retrieved_b = [
        "insurance_policy_page_1_chunk_1",
        "insurance_policy_page_2_chunk_2",
        "insurance_policy_page_3_chunk_1"
    ]

    # Test C: Correct chunk absent.
    retrieved_c = [
        "insurance_policy_page_1_chunk_1",
        "insurance_policy_page_2_chunk_2",
        "insurance_policy_page_4_chunk_3"
    ]



    print("Test A:", hit_rate_at_k(retrieved_a, relevant_ids, k=3))
    print("Test B:", hit_rate_at_k(retrieved_b, relevant_ids, k=3))
    print("Test C:", hit_rate_at_k(retrieved_c, relevant_ids, k=3))

    print("\nRECIPROCAL RANK RESULTS")

    rr_a = reciprocal_rank_at_k(
        retrieved_a, relevant_ids, k=3
    )

    rr_b = reciprocal_rank_at_k(
        retrieved_b, relevant_ids, k=3
    )

    rr_c = reciprocal_rank_at_k(
        retrieved_c, relevant_ids, k=3
    )

    print(f"Test A: {rr_a:.4f}")
    print(f"Test B: {rr_b:.4f}")
    print(f"Test C: {rr_c:.4f}")

    mrr = mean_reciprocal_rank([rr_a, rr_b, rr_c])

    print(f"\nMean Reciprocal Rank@3: {mrr:.4f}")