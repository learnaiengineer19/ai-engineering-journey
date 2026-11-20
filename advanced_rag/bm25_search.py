
"""
LESSON 13 - PART 2: BM25 KEYWORD SEARCH
=======================================

OBJECTIVE:
Build a reusable keyword-based document retrieval
system using the BM25 ranking algorithm.

CONCEPTS COVERED:
1. Lexical (keyword) search
2. Text tokenisation
3. BM25Okapi ranking
4. Document relevance scores
5. Sorting and top-K retrieval
6. Reusable Python functions
7. Input validation
8. Search result formatting

AI ENGINEERING CONNECTION:
BM25 complements vector-based semantic search.
It is useful for exact keywords, technical terms,
policy clauses and product identifiers.

LATER IN LESSON 13:
We will combine BM25 and ChromaDB using
Reciprocal Rank Fusion (RRF).

LIBRARY:
rank-bm25

RUN FROM PROJECT ROOT:
python advanced_rag/bm25_search.py
"""

import re
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# STEP 1: DEFINE SAMPLE INSURANCE DOCUMENTS
# --------------------------------------------------

# These are fictional examples for learning.
# They are not excerpts from our insurance PDF.

documents = [
    "Mechanical breakdown caused by wear and tear is excluded.",
    "Accident claims require photographs of vehicle damage.",
    "A police report may be needed for certain accident claims.",
    "Policy premiums are renewed annually.",
    "Engine damage caused by an insured collision may be covered."
]


# --------------------------------------------------
# STEP 2: CREATE TOKENISATION FUNCTION
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Convert a string into lowercase word tokens.

    Example:
        Input:  "Engine damage!"
        Output: ["engine", "damage"]

    Args:
        text: Input text.

    Returns:
        List of lowercase tokens.

    Raises:
        TypeError: If text is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# STEP 3: BUILD BM25 SEARCH INDEX
# --------------------------------------------------

def build_bm25_index(documents: list[str]) -> BM25Okapi:
    """
    Create a BM25 index from a collection of documents.

    Args:
        documents: List of document texts.

    Returns:
        Initialized BM25Okapi search index.

    Raises:
        ValueError: If no documents are supplied.
    """

    if not documents:
        raise ValueError("Documents cannot be empty")

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # BM25 builds statistics over the tokenized
    # collection, including term frequency,
    # document frequency and document lengths.

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


# --------------------------------------------------
# STEP 4: IMPLEMENT REUSABLE KEYWORD SEARCH
# --------------------------------------------------

def keyword_search(
    query: str,
    documents: list[str],
    bm25: BM25Okapi,
    top_k: int = 3
) -> list[tuple[str, float]]:
    """
    Retrieve the top-K documents ranked by BM25 score.

    Args:
        query: Search question.
        documents: Original document texts.
        bm25: Pre-built BM25 index.
        top_k: Maximum results to return.

    Returns:
        List of tuples:
            (document_text, bm25_score)

        Results are sorted by score, highest first.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty or top_k invalid.
    """

    # ----------------------------------------------
    # STEP 4.1: VALIDATE QUERY
    # ----------------------------------------------

    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if type(top_k) is not int or top_k < 1:
        raise ValueError(
            "top_k must be a positive integer"
        )

    # ----------------------------------------------
    # STEP 4.2: TOKENIZE QUERY
    # ----------------------------------------------

    query_tokens = tokenize(query)

    # ----------------------------------------------
    # STEP 4.3: CALCULATE BM25 SCORES
    # ----------------------------------------------

    # Returns one score for each original document.
    # The scores are in document-list order.

    scores = bm25.get_scores(query_tokens)

    # ----------------------------------------------
    # STEP 4.4: COMBINE DOCUMENTS WITH SCORES
    # ----------------------------------------------

    ranked_documents = []

    for index, score in enumerate(scores):

        ranked_documents.append(
            (
                documents[index],
                float(score)
            )
        )

    # ----------------------------------------------
    # STEP 4.5: SORT BY RELEVANCE
    # ----------------------------------------------

    # Higher BM25 score = stronger lexical match.

    ranked_documents = sorted(
        ranked_documents,
        key=lambda item: item[1],
        reverse=True
    )

    # ----------------------------------------------
    # STEP 4.6: RETURN TOP-K
    # ----------------------------------------------

    # Keep only documents with positive BM25 scores.
    relevant_documents = [
        (document, score)
        for document, score in ranked_documents
        if score > 0
    ]

    # Return at most top_k relevant documents.
    return relevant_documents[:top_k]


# --------------------------------------------------
# STEP 5: RUN A SAMPLE SEARCH
# --------------------------------------------------

if __name__ == "__main__":

    # Build the index once for these documents.
    bm25 = build_bm25_index(documents)

    # Define our insurance-related question.
    query = "Does my policy cover damage caused by a collision?"

    # Search the indexed documents.
    results = keyword_search(
        query=query,
        documents=documents,
        bm25=bm25,
        top_k=3
    )

    # ----------------------------------------------
    # STEP 6: DISPLAY SEARCH RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("USER QUESTION")
    print("=" * 60)
    print(query)

    print("\n" + "=" * 60)
    print("BM25 KEYWORD SEARCH RESULTS")
    print("=" * 60)

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"\nRank: {rank}")
        print(f"Document: {document}")
        print(f"BM25 Score: {score:.4f}")
        print("-" * 50)

    print("\nTotal results:", len(results))
