
"""
LESSON 13 - PART 9: ADVANCED PDF RAG INTEGRATION
================================================

OBJECTIVE:
Integrate the retrieval techniques developed
throughout Lesson 13 into a single pipeline.

PIPELINE:
1. Accept an original user question.
2. Rewrite the question using Ollama Qwen2.
3. Run BM25 keyword search.
4. Run ChromaDB semantic search.
5. Combine search rankings using RRF.
6. Repeat retrieval using the rewritten query.
7. Merge and deduplicate retrieved candidates.
8. Rerank candidates using a cross-encoder.
9. Select and display the top-K evidence chunks.

IMPORTANT:
- Always preserve the original user question.
- Query rewriting is used only for retrieval.
- Duplicate chunks are identified using chunk_id.
- RRF scores are not summed across separate
  original-query and rewritten-query searches.
- Reranking uses the original question.
- No hard relevance threshold is applied yet.

DEPENDENCIES:
chromadb
ollama
rank-bm25
sentence-transformers

RUN FROM PROJECT ROOT:
python -m advanced_rag.advanced_pdf_rag

NEXT:
Connect selected chunks to Qwen2 for grounded
answer generation with source citations.
"""

from ollama import chat
from sentence_transformers import CrossEncoder

from advanced_rag.query_rewrite_hybrid import (
    retrieve_with_rewriting
)

from advanced_rag.hybrid_reranker import (
    rerank_documents
)

LLM_MODEL = "qwen2:7b"

# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Number of results requested from each retrieval method.
RETRIEVAL_TOP_K = 5

# Maximum number of evidence chunks to retain.
FINAL_TOP_K = 3


# --------------------------------------------------
# STEP 2: MERGE RETRIEVAL RESULTS
# --------------------------------------------------

def merge_retrieval_results(
    original_results: list[dict],
    rewritten_results: list[dict]
) -> list[dict]:
    """
    Merge retrieved results from the original and
    rewritten search queries.

    WHY DO WE NEED THIS?

    The same document chunk can be retrieved by:

    - Original BM25 search
    - Original semantic search
    - Rewritten BM25 search
    - Rewritten semantic search

    Each query's BM25 and semantic results have
    already been combined using RRF.

    Here, we merge both candidate lists without
    duplicating chunks.

    Args:
        original_results:
            RRF results from the original question.

        rewritten_results:
            RRF results from the rewritten query.

    Returns:
        List of unique document dictionaries.

    Example:
        Original chunks:  [A, B, C]
        Rewritten chunks: [A, D, E]

        Merged chunks:    [A, B, C, D, E]
    """

    # Use chunk_id as the dictionary key.
    # A dictionary cannot contain duplicate keys.
    merged = {}

    # Concatenate both lists.
    #
    # Python processes original results first.
    # Rewritten results are processed afterwards.
    for result in original_results + rewritten_results:

        chunk_id = result["chunk_id"]

        # Only create an entry if we have not
        # encountered this chunk_id previously.
        if chunk_id not in merged:

            # .copy() creates a separate dictionary
            # so we don't accidentally modify the
            # original retrieval result.
            merged[chunk_id] = result.copy()

    # Return unique documents as a list.
    #
    # Dictionary insertion order is preserved,
    # so original-query candidates appear first.
    return list(merged.values())


# --------------------------------------------------
# STEP 3: CREATE ADVANCED RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve_and_rerank(
    question: str,
    reranker: CrossEncoder,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    final_top_k: int = FINAL_TOP_K
) -> dict:
    """
    Run the complete advanced retrieval pipeline.

    This function combines:
    - Query rewriting
    - BM25 retrieval
    - Semantic retrieval
    - RRF fusion
    - Candidate deduplication
    - Cross-encoder reranking

    Args:
        question:
            Original user question.

        reranker:
            Preloaded CrossEncoder model.

        retrieval_top_k:
            Number of candidates requested from
            each BM25 and semantic search.

        final_top_k:
            Maximum number of reranked chunks
            retained for final evidence.

    Returns:
        Dictionary containing:
        - original_question
        - rewritten_query
        - original_candidate_count
        - rewritten_candidate_count
        - merged_candidate_count
        - reranked_results
        - final_results
    """

    # ----------------------------------------------
    # STEP 3.1: VALIDATE USER INPUT
    # ----------------------------------------------

    if not isinstance(question, str):
        raise TypeError("Question must be a string")

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if type(retrieval_top_k) is not int or retrieval_top_k < 1:
        raise ValueError(
            "retrieval_top_k must be a positive integer"
        )

    if type(final_top_k) is not int or final_top_k < 1:
        raise ValueError(
            "final_top_k must be a positive integer"
        )

    # ----------------------------------------------
    # STEP 3.2: ORIGINAL + REWRITTEN RETRIEVAL
    # ----------------------------------------------

    # This existing function:
    #
    # 1. Rewrites the original question.
    # 2. Runs BM25 and semantic search for the
    #    original question.
    # 3. Combines original rankings using RRF.
    # 4. Runs BM25 and semantic search for the
    #    rewritten query.
    # 5. Combines rewritten rankings using RRF.
    #
    # It returns both ranked candidate lists.

    retrieval = retrieve_with_rewriting(
        question=question,
        top_k=retrieval_top_k
    )

    original_results = retrieval["original_results"]
    rewritten_results = retrieval["rewritten_results"]

    # ----------------------------------------------
    # STEP 3.3: MERGE UNIQUE CHUNKS
    # ----------------------------------------------

    merged_results = merge_retrieval_results(
        original_results=original_results,
        rewritten_results=rewritten_results
    )

    # ----------------------------------------------
    # STEP 3.4: HANDLE EMPTY RESULTS
    # ----------------------------------------------

    # We should never ask the reranker to score
    # an empty candidate collection.
    if not merged_results:

        return {
            "original_question": question,
            "rewritten_query": retrieval["rewritten_query"],
            "original_candidate_count": len(original_results),
            "rewritten_candidate_count": len(rewritten_results),
            "merged_candidate_count": 0,
            "reranked_results": [],
            "final_results": []
        }

    # ----------------------------------------------
    # STEP 3.5: CROSS-ENCODER RERANKING
    # ----------------------------------------------

    # IMPORTANT:
    # Rerank using the original user question,
    # not the rewritten search query.
    #
    # This reduces the risk of changing the
    # user's original intended meaning.
    #
    # Rerank ALL merged candidates first.
    # Do not limit them to the final top-K yet.

    reranked_results = rerank_documents(
        question=question,
        documents=merged_results,
        model=reranker,
        top_k=len(merged_results)
    )

    # ----------------------------------------------
    # STEP 3.6: SELECT FINAL TOP-K CHUNKS
    # ----------------------------------------------

    # These results are already sorted by
    # reranker score in descending order.
    #
    # We deliberately do not use a hard score
    # threshold in this lesson stage.
    #
    # A low reranker score does not automatically
    # prove that a chunk is irrelevant.

    final_results = reranked_results[:final_top_k]

    # ----------------------------------------------
    # STEP 3.7: RETURN STRUCTURED OUTPUT
    # ----------------------------------------------

    return {
        "original_question": question,
        "rewritten_query": retrieval["rewritten_query"],
        "original_candidate_count": len(original_results),
        "rewritten_candidate_count": len(rewritten_results),
        "merged_candidate_count": len(merged_results),
        "reranked_results": reranked_results,
        "final_results": final_results
    }


# --------------------------------------------------
# STEP 4: DISPLAY RETRIEVAL RESULTS
# --------------------------------------------------

def display_results(result: dict) -> None:
    """
    Print the outputs of the advanced retrieval
    pipeline in a readable format.
    """

    print("\n" + "=" * 60)
    print("ADVANCED PDF RAG - RETRIEVAL RESULTS")
    print("=" * 60)

    print("\nORIGINAL QUESTION:")
    print(result["original_question"])

    print("\nREWRITTEN QUERY:")
    print(result["rewritten_query"])

    # ----------------------------------------------
    # DISPLAY CANDIDATE COUNTS
    # ----------------------------------------------

    print("\n" + "-" * 60)
    print("RETRIEVAL STATISTICS")
    print("-" * 60)

    print(
        "Original query candidates:",
        result["original_candidate_count"]
    )

    print(
        "Rewritten query candidates:",
        result["rewritten_candidate_count"]
    )

    print(
        "Unique merged candidates:",
        result["merged_candidate_count"]
    )

    print(
        "Final selected chunks:",
        len(result["final_results"])
    )

    # ----------------------------------------------
    # DISPLAY FINAL TOP-K EVIDENCE
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("TOP RERANKED PDF CHUNKS")
    print("=" * 60)

    if not result["final_results"]:

        print("\nNo document chunks were retrieved.")
        return

    for rank, document in enumerate(
        result["final_results"],
        start=1
    ):

        # Some ChromaDB records might not have
        # metadata, so default to an empty dict.
        metadata = document["metadata"] or {}

        print(f"\nRank: {rank}")

        print(
            f"Chunk ID: {document['chunk_id']}"
        )

        print(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: {metadata.get('page_number', 'Unknown')}"
        )

        print(
            f"RRF Score: {document['rrf_score']:.6f}"
        )

        print(
            f"Reranker Score: "
            f"{document['reranker_score']:.4f}"
        )

        print(
            f"Retrieved By: {document['retrieved_by']}"
        )

        print(
            f"Text Preview: {document['text'][:300]}"
        )

        print("-" * 60)



def build_context(documents: list[dict]) -> str:
    """
    Convert selected PDF chunks into a context string
    containing source labels and page metadata.

    The labels allow Qwen2 to cite its evidence.
    """

    context_parts = []

    for index, document in enumerate(documents, start=1):

        metadata = document["metadata"] or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page_number", "Unknown")

        context_parts.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Text:\n{document['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate an insurance-policy answer using
    retrieved evidence only.

    If the context does not support the answer,
    instruct the model to abstain.
    """

    if not context.strip():
        return (
            "I couldn't find sufficient evidence "
            "in the provided insurance policy."
        )

    system_prompt = (
        "You are a document-grounded insurance policy assistant. "

        "Answer the user's question using only the supplied "
        "insurance policy excerpts. "

        "First identify all policy provisions that directly "
        "address the question. "

        "State those supported provisions clearly before "
        "discussing missing or uncertain information. "

        "If the question is broad but the policy gives only "
        "specific examples, explain those examples and "
        "clearly state the limits of what can be concluded. "

        "Do not generalise a specific exclusion into a "
        "universal rule unless the context supports it. "

        "Cite every factual claim using [Source N], where "
        "N is a source number in the supplied context. "

        "Never invent source numbers or policy provisions. "

        "If none of the supplied excerpts contains evidence "
        "relevant to the question, reply exactly: "
        "'I couldn't find sufficient evidence in the "
        "provided insurance policy.'"
    )

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the context and include "
                    "source citations."
                )
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response.message.content.strip()
# --------------------------------------------------
# STEP 5: MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------
    # STEP 5.1: DEFINE USER QUESTION
    # ----------------------------------------------

    question = (
        "Does normal wear and tear qualify "
        "for insurance coverage?"
    )

    # ----------------------------------------------
    # STEP 5.2: LOAD CROSS-ENCODER MODEL
    # ----------------------------------------------

    # Loading the model once allows us to reuse it
    # for subsequent user questions.
    #
    # The first run may download model files.
    # Later runs use the local Hugging Face cache.

    print("\nLoading cross-encoder reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu"
    )

    # ----------------------------------------------
    # STEP 5.3: EXECUTE ADVANCED RETRIEVAL
    # ----------------------------------------------

    result = retrieve_and_rerank(
        question=question,
        reranker=reranker,
        retrieval_top_k=RETRIEVAL_TOP_K,
        final_top_k=FINAL_TOP_K
    )

    # ----------------------------------------------
    # STEP 5.4: DISPLAY RESULTS
    # ----------------------------------------------

    display_results(result)

    # Select the reranked evidence.
    selected_documents = result["final_results"][:1]

    # Build the LLM context.
    context = build_context(selected_documents)

    # Generate an answer using the original question.
    answer = generate_answer(
        question=question,
        context=context
    )

    print("\n" + "=" * 60)
    print("ADVANCED RAG ANSWER")
    print("=" * 60)
    print(answer)