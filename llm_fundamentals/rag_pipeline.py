
"""
LESSON 10: REUSABLE RAG PIPELINE
===============================

OBJECTIVE:
Build a reusable RAG function that combines
saved document embeddings, semantic retrieval
and LLM-based answer generation.

CONCEPTS:
1. Function abstraction
2. Semantic search
3. Context construction
4. Prompt engineering
5. Local LLM integration
6. Error handling

AI ENGINEERING CONNECTION:
A reusable RAG pipeline can power chatbots,
document question-answering systems and
AI-powered internal knowledge assistants.

MODELS:
Embedding: nomic-embed-text
Generation: qwen2:7b
"""

from ollama import chat, ResponseError
from semantic_search_cached import search_saved_documents


def ask_rag(
    query: str,
    filename: str = "llm_fundamentals/embeddings.json",
    top_k: int = 2
) -> str | None:
    """
    Answer a question using retrieved documents.

    Args:
        query: User's question.
        filename: Saved embeddings JSON file.
        top_k: Number of documents to retrieve.

    Returns:
        Generated answer or None if a model request fails.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    retrieved_documents = search_saved_documents(
        query=query,
        filename=filename,
        top_k=top_k
        )


    context = ""

    for index, (document, score) in enumerate(
        retrieved_documents,
        start=1
        ):
        context += f"[Document {index}] {document}\n"

    messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance information assistant. "
            "Answer using only the provided context. "
            "If the context does not contain enough information, "
            "say so explicitly. "
            "Do not invent insurance requirements. "
            "Cite supporting document numbers."
        )
    },
    {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    }
    ]

    try:
        response = chat(
            model="qwen2:7b",
            messages=messages)
        return response.message.content

    except ResponseError as error:
        print("error:", error)
        return None
    except ConnectionError as error:
        print("error:", error)
        return None


if __name__ == "__main__":

    question = "What documents do I need after a car crash?"

    answer = ask_rag(question)

    print("\nQuestion:", question)
    print("\nRAG Answer:", answer)
