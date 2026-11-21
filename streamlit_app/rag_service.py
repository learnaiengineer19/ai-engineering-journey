
"""
LESSON 14 - PART 3: STREAMLIT RAG SERVICE
=========================================

OBJECTIVE:
Create a reusable service layer that connects
the Streamlit frontend to our existing Advanced
RAG retrieval and generation pipeline.

RESPONSIBILITIES:
1. Accept a user question.
2. Execute advanced hybrid retrieval.
3. Rerank retrieved document chunks.
4. Build grounded context.
5. Generate an answer using Ollama Qwen2.
6. Return the answer and supporting sources.

IMPORTANT:
No Streamlit UI code should be placed here.

RUN:
This module is imported by Streamlit.
"""

from sentence_transformers import CrossEncoder

from advanced_rag.advanced_pdf_rag import (
    retrieve_and_rerank,
    build_context,
    generate_answer,
)

from ollama import chat
import logging
from ollama import ResponseError

from streamlit_app.errors import (
    OllamaUnavailableError,
    PolicyIndexUnavailableError,
    RAGProcessingError,
)

logger = logging.getLogger(__name__)


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_reranker() -> CrossEncoder:
    """
    Load our cross-encoder model on the CPU.

    We use CPU to avoid the MPS out-of-memory
    error previously encountered on the Mac.

    Streamlit will cache this resource later.
    """

    return CrossEncoder(
        RERANKER_MODEL,
        device="cpu"
    )



def contextualize_question(
    question: str,
    chat_history: list[dict]
) -> str:
    """
    LESSON 14 - PART 5.2: CONVERSATIONAL QUERY REWRITING

    OBJECTIVE:
    Convert a context-dependent follow-up question
    into a standalone question using chat history.

    EXAMPLE:
        Previous: Is engine breakdown covered?
        Current: What if it happened in an accident?

        Standalone:
        Is engine damage caused by an accident covered?

    IMPORTANT:
    - Do not invent facts.
    - Preserve the user's original intent.
    - History is used to understand references,
      not as evidence of insurance coverage.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    # No previous conversation means no need
    # to resolve references from history.
    if not chat_history:
        return question

    # Keep only the most recent six messages.
    # This prevents unlimited conversation growth
    # from expanding the LLM prompt.
    recent_history = chat_history[-6:]

    # Convert historical messages into readable text.
    # Note: only role and content are included.
    # We do not pass source dictionaries to this rewriter.
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )

    response = chat(
        model="qwen2:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the current user question as "
                    "a standalone question using the "
                    "conversation history only when needed. "
                    "Resolve references such as 'it', "
                    "'that', and 'they'. "
                    "Preserve the original meaning. "
                    "Do not invent policy facts or coverage "
                    "conditions. "
                    "Do not answer the question. "
                    "If the question is already standalone, "
                    "return it unchanged. "
                    "Return only the standalone question."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Current question:\n{question}"
                )
            }
        ],
        options={"temperature": 0}
    )

    return response.message.content.strip() or question



def ask_insurance_assistant(
    question: str,
    reranker: CrossEncoder,
    chat_history: list[dict] | None = None
) -> dict:
    """
    Execute conversational RAG with basic
    application-level error handling.
    """

    # Validate before calling the AI models.
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        # STEP 1: Resolve conversational references.
        standalone_question = contextualize_question(
            question=question,
            chat_history=chat_history or []
        )

        # STEP 2: Execute existing hybrid retrieval.
        result = retrieve_and_rerank(
            question=standalone_question,
            reranker=reranker,
            retrieval_top_k=5,
            final_top_k=3
        )

        # STEP 3: Extract selected evidence.
        sources = result["final_results"]

        # Avoid generating a grounded answer
        # when no policy evidence was retrieved.
        if not sources:
            return {
                "question": question,
                "standalone_question": standalone_question,
                "rewritten_query": result["rewritten_query"],
                "answer": (
                    "I couldn't find relevant information "
                    "in the provided insurance policy."
                ),
                "sources": []
            }

        # STEP 4: Build PDF context.
        context = build_context(sources)

        # STEP 5: Generate the answer using Ollama.
        answer = generate_answer(
            question=standalone_question,
            context=context
        )

        # STEP 6: Return structured response.
        return {
            "question": question,
            "standalone_question": standalone_question,
            "rewritten_query": result["rewritten_query"],
            "answer": answer,
            "sources": sources
        }

    except ConnectionError as exc:
        # Ollama's Python client raises ConnectionError
        # when it cannot connect to the local server.
        logger.exception("Ollama connection failed")

        raise OllamaUnavailableError(
            "Cannot connect to Ollama. "
            "Please check that Ollama is running."
        ) from exc

    except ResponseError as exc:
        # For example: Ollama reports a missing model.
        logger.exception("Ollama returned an error")

        raise RAGProcessingError(
            "Ollama could not complete the request. "
            "Please check the configured models."
        ) from exc

    except FileNotFoundError as exc:
        logger.exception("Required local file is missing")

        raise PolicyIndexUnavailableError(
            "A required policy file or search-index "
            "resource is missing."
        ) from exc

    except Exception as exc:
        # Unexpected errors are recorded for debugging.
        # We do not expose their raw details to users.
        logger.exception("Unexpected RAG processing error")

        raise RAGProcessingError(
            "The AI assistant encountered an unexpected "
            "processing error."
        ) from exc

