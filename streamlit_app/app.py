
"""
LESSON 14 - PART 5: STREAMLIT CHAT HISTORY
==========================================

OBJECTIVE:
Preserve user questions, assistant answers, and
supporting PDF sources across Streamlit reruns.

CONCEPTS:
1. st.session_state
2. Storing messages as dictionaries
3. Appending user and assistant messages
4. Reconstructing chat history on every rerun
5. Preserving PDF source metadata
6. Clearing conversation history
7. Reusing a cached cross-encoder model

IMPORTANT:
This part implements visible chat history.

It does NOT yet pass previous messages into the
RAG backend for context-aware follow-up questions.
That will be our next exercise.

RUN FROM PROJECT ROOT:
python -m streamlit run streamlit_app/app.py
"""

import streamlit as st
from streamlit_app.citations import analyse_citations

from streamlit_app.rag_service import (
    ask_insurance_assistant,
    load_reranker,
)

from streamlit_app.errors import (
    OllamaUnavailableError,
    PolicyIndexUnavailableError,
    RAGProcessingError,
)

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------
# STEP 1: CONFIGURE STREAMLIT
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Policy AI Assistant",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Policy AI Assistant")

st.caption(
    "Advanced RAG with ChromaDB, BM25, "
    "Ollama and cross-encoder reranking"
)


# --------------------------------------------------
# STEP 2: LOAD AND CACHE THE RERANKER
# --------------------------------------------------

@st.cache_resource
def get_reranker():
    """
    Load the cross-encoder once and reuse it
    across Streamlit script reruns.

    CPU is used to avoid the MPS out-of-memory
    error we encountered during Lesson 13.
    """

    return load_reranker()


# --------------------------------------------------
# STEP 3: INITIALISE CHAT HISTORY
# --------------------------------------------------

# Session State works like a dictionary.
#
# The message list is created once per session.
# Existing messages survive script reruns.

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# STEP 4: CREATE CLEAR HISTORY BUTTON
# --------------------------------------------------

# This button clears the conversation stored in
# the current Streamlit session.


# --------------------------------------------------
# LESSON 14 - PART 9: APPLICATION SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🛡️ Insurance AI Assistant")

    st.write(
        "Ask questions about the sample motor "
        "insurance policy using Advanced RAG."
    )

    st.divider()

    st.subheader("Technology Stack")

    st.write("• Streamlit")
    st.write("• Ollama (Qwen2)")
    st.write("• ChromaDB")
    st.write("• BM25 + Hybrid Search")
    st.write("• Cross-Encoder Reranking")

    st.divider()

    st.subheader("Example Questions")

    st.write("1. Is mechanical engine breakdown covered?")
    st.write("2. What documents are required for a claim?")
    st.write("3. Is normal tyre wear covered?")

    st.divider()

    # Display conversation statistics.
    message_count = len(st.session_state.messages)

    st.metric(
        "Messages in Conversation",
        message_count
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()



# --------------------------------------------------
# STEP 5: CREATE SOURCE DISPLAY FUNCTION
# --------------------------------------------------

def display_sources(
    sources: list[dict],
    answer: str
) -> None:
    """
    Render expandable supporting PDF sources.

    Called for both:
    - Previously saved assistant messages
    - Newly generated assistant messages

    Args:
        sources: Retrieved PDF chunks with metadata.
    """

    if not sources:
        return

    with st.expander("View supporting PDF sources"):

        for index, source in enumerate(sources, start=1):

            metadata = source.get("metadata") or {}

            page_number = metadata.get(
                "page_number",
                "Unknown"
            )

            filename = metadata.get(
                "source",
                "Unknown"
            )

            st.markdown(
                f"**Source {index} — Page {page_number}**"
            )

            st.write(f"File: {filename}")

            st.write(
                f"Chunk ID: {source['chunk_id']}"
            )

            st.write(
                f"Reranker Score: "
                f"{source['reranker_score']:.4f}"
            )

            st.write(source["text"])

            st.divider()


# --------------------------------------------------
# STEP 6: DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

# On every rerun, Streamlit rebuilds the screen.
#
# We loop through saved messages and render each
# one using its stored role and content.

if not st.session_state.messages:

    with st.chat_message("assistant"):

        st.write(
            "👋 Welcome! I'm your Insurance Policy AI Assistant."
        )

        st.write(
            "You can ask me about coverage, exclusions, "
            "accident claims and policy conditions."
        )

        st.info(
            "My answers are based on the provided sample "
            "insurance policy. Please check the cited "
            "PDF evidence for important details."
        )

else:

    for message in st.session_state.messages:

        # Example:
        # message["role"] could be "user"
        # or "assistant".
        with st.chat_message(message["role"]):

            st.write(message["content"])

            # User messages have no PDF sources.
            # Assistant messages may contain them.
            if message["role"] == "assistant":

                display_sources(
                    message.get("sources", [])
                )


# --------------------------------------------------
# STEP 7: ACCEPT A NEW USER QUESTION
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your insurance policy..."
)


# --------------------------------------------------
# STEP 8: HANDLE NEW USER MESSAGE
# --------------------------------------------------


# --------------------------------------------------
# LESSON 14 - PART 8: SAFE REQUEST PROCESSING
# --------------------------------------------------

if question and question.strip():

    previous_history = st.session_state.messages.copy()

    # Save and display the user's question.
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Searching insurance policy..."):

                # Loading is inside try because model
                # initialisation can also fail.
                reranker = get_reranker()

                response = ask_insurance_assistant(
                    question=question,
                    reranker=reranker,
                    chat_history=previous_history
                )

            st.write(response["answer"])

            display_sources(
                sources=response["sources"],
                answer=response["answer"]
            )

            # Existing contextual-question expander.
            with st.expander("View contextualised question"):
                st.write(response["standalone_question"])

            # Save only after a successful response.
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            })

        except OllamaUnavailableError as exc:
            st.error(str(exc))

        except PolicyIndexUnavailableError as exc:
            st.error(str(exc))

        except RAGProcessingError as exc:
            st.error(str(exc))

        except Exception:
            # Also catches failures occurring during
            # reranker loading and UI integration.
            logger.exception("Unexpected Streamlit request error")

            st.error(
                "Something went wrong while processing "
                "your question. Please try again."
            )

