
"""
LESSON 08: CONTEXT WINDOWS AND CONTEXT ENGINEERING
==================================================

OBJECTIVE:
Understand how LLM applications manage conversation
history and operate within context limits.

CONCEPTS COVERED:

1. CONTEXT WINDOW
   - Maximum amount of information a model can process.
   - Usually measured in tokens.

2. CONVERSATION HISTORY
   - Previous messages supplied to an LLM.
   - Often resent with each API request.

3. TRUNCATION
   - Removing older information to reduce context size.

4. CONTEXT ENGINEERING
   - Selecting and organizing relevant information.
   - Helps manage relevance, latency and cost.

AI ENGINEERING CONNECTION:
RAG applications and AI agents must manage their context
to avoid unnecessary information and token-limit errors.

NOTE:
This exercise limits the number of messages.
Real LLM systems generally budget tokens, not messages.
"""


# --------------------------------------------------
# 1. DEFINE CONVERSATION HISTORY
# --------------------------------------------------

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG combines retrieval and generation."},
    {"role": "user", "content": "What are embeddings?"},
    {"role": "assistant", "content": "Embeddings represent information as vectors."},
    {"role": "user", "content": "What is semantic search?"}
]


# --------------------------------------------------
# 2. DISPLAY CONVERSATION
# --------------------------------------------------

print("Total messages:", len(messages))

for message in messages:
    print(message["role"], ":", message["content"])



def manage_context(
    messages: list[dict[str, str]],
    max_messages: int
) -> list[dict[str, str]]:

    if max_messages < 1:
        raise ValueError("max_messages must be at least 1")

    system_messages = []
    conversation_history = []

    # Separate system messages and conversation history
    for message in messages:

        if message["role"] == "system":
            system_messages.append(message)
        else:
            conversation_history.append(message)

    # Keep only the latest messages
    recent_messages = conversation_history[-max_messages:]

    # Combine system messages and recent history
    managed_context = system_messages + recent_messages

    return managed_context




result = manage_context(messages, 3)

for message in result:
    print(message["role"], ":", message["content"])