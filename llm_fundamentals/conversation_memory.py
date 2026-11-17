
"""
LESSON 08: CONVERSATION MEMORY
==============================

OBJECTIVE:
Understand how important information can be preserved
when conversation history is shortened.

CONCEPTS COVERED:
1. Context truncation
2. Conversation summarization
3. External memory
4. Rule-based information extraction
5. Combining memory with recent conversation history

AI ENGINEERING CONNECTION:
AI agents and conversational RAG systems often need
to retain relevant information across interactions.

NOTE:
This exercise uses manually defined rules.
It does not use an LLM or semantic retrieval.
"""


# --------------------------------------------------
# 1. SAMPLE CONVERSATION
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "I am travelling to Japan with my family."
    },
    {
        "role": "user",
        "content": "My total budget is $5000."
    },
    {
        "role": "user",
        "content": "I prefer vegetarian food."
    },
    {
        "role": "user",
        "content": "We want to visit Tokyo and Kyoto."
    },
    {
        "role": "user",
        "content": "Can you recommend restaurants?"
    }
]

memory = {
    "destination": None,
    "budget": None,
    "food_preference": None
}

def extract_memory(
    messages: list[dict[str, str]]
) -> dict[str, str | None]:

    memory = {
        "destination": None,
        "budget": None,
        "food_preference": None
    }

    for message in messages:

        content = message["content"].lower()

        if "japan" in content:
            memory["destination"] = "Japan"

        if "$5000" in content:
            memory["budget"] = "$5000"

        if "vegetarian" in content:
            memory["food_preference"] = "vegetarian"

    return memory

result = extract_memory(messages)

print("Extracted memory:", result)

