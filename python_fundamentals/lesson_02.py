
"""
LESSON 02: PYTHON LISTS AND DICTIONARIES
========================================

OBJECTIVE:
Learn how lists and dictionaries work and how they are used
to represent conversations in LLM applications.

CONCEPTS COVERED:

1. LIST
   - An ordered collection of items.
   - Defined using square brackets [].
   - Indexing starts at 0.
   - Example: messages[1] accesses the second item.

2. DICTIONARY
   - Stores information as key-value pairs.
   - Defined using curly braces {}.
   - Values are accessed using their keys.
   - Example: message["content"] retrieves the content.

3. LIST OF DICTIONARIES
   - Combines multiple structured records into one collection.
   - Commonly used for LLM conversation messages.
   - Each message contains a role and content.

4. FOR LOOP
   - Iterates through items in a collection.
   - Allows us to process each message individually.

AI ENGINEERING CONNECTION:
LLM APIs commonly represent conversations as lists of messages.
Each message identifies the sender (role) and message text (content).

"""

# --------------------------------------------------
# 1. Create a list of dictionaries
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "What is RAG?"
    },
    {
        "role": "assistant",
        "content": "RAG stands for Retrieval-Augmented Generation."
    }
]


# --------------------------------------------------
# 2. Find the total number of messages
# --------------------------------------------------

print("Total messages:", len(messages))


# --------------------------------------------------
# 3. Access individual messages using list indexing
# --------------------------------------------------

# Index 1 = second message (user)
print("User question:", messages[1]["content"])

# Index 2 = third message (assistant)
print("Assistant response:", messages[2]["content"])


# --------------------------------------------------
# 4. Iterate through every message
# --------------------------------------------------

print("\nFull conversation:")

for message in messages:
    print("Role:", message["role"])
    print("Content:", message["content"])
    print("--------------------")
