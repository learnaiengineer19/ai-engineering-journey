
"""
LESSON 03: FUNCTIONS, LOOPS AND JSON
====================================

OBJECTIVE:
Learn how to write reusable functions, process collections,
and convert Python objects to and from JSON.

CONCEPTS COVERED:

1. FUNCTIONS
   - Reusable blocks of code defined using 'def'.
   - Accept inputs through parameters.
   - Can return results using 'return'.

2. FOR LOOPS
   - Iterate through lists or other iterable objects.
   - Useful for processing multiple records.

3. IF CONDITIONS
   - Execute code only when a condition is satisfied.
   - The == operator checks equality.

4. JSON
   - A standard text format for exchanging structured data.
   - json.dumps(): Python object -> JSON string.
   - json.loads(): JSON string -> Python object.

5. LIST METHODS
   - append(): Adds an element to the end of a list.

AI ENGINEERING CONNECTION:
   - Functions help organize AI pipelines into components.
   - Loops process conversations and retrieved documents.
   - JSON is widely used in API requests, tool arguments,
     structured LLM outputs and application configuration.

"""


# --------------------------------------------------
# 1. Functions: Create a conversation message
# --------------------------------------------------

def create_message(role, content):
    message = {
        "role": role,
        "content": content
    }
    return message


user_message = create_message("user", "What is RAG?")

print("Created message:", user_message)


# --------------------------------------------------
# 2. Lists and loops: Process a conversation
# --------------------------------------------------

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What is an LLM?"}
]

print("\nFull conversation:")

for message in messages:
    print(message["role"], ":", message["content"])


# --------------------------------------------------
# 3. Conditions: Find user messages
# --------------------------------------------------

print("\nQuestions asked by the user:")

for message in messages:
    if message["role"] == "user":
        print(message["content"])


# --------------------------------------------------
# 4. JSON: Convert Python objects to JSON
# --------------------------------------------------

import json

sample_message = {
    "role": "user",
    "content": "Explain embeddings"
}

# Convert Python dictionary into JSON text
json_text = json.dumps(sample_message)

print("\nJSON string:", json_text)
print("JSON type:", type(json_text))


# Convert JSON text back into a Python dictionary
python_dict = json.loads(json_text)

print("Converted dictionary:", python_dict)
print("Content:", python_dict["content"])


# --------------------------------------------------
# 5. HANDS-ON EXERCISE
# --------------------------------------------------

# TASK:
# Create a function called get_user_messages(messages).
#
# Requirements:
# 1. Accept a list of message dictionaries.
# 2. Create an empty list to store user messages.
# 3. Iterate through each message.
# 4. Check whether the role is "user".
# 5. Append the content of matching messages.
# 6. Return the resulting list.
#
# EXPECTED OUTPUT:
# ['What is Python?', 'What is an LLM?']


def get_user_messages(messages: list[dict[str, str]]) -> list[str]:
    
    user_messages = []
    for message in messages:
        if message["role"] == "user":
            user_messages.append(message["content"])
    return user_messages


result = get_user_messages(messages)

print("\nExtracted user messages:", result)

test_messages = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "assistant", "content": "Hello!"}
]

print(get_user_messages(test_messages))
print(get_user_messages([]))
