
"""
LESSON 09: REUSABLE LLM CLIENT
==============================

OBJECTIVE:
Build a reusable Python function to interact
with a locally hosted LLM using Ollama.

CONCEPTS:
1. Input validation
2. Reusable functions
3. API requests
4. Exception handling
5. Type hints

MODEL: qwen2:7b
"""

from ollama import chat, ResponseError



def generate_response(
    prompt: str,
    model: str = "qwen2:7b"
) -> str | None:

    # Step 1: Validate data type
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    # Step 2: Validate empty input
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Step 3: Clean the prompt
    prompt = prompt.strip()

    messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": prompt
    }
]

    try:
        response = chat(
            model=model,
            messages=messages
            )   

    except ResponseError as error:
        print("Ollama error:", error)
        return None

    except ConnectionError as error:
        print("Connection error:", error)
        return None

    # Temporary return for testing
    return response.message.content


print(generate_response("Explain embeddings in two sentences."))

try:
    generate_response(123)
except TypeError as error:
    print("Type error:", error)

try:
    generate_response("   ")
except ValueError as error:
    print("Value error:", error)


answer = generate_response(
    prompt="Hello",
    model="invalid-model"
)

print("Result:", answer)