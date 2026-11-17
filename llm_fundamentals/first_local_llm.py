
"""
LESSON 09: FIRST LOCAL LLM API CALL
===================================

OBJECTIVE:
Send a prompt to a locally hosted LLM using Ollama
and read its generated response.

CONCEPTS COVERED:
1. Local LLM inference
2. Ollama Python SDK
3. Model selection
4. Message roles
5. API requests and responses

AI ENGINEERING CONNECTION:
Understand how Python applications interact with LLMs
before building RAG pipelines and AI agents.
"""

from ollama import chat


# --------------------------------------------------
# 1. CONFIGURE MODEL
# --------------------------------------------------

model_name = "qwen2:7b"


# --------------------------------------------------
# 2. DEFINE CONVERSATION
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are an experienced AI engineer. "
            "You specialize in Large Language Models, "
            "Retrieval-Augmented Generation, and AI applications. "
            "Explain technical concepts accurately and simply."
        )
    },
    {
        "role": "user",
        "content": (
            "Explain Retrieval-Augmented Generation (RAG) "
            "in exactly three simple sentences."
        )
    }
]


# --------------------------------------------------
# 3. CALL THE LOCAL LLM
# --------------------------------------------------

response = chat(
    model=model_name,
    messages=messages
)


# --------------------------------------------------
# 4. DISPLAY RESPONSE
# --------------------------------------------------

print("\nLLM Response:")
print(response.message.content)
print("\n--- Model Statistics ---")

print("Input tokens:", response.prompt_eval_count)
print("Output tokens:", response.eval_count)
print("Total duration (ns):", response.total_duration)
