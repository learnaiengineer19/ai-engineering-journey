
"""
Compare direct Ollama SDK with LangChain.

Both approaches use the same local Qwen2 model.
"""

import time

from ollama import chat
from langchain_ollama import ChatOllama


QUESTION = (
    "What is retrieval augmented generation? "
    "Explain in two sentences."
)


# --------------------------------------------------
# APPROACH 1: DIRECT OLLAMA
# --------------------------------------------------

start = time.perf_counter()

ollama_response = chat(
    model="qwen2:7b",
    messages=[
        {
            "role": "user",
            "content": QUESTION
        }
    ],
    options={"temperature": 0}
)

ollama_time = time.perf_counter() - start

print("\nDIRECT OLLAMA RESPONSE:")
print(ollama_response.message.content)

print(f"\nOllama elapsed: {ollama_time:.2f}s")


# --------------------------------------------------
# APPROACH 2: LANGCHAIN
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

start = time.perf_counter()

langchain_response = model.invoke(QUESTION)

langchain_time = time.perf_counter() - start

print("\nLANGCHAIN RESPONSE:")
print(langchain_response.content)

print(f"\nLangChain elapsed: {langchain_time:.2f}s")


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print("\n========== COMPARISON ==========")

print(f"Direct Ollama: {ollama_time:.2f}s")
print(f"LangChain:     {langchain_time:.2f}s")
