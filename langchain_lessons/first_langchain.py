
"""
LESSON 18 - PART 1: FIRST LANGCHAIN APPLICATION
================================================

OBJECTIVE:
Understand how LangChain integrates with Ollama.

CONCEPTS:
1. ChatOllama
2. Model initialization
3. invoke()
4. AIMessage
5. Response metadata
6. Token usage

RUN:
python -m langchain_lessons.first_langchain
"""

from langchain_ollama import ChatOllama


# --------------------------------------------------
# STEP 1: INITIALIZE THE LLM
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)


# --------------------------------------------------
# STEP 2: DEFINE THE QUESTION
# --------------------------------------------------

question = (
    "Explain what an insurance deductible is "
    "in two simple sentences."
)


# --------------------------------------------------
# STEP 3: INVOKE THE MODEL
# --------------------------------------------------

print("\nUSER QUESTION:")
print(question)

response = model.invoke(question)


# --------------------------------------------------
# STEP 4: DISPLAY THE RESPONSE
# --------------------------------------------------

print("\nAI RESPONSE:")
print(response.content)


# --------------------------------------------------
# STEP 5: INSPECT THE LANGCHAIN RESPONSE OBJECT
# --------------------------------------------------

print("\n========== RESPONSE DETAILS ==========")

print("Response type:", type(response).__name__)

print("Content:", response.content)

print("Response metadata:", response.response_metadata)

print("Token usage:", response.usage_metadata)
