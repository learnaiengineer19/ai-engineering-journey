
"""
LESSON 18 - PART 8: MODEL COMPARISON
===================================

RUN:
python -m langchain_lessons.model_comparison

Measures:
- Response latency
- Input tokens
- Output tokens
- Optional estimated inference cost
"""

import os
import time
import json

from langchain_core.prompts import ChatPromptTemplate

from langchain_lessons.model_factory import get_model


# --------------------------------------------------
# STEP 1: LOAD CONFIGURED MODEL
# --------------------------------------------------

provider = os.getenv("LLM_PROVIDER", "ollama")

model = get_model()

model_name = {
    "ollama": "OLLAMA_MODEL",
    "openai": "OPENAI_MODEL",
    "bedrock": "BEDROCK_MODEL_ID"
}.get(provider.lower())

selected_model = (
    os.getenv(model_name, "qwen2:7b")
    if model_name else "unknown"
)


# --------------------------------------------------
# STEP 2: CREATE A FIXED TEST PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise insurance analyst. "
        "Give a clear answer in no more than "
        "three sentences."
    ),
    (
        "human",
        "{question}"
    )
])

chain = prompt | model

question = (
    "Explain the difference between an insurance "
    "premium and a deductible. Give one example."
)


# --------------------------------------------------
# STEP 3: EXECUTE AND MEASURE
# --------------------------------------------------

start = time.perf_counter()

response = chain.invoke({
    "question": question
})

elapsed = time.perf_counter() - start


# --------------------------------------------------
# STEP 4: EXTRACT USAGE METADATA
# --------------------------------------------------

usage = response.usage_metadata or {}

input_tokens = usage.get("input_tokens")
output_tokens = usage.get("output_tokens")
total_tokens = usage.get("total_tokens")


# --------------------------------------------------
# STEP 5: OPTIONAL COST ESTIMATION
# --------------------------------------------------

# Provide real, current per-million-token rates
# when evaluating a hosted provider.
#
# These environment variables default to unset,
# so we won't mistake unknown prices for $0.

input_rate = os.getenv("INPUT_USD_PER_1M")
output_rate = os.getenv("OUTPUT_USD_PER_1M")

estimated_cost = None

if (
    input_rate is not None
    and output_rate is not None
    and input_tokens is not None
    and output_tokens is not None
):

    estimated_cost = (
        input_tokens / 1_000_000 * float(input_rate)
        +
        output_tokens / 1_000_000 * float(output_rate)
    )


# --------------------------------------------------
# STEP 6: PRINT RESULT
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

print("Provider:", provider)
print("Model:", selected_model)

print("\nANSWER:")
print(response.content)

print("\n========== METRICS ==========")

print(f"Latency: {elapsed:.2f}s")
print("Input tokens:", input_tokens)
print("Output tokens:", output_tokens)
print("Total tokens:", total_tokens)

if estimated_cost is None:
    print("Estimated API cost: not configured")
else:
    print(
        f"Estimated API cost: "
        f"${estimated_cost:.6f} USD"
    )


# --------------------------------------------------
# STEP 7: SAVE EXPERIMENT
# --------------------------------------------------

record = {
    "provider": provider,
    "model": selected_model,
    "question": question,
    "latency_seconds": round(elapsed, 3),
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "total_tokens": total_tokens,
    "estimated_api_cost_usd": estimated_cost,
    "answer": response.content
}

with open(
    "langchain_lessons/model_comparison_result.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        record,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("\nExperiment saved successfully.")
