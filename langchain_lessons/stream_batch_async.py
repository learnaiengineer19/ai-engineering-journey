
"""
LESSON 18 - PART 7: STREAMING, BATCH AND ASYNC
==============================================

OBJECTIVES:
1. Compare invoke() with stream().
2. Run several requests sequentially.
3. Compare sequential execution with batch().
4. Understand async methods and asyncio.

RUN:
python -m langchain_lessons.stream_batch_async

NOTE:
Local model concurrency depends on hardware and
Ollama scheduling. Batch is not guaranteed faster.
"""

import asyncio
import time

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# STEP 1: BUILD A REUSABLE LCEL CHAIN
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an insurance assistant. "
        "Answer briefly in two sentences."
    ),
    (
        "human",
        "{question}"
    )
])

chain = prompt | model | StrOutputParser()


QUESTIONS = [
    {"question": "What is an insurance deductible?"},
    {"question": "What is an insurance premium?"},
]


# --------------------------------------------------
# STEP 2: NORMAL INVOCATION
# --------------------------------------------------

def test_invoke():

    print("\n========== INVOKE ==========")

    start = time.perf_counter()

    result = chain.invoke(QUESTIONS[0])

    elapsed = time.perf_counter() - start

    print("ANSWER:")
    print(result)

    print(f"\nTotal time: {elapsed:.2f}s")


# --------------------------------------------------
# STEP 3: STREAMING
# --------------------------------------------------

def test_stream():

    print("\n========== STREAM ==========")

    start = time.perf_counter()

    first_chunk_time = None
    chunk_count = 0

    print("ANSWER:")

    for chunk in chain.stream(QUESTIONS[0]):

        if first_chunk_time is None:
            first_chunk_time = (
                time.perf_counter() - start
            )

        chunk_count += 1

        # Print each chunk immediately.
        print(chunk, end="", flush=True)

    elapsed = time.perf_counter() - start

    print("\n")

    print("Chunks received:", chunk_count)

    print(
        f"Time to first chunk: "
        f"{first_chunk_time:.2f}s"
        if first_chunk_time is not None
        else "No chunks received"
    )

    print(f"Total time: {elapsed:.2f}s")


# --------------------------------------------------
# STEP 4: SEQUENTIAL EXECUTION
# --------------------------------------------------

def test_sequential():

    print("\n========== SEQUENTIAL ==========")

    start = time.perf_counter()

    results = []

    for question in QUESTIONS:
        answer = chain.invoke(question)
        results.append(answer)

    elapsed = time.perf_counter() - start

    print("Answers received:", len(results))
    print(f"Total time: {elapsed:.2f}s")


# --------------------------------------------------
# STEP 5: BATCH EXECUTION
# --------------------------------------------------

def test_batch():

    print("\n========== BATCH ==========")

    start = time.perf_counter()

    results = chain.batch(
        QUESTIONS,
        config={
            "max_concurrency": 2
        }
    )

    elapsed = time.perf_counter() - start

    for index, result in enumerate(results, start=1):
        print(f"\nAnswer {index}: {result}")

    print(f"\nTotal time: {elapsed:.2f}s")


# --------------------------------------------------
# STEP 6: ASYNC INVOCATION
# --------------------------------------------------

async def test_async():

    print("\n========== ASYNC ==========")

    start = time.perf_counter()

    results = await chain.abatch(
        QUESTIONS,
        config={
            "max_concurrency": 2
        }
    )

    elapsed = time.perf_counter() - start

    print("Answers received:", len(results))
    print(f"Async total time: {elapsed:.2f}s")


# --------------------------------------------------
# STEP 7: ASYNC STREAMING
# --------------------------------------------------

async def test_async_stream():

    print("\n========== ASYNC STREAM ==========")

    chunk_count = 0
    start = time.perf_counter()

    try:
        async for chunk in chain.astream(QUESTIONS[0]):

            chunk_count += 1
            print(chunk, end="", flush=True)

        print("\n")
        print("Chunks received:", chunk_count)

        elapsed = time.perf_counter() - start
        print(f"Total time: {elapsed:.2f}s")

    except Exception as exc:
        print("\n\nASYNC STREAM ERROR")
        print("Exception type:", type(exc).__name__)
        print("Exception message:", str(exc))

        # Print the complete traceback.
        import traceback
        traceback.print_exc()

        raise


# --------------------------------------------------
# STEP 8: RUN EXPERIMENTS
# --------------------------------------------------

if __name__ == "__main__":

    # Warm up the local model before timing.
    print("[WARMUP] Loading Qwen2")
    model.invoke("Reply with OK.")

    # test_invoke()
    # test_stream()
    # test_sequential()
    # test_batch()

    # asyncio.run(test_async())
    asyncio.run(test_async_stream())
