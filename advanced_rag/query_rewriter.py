
"""
LESSON 13 - PART 8: QUERY REWRITING
===================================

OBJECTIVE:
Use a local Ollama language model to generate a
search-friendly version of the user's question.

IMPORTANT:
The original question is never overwritten.

Rewritten queries are used only for retrieval.
The original question remains the source of truth
for answer generation.

MODEL:
qwen2:7b

RUN:
python -m advanced_rag.query_rewriter
"""

from ollama import chat


LLM_MODEL = "qwen2:7b"


def rewrite_query(question: str) -> str:
    """
    Generate a search-friendly version of a question.

    Args:
        question: Original user question.

    Returns:
        Rewritten search query.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string")

    if not question.strip():
        raise ValueError("Question cannot be empty")

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite user questions for searching "
                    "a motor insurance policy document. "
                    "Preserve the original intent and scope. "
                    "Do not invent facts, policy clauses, "
                    "coverage details, or assumptions. "
                    "If the question is unrelated to motor "
                    "insurance, preserve its meaning rather "
                    "than forcing it into the insurance domain. "
                    "Return only one rewritten search query "
                    "without explanation."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature": 0
        }
    )

    rewritten_query = response.message.content.strip()

    # Fall back to the original question if
    # the model generates an empty response.
    return rewritten_query or question


if __name__ == "__main__":

    questions = [
        "Is mechanical engine breakdown covered?",
        "Does normal wear and tear qualify for insurance coverage?",
        "What documents are required for an accident claim?",
        "What is the capital of France?"
    ]

    for question in questions:

        rewritten = rewrite_query(question)

        print("\n" + "=" * 60)
        print("ORIGINAL QUESTION")
        print(question)

        print("\nREWRITTEN SEARCH QUERY")
        print(rewritten)
