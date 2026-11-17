
"""
LESSON 10: BASIC RAG EVALUATION
===============================

OBJECTIVE:
Evaluate whether a RAG application correctly
answers or abstains when supporting information
is missing.

CONCEPTS:
1. Test datasets
2. Expected behaviour
3. Automated evaluation
4. Grounding and answerability

NOTE:
This initial evaluation uses labelled expectations
and simple keyword checks.

It does NOT prove factual correctness.
"""

test_cases = [
    {
        "question": "What documents do I need after a car crash?",
        "expected_answerable": True,
        "expected_keywords": ["police report", "photographs"]
    },
    {
        "question": "Does my car insurance cover engine replacement?",
        "expected_answerable": False,
        "expected_keywords": []
    }
]

def evaluate_keywords(
    answer: str,
    expected_keywords: list[str]
) -> bool:

    for keyword in expected_keywords:
        if keyword not in answer.lower():
            return False
    return True


answer_a = (
    "You will need a police report and "
    "photographs of the vehicle damage."
)

answer_b = (
    "The provided documents do not specify "
    "whether engine replacement is covered."
)

print(
    evaluate_keywords(
        answer_a,
        ["police report", "photographs"]
    )
)

print(
    evaluate_keywords(
        answer_a,
        ["police report", "repair invoice"]
    )
)


abstention_phrases = [
    "does not specify",
    "not enough information",
    "cannot determine",
    "not provided",
    "insufficient information"
]


def evaluate_abstention(answer: str) -> bool:
    answer_lower = answer.lower()

    for phrase in abstention_phrases:
        if phrase in answer_lower:
            return True

    return False


# --------------------------------------------------
# TEST CASES
# --------------------------------------------------

answer_a = (
    "You need a police report and photographs."
)

answer_b = (
    "The context provided does not specify "
    "whether engine replacement is covered."
)

print("Answer A — Keyword check:")
print(
    evaluate_keywords(
        answer_a,
        ["police report", "photographs"]
    )
)

print("Answer B — Abstention check:")
print(evaluate_abstention(answer_b))
