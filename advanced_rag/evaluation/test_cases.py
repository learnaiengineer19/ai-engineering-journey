
"""
LESSON 13 - PART 10.1: RAG EVALUATION DATASET
=============================================

OBJECTIVE:
Create a collection of evaluation questions and
their expected supporting document chunks.

WHY DO WE NEED THIS?

Without expected results, we cannot reliably
measure retrieval accuracy.

Each test case contains:
- question: User's search question.
- relevant_chunk_ids: Known supporting chunks.

IMPORTANT:
These labels come from our previous manual
inspection of the insurance policy.

We will expand and review the dataset later.
"""

EVALUATION_CASES = [
    {
        "question": "Is mechanical engine breakdown covered?",
        "relevant_chunk_ids": [
            "insurance_policy_page_3_chunk_1"
        ]
    },
    {
        "question": (
            "Does normal wear and tear qualify "
            "for insurance coverage?"
        ),
        "relevant_chunk_ids": [
            "insurance_policy_page_3_chunk_1",
            "insurance_policy_page_3_chunk_2"
        ]
    },
    {
        "question": (
            "What documents are required "
            "for an accident claim?"
        ),
        "relevant_chunk_ids": [
            "insurance_policy_page_2_chunk_1"
        ]
    }
]
