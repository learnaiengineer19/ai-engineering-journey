
"""
LESSON 13 - PART 10.5: CITATION VALIDATION
==========================================

OBJECTIVE:
Validate source citations in RAG-generated answers.

CHECKS:
1. Detect citations using [Source N].
2. Check that citation numbers exist in the
   supplied retrieved context.
3. Identify malformed Source citations.
4. Detect answers with no citations.

LIMITATION:
This validates citation structure and references,
not whether the cited evidence supports a claim.

RUN:
python -m advanced_rag.evaluation.citation_validator
"""

import re


def validate_citations(
    answer: str,
    total_sources: int
) -> dict:
    """
    Validate citation references in an LLM answer.

    Args:
        answer: Generated answer text.
        total_sources: Number of chunks supplied to
            the LLM as numbered sources.

    Returns:
        Dictionary with validation details.
    """

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string")

    if type(total_sources) is not int or total_sources < 0:
        raise ValueError("total_sources must be non-negative")

    # Find bracketed references that start with
    # "Source", including malformed examples.
    source_references = re.findall(
        r"\[Source[^\]]*\]",
        answer
    )

    # Exact citation format required: [Source N]
    valid_pattern = re.compile(r"\[Source ([1-9]\d*)\]")

    valid_source_ids = []
    malformed_citations = []
    invalid_source_ids = []

    for citation in source_references:

        match = valid_pattern.fullmatch(citation)

        if match is None:
            malformed_citations.append(citation)
            continue

        source_id = int(match.group(1))

        if 1 <= source_id <= total_sources:
            valid_source_ids.append(source_id)
        else:
            invalid_source_ids.append(source_id)

    # An answer with no citations is marked invalid.
    is_valid = (
        bool(valid_source_ids)
        and not malformed_citations
        and not invalid_source_ids
    )

    return {
        "is_valid": is_valid,
        "valid_source_ids": valid_source_ids,
        "malformed_citations": malformed_citations,
        "invalid_source_ids": invalid_source_ids,
        "has_citations": bool(source_references)
    }


if __name__ == "__main__":

    # Assume the LLM received three source chunks.
    total_sources = 3

    test_answers = [
        "Engine wear and tear is excluded [Source 1].",
        "Ordinary tyre wear is excluded [Source 2].",
        "Engine breakdown is excluded [Source 5].",
        "Engine failure is excluded [Source 1, Page 3].",
        "Engine failure is excluded."
    ]

    for index, answer in enumerate(test_answers, start=1):

        result = validate_citations(
            answer=answer,
            total_sources=total_sources
        )

        print(f"\nTest {index}")
        print("Answer:", answer)
        print("Citation valid:", result["is_valid"])
        print("Valid source IDs:", result["valid_source_ids"])
        print("Malformed:", result["malformed_citations"])
        print("Invalid IDs:", result["invalid_source_ids"])