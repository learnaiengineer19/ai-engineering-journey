
"""
LESSON 12 - PART 3: TEXT CLEANING FOR RAG
========================================

OBJECTIVE:
Clean text extracted from PDF documents without
removing important policy information.

CONCEPTS:
1. Whitespace normalization
2. Regular expressions
3. Preserving paragraph boundaries
4. Reusable Python functions
5. Input validation

AI ENGINEERING CONNECTION:
Clean text improves consistency before chunking
and embedding generation.

IMPORTANT:
Do not remove insurance exclusions, conditions,
dates, amounts or other meaningful information.
"""

import re


# --------------------------------------------------
# FUNCTION 1: CLEAN EXTRACTED TEXT
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraphs.

    Args:
        text: Extracted PDF text.

    Returns:
        Cleaned text with consistent whitespace.

    Raises:
        TypeError: If text is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    # Normalize Windows and older Mac line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace consecutive spaces or tabs with one space.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately before newlines.
    text = re.sub(r" *\n", "\n", text)

    # Reduce 3+ newlines to 2 (paragraph separation).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace.
    return text.strip()


# --------------------------------------------------
# EXAMPLE: CLEAN A SAMPLE TEXT
# --------------------------------------------------

if __name__ == "__main__":

    sample_text = (
        "Motor Insurance\n\n\n"
        "Accidental    damage is covered.\n"
        "Claims   must be reported.\n\n\n\n"
        "Section 2: Exclusions"
    )

    cleaned_text = clean_text(sample_text)

    print("BEFORE CLEANING:")
    print(repr(sample_text))

    print("\nAFTER CLEANING:")
    print(repr(cleaned_text))
 