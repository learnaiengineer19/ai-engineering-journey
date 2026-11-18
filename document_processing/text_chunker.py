
"""
LESSON 12 - PART 5: TEXT CHUNKING
================================

OBJECTIVE:
Split long text into smaller overlapping chunks.

CONCEPTS:
1. String slicing
2. Chunk size
3. Chunk overlap
4. Sliding window
5. Input validation
6. Reusable functions

AI ENGINEERING CONNECTION:
Chunking prepares extracted PDF text for
embedding generation and vector storage.
"""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared by adjacent chunks.

    Returns:
        A list of text chunks.

    Raises:
        TypeError: If text is not a string.
        ValueError: If chunk settings are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if (
        type(chunk_overlap) is not int
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size - 1"
        )

    chunks = []

    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        end = start + chunk_size

        chunks.append(text[start:end])

        if end >= len(text):
            break

    return chunks


if __name__ == "__main__":
    sample_text = "ABCDEFGHIJKL"

    result = chunk_text(
        sample_text,
        chunk_size=5,
        chunk_overlap=2
    )

    print("Chunks:", result)
    print("Total chunks:", len(result))
