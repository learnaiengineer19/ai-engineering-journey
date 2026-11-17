
"""
LESSON 10: SAVING DOCUMENT EMBEDDINGS
====================================

OBJECTIVE:
Generate document embeddings once and save them
to a JSON file for future semantic searches.

CONCEPTS COVERED:
1. Embedding generation using Ollama
2. Batch processing
3. Python functions and type hints
4. Dictionary creation
5. JSON file handling
6. Embedding persistence

AI ENGINEERING CONNECTION:
RAG systems separate document ingestion from query
processing to avoid regenerating embeddings for
unchanged documents.

MODEL:
nomic-embed-text (locally hosted using Ollama)
"""

import json
from ollama import embed


# --------------------------------------------------
# 1. DEFINE SAMPLE DOCUMENTS
# --------------------------------------------------

documents = [
    "Vehicle accident claims require a police report.",
    "Car insurance premiums are renewed annually.",
    "Travel insurance covers certain medical emergencies.",
    "Home insurance protects against fire and flood damage.",
    "Vehicle collision claims require photographs of the damage."
]


# --------------------------------------------------
# 2. GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(
    sentences: list[str]
) -> list[list[float]]:
    """
    Generate vector embeddings for a list of texts.

    Args:
        sentences: List of document texts.

    Returns:
        List of embedding vectors.
    """

    # Send all texts to Ollama in one request.
    response = embed(
        model="nomic-embed-text",
        input=sentences
    )

    # Convert Ollama's embedding vectors to
    # ordinary Python lists.
    result_list = []

    for embedding in response.embeddings:
        result_list.append(list(embedding))

    return result_list


# --------------------------------------------------
# 3. SAVE EMBEDDINGS TO JSON
# --------------------------------------------------

def save_embeddings(
    documents: list[str],
    filename: str = "embeddings.json"
) -> None:
    """
    Generate and save document embeddings to JSON.

    Args:
        documents: List of documents to embed.
        filename: Destination JSON filename.

    Returns:
        None. Data is saved to a file.
    """

    # STEP 1: Generate document embeddings.
    embeddings = generate_embeddings(documents)

    # STEP 2: Create the data structure.
    data = {
        "model": "nomic-embed-text",
        "documents": documents,
        "embeddings": embeddings
    }

    # STEP 3: Open the destination file.
    # STEP 4: Write the dictionary as JSON.
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    # STEP 5: Display confirmation.
    print(f"Embeddings saved successfully to {filename}")
    print(f"Total documents: {len(documents)}")
    print(f"Total embeddings: {len(embeddings)}")


# --------------------------------------------------
# 4. RUN THE INGESTION PROCESS
# --------------------------------------------------

if __name__ == "__main__":

    save_embeddings(
        documents,
        filename="llm_fundamentals/embeddings.json"
    )
