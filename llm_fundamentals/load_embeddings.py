
"""
LESSON 10: LOADING SAVED EMBEDDINGS
==================================

OBJECTIVE:
Load previously generated document embeddings
from a JSON file.

CONCEPTS:
1. JSON file reading
2. File handling using with open()
3. Reusable functions
4. Embedding persistence
5. Separating ingestion from querying

AI ENGINEERING CONNECTION:
RAG applications should reuse stored document
embeddings rather than regenerate them for
every user question.
"""

import json


def load_embeddings(filename: str) -> dict:
    """
    Load documents and embeddings from a JSON file.

    Args:
        filename: Path to the saved embeddings file.

    Returns:
        Dictionary containing model, documents
        and embeddings.
    """

    # TODO 1: Open the file in read mode.
    # 
    with open(filename, "r", encoding="utf-8") as file:
        file_content = json.load(file)

        return file_content

    # TODO 2: Load its contents using json.load().

    # TODO 3: Return the dictionary.

    pass


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    data = load_embeddings(
        "llm_fundamentals/embeddings.json"
    )

    print("Model:", data["model"])
    print("Total documents:", len(data["documents"]))
    print("Total embeddings:", len(data["embeddings"]))
    print("Dimensions:", len(data["embeddings"][0]))
