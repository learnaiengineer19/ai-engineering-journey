
"""
LESSON 05: MINI PROJECT - DOCUMENT PROCESSOR
============================================

OBJECTIVE:
Combine modules, file handling, JSON and error
handling to process a document.

CONCEPTS COVERED:
1. Importing functions from another Python module
2. Loading text files
3. Reading JSON configuration
4. Validating document content
5. Counting words and characters
6. Writing structured output to JSON

AI ENGINEERING CONNECTION:
This is a simplified document ingestion pipeline.
Later, we'll extend it with document chunking,
embeddings, retrieval and LLM generation.
"""

import json
from pathlib import Path

from document_loader import load_document

# --------------------------------------------------
# 1. DEFINE FILE PATHS
# --------------------------------------------------

base_path = Path(__file__).parent

document_path = base_path / "sample_document.txt"
config_path = base_path / "model_config.json"
output_path = base_path / "document_metadata.json"


# --------------------------------------------------
# 2. LOAD JSON CONFIGURATION
# --------------------------------------------------

def load_config(config_path):
    """
    Load configuration settings from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        dict: Configuration data if successfully loaded.
        None: If the file cannot be read or parsed.
    """

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as error:
        print("Configuration file not found:", error)
        return None

    except json.JSONDecodeError as error:
        print("Invalid JSON configuration:", error)
        return None


# --------------------------------------------------
# 3. PROCESS DOCUMENT
# --------------------------------------------------


def process_document(document_path):
    """
    Load a text document and calculate basic metadata.

    Args:
        document_path: Path to the text document.

    Returns:
        dict: Metadata for a valid document.
        None: If the document is missing or empty.
    """

    # Step 1: Load document
    result = load_document(document_path)

    # Step 2: Check if document loading failed
    if result is None:
        return None

    # Step 3: Check if document is empty
    if not result.strip():
        print("Document is empty")
        return None

    # Step 4: Create metadata for a valid document
    metadata = {
        "filename": document_path.name,
        "character_count": len(result),
        "word_count": len(result.split())
    }

    # Step 5: Return metadata
    return metadata




# --------------------------------------------------
# 4. RUN THE DOCUMENT PROCESSING APPLICATION
# --------------------------------------------------

config = load_config(config_path)
metadata = process_document(document_path)

if config is not None and metadata is not None:

    print("Model:", config["model"])
    print("Document metadata:", metadata)

    # Save metadata to JSON
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Metadata saved successfully!")

else:
    print("Document processing could not be completed.")
