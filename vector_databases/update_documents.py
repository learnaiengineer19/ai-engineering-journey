
"""
LESSON 11: UPDATING AND DELETING CHROMADB RECORDS
=================================================

OBJECTIVE:
Learn how to maintain records in a vector database.

CONCEPTS:
1. add() - Insert new records
2. update() - Modify existing records
3. upsert() - Insert or update records
4. delete() - Remove records
5. Re-generating embeddings after text changes

AI ENGINEERING CONNECTION:
Real-world RAG applications need reliable
document updates when source data changes.

EMBEDDING MODEL:
nomic-embed-text using Ollama.
"""

import chromadb
from ollama import embed


# --------------------------------------------------
# STEP 1: CONNECT TO DATABASE
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

collection = client.get_or_create_collection(
    name="insurance_documents_practice",
    embedding_function=None
)


# --------------------------------------------------
# STEP 2: DEFINE ORIGINAL DOCUMENT
# --------------------------------------------------

document_id = "motor_practice_001"

original_document = (
    "Vehicle accident claims require a police report."
)


# --------------------------------------------------
# STEP 3: GENERATE ORIGINAL EMBEDDING
# --------------------------------------------------

response = embed(
    model="nomic-embed-text",
    input=original_document
)

original_embedding = list(response.embeddings[0])


# --------------------------------------------------
# STEP 4: INSERT ORIGINAL DOCUMENT
# --------------------------------------------------

collection.upsert(
    ids=[document_id],
    documents=[original_document],
    embeddings=[original_embedding],
    metadatas=[{"category": "motor", "version": 1}]
)

print("Before update:")
print(collection.get(
    ids=[document_id],
    include=["documents", "metadatas"]
))


# --------------------------------------------------
# STEP 5: DEFINE UPDATED DOCUMENT
# --------------------------------------------------

updated_document = (
    "Vehicle accident claims require a police report "
    "and photographs of the damaged vehicle."
)


# --------------------------------------------------
# STEP 6: GENERATE UPDATED EMBEDDING
# --------------------------------------------------

response = embed(
    model="nomic-embed-text",
    input=updated_document
)

updated_embedding = list(response.embeddings[0])


# --------------------------------------------------
# STEP 7: YOUR IMPLEMENTATION
# --------------------------------------------------

collection.update(
    ids=[document_id],
    documents=[updated_document],
    embeddings=[updated_embedding],
    metadatas=[{"category": "motor", "version": 2}]
)

collection.update(
    ids=[document_id],
    metadatas=[{"category": "motor", "version": 2, "status":"approved"}]
)


new_document_id = "home_practice_001"

new_document = (
    "Home insurance policies cover certain "
    "fire and storm damage."
)

response = embed(
    model="nomic-embed-text",
    input=new_document
)

new_embedding = list(response.embeddings[0])

collection.upsert(
    ids=[new_document_id],
    documents=[new_document],
    embeddings=[new_embedding],
    metadatas=[{"category": "home", "version": 1}]
)

print("Total records:", collection.count())

print(
    collection.get(
        ids=[new_document_id],
        include=["documents", "metadatas"]
    )
)

# TODO: Update the existing record.
#
# Use:
# collection.update(...)
#
# Supply:
# - document ID
# - updated document
# - updated embedding
# - metadata with version = 2


# --------------------------------------------------
# STEP 8: VERIFY THE UPDATE
# --------------------------------------------------

print("\nAfter update:")

print(collection.get(
    ids=[document_id],
    include=["documents", "metadatas"]
))


collection.delete(
    ids=["home_practice_001"]
)

print("Total records:", collection.count())

result = collection.get(
    ids=["home_practice_001"],
    include=["documents", "metadatas"]
)

print("Remaining matching IDs:", result["ids"])





"""
LESSON 11: DELETING DOCUMENTS FROM CHROMADB
==========================================

OBJECTIVE:
Learn how to delete documents from a vector
database using unique document IDs.

CONCEPTS COVERED:
1. collection.get() - Check if a record exists
2. collection.delete() - Delete a record
3. Reusable Python functions
4. Boolean return values
5. Idempotent deletion behaviour
6. Verifying database operations

AI ENGINEERING CONNECTION:
In production RAG systems, outdated or invalid
documents must be removed so the LLM does not
retrieve obsolete information.

This exercise uses our separate practice
collection to preserve the original documents.
"""

import chromadb


# --------------------------------------------------
# STEP 1: CONNECT TO PERSISTENT CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_databases/chroma_data"
)

collection = client.get_collection(
    name="insurance_documents_practice",
    embedding_function=None
)


# --------------------------------------------------
# STEP 2: DEFINE REUSABLE DELETE FUNCTION
# --------------------------------------------------

def delete_document(collection, document_id: str) -> bool:
    """
    Delete a document from a ChromaDB collection.

    Args:
        collection:
            The ChromaDB collection containing records.

        document_id (str):
            The unique identifier of the record to delete.

    Returns:
        bool:
            True if the record existed and was deleted.
            False if the record was not found.

    Note:
        This function assumes no other process changes
        the record between checking and deleting it.
    """

    # Check whether the document exists.
    result = collection.get(
        ids=[document_id]
    )

    # IDs are returned as a list.
    # An empty list means no matching record was found.
    if not result["ids"]:
        return False

    # Delete the existing document.
    collection.delete(
        ids=[document_id]
    )

    return True


# --------------------------------------------------
# STEP 3: DELETE THE DOCUMENT
# --------------------------------------------------

document_id = "home_practice_001"

print("Records before deletion:", collection.count())

deleted = delete_document(
    collection,
    document_id
)

print("First deletion result:", deleted)
print("Records after deletion:", collection.count())


# --------------------------------------------------
# STEP 4: TRY DELETING THE SAME DOCUMENT AGAIN
# --------------------------------------------------

deleted_again = delete_document(
    collection,
    document_id
)

print("Second deletion result:", deleted_again)


# --------------------------------------------------
# STEP 5: VERIFY THE DOCUMENT NO LONGER EXISTS
# --------------------------------------------------

result = collection.get(
    ids=[document_id],
    include=["documents", "metadatas"]
)

print("Matching IDs after deletion:", result["ids"])
print("Final record count:", collection.count())
