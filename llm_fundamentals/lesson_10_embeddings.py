
"""
LESSON 10: EMBEDDINGS AND SEMANTIC SEARCH
========================================

OBJECTIVE:
Understand how text can be converted into numerical
vectors using an embedding model.

CONCEPTS COVERED:
1. Embeddings
2. Numerical vectors
3. Embedding models vs generative LLMs
4. Ollama embedding API
5. Vector dimensions
6. Semantic similarity (upcoming)

WHY THIS MATTERS:
Embeddings are fundamental to semantic search,
vector databases and Retrieval-Augmented Generation.

LOCAL MODELS:
- qwen2:7b            -> Text generation
- nomic-embed-text     -> Text embeddings

AI ENGINEERING CONNECTION:
In a RAG application, document embeddings can be
used to retrieve information relevant to a question.

We then provide the retrieved information to an LLM
to generate a grounded response.
"""

from ollama import embed


# --------------------------------------------------
# 1. DEFINE SAMPLE TEXT
# --------------------------------------------------

text = "My car was damaged in an accident."


# --------------------------------------------------
# 2. GENERATE EMBEDDING
# --------------------------------------------------

response = embed(
    model="nomic-embed-text",
    input=text
)


# --------------------------------------------------
# 3. EXTRACT THE VECTOR
# --------------------------------------------------

embedding = response.embeddings[0]


# --------------------------------------------------
# 4. DISPLAY THE RESULTS
# --------------------------------------------------

print("Original text:", text)
print("First 10 vector values:", embedding[:10])
print("Vector dimensions:", len(embedding))


from ollama import embed
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(
    sentences: list[str]
) -> list[list[float]]:

    # Generate embeddings for all sentences
    # in a single Ollama request.
    response = embed(
        model="nomic-embed-text",
        input=sentences
    )

    # Convert each embedding into a Python list.
    result_list = []

    for embedding in response.embeddings:
        result_list.append(list(embedding))

    return result_list


# --------------------------------------------------
# TEST
# --------------------------------------------------

sentences = [
    "My car was damaged in an accident.",
    "I was involved in a vehicle collision.",
    "I purchased a new laptop."
]

embeddings = generate_embeddings(sentences)
similarity_matrix = cosine_similarity(embeddings)
print("\nSimilarity Matrix:")
print(similarity_matrix)

print(
    f"Car accident vs vehicle collision: "
    f"{similarity_matrix[0][1]:.4f}"
)

print(
    f"Car accident vs laptop purchase: "
    f"{similarity_matrix[0][2]:.4f}"
)


# print(embeddings)

print("Number of embeddings:", len(embeddings))
print("Dimensions of first embedding:", len(embeddings[0]))
print("First 5 values:", embeddings[0][:5])
