from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

question = "Is mechanical engine breakdown covered?"

documents = [
    "Mechanical breakdown caused by wear and tear is excluded.",
    "Policy premiums are renewed annually.",
    "Engine damage caused by an insured collision may be covered."
]

# Construct question-document pairs.
pairs = [
    (question, document)
    for document in documents
]

# Predict relevance scores.
scores = model.predict(pairs)

# --------------------------------------------------
# STEP 1: COMBINE EACH DOCUMENT WITH ITS SCORE
# --------------------------------------------------

document_with_score = []

for index, score in enumerate(scores):
    document_with_score.append(
        (documents[index], float(score))
    )


# --------------------------------------------------
# STEP 2: SORT DOCUMENTS BY RELEVANCE SCORE
# --------------------------------------------------

# item[0] = document text
# item[1] = reranker score
#
# reverse=True means highest score first.

ranked_documents = sorted(
    document_with_score,
    key=lambda item: item[1],
    reverse=True
)


# --------------------------------------------------
# STEP 3: PRINT RERANKED DOCUMENTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CROSS-ENCODER RERANKING RESULTS")
print("=" * 60)

for rank, (document, score) in enumerate(
    ranked_documents,
    start=1
):
    print(f"\nRank: {rank}")
    print(f"Document: {document}")
    print(f"Reranker Score: {score:.4f}")
    print("-" * 50)