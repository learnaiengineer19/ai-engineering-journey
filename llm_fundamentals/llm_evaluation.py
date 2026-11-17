def evaluate_rag_response(response_text: str) -> dict[str, bool]:
    text = "Retrieval-Augmented Generation uses external information."
    response = {"mentions_retrieval": None,"mentions_generation": None,"mentions_external": None}

    if "retrieval" in text.lower():
        response["mentions_retrieval"]= True
    if "generation" in text.lower():
        response["mentions_generation"]= True
    if "external" in text.lower():
        response["mentions_external"]= True

    return response
