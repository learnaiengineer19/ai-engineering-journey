
"""
LESSON 18 - PART 3: STRUCTURED OUTPUTS
=====================================

OBJECTIVE:
Use LangChain and Pydantic to extract structured
insurance claim information from unstructured text.

CONCEPTS:
1. Pydantic BaseModel
2. Field descriptions
3. Literal and validation constraints
4. with_structured_output()
5. LCEL pipelines
6. Handling missing information

RUN:
python -m langchain_lessons.structured_output
"""

from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)


# --------------------------------------------------
# STEP 1: DEFINE THE OUTPUT SCHEMA
# --------------------------------------------------

class ClaimRecord(BaseModel):
    """
    Structure we expect from the LLM.

    This acts as our output contract.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    claim_id: str = Field(
        description="Insurance claim identifier",
        pattern=r"^CLM[0-9]{4}$"
    )

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    amount: float | None = Field(
        description=(
            "Claim amount. Use null when "
            "the amount is not provided."
        ),
        ge=0
    )

    claim_type: str = Field(
        description="Type of insurance claim"
    )


# --------------------------------------------------
# STEP 2: DEFINE THE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract insurance claim information "
        "from document text. "
        "Use only information explicitly present "
        "in the supplied document. "
        "Never invent missing information. "
        "For an unknown amount, return null. "
        "Return values matching the required schema."
    ),
    (
        "human",
        "Extract the claim information:\n\n"
        "{document_text}"
    )
])


# --------------------------------------------------
# STEP 3: INITIALIZE QWEN2
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)


# --------------------------------------------------
# STEP 4: CREATE STRUCTURED MODEL
# --------------------------------------------------

structured_model = model.with_structured_output(
    ClaimRecord,
    method="json_schema"
)


# --------------------------------------------------
# STEP 5: BUILD LCEL CHAIN
# --------------------------------------------------

chain = prompt | structured_model


# --------------------------------------------------
# STEP 6: TEST DOCUMENTS
# --------------------------------------------------

documents = [
    (
        "Claim CLM1004 relates to a Motor Accident. "
        "The amount is $3200 and the claim is Approved."
    ),
    (
        "Claim CLM1005 is for Vehicle Damage. "
        "Its status is Under Review. "
        "The claim amount has not yet been determined."
    ),
    (
        "Claim CLM1006 is related to Motor Accident. "
        "The claim status is Approved. "
        "No amount is mentioned."
    )
]


# --------------------------------------------------
# STEP 7: RUN EXTRACTION
# --------------------------------------------------

for index, document in enumerate(documents, start=1):

    print(f"\n========== DOCUMENT {index} ==========")
    print(document)

    try:

        result = chain.invoke({
            "document_text": document
        })

        print("\nOUTPUT TYPE:")
        print(type(result).__name__)

        print("\nVALIDATED PYDANTIC OBJECT:")
        print(result)

        print("\nSTRUCTURED JSON:")
        print(result.model_dump_json(indent=2))

    except ValidationError as exc:

        print("\nVALIDATION ERROR:")
        print(exc)

    except ValueError as exc:

        print("\nSTRUCTURED OUTPUT ERROR:")
        print(exc)
