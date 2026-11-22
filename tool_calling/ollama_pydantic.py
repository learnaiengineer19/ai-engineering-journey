
"""
LESSON 16 - PART 6.2: OLLAMA + PYDANTIC
======================================

OBJECTIVE:
Use a Pydantic model to:
1. Generate a JSON Schema for Ollama.
2. Constrain the LLM's output structure.
3. Validate the model-generated JSON.
4. Handle missing values safely.

CONCEPTS:
- BaseModel
- ConfigDict
- Field
- Literal
- model_json_schema()
- model_validate_json()
- ValidationError

RUN FROM PROJECT ROOT:
python -m tool_calling.ollama_pydantic
"""

from typing import Literal

from ollama import chat
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)


# --------------------------------------------------
# STEP 1: DEFINE OUR PYDANTIC MODEL
# --------------------------------------------------

class InsuranceClaim(BaseModel):
    """
    Structured representation of an insurance claim.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    claim_id: str = Field(
        description="Claim identifier, e.g. CLM1006"
    )

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    # None represents missing or unknown information.
    # An explicit amount of zero is still allowed.
    amount: float | None = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description=(
            "Claim amount when explicitly provided. "
            "Use null when the amount is unknown. "
            "Never invent an amount."
        )
    )


# --------------------------------------------------
# STEP 2: DEFINE INPUT DESCRIPTION
# --------------------------------------------------

claim_description = (
    "Claim CLM1006 has been approved"
)


# --------------------------------------------------
# STEP 3: GENERATE JSON SCHEMA FROM PYDANTIC
# --------------------------------------------------

schema = InsuranceClaim.model_json_schema()

print("\nGENERATED PYDANTIC JSON SCHEMA:")
print(schema)


# --------------------------------------------------
# STEP 4: SEND REQUEST TO OLLAMA
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=[
        {
            "role": "system",
            "content": (
                "Extract insurance claim information "
                "from the provided text. "
                "Return only JSON matching the schema. "
                "Use only explicitly stated facts. "
                "If the amount is missing, return null. "
                "Never assume an unknown amount is zero."
            )
        },
        {
            "role": "user",
            "content": claim_description
        }
    ],
    format=schema,
    options={"temperature": 0}
)


# --------------------------------------------------
# STEP 5: GET RAW LLM RESPONSE
# --------------------------------------------------

raw_json = response.message.content

print("\nRAW LLM RESPONSE:")
print(raw_json)


# --------------------------------------------------
# STEP 6: VALIDATE USING PYDANTIC
# --------------------------------------------------

try:

    # Parse the JSON and validate the fields.
    claim = InsuranceClaim.model_validate_json(
        raw_json
    )

    print("\nVALIDATION PASSED!")

    print("\nVALIDATED CLAIM:")
    print(claim.model_dump_json(indent=2))

except ValidationError as exc:

    print("\nVALIDATION FAILED!")

    for error in exc.errors():

        print(
            f"Field: {error['loc']}, "
            f"Error: {error['msg']}"
        )

    raise


# --------------------------------------------------
# STEP 7: ACCESS VALIDATED PYTHON FIELDS
# --------------------------------------------------

print("\nCLAIM DETAILS:")

print("Claim ID:", claim.claim_id)
print("Status:", claim.status)
print("Amount:", claim.amount)
