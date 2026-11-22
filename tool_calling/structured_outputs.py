
"""
LESSON 16 - PART 5: STRUCTURED JSON OUTPUTS
===========================================

OBJECTIVE:
Generate predictable JSON responses using
Ollama and an explicit JSON Schema.

CONCEPTS:
1. Structured output vs free-form text
2. JSON Schema
3. Ollama's format parameter
4. json.loads()
5. Accessing structured fields
6. Basic schema validation

RUN FROM PROJECT ROOT:
python -m tool_calling.structured_outputs
"""

import json
from ollama import chat


# --------------------------------------------------
# STEP 1: DEFINE A JSON OUTPUT SCHEMA
# --------------------------------------------------

# This schema describes the structure we want
# Qwen2 to return.

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {
            "type": "string",
            "description": "Insurance claim identifier"
        },
        "status": {
            "type": "string",
            "enum": [
                "Approved",
                "Rejected",
                "Under Review"
            ]
        },
        "amount": {
            "type": ["number", "null"],
            "minimum": 0,
            "description": (
                "Claim amount if explicitly provided. "
                "Return null if the amount is missing. "
                "Never invent a value or assume zero."
            )
        }
    },
    "required": [
        "claim_id",
        "status",
        "amount"
    ],
    "additionalProperties": False
}


# --------------------------------------------------
# STEP 2: DEFINE INPUT TEXT
# --------------------------------------------------

claim_description = (
    "Claim CLM1006 has been approved"
)


# --------------------------------------------------
# STEP 3: SEND THE REQUEST TO QWEN2
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=[
        {
            "role": "system",
            "content": (
                "You extract structured insurance "
                "claim information from text. "
                "Extract only information explicitly present "
                "in the input text. "
                "If the claim amount is not mentioned, "
                "return null for amount. "
                "Never assume an unknown amount is zero."
                "Return a JSON object matching "
                "the provided schema."
            )
        },
        {
            "role": "user",
            "content": claim_description
        }
    ],
    format=CLAIM_SCHEMA,
    options={"temperature": 0}
)


# --------------------------------------------------
# STEP 4: INSPECT THE RAW JSON STRING
# --------------------------------------------------

raw_response = response.message.content

print("\nRAW LLM RESPONSE:")
print(raw_response)


# --------------------------------------------------
# STEP 5: CONVERT JSON STRING TO PYTHON DICTIONARY
# --------------------------------------------------

try:
    claim = json.loads(raw_response)

except json.JSONDecodeError as exc:
    raise ValueError(
        "The model returned invalid JSON."
    ) from exc


# --------------------------------------------------
# STEP 6: ACCESS THE FIELDS
# --------------------------------------------------

print("\nPARSED PYTHON DATA:")

print("Claim ID:", claim["claim_id"])
print("Status:", claim["status"])
print("Amount:", claim["amount"])


# --------------------------------------------------
# STEP 7: BASIC VALIDATION
# --------------------------------------------------

# JSON parsing alone does not prove the
# values meet our expected requirements.

if not isinstance(claim["claim_id"], str):
    raise TypeError("claim_id must be a string")

if claim["status"] not in {
    "Approved",
    "Rejected",
    "Under Review"
}:
    raise ValueError("Invalid claim status")

amount = claim["amount"]

if amount is not None:

    if (
        type(amount) not in (int, float)
        or not (0 <= amount < float("inf"))
    ):
        raise ValueError("Invalid claim amount")


print("\nVALIDATION PASSED!")

print("\nFORMATTED JSON:")
print(json.dumps(claim, indent=2))
