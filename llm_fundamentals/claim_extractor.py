
"""
LESSON 09: LLM-POWERED INSURANCE CLAIM EXTRACTION
=================================================

OBJECTIVE:
Extract structured insurance claim information
using a locally hosted LLM and validate it.

PIPELINE:
1. Receive an insurance claim description
2. Send the description to Qwen2:7B through Ollama
3. Receive a JSON response
4. Parse JSON into a Python dictionary
5. Validate the dictionary using Pydantic
6. Display the validated information

CONCEPTS:
- Ollama API integration
- JSON parsing
- Pydantic BaseModel
- Date validation
- Field constraints
- Exception handling

AI ENGINEERING CONNECTION:
This pattern is useful for document processing,
insurance claim ingestion and AI-powered ETL.

NOTE:
Schema validation does not guarantee that the
extracted facts are accurate.
"""

import json
from datetime import date

from ollama import chat, ResponseError
from pydantic import BaseModel, Field, ValidationError


# --------------------------------------------------
# 1. DEFINE THE EXPECTED DATA STRUCTURE
# --------------------------------------------------

class InsuranceClaim(BaseModel):

    claim_type: str

    # Pydantic converts valid ISO date strings
    # into Python date objects.
    incident_date: date

    # Estimated cost must be zero or greater.
    estimated_cost: int = Field(ge=0)


# --------------------------------------------------
# 2. DEFINE SAMPLE INPUT
# --------------------------------------------------

claim_description = (
    "My car was damaged in an accident on "
    "15 October 2026. The estimated repair "
    "cost is $3,500."
)


# --------------------------------------------------
# 3. PREPARE THE LLM PROMPT
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claim information extractor. "
            "Extract claim_type, incident_date, and estimated_cost. "
            "Use YYYY-MM-DD for dates and integers for costs. "
            "Do not invent missing information. "
            "Return only a JSON object."
        )
    },
    {
        "role": "user",
        "content": claim_description
    }
]


# --------------------------------------------------
# 4. CALL OLLAMA
# --------------------------------------------------

try:
    response = chat(
        model="qwen2:7b",
        messages=messages,
        format="json"
    )

    # --------------------------------------------------
    # 5. CONVERT JSON STRING TO PYTHON DICTIONARY
    # --------------------------------------------------

    claim_data = json.loads(response.message.content)

    print("\nExtracted JSON:")
    print(claim_data)

    # --------------------------------------------------
    # 6. VALIDATE USING PYDANTIC
    # --------------------------------------------------

    validated_claim = InsuranceClaim(**claim_data)

    # --------------------------------------------------
    # 7. DISPLAY VALIDATED DATA
    # --------------------------------------------------

    print("\nValidated Insurance Claim:")
    print(validated_claim)

    print("\nIndividual Fields:")
    print("Claim type:", validated_claim.claim_type)
    print("Incident date:", validated_claim.incident_date)
    print("Estimated cost:", validated_claim.estimated_cost)

    print("\nPython Data Types:")
    print("Incident date:", type(validated_claim.incident_date))
    print("Estimated cost:", type(validated_claim.estimated_cost))

# --------------------------------------------------
# 8. HANDLE ERRORS
# --------------------------------------------------

except ResponseError as error:
    print("Ollama error:", error)

except ConnectionError as error:
    print("Connection error:", error)

except json.JSONDecodeError as error:
    print("Invalid JSON returned by LLM:", error)

except ValidationError as error:
    print("Pydantic validation failed:")
    print(error)

except TypeError as error:
    print("Unexpected JSON structure:", error)
