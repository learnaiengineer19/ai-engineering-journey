
"""
LESSON 16 - PART 8.2: LLM REST API TOOL
======================================

OBJECTIVE:
Use Ollama tool calling to retrieve insurance
claims through an HTTP REST API.

CONCEPTS:
1. requests.get()
2. Query parameters
3. HTTP status codes
4. Request timeouts
5. Handling connection errors
6. Pydantic validation
7. Returning API results to an LLM

PREREQUISITE:
The local claims_api.py server must be running.

RUN:
python -m tool_calling.api_tool_calling
"""

import json

import requests
from ollama import chat
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal


API_URL = "http://127.0.0.1:8765/claims"


# --------------------------------------------------
# STEP 1: DEFINE ARGUMENT VALIDATION
# --------------------------------------------------

class FindClaimsArgs(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    min_amount: float = Field(
        ge=0,
        allow_inf_nan=False
    )


# --------------------------------------------------
# STEP 2: DEFINE THE API TOOL
# --------------------------------------------------

def find_claims_api(
    status: str,
    min_amount: float
) -> dict:
    """
    Search insurance claims through the local REST API.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount, inclusive.

    Returns:
        JSON data from the claims API, or an error.
    """

    print("\n[PYTHON] Calling claims REST API...")

    try:

        response = requests.get(
            API_URL,
            params={
                "status": status,
                "min_amount": min_amount
            },
            timeout=5
        )

        print("HTTP STATUS:", response.status_code)

        # Raise an exception for HTTP error statuses.
        response.raise_for_status()

        return response.json()

    except requests.Timeout:
        return {
            "error": "Claims API request timed out"
        }

    except requests.ConnectionError:
        return {
            "error": "Cannot connect to claims API"
        }

    except requests.HTTPError as exc:
        return {
            "error": "Claims API returned an HTTP error",
            "status_code": exc.response.status_code
        }

    except requests.exceptions.JSONDecodeError:
        return {
            "error": "Claims API returned invalid JSON"
        }

    except requests.RequestException:
        return {
            "error": "Claims API request failed"
        }


# --------------------------------------------------
# STEP 3: DEFINE USER QUESTION
# --------------------------------------------------

question = "Find approved insurance claims above $1000."

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "Use find_claims_api to look up claim records. "
            "Only report claim records returned by the API. "
            "If the tool returns an error, explain that "
            "the information could not be retrieved. "
            "Do not invent claim information."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 4: ASK QWEN2 TO SELECT A TOOL
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages,
    tools=[find_claims_api],
    options={"temperature": 0}
)

tool_calls = response.message.tool_calls or []


# --------------------------------------------------
# STEP 5: EXECUTE APPROVED TOOL CALLS
# --------------------------------------------------

if not tool_calls:

    print("\nNo tool requested.")
    print(response.message.content)

else:

    messages.append(response.message)

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nTOOL SELECTED:", name)
        print("GENERATED ARGUMENTS:", arguments)

        if name != "find_claims_api":

            result = {
                "error": "Unknown or unauthorized tool"
            }

        else:

            try:
                validated = FindClaimsArgs.model_validate(
                    arguments
                )

                result = find_claims_api(
                    **validated.model_dump()
                )

            except ValidationError:
                result = {
                    "error": "Invalid tool arguments"
                }

        print("\nAPI TOOL RESULT:")
        print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

    # --------------------------------------------------
    # STEP 6: GENERATE FINAL RESPONSE
    # --------------------------------------------------

    final_response = chat(
        model="qwen2:7b",
        messages=messages,
        options={"temperature": 0}
    )

    print("\nFINAL AI ANSWER:")
    print(final_response.message.content)
