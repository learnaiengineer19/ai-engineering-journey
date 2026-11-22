
"""
LESSON 16 - PART 2: TOOL SCHEMAS AND ARGUMENTS
==============================================

OBJECTIVE:
Teach an LLM to call a Python function using
multiple arguments and an explicit JSON schema.

CONCEPTS:
1. JSON tool schemas
2. Required arguments
3. String and number types
4. Enum constraints
5. Validating LLM-generated arguments
6. Safe tool execution

RUN FROM PROJECT ROOT:
python -m tool_calling.tool_schemas
"""

import json
from ollama import chat


# --------------------------------------------------
# STEP 1: SAMPLE INSURANCE CLAIMS DATABASE
# --------------------------------------------------

CLAIMS_DATABASE = {
    "CLM1001": {
        "status": "Under Review",
        "amount": 4500
    },
    "CLM1002": {
        "status": "Approved",
        "amount": 850
    },
    "CLM1003": {
        "status": "Rejected",
        "amount": 2200
    },
    "CLM1004": {
        "status": "Approved",
        "amount": 3200
    },
    "CLM1005": {
        "status": "Approved",
        "amount": 1800
    }
}


# --------------------------------------------------
# STEP 2: DEFINE THE PYTHON FUNCTION
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find claims matching a status and minimum amount.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount, inclusive.

    Returns:
        List of matching claim records.
    """

    results = []

    for claim_id, claim in CLAIMS_DATABASE.items():

        if (
            claim["status"] == status
            and claim["amount"] >= min_amount
        ):

            results.append({
                "claim_id": claim_id,
                **claim
            })

    return results


# --------------------------------------------------
# STEP 3: DEFINE AN EXPLICIT TOOL SCHEMA
# --------------------------------------------------

# JSON Schema describes the function to the LLM.
# This dictionary is sent to Ollama.

FIND_CLAIMS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_claims",
        "description": (
            "Search insurance claims using their "
            "status and minimum claim amount."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Insurance claim status",
                    "enum": [
                        "Approved",
                        "Rejected",
                        "Under Review"
                    ]
                },
                "min_amount": {
                    "type": "number",
                    "description": (
                        "Minimum claim amount, inclusive"
                    ),
                    "minimum": 0
                }
            },
            "required": [
                "status",
                "min_amount"
            ]
        }
    }
}


# --------------------------------------------------
# STEP 4: ASK THE USER'S QUESTION
# --------------------------------------------------

question = "what do you know about claims"

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claim assistant. "
            "Use the available tool to search claims. "
            "Do not invent claim records. "
            "If the user omits a required filter, "
            "ask them for it."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 5: SEND TOOL SCHEMA TO OLLAMA
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages,
    tools=[FIND_CLAIMS_TOOL],
    options={"temperature": 0}
)

tool_calls = response.message.tool_calls or []


# --------------------------------------------------
# STEP 6: INSPECT GENERATED ARGUMENTS
# --------------------------------------------------

if not tool_calls:

    print("No tool call returned.")
    print(response.message.content)

else:

    messages.append(response.message)

    for tool_call in tool_calls:

        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nFUNCTION:", function_name)
        print("ARGUMENTS:", arguments)

        # ------------------------------------------
        # STEP 7: VALIDATE ARGUMENTS
        # ------------------------------------------

        if function_name != "find_claims":

            result = {
                "error": "Unknown tool requested"
            }

        else:

            status = arguments.get("status")
            min_amount = arguments.get("min_amount")

            allowed_statuses = {
                "Approved",
                "Rejected",
                "Under Review"
            }

            if status not in allowed_statuses:

                result = {
                    "error": "Invalid claim status"
                }

            elif (
                type(min_amount) not in (int, float)
                or min_amount < 0
            ):

                result = {
                    "error": "Invalid minimum amount"
                }

            else:

                # Execute the approved function.
                result = find_claims(
                    status=status,
                    min_amount=min_amount
                )

        print("\nTOOL RESULT:")
        print(json.dumps(result, indent=2))

        # ------------------------------------------
        # STEP 8: RETURN RESULT TO THE MODEL
        # ------------------------------------------

        messages.append({
            "role": "tool",
            "tool_name": function_name,
            "content": json.dumps(result)
        })

    final_response = chat(
        model="qwen2:7b",
        messages=messages,
        options={"temperature": 0}
    )

    print("\nFINAL AI ANSWER:")
    print(final_response.message.content)
