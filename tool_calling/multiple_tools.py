
"""
LESSON 16 - PART 3: MULTIPLE TOOLS
==================================

OBJECTIVE:
Allow an LLM to choose between multiple
Python functions based on a user request.

CONCEPTS:
1. Registering multiple tools
2. Dynamic function selection
3. Tool registry using a dictionary
4. Validating generated arguments
5. Executing the chosen function
6. Returning tool results to the LLM

RUN:
python -m tool_calling.multiple_tools
"""

import json
from ollama import chat


# --------------------------------------------------
# STEP 1: SAMPLE CLAIMS DATABASE
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
# STEP 2: TOOL 1 - LOOK UP A SINGLE CLAIM
# --------------------------------------------------

def get_claim_status(claim_id: str) -> dict:
    """
    Retrieve an insurance claim by its unique ID.

    Args:
        claim_id: Claim identifier, e.g. CLM1001.
    """

    claim = CLAIMS_DATABASE.get(claim_id)

    if claim is None:
        return {"error": "Claim not found"}

    return {
        "claim_id": claim_id,
        **claim
    }


# --------------------------------------------------
# STEP 3: TOOL 2 - SEARCH CLAIMS USING FILTERS
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find claims by status and minimum amount.

    Args:
        status: Approved, Rejected, Under Review.
        min_amount: Minimum amount, inclusive.
    """

    return [
        {"claim_id": claim_id, **claim}
        for claim_id, claim in CLAIMS_DATABASE.items()
        if (
            claim["status"] == status
            and claim["amount"] >= min_amount
        )
    ]


# --------------------------------------------------
# STEP 4: REGISTER APPROVED FUNCTIONS
# --------------------------------------------------

AVAILABLE_TOOLS = {
    "get_claim_status": get_claim_status,
    "find_claims": find_claims
}


# --------------------------------------------------
# STEP 5: DEFINE VALIDATION AND EXECUTION
# --------------------------------------------------

def execute_tool(name: str, arguments: dict):
    """
    Validate LLM-generated arguments and execute
    only functions registered by our application.
    """

    if not isinstance(arguments, dict):
        return {"error": "Invalid tool arguments"}

    if name == "get_claim_status":

        claim_id = arguments.get("claim_id")

        if not isinstance(claim_id, str):
            return {"error": "Invalid claim ID"}

        return AVAILABLE_TOOLS[name](
            claim_id=claim_id
        )

    elif name == "find_claims":

        status = arguments.get("status")
        min_amount = arguments.get("min_amount")

        allowed_statuses = {
            "Approved",
            "Rejected",
            "Under Review"
        }

        if status not in allowed_statuses:
            return {"error": "Invalid status"}

        if (
            type(min_amount) not in (int, float)
            or min_amount < 0
        ):
            return {"error": "Invalid minimum amount"}

        return AVAILABLE_TOOLS[name](
            status=status,
            min_amount=min_amount
        )

    return {"error": f"Unknown tool: {name}"}


# --------------------------------------------------
# STEP 6: ACCEPT USER QUESTION
# --------------------------------------------------

question = "What is an insurance deductible"

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "Use get_claim_status for individual claims. "
            "Use find_claims to search claims by filters. "
            "Do not invent claim information. "
            "If required information is missing, "
            "ask the user for clarification."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 7: PROVIDE MULTIPLE TOOLS TO QWEN2
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages,
    tools=[
        get_claim_status,
        find_claims
    ],
    options={"temperature": 0}
)


# --------------------------------------------------
# STEP 8: EXECUTE REQUESTED TOOL CALLS
# --------------------------------------------------

tool_calls = response.message.tool_calls or []

if not tool_calls:

    print("\nNo tool call requested.")
    print(response.message.content)

else:

    messages.append(response.message)

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nSELECTED TOOL:", name)
        print("ARGUMENTS:", arguments)

        result = execute_tool(
            name=name,
            arguments=arguments
        )

        print("\nTOOL RESULT:")
        print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

    # ----------------------------------------------
    # STEP 9: GENERATE FINAL ANSWER
    # ----------------------------------------------

    final_response = chat(
        model="qwen2:7b",
        messages=messages,
        options={"temperature": 0}
    )

    print("\nFINAL AI ANSWER:")
    print(final_response.message.content)
