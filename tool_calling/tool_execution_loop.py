
"""
LESSON 16 - PART 4: MULTI-STEP TOOL EXECUTION
============================================

OBJECTIVE:
Allow an LLM to request multiple rounds of
Python tool execution before answering.

CONCEPTS:
1. while loops
2. Multi-step tool calling
3. Tool registry
4. Conversation/message history
5. Sending tool results back to the LLM
6. Argument validation
7. Maximum tool-execution rounds

RUN:
python -m tool_calling.tool_execution_loop

NOTE:
Insurance data is fictional and for learning only.
"""

import json
from ollama import chat
import time

# --------------------------------------------------
# STEP 1: CREATE SAMPLE DATABASE
# --------------------------------------------------

CLAIMS_DATABASE = {
    "CLM1001": {
        "status": "Under Review",
        "amount": 4500,
        "claim_type": "Motor Accident",
        "description": "Rear-end collision",
        "assigned_team": "Motor Claims Team"
    },
    "CLM1002": {
        "status": "Approved",
        "amount": 850,
        "claim_type": "Windshield Damage",
        "description": "Cracked windshield",
        "assigned_team": "Glass Claims Team"
    },
    "CLM1003": {
        "status": "Rejected",
        "amount": 2200,
        "claim_type": "Mechanical Breakdown",
        "description": "Engine wear and tear",
        "assigned_team": "Motor Assessment Team"
    },
    "CLM1004": {
        "status": "Approved",
        "amount": 3200,
        "claim_type": "Motor Accident",
        "description": "Side impact collision",
        "assigned_team": "Motor Claims Team"
    },
    "CLM1005": {
        "status": "Approved",
        "amount": 1800,
        "claim_type": "Vehicle Damage",
        "description": "Parking-related damage",
        "assigned_team": "Motor Claims Team"
    }
}


# --------------------------------------------------
# STEP 2: TOOL 1 - SEARCH CLAIMS
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find insurance claims by status and minimum amount.

    Returns only matching claim IDs, status and amount.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount (inclusive).
    """

    print("\n[PYTHON] Executing find_claims()")

    return [
        {
            "claim_id": claim_id,
            "status": claim["status"],
            "amount": claim["amount"]
        }
        for claim_id, claim in CLAIMS_DATABASE.items()
        if (
            claim["status"] == status
            and claim["amount"] >= min_amount
        )
    ]


# --------------------------------------------------
# STEP 3: TOOL 2 - GET CLAIM DETAILS
# --------------------------------------------------

def get_claim_details(claim_id: str) -> dict:
    """
    Retrieve the full details of a claim by claim ID.

    Args:
        claim_id: Insurance claim ID, e.g. CLM1004.
    """

    print(f"\n[PYTHON] Executing get_claim_details({claim_id})")

    claim = CLAIMS_DATABASE.get(claim_id)

    if claim is None:
        return {
            "claim_id": claim_id,
            "error": "Claim not found"
        }

    return {
        "claim_id": claim_id,
        **claim
    }


# --------------------------------------------------
# STEP 4: TOOL REGISTRY
# --------------------------------------------------

AVAILABLE_TOOLS = {
    "find_claims": find_claims,
    "get_claim_details": get_claim_details
}


# --------------------------------------------------
# STEP 5: VALIDATE AND EXECUTE TOOL REQUESTS
# --------------------------------------------------

def execute_tool(name: str, arguments: dict):
    """
    Validate the LLM-generated arguments before
    executing an explicitly approved Python function.
    """

    if not isinstance(arguments, dict):
        return {"error": "Arguments must be an object"}

    if name == "find_claims":

        status = arguments.get("status")
        min_amount = arguments.get("min_amount")

        if status not in {
            "Approved",
            "Rejected",
            "Under Review"
        }:
            return {"error": "Invalid claim status"}

        if (
            type(min_amount) not in (int, float)
            or not (0 <= min_amount < float("inf"))
        ):
            return {"error": "Invalid minimum amount"}

        return AVAILABLE_TOOLS[name](
            status=status,
            min_amount=min_amount
        )

    if name == "get_claim_details":

        claim_id = arguments.get("claim_id")

        if not isinstance(claim_id, str):
            return {"error": "Invalid claim ID"}

        return AVAILABLE_TOOLS[name](
            claim_id=claim_id
        )

    return {"error": f"Unknown tool: {name}"}


# --------------------------------------------------
# STEP 6: DEFINE QUESTION AND MESSAGE HISTORY
# --------------------------------------------------

question = (
    "Find all approved insurance claims above "
    "$1000, then retrieve the full details "
    "of each matching claim."
)

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "Use find_claims to search by status and amount. "
            "That tool returns only summary information. "
            "When the user requests full details, "
            "use get_claim_details for each matching claim. "
            "Do not invent claim information. "
            "Use tool results to produce your final answer."
        )
    },
    {
        "role": "user",
        "content": question
    }
]

print("\nUSER QUESTION:")
print(question)


# --------------------------------------------------
# STEP 7: TOOL EXECUTION LOOP
# --------------------------------------------------

MAX_ROUNDS = 5

round_number = 0

start_time = time.perf_counter()
llm_calls = 0
tool_executions = 0

while round_number < MAX_ROUNDS:

    round_number += 1

    print(f"\n========== ROUND {round_number} ==========")

    # Ask Qwen2 whether it needs a tool.
    response = chat(
        model="qwen2:7b",
        messages=messages,
        tools=[
            find_claims,
            get_claim_details
        ],
        options={"temperature": 0}
    )
    llm_calls += 1

    # Preserve the assistant's message,
    # including any structured tool requests.
    messages.append(response.message)

    tool_calls = response.message.tool_calls or []

    # ----------------------------------------------
    # STEP 8: STOP IF NO TOOL IS REQUESTED
    # ----------------------------------------------

    if not tool_calls:

        print("\nFINAL AI ANSWER:")
        print(response.message.content)

        break

    # ----------------------------------------------
    # STEP 9: EXECUTE EVERY REQUESTED TOOL
    # ----------------------------------------------

    for tool_call in tool_calls:

        tool_executions += 1
        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nTOOL REQUESTED:", name)
        print("ARGUMENTS:", arguments)

        result = execute_tool(
            name=name,
            arguments=arguments
        )

        print("\nTOOL RESULT:")
        print(json.dumps(result, indent=2))

        # Return the result to the LLM.
        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

else:
    print(
        "\nMaximum tool execution rounds reached. "
        "Stopping without claiming completion."
    )

elapsed = time.perf_counter() - start_time

print("\n========== PERFORMANCE ==========")
print("LLM calls:", llm_calls)
print("Tool executions:", tool_executions)
print(f"Total execution time: {elapsed:.2f} seconds")