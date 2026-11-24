
"""
LESSON 17 - PART 1: OUR FIRST AI AGENT
=====================================

OBJECTIVE:
Build a goal-driven agent using Python and Ollama.

THE AGENT WILL:
1. Receive a goal.
2. Decide which tool to request.
3. Execute approved tools.
4. Observe the results.
5. Decide whether more actions are needed.
6. Produce a final answer.

CONCEPTS:
- Agent goals
- Actions and observations
- Tool registry
- Agent execution loop
- Message history as working context
- Stopping conditions
- Basic safety boundaries

RUN FROM PROJECT ROOT:
python -m agents.first_agent
"""

import json
import sqlite3

from ollama import chat

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MODEL = "qwen2:7b"

MAX_ROUNDS = 6

GOAL = (
    "Find all approved insurance claims with amounts "
    "of at least $1000, calculate their total amount, "
    "and provide a summary."
)


# --------------------------------------------------
# STEP 2: TOOL 1 - SEARCH CLAIMS
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Search insurance claims by status and minimum amount.

    Args:
        status: Claim status, e.g. Approved.
        min_amount: Minimum amount, inclusive.
    """

    if status not in {"Approved", "Rejected", "Under Review"}:
        return [{"error": "Invalid claim status"}]

    if (
        type(min_amount) not in (int, float)
        or not (0 <= min_amount < float("inf"))
    ):
        return [{"error": "Invalid amount"}]

    connection = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro",
        uri=True
    )

    try:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT claim_id, status, amount, claim_type
            FROM claims
            WHERE status = ?
              AND amount >= ?
            ORDER BY claim_id
            """,
            (status, min_amount)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


# --------------------------------------------------
# STEP 3: TOOL 2 - CALCULATE CLAIM TOTAL
# --------------------------------------------------

def calculate_total(claim_ids: list[str]) -> dict:
    """
    Calculate total amount for specified claim IDs
    using the insurance claims database.

    Args:
        claim_ids: List of claim identifiers.
    """

    if (
        not isinstance(claim_ids, list)
        or not claim_ids
        or any(
            not isinstance(claim_id, str)
            for claim_id in claim_ids
        )
    ):
        return {"error": "Invalid claim IDs"}

    if len(claim_ids) > 50:
        return {"error": "Too many claim IDs"}

    # Remove duplicate IDs.
    unique_ids = list(dict.fromkeys(claim_ids))

    placeholders = ",".join("?" for _ in unique_ids)

    connection = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro",
        uri=True
    )

    try:
        rows = connection.execute(
            f"""
            SELECT claim_id, amount
            FROM claims
            WHERE claim_id IN ({placeholders})
            """,
            unique_ids
        ).fetchall()

        found_ids = {row[0] for row in rows}
        missing_ids = [
            claim_id
            for claim_id in unique_ids
            if claim_id not in found_ids
        ]

        if missing_ids:
            return {
                "error": "Some claim IDs were not found",
                "missing_ids": missing_ids
            }

        total = sum(row[1] for row in rows)

        return {
            "claim_count": len(rows),
            "total_amount": round(total, 2)
        }

    finally:
        connection.close()


# --------------------------------------------------
# STEP 4: REGISTER APPROVED TOOLS
# --------------------------------------------------

TOOLS = {
    "find_claims": find_claims,
    "calculate_total": calculate_total
}


# --------------------------------------------------
# STEP 5: DEFINE THE AGENT'S GOAL AND RULES
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance operations AI agent. "
            "Your goal is to complete the user's task "
            "using the available tools. "
            "First retrieve matching claims using find_claims. "
            "Use calculate_total to compute the total amount. "
            "Do not invent claim records or amounts. "
            "If a tool fails, explain what could not be completed. "
            "Only report successful completion when "
            "the required information is available."
        )
    },
    {
        "role": "user",
        "content": GOAL
    }
]


# --------------------------------------------------
# STEP 6: EXECUTE THE AGENT LOOP
# --------------------------------------------------

print("\nAGENT GOAL:")
print(GOAL)

round_number = 0
tool_executions = 0

while round_number < MAX_ROUNDS:

    round_number += 1

    print(f"\n========== AGENT ROUND {round_number} ==========")

    # The agent decides what action is needed.
    response = chat(
        model=MODEL,
        messages=messages,
        tools=list(TOOLS.values()),
        options={"temperature": 0}
    )

    # Save the assistant response, including tool calls.
    messages.append(response.message)

    tool_calls = response.message.tool_calls or []

    # ----------------------------------------------
    # STEP 7: HANDLE FINAL ANSWER
    # ----------------------------------------------

    if not tool_calls:

        print("\nAGENT FINAL ANSWER:")
        print(response.message.content)

        break

    # ----------------------------------------------
    # STEP 8: EXECUTE REQUESTED ACTIONS
    # ----------------------------------------------

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nAGENT ACTION:", name)
        print("ARGUMENTS:", arguments)

        if name not in TOOLS:

            result = {"error": "Unknown tool"}

        else:

            # Restrict each tool to its known arguments.
            try:
                if not isinstance(arguments, dict):
                    result = {"error": "Invalid arguments"}

                elif name == "find_claims":
                    result = find_claims(
                        status=arguments.get("status"),
                        min_amount=arguments.get("min_amount")
                    )
                    tool_executions += 1

                elif name == "calculate_total":
                    result = calculate_total(
                        claim_ids=arguments.get("claim_ids")
                    )
                    tool_executions += 1

            except (TypeError, ValueError, sqlite3.Error):
                result = {"error": "Tool execution failed"}

        # ------------------------------------------
        # STEP 9: RETURN OBSERVATION TO AGENT
        # ------------------------------------------

        print("OBSERVATION:")
        print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

else:
    print(
        "\nAgent stopped after reaching "
        "the maximum number of rounds."
    )


# --------------------------------------------------
# STEP 10: DISPLAY EXECUTION METRICS
# --------------------------------------------------

print("\n========== AGENT METRICS ==========")
print("LLM rounds:", round_number)
print("Tool executions:", tool_executions)
