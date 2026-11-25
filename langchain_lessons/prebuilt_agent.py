
"""
LESSON 18 - PART 5: PREBUILT LANGCHAIN AGENT
============================================

OBJECTIVE:
Replace our manual tool-calling loop with
LangChain's create_agent().

CONCEPTS:
1. create_agent()
2. Built-in model/tool execution loop
3. Automatic ToolMessage handling
4. Agent state and message history
5. Execution limits
6. Inspecting the agent's actions

RUN:
python -m langchain_lessons.prebuilt_agent
"""

import sqlite3
from typing import Literal

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.errors import GraphRecursionError

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: DEFINE TOOL INPUT SCHEMAS
# --------------------------------------------------

class FindClaimsInput(BaseModel):

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    min_amount: float = Field(
        ge=0,
        allow_inf_nan=False
    )


class CalculateTotalInput(BaseModel):

    claim_ids: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Claim IDs to include in the total"
    )


# --------------------------------------------------
# STEP 2: TOOL 1 - SEARCH CLAIMS
# --------------------------------------------------

@tool(args_schema=FindClaimsInput)
def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find insurance claims matching the provided
    status and minimum amount.
    """

    print("\n[TOOL EXECUTED] find_claims")

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
            ORDER BY amount DESC
            """,
            (status, min_amount)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


# --------------------------------------------------
# STEP 3: TOOL 2 - CALCULATE TOTAL
# --------------------------------------------------

@tool(args_schema=CalculateTotalInput)
def calculate_claim_total(
    claim_ids: list[str]
) -> dict:
    """
    Calculate total claim amount for specified
    insurance claim IDs using the database.
    """

    print("\n[TOOL EXECUTED] calculate_claim_total")

    unique_ids = list(dict.fromkeys(claim_ids))

    # Placeholders are created from the bounded list.
    # Claim ID values remain parameterized.
    placeholders = ",".join(
        "?" for _ in unique_ids
    )

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

        missing_ids = sorted(
            set(unique_ids) - found_ids
        )

        if missing_ids:
            return {
                "success": False,
                "error": "Some claims were not found",
                "missing_ids": missing_ids
            }

        return {
            "success": True,
            "claim_count": len(rows),
            "total_amount": round(
                sum(row[1] for row in rows),
                2
            )
        }

    finally:
        connection.close()


# --------------------------------------------------
# STEP 4: INITIALIZE QWEN2
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)


# --------------------------------------------------
# STEP 5: CREATE OUR PREBUILT AGENT
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        find_claims,
        calculate_claim_total
    ],
    system_prompt=(
        "You are an insurance claims assistant. "
        "Use find_claims to retrieve claim records. "
        "When asked for a combined total, use "
        "calculate_claim_total with the retrieved IDs. "
        "Never invent claim amounts or records. "
        "If a tool reports an error, explain it. "
        "Once the required information is available, "
        "provide a concise final answer."
    )
)


# --------------------------------------------------
# STEP 6: DEFINE THE GOAL
# --------------------------------------------------

question = (
    "Find all approved insurance claims with amounts "
    "of at least $1000, calculate their combined total, "
    "and summarize the results."
)


# --------------------------------------------------
# STEP 7: INVOKE THE AGENT
# --------------------------------------------------

print("\nUSER QUESTION:")
print(question)

try:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config={
            # Limit how long the graph may continue.
            # This is a graph-step limit, not
            # an exact count of LLM/tool calls.
            "recursion_limit": 12
        }
    )


    print("\nTYPE OF RESULT:")
    print(type(result))

    print("\nTYPE OF result['messages']:")
    print(type(result["messages"]))

    print("\nTOTAL MESSAGES:")
    print(len(result["messages"]))

    print("\nMESSAGE TYPES:")
    for message in result["messages"]:
        print(type(message).__name__)
    # ----------------------------------------------
    # STEP 8: DISPLAY FINAL RESPONSE
    # ----------------------------------------------

    print("\n========== FINAL ANSWER ==========")

    print(" raw result ", result["messages"] )

    final_message = result["messages"][-1]

    print(final_message.content)

    # ----------------------------------------------
    # STEP 9: INSPECT EXECUTION HISTORY
    # ----------------------------------------------

    print("\n========== AGENT EXECUTION HISTORY ==========")

    llm_calls = 0
    tool_requests = 0
    tool_results = 0

    for message in result["messages"]:

        if isinstance(message, AIMessage):

            llm_calls += 1

            for call in message.tool_calls:

                tool_requests += 1

                print("\nLLM REQUESTED:", call["name"])
                print("ARGUMENTS:", call["args"])

        elif isinstance(message, ToolMessage):

            tool_results += 1

            print("\nTOOL RESPONSE:")
            print(message.content)

    # ----------------------------------------------
    # STEP 10: PERFORMANCE METRICS
    # ----------------------------------------------

    print("\n========== METRICS ==========")

    print("LLM responses:", llm_calls)
    print("Tool requests:", tool_requests)
    print("Tool result messages:", tool_results)

except GraphRecursionError:

    print(
        "\nAgent reached the graph execution limit "
        "before completing the request."
    )
