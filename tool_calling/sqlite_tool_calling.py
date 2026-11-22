
"""
LESSON 16 - PART 8.1: LLM DATABASE TOOL
=======================================

OBJECTIVE:
Allow Qwen2 to retrieve insurance claims
from a real SQLite database.

CONCEPTS:
1. Database-backed Python tools
2. Parameterized SQL
3. Pydantic argument validation
4. Tool selection using Ollama
5. Structured tool results

RUN:
python -m tool_calling.sqlite_tool_calling

PREREQUISITE:
python -m tool_calling.setup_claims_db
"""

import json
import sqlite3

from ollama import chat
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: VALIDATE TOOL ARGUMENTS
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
# STEP 2: DEFINE DATABASE TOOL
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find insurance claims by status and minimum
    amount using our local SQLite database.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum amount, inclusive.
    """

    # Open database in read-only mode.
    # No INSERT, UPDATE or DELETE is allowed.
    connection = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro",
        uri=True
    )

    try:

        connection.row_factory = sqlite3.Row

        cursor = connection.execute("""
            SELECT
                claim_id,
                status,
                amount,
                claim_type
            FROM claims
            WHERE status = ?
              AND amount >= ?
            ORDER BY amount DESC
        """, (status, min_amount))

        # Convert SQLite rows into dictionaries.
        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


# --------------------------------------------------
# STEP 3: DEFINE APPROVED TOOL REGISTRY
# --------------------------------------------------

AVAILABLE_TOOLS = {
    "find_claims": find_claims
}


# --------------------------------------------------
# STEP 4: RECEIVE USER QUESTION
# --------------------------------------------------

question = "Find claims under review above $3000"

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "Use find_claims to retrieve claim records "
            "from the database. "
            "Do not invent claim data."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 5: ASK QWEN2 TO SELECT A TOOL
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages,
    tools=[find_claims],
    options={"temperature": 0}
)

tool_calls = response.message.tool_calls or []


# --------------------------------------------------
# STEP 6: VALIDATE AND EXECUTE TOOL REQUESTS
# --------------------------------------------------

if not tool_calls:

    print("No tool was requested.")
    print(response.message.content)

else:

    messages.append(response.message)

    for tool_call in tool_calls:

        name = tool_call.function.name
        args = tool_call.function.arguments

        print("\nSELECTED TOOL:", name)
        print("GENERATED ARGUMENTS:", args)

        if name not in AVAILABLE_TOOLS:

            result = {"error": "Unknown tool"}

        else:

            try:
                validated = FindClaimsArgs.model_validate(
                    args
                )

                result = AVAILABLE_TOOLS[name](
                    **validated.model_dump()
                )

            except Exception:
                # In production, log internal details
                # and handle validation and DB errors
                # separately.
                result = {
                    "error": "Tool request failed"
                }

        print("\nDATABASE RESULT:")
        print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

    # ----------------------------------------------
    # STEP 7: GENERATE FINAL ANSWER
    # ----------------------------------------------

    final_response = chat(
        model="qwen2:7b",
        messages=messages,
        options={"temperature": 0}
    )

    print("\nFINAL AI ANSWER:")
    print(final_response.message.content)
