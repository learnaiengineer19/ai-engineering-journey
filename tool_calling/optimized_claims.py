
"""
LESSON 16 - PART 9.1: OPTIMIZED TOOL CALLING
===========================================

OBJECTIVE:
Reduce the number of LLM calls by designing
a tool that returns all required claim details
in a single database query.

CONCEPTS:
1. Coarse-grained tool design
2. Database-backed tools
3. Pydantic argument validation
4. Measuring LLM calls
5. Measuring end-to-end execution time
6. Comparing multi-step vs optimized workflows

EXPECTED:
One tool request + one final LLM response.

RUN FROM PROJECT ROOT:
python -m tool_calling.optimized_claims

PREREQUISITE:
python -m tool_calling.setup_claims_db
"""

import json
import sqlite3
import time

from typing import Literal

from ollama import chat
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError
)

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: DEFINE TOOL ARGUMENT SCHEMA
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
# STEP 2: OPTIMIZED DATABASE FUNCTION
# --------------------------------------------------

def find_claims_with_details(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Find matching insurance claims and return
    complete available details in one query.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount, inclusive.

    Returns:
        Matching claim records with their available
        database fields.
    """

    print("\n[PYTHON] Executing optimized database query")

    connection = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro",
        uri=True
    )

    try:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                claim_id,
                status,
                amount,
                claim_type
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
# STEP 3: DEFINE USER QUESTION
# --------------------------------------------------

question = (
    "Find all approved insurance claims "
    "above $1000 and provide their details."
)

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "Use find_claims_with_details to retrieve "
            "matching records from the database. "
            "The tool already returns all available "
            "claim details, so do not request "
            "additional lookups. "
            "Never invent claim information."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 4: INITIALISE PERFORMANCE COUNTERS
# --------------------------------------------------

llm_calls = 0
tool_calls_count = 0

start_time = time.perf_counter()


# --------------------------------------------------
# STEP 5: REQUEST THE TOOL CALL
# --------------------------------------------------

response = chat(
    model="qwen2:7b",
    messages=messages,
    tools=[find_claims_with_details],
    options={"temperature": 0}
)

llm_calls += 1

tool_calls = response.message.tool_calls or []

print("\nUSER QUESTION:")
print(question)


# --------------------------------------------------
# STEP 6: VALIDATE AND EXECUTE TOOL
# --------------------------------------------------

if not tool_calls:

    print("\nNo tool call requested.")
    print(response.message.content)

else:

    messages.append(response.message)

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nTOOL SELECTED:", name)
        print("ARGUMENTS:", arguments)

        if name != "find_claims_with_details":

            result = {
                "error": "Unauthorized tool"
            }

        else:

            try:
                validated = FindClaimsArgs.model_validate(
                    arguments
                )

                result = find_claims_with_details(
                    **validated.model_dump()
                )

                tool_calls_count += 1

            except ValidationError as exc:

                result = {
                    "error": "Invalid tool arguments",
                    "details": [
                        error["msg"]
                        for error in exc.errors()
                    ]
                }

            except sqlite3.Error:

                result = {
                    "error": "Database operation failed"
                }

        print("\nTOOL RESULT:")
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

    llm_calls += 1

    print("\nFINAL AI ANSWER:")
    print(final_response.message.content)


# --------------------------------------------------
# STEP 8: DISPLAY PERFORMANCE METRICS
# --------------------------------------------------

elapsed = time.perf_counter() - start_time

print("\n========== PERFORMANCE ==========")

print("LLM calls:", llm_calls)

print("Successful tool executions:", tool_calls_count)

print(f"Total execution time: {elapsed:.2f} seconds")
