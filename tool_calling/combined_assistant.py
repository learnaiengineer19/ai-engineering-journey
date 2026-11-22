
"""
LESSON 16 - PART 9.3: COMBINED INSURANCE ASSISTANT
=================================================

OBJECTIVE:
Allow Qwen2 to choose between:
1. A local SQLite database tool
2. A REST API tool

CONCEPTS:
- Multiple tool selection
- Pydantic argument validation
- SQL queries and HTTP requests
- Tool execution registry
- Error handling
- Bounded tool execution loop
- LLM call and execution-time measurement

PREREQUISITES:
1. SQLite database created in Part 8.1
2. Local claims API running for API requests

RUN FROM PROJECT ROOT:
python -m tool_calling.combined_assistant
"""

import json
import sqlite3
import time

from typing import Literal

import requests
from ollama import chat
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MODEL = "qwen2:7b"

API_URL = "http://127.0.0.1:8765/claims"

MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 5


# --------------------------------------------------
# STEP 2: VALIDATE TOOL ARGUMENTS
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
# STEP 3: TOOL 1 - SQLITE DATABASE
# --------------------------------------------------

def find_claims_database(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Search insurance claims directly in the local
    SQLite database.

    Use this tool for database or SQL-based lookups.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount, inclusive.
    """

    print("\n[DATABASE TOOL] Querying SQLite")

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
# STEP 4: TOOL 2 - REST API
# --------------------------------------------------

def find_claims_api(
    status: str,
    min_amount: float
) -> dict:
    """
    Search insurance claims through the local REST API.

    Use this tool when the user explicitly asks for
    an API-based lookup.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum claim amount, inclusive.
    """

    print("\n[API TOOL] Calling claims REST API")

    response = requests.get(
        API_URL,
        params={
            "status": status,
            "min_amount": min_amount
        },
        timeout=5
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# STEP 5: APPROVED TOOL REGISTRY
# --------------------------------------------------

AVAILABLE_TOOLS = {
    "find_claims_database": find_claims_database,
    "find_claims_api": find_claims_api
}


# --------------------------------------------------
# STEP 6: SAFE TOOL EXECUTION
# --------------------------------------------------

def execute_tool(name: str, arguments: dict) -> dict:
    """
    Validate a tool request and execute only
    approved Python functions.

    Return a structured result for the LLM.
    """

    if name not in AVAILABLE_TOOLS:
        return {
            "success": False,
            "error": "Unknown or unauthorized tool"
        }

    try:
        validated = FindClaimsArgs.model_validate(
            arguments
        )

    except ValidationError as exc:
        return {
            "success": False,
            "error": "Invalid tool arguments",
            "details": [
                error["msg"]
                for error in exc.errors()
            ]
        }

    try:
        function = AVAILABLE_TOOLS[name]

        data = function(
            **validated.model_dump()
        )

        return {
            "success": True,
            "data": data
        }

    except sqlite3.Error:
        return {
            "success": False,
            "error": "Database query failed"
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "API request timed out"
        }

    except requests.RequestException:
        return {
            "success": False,
            "error": "API request failed"
        }


# --------------------------------------------------
# STEP 7: DEFINE USER QUESTION
# --------------------------------------------------

question = (
    "Find approved claims above $1000 using the REST API."
)

messages = [
    {
        "role": "system",
        "content": (
            "You are an insurance claims assistant. "
            "For database or SQL requests, use "
            "find_claims_database. "
            "If the user explicitly requests the REST API, "
            "do not switch to SQLite when the API fails. "
            "Do not suggest an alternative data source "
            "unless the user asks for one. "
            "When the requested source fails, explain "
            "the failure and stop."
            "For ordinary claim searches without "
            "a specified source, prefer the database. "
            "Do not call both tools unnecessarily. "
            "Never invent claims or results. "
            "If a tool fails, explain the failure."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# --------------------------------------------------
# STEP 8: EXECUTE BOUNDED TOOL-CALLING LOOP
# --------------------------------------------------

llm_calls = 0
tool_calls_count = 0

start_time = time.perf_counter()

print("\nUSER QUESTION:")
print(question)

while llm_calls < MAX_LLM_CALLS:

    llm_calls += 1

    response = chat(
        model=MODEL,
        messages=messages,
        tools=[
            find_claims_database,
            find_claims_api
        ],
        options={"temperature": 0}
    )

    messages.append(response.message)

    tool_calls = response.message.tool_calls or []

    # If the model doesn't request tools,
    # the workflow has reached its final answer.
    if not tool_calls:

        print("\nFINAL AI ANSWER:")
        print(response.message.content)
        break

    # Bound total tool calls, not just LLM rounds.
    if tool_calls_count + len(tool_calls) > MAX_TOOL_CALLS:

        print("\nTool execution limit exceeded.")
        break

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        tool_calls_count += 1

        print("\nTOOL SELECTED:", name)
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

else:
    print("\nMaximum LLM call limit reached.")


# --------------------------------------------------
# STEP 9: PERFORMANCE METRICS
# --------------------------------------------------

elapsed = time.perf_counter() - start_time

print("\n========== PERFORMANCE ==========")

print("LLM calls:", llm_calls)
print("Tool requests:", tool_calls_count)
print(f"Total time: {elapsed:.2f} seconds")
