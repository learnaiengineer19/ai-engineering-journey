
"""
LESSON 16 - PART 9.4: STRUCTURED TOOL RESULTS
============================================

OBJECTIVE:
1. Let Qwen2 select an approved data source.
2. Validate LLM-generated tool arguments.
3. Execute a SQLite or REST API lookup.
4. Validate the returned records with Pydantic.
5. Construct a structured JSON response in Python.
6. Handle failures without inventing claim details.

KEY LESSON:
Use the LLM to interpret intent.
Use Python for data validation and deterministic output.

RUN:
python -m tool_calling.structured_claims_assistant

PREREQUISITES:
- claims.db exists
- Local API running for REST API questions
"""

import json
import sqlite3
import time

from typing import Literal

import requests
from ollama import chat
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MODEL = "qwen2:7b"
API_URL = "http://127.0.0.1:8765/claims"

# Change this to test different questions.
QUESTION = (
    "Find approved claims above $10000 using SQLite"
)


# --------------------------------------------------
# STEP 2: PYDANTIC INPUT VALIDATION
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
# STEP 3: PYDANTIC OUTPUT VALIDATION
# --------------------------------------------------

class ClaimRecord(BaseModel):
    """Validate each record returned by a tool."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    claim_id: str = Field(
        pattern=r"^CLM[0-9]{4}$"
    )

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    amount: float = Field(
        ge=0,
        allow_inf_nan=False
    )

    claim_type: str


class ClaimsResponse(BaseModel):
    """Standard output contract for our application."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    success: bool
    source: Literal["database", "api"]
    count: int = Field(ge=0)
    claims: list[ClaimRecord]
    message: str


# --------------------------------------------------
# STEP 4: DEFINE DATABASE TOOL
# --------------------------------------------------

def find_claims_database(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Search the local SQLite claims database.

    Use for direct database lookups.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum amount, inclusive.
    """

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
# STEP 5: DEFINE API TOOL
# --------------------------------------------------

def find_claims_api(
    status: str,
    min_amount: float
) -> dict:
    """
    Search claims through the local REST API.

    Use when the user explicitly requests an API lookup.

    Args:
        status: Approved, Rejected or Under Review.
        min_amount: Minimum amount, inclusive.
    """

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
# STEP 6: REGISTER APPROVED TOOLS
# --------------------------------------------------

TOOLS = {
    "find_claims_database": find_claims_database,
    "find_claims_api": find_claims_api
}

TOOL_SOURCES = {
    "find_claims_database": "database",
    "find_claims_api": "api"
}


# --------------------------------------------------
# STEP 7: DETERMINE ALLOWED TOOLS
# --------------------------------------------------

def choose_allowed_tools(question: str) -> dict:
    """
    Demo-only source routing.

    Explicit source selection is recognized using
    simple keywords. Production routing needs a
    more robust, trusted application-level contract.
    """

    text = question.lower()

    if "api" in text or "rest" in text:
        return {
            "find_claims_api": find_claims_api
        }

    # Default to SQLite for this exercise.
    return {
        "find_claims_database": find_claims_database
    }


# --------------------------------------------------
# STEP 8: BUILD A STRUCTURED RESULT
# --------------------------------------------------

def build_result(
    source: str,
    raw_records: list[dict]
) -> ClaimsResponse:
    """
    Validate tool results and build a response.

    Pydantic checks each returned record.
    Python calculates count from the actual data.
    """

    claims = [
        ClaimRecord.model_validate(record)
        for record in raw_records
    ]

    return ClaimsResponse(
        success=True,
        source=source,
        count=len(claims),
        claims=claims,
        message=f"Found {len(claims)} matching claims."
    )


# --------------------------------------------------
# STEP 9: MAIN WORKFLOW
# --------------------------------------------------

def run_assistant(question: str) -> dict:

    allowed_tools = choose_allowed_tools(question)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an insurance claims assistant. "
                "Select an available tool to retrieve "
                "insurance claim records. "
                "Never invent claim information. "
                "Use only the tools provided."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    start_time = time.perf_counter()

    llm_calls = 0
    tool_executions = 0

    # ----------------------------------------------
    # STEP 9.1: REQUEST TOOL SELECTION
    # ----------------------------------------------

    response = chat(
        model=MODEL,
        messages=messages,
        tools=list(allowed_tools.values()),
        options={"temperature": 0}
    )

    llm_calls += 1

    calls = response.message.tool_calls or []

    if len(calls) != 1:
        return {
            "success": False,
            "error": (
                "Expected exactly one tool request."
            ),
            "llm_calls": llm_calls
        }

    tool_call = calls[0]

    name = tool_call.function.name
    arguments = tool_call.function.arguments

    print("\nSELECTED TOOL:", name)
    print("GENERATED ARGUMENTS:", arguments)

    # ----------------------------------------------
    # STEP 9.2: ENFORCE SOURCE RESTRICTIONS
    # ----------------------------------------------

    if name not in allowed_tools:
        return {
            "success": False,
            "error": "Tool is not permitted",
            "llm_calls": llm_calls
        }

    # ----------------------------------------------
    # STEP 9.3: VALIDATE INPUT ARGUMENTS
    # ----------------------------------------------

    try:
        validated = FindClaimsArgs.model_validate(
            arguments
        )

        # ------------------------------------------
        # STEP 9.4: EXECUTE THE SELECTED TOOL
        # ------------------------------------------

        raw_result = allowed_tools[name](
            **validated.model_dump()
        )

        tool_executions += 1

        # The API returns a wrapper containing
        # count and claims. SQLite returns a list.
        if name == "find_claims_api":

            if (
                not isinstance(raw_result, dict)
                or not isinstance(
                    raw_result.get("claims"), list
                )
            ):
                raise ValueError(
                    "Unexpected API response format"
                )

            records = raw_result["claims"]

            if raw_result.get("count") != len(records):
                raise ValueError(
                    "API count does not match records"
                )

        else:
            records = raw_result

        # ------------------------------------------
        # STEP 9.5: VALIDATE STRUCTURED OUTPUT
        # ------------------------------------------

        result = build_result(
            source=TOOL_SOURCES[name],
            raw_records=records
        )

        elapsed = time.perf_counter() - start_time

        return {
            "response": result.model_dump(),
            "metrics": {
                "llm_calls": llm_calls,
                "tool_executions": tool_executions,
                "elapsed_seconds": round(elapsed, 2)
            }
        }

    except ValidationError:
        error = "Input or output validation failed"

    except requests.Timeout:
        error = "REST API request timed out"

    except requests.RequestException:
        error = "REST API request failed"

    except sqlite3.Error:
        error = "Database query failed"

    except (TypeError, ValueError, KeyError):
        error = "Unexpected tool response"

    # Return a controlled failure, not invented claims.
    return {
        "success": False,
        "error": error,
        "llm_calls": llm_calls,
        "tool_executions": tool_executions
    }


# --------------------------------------------------
# STEP 10: RUN AND DISPLAY
# --------------------------------------------------

if __name__ == "__main__":

    result = run_assistant(QUESTION)

    print("\nSTRUCTURED ASSISTANT RESULT:")
    print(json.dumps(result, indent=2))
