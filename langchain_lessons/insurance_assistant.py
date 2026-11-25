
"""
LESSON 18 - PART 9: INSURANCE ASSISTANT
=====================================

Combines:
- LangChain model factory
- Agent tool calling
- Middleware
- SQLite
- Pydantic reports
- Independent data verification

Run from project root:

python -m langchain_lessons.insurance_assistant
"""

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_lessons.model_factory import get_model
from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

DB_FILE = Path(DB_PATH)

REQUESTED_STATUS = "Approved"
MIN_AMOUNT = 1000.0

QUESTION = (
    "Find all Approved insurance claims with "
    "amounts of at least $1000. "
    "Use the database tools to retrieve matching "
    "claims and calculate their combined total. "
    "Then provide a short summary."
)


# --------------------------------------------------
# STEP 2: DEFINE TOOL INPUT SCHEMAS
# --------------------------------------------------

class SearchInput(BaseModel):
    status: Literal[
        "Approved",
        "Rejected",
        "Under Review",
    ]
    min_amount: float = Field(
        ge=0,
        allow_inf_nan=False
    )


class TotalInput(BaseModel):
    claim_ids: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Claim IDs to total"
    )


# --------------------------------------------------
# STEP 3: CENTRAL READ-ONLY DATABASE FUNCTION
# --------------------------------------------------

def read_claims(
    status: str,
    min_amount: float
) -> list[dict]:

    # SQLite URI read-only mode prevents writes.
    uri = DB_FILE.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:

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


# --------------------------------------------------
# STEP 4: LANGCHAIN TOOL - FIND CLAIMS
# --------------------------------------------------

@tool(args_schema=SearchInput)
def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """
    Retrieve insurance claims matching a
    given status and minimum amount.
    """

    print(
        f"\n[TOOL] find_claims "
        f"status={status}, min_amount={min_amount}"
    )

    return read_claims(
        status=status,
        min_amount=min_amount
    )


# --------------------------------------------------
# STEP 5: LANGCHAIN TOOL - CALCULATE TOTAL
# --------------------------------------------------

@tool(args_schema=TotalInput)
def calculate_total(
    claim_ids: list[str]
) -> dict:
    """
    Calculate the total amount of the specified
    claims directly from the database.
    """

    print(
        f"\n[TOOL] calculate_total "
        f"claim_ids={claim_ids}"
    )

    # Avoid counting duplicate IDs twice.
    unique_ids = list(dict.fromkeys(claim_ids))

    placeholders = ",".join(
        "?" for _ in unique_ids
    )

    uri = DB_FILE.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:

        rows = connection.execute(
            f"""
            SELECT claim_id, amount
            FROM claims
            WHERE claim_id IN ({placeholders})
            """,
            unique_ids
        ).fetchall()

    found_ids = {row[0] for row in rows}
    missing_ids = sorted(set(unique_ids) - found_ids)

    if missing_ids:
        return {
            "success": False,
            "missing_ids": missing_ids,
            "error": "Unknown claim IDs"
        }

    return {
        "success": True,
        "claim_ids": sorted(unique_ids),
        "claim_count": len(rows),
        "total_amount": round(
            sum(float(row[1]) for row in rows),
            2
        ),
    }


# --------------------------------------------------
# STEP 6: DEFINE FINAL OUTPUT SCHEMA
# --------------------------------------------------

class ClaimRecord(BaseModel):
    claim_id: str
    claim_type: str
    status: str
    amount: float


class VerifiedReport(BaseModel):
    verification_status: Literal["verified"]
    claim_count: int = Field(ge=0)
    claims: list[ClaimRecord]
    total_amount: float = Field(ge=0)
    agent_summary: str
    tool_requests: int = Field(ge=0)
    llm_responses: int = Field(ge=0)


# --------------------------------------------------
# STEP 7: BUILD THE AGENT
# --------------------------------------------------

model = get_model()

agent = create_agent(
    model=model,
    tools=[
        find_claims,
        calculate_total
    ],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=5,
            exit_behavior="error"
        ),
        ToolCallLimitMiddleware(
            run_limit=4,
            exit_behavior="error"
        ),
    ],
    system_prompt=(
        "You are a read-only insurance analyst. "
        "Use find_claims to retrieve records. "
        "Use calculate_total with the retrieved "
        "claim IDs when asked for a total. "
        "Never invent claim data, IDs or amounts. "
        "Use only tool results as evidence. "
        "Explain errors instead of guessing. "
        "Give a concise final response."
    )
)


# --------------------------------------------------
# STEP 8: EXECUTE AGENT
# --------------------------------------------------

def execute_agent() -> tuple[dict, float]:

    start = time.perf_counter()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": QUESTION
                }
            ]
        },
        config={
            "recursion_limit": 20
        }
    )

    elapsed = time.perf_counter() - start

    return result, elapsed


# --------------------------------------------------
# STEP 9: VERIFY THE AGENT'S WORK
# --------------------------------------------------

def verify_and_build_report(result: dict) -> VerifiedReport:
    """
    Verify that the agent called the correct tools
    with correct arguments, and that the tool
    outputs match authoritative SQLite records.

    The report is only produced if every check passes.
    """

    messages = result["messages"]

    # Independently calculate expected records.
    expected = read_claims(
        REQUESTED_STATUS,
        MIN_AMOUNT
    )

    expected_ids = sorted(
        row["claim_id"] for row in expected
    )

    expected_total = round(
        sum(float(row["amount"]) for row in expected),
        2
    )

    # Capture requests by their unique tool-call IDs.
    tool_calls = {}

    # Capture tool results by matching call IDs.
    tool_results = {}

    llm_responses = 0

    for message in messages:

        if isinstance(message, AIMessage):

            llm_responses += 1

            for call in message.tool_calls:
                tool_calls[call["id"]] = call

        elif isinstance(message, ToolMessage):
            tool_results[message.tool_call_id] = message

    verified_search = False
    verified_total = False

    for call_id, call in tool_calls.items():

        name = call["name"]
        args = call["args"]

        response = tool_results.get(call_id)

        if response is None:
            continue

        if response.status == "error":
            continue

        try:
            payload = json.loads(response.content)
        except (TypeError, ValueError):
            continue

        # Validate find_claims request and result.
        if name == "find_claims":

            if (
                args.get("status") != REQUESTED_STATUS
                or args.get("min_amount") != MIN_AMOUNT
            ):
                continue

            if payload == expected:
                verified_search = True

        # Validate total calculation.
        elif name == "calculate_total":

            supplied_ids = args.get("claim_ids", [])

            if (
                not isinstance(supplied_ids, list)
                or len(supplied_ids) != len(expected_ids)
                or sorted(supplied_ids) != expected_ids
            ):
                continue

            if (
                isinstance(payload, dict)
                and payload.get("success") is True
                and payload.get("claim_count") == len(expected)
                and payload.get("claim_ids") == expected_ids
                and math.isclose(
                    float(payload["total_amount"]),
                    expected_total,
                    abs_tol=0.01
                )
            ):
                verified_total = True

    if not verified_search:
        raise ValueError(
            "Verification failed: no matching "
            "search request and result."
        )

    if not verified_total:
        raise ValueError(
            "Verification failed: no matching "
            "total calculation."
        )

    # We trust the verified database for financial
    # figures, not the model's natural-language answer.
    final_message = messages[-1]

    if not isinstance(final_message, AIMessage):
        raise ValueError("Missing final AI answer")

    if final_message.tool_calls:
        raise ValueError(
            "Agent ended with unexecuted tool requests"
        )

    return VerifiedReport(
        verification_status="verified",
        claim_count=len(expected),
        claims=[
            ClaimRecord.model_validate(row)
            for row in expected
        ],
        total_amount=expected_total,
        agent_summary=str(final_message.content),
        tool_requests=len(tool_calls),
        llm_responses=llm_responses,
    )


# --------------------------------------------------
# STEP 10: MAIN APPLICATION
# --------------------------------------------------

def main():

    print("\n========== INSURANCE ASSISTANT ==========")
    print("QUESTION:", QUESTION)

    try:

        result, elapsed = execute_agent()

        print("\n========== AGENT HISTORY ==========")

        for message in result["messages"]:

            if isinstance(message, AIMessage):

                for call in message.tool_calls:
                    print(
                        "REQUEST:",
                        call["name"],
                        call["args"]
                    )

            elif isinstance(message, ToolMessage):

                print(
                    "RESULT:",
                    message.name,
                    message.content
                )

        # Verify before producing a report.
        report = verify_and_build_report(result)

        print("\n========== VERIFIED REPORT ==========")
        print(report.model_dump_json(indent=2))

        print(f"\nExecution time: {elapsed:.2f}s")

        output_path = (
            Path("langchain_lessons")
            / "insurance_report.json"
        )

        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8"
        )

        print(f"Report saved to: {output_path}")

    except Exception as exc:

        # Fail closed: no unverified report.
        print("\n[REPORT NOT GENERATED]")
        print(type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
