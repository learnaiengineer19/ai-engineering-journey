
"""
LESSON 18 - PART 4: LANGCHAIN TOOL CALLING
==========================================

OBJECTIVE:
Learn how LangChain converts Python functions into
LLM-callable tools and executes their tool calls.

CONCEPTS:
1. @tool decorator
2. Pydantic input schemas
3. bind_tools()
4. AIMessage.tool_calls
5. ToolMessage
6. Validated tool execution

RUN:
python -m langchain_lessons.langchain_tools
"""

import json
import sqlite3
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: DEFINE TOOL INPUT SCHEMA
# --------------------------------------------------

class FindClaimsInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    min_amount: float = Field(
        ge=0,
        description="Minimum claim amount, inclusive"
    )


# --------------------------------------------------
# STEP 2: CREATE TOOL 1 - FIND CLAIMS
# --------------------------------------------------

@tool(args_schema=FindClaimsInput)
def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """Search insurance claims by status and amount."""

    print("\n[PYTHON] Executing find_claims")

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
# STEP 3: CREATE TOOL 2 - GET CLAIM DETAILS
# --------------------------------------------------

class ClaimDetailsInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(
        pattern=r"^CLM[0-9]{4}$",
        description="Insurance claim identifier"
    )


@tool(args_schema=ClaimDetailsInput)
def get_claim_details(claim_id: str) -> dict:
    """Retrieve details of one insurance claim by ID."""

    print("\n[PYTHON] Executing get_claim_details")

    connection = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro",
        uri=True
    )

    try:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT claim_id, status, amount, claim_type
            FROM claims
            WHERE claim_id = ?
            """,
            (claim_id,)
        ).fetchone()

        return (
            dict(row)
            if row is not None
            else {"error": "Claim not found"}
        )

    finally:
        connection.close()


# --------------------------------------------------
# STEP 4: REGISTER TOOLS
# --------------------------------------------------

tools = [
    find_claims,
    get_claim_details
]

tool_registry = {
    selected_tool.name: selected_tool
    for selected_tool in tools
}


# --------------------------------------------------
# STEP 5: INSPECT LANGCHAIN TOOL METADATA
# --------------------------------------------------

for selected_tool in tools:

    print("\n========== TOOL ==========")
    print("Name:", selected_tool.name)
    print("Description:", selected_tool.description)

    print("Input schema:")
    print(
        json.dumps(
            selected_tool.args,
            indent=2
        )
    )


# --------------------------------------------------
# STEP 6: INITIALIZE MODEL AND BIND TOOLS
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

model_with_tools = model.bind_tools(tools)


# --------------------------------------------------
# STEP 7: ASK QUESTION
# --------------------------------------------------

question = (
    "Get details of insurance claim CLM1004."
)

messages = [
    SystemMessage(
        content=(
            "You are an insurance claims assistant. "
            "Use the available tools to retrieve claim data. "
            "Do not invent insurance records. "
            "For a list of claims, use find_claims."
        )
    ),
    HumanMessage(content=question)
]

print("\nUSER QUESTION:")
print(question)


# --------------------------------------------------
# STEP 8: LET QWEN2 SELECT TOOL
# --------------------------------------------------

response = model_with_tools.invoke(messages)

print("\nTOOL CALLS PROPOSED BY QWEN2:")
print(json.dumps(response.tool_calls, indent=2))

# Save the assistant message containing tool requests.
messages.append(response)


# --------------------------------------------------
# STEP 9: EXECUTE APPROVED TOOL CALLS
# --------------------------------------------------

if not response.tool_calls:
    print("\nNo tool requested by Qwen2.")

else:
    for tool_call in response.tool_calls:

        name = tool_call["name"]

        print("\nTOOL SELECTED:", name)
        print("ARGUMENTS:", tool_call["args"])

        if name not in tool_registry:
            # Never execute unknown model-requested tools.
            from langchain_core.messages import ToolMessage

            result = ToolMessage(
                content=json.dumps({
                    "error": "Unauthorized tool"
                }),
                name=name,
                tool_call_id=tool_call["id"]
            )

        else:
            try:
                # Passing the complete ToolCall lets
                # LangChain validate arguments and
                # return a ToolMessage with the
                # corresponding tool_call_id.
                result = tool_registry[name].invoke(
                    tool_call
                )
                

            except (ValidationError, ValueError):
                from langchain_core.messages import ToolMessage

                result = ToolMessage(
                    content=json.dumps({
                        "error": "Invalid tool arguments"
                    }),
                    name=name,
                    tool_call_id=tool_call["id"]
                )

            except sqlite3.Error:
                from langchain_core.messages import ToolMessage

                result = ToolMessage(
                    content=json.dumps({
                        "error": "Database query failed"
                    }),
                    name=name,
                    tool_call_id=tool_call["id"]
                )

        print("\nTOOL RESULT:")
        print(result.content)

        messages.append(result)


    # --------------------------------------------------
    # STEP 10: ASK THE MODEL TO SUMMARIZE TOOL RESULTS
    # --------------------------------------------------

    final_response = model_with_tools.invoke(messages)

    if final_response.tool_calls:
        print(
            "\nQwen2 requested additional tool calls. "
            "This example stops after one execution batch."
        )
    else:
        print("\nFINAL AI ANSWER:")
        print(final_response.content)
