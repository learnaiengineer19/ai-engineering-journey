
"""
LESSON 18 - PART 6: LANGCHAIN AGENT MIDDLEWARE
=============================================

OBJECTIVE:
Learn how built-in middleware handles:
- Model-call limits
- Tool-call limits
- Transient tool failures
- Safe stopping conditions

RUN:
python -m langchain_lessons.safe_agent

PREREQUISITES:
- Local Ollama and qwen2:7b
- Existing fictional insurance claims database
- Compatible LangChain 1.x version
"""

from langchain.agents import create_agent

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.errors import GraphRecursionError

from agents.llm_planning_agent import find_claims, calculate_total


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MODEL = "qwen2:7b"

MAX_MODEL_CALLS = 4
MAX_TOOL_CALLS = 3

# Options: "normal" or "transient"
SCENARIO = "normal"

# This counter is only for our local experiment.
# It tracks actual attempts inside the tool.
database_attempts = 0


# --------------------------------------------------
# STEP 2: DEFINE LANGCHAIN TOOLS
# --------------------------------------------------

@tool
def search_approved_claims() -> list[dict]:
    """
    Retrieve approved insurance claims worth
    at least $1000 from the local database.
    """

    global database_attempts

    database_attempts += 1

    print(
        f"\n[DATABASE] Attempt {database_attempts}"
    )

    # Simulate a temporary connection failure.
    if (
        SCENARIO == "transient"
        and database_attempts == 1
    ):
        raise ConnectionError(
            "Simulated temporary database failure"
        )

    # Reuse the existing read-only SQLite function.
    return find_claims()


class TotalInput(BaseModel):
    amounts: list[float] = Field(
        min_length=1,
        max_length=50,
        description="Amounts of matching claims"
    )


@tool(args_schema=TotalInput)
def calculate_claim_exposure(
    amounts: list[float]
) -> dict:
    """
    Calculate the combined amount of insurance
    claims using the amounts already retrieved.
    """

    if any(
        not isinstance(amount, (int, float))
        or not (0 <= amount < float("inf"))
        for amount in amounts
    ):
        raise ValueError("Invalid claim amount")

    return {
        "claim_count": len(amounts),
        "total_amount": round(sum(amounts), 2)
    }


# --------------------------------------------------
# STEP 3: INITIALIZE MODEL
# --------------------------------------------------

model = ChatOllama(
    model=MODEL,
    temperature=0
)


# --------------------------------------------------
# STEP 4: CONFIGURE MIDDLEWARE
# --------------------------------------------------

middleware = [

    # Allow up to four LLM calls in this run.
    ModelCallLimitMiddleware(
        run_limit=MAX_MODEL_CALLS,
        exit_behavior="error"
    ),

    # Allow at most three tool calls per run.
    ToolCallLimitMiddleware(
        run_limit=MAX_TOOL_CALLS,
        exit_behavior="error"
    ),

    # Retry only ConnectionError.
    # One retry = two total attempts.
    ToolRetryMiddleware(
        tools=["search_approved_claims"],
        max_retries=1,
        retry_on=(ConnectionError,),
        initial_delay=0.1,
        jitter=False,
        on_failure="error"
    ),
]


# --------------------------------------------------
# STEP 5: CREATE THE AGENT
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        search_approved_claims,
        calculate_claim_exposure
    ],
    middleware=middleware,
    system_prompt=(
        "You are a read-only insurance assistant. "
        "First use search_approved_claims to retrieve "
        "approved claims. "
        "When a total is requested, use "
        "calculate_claim_exposure with the "
        "amounts returned by the search tool. "
        "Never invent claim records or amounts. "
        "Do not request unnecessary repeated tools. "
        "Provide a concise final answer."
    )
)


# --------------------------------------------------
# STEP 6: DEFINE THE GOAL
# --------------------------------------------------

question = (
    "Find approved claims above $1000, "
    "calculate their total exposure, "
    "and summarize the results."
)


# --------------------------------------------------
# STEP 7: RUN AGENT
# --------------------------------------------------

print("\nSCENARIO:", SCENARIO)
print("USER QUESTION:", question)

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
            "recursion_limit": 15
        }
    )

    print("\n========== FINAL ANSWER ==========")
    print(result["messages"][-1].content)

    # ----------------------------------------------
    # STEP 8: INSPECT EXECUTION HISTORY
    # ----------------------------------------------

    llm_responses = 0
    tool_requests = 0
    tool_results = 0

    for message in result["messages"]:

        if isinstance(message, AIMessage):

            llm_responses += 1

            for call in message.tool_calls:

                tool_requests += 1

                print(
                    "\nLLM REQUESTED:",
                    call["name"]
                )

        elif isinstance(message, ToolMessage):

            tool_results += 1

            print(
                "\nTOOL RESULT:",
                message.content
            )

    print("\n========== METRICS ==========")

    print("LLM responses:", llm_responses)
    print("Tool requests:", tool_requests)
    print("Tool result messages:", tool_results)
    print("Database attempts:", database_attempts)

except GraphRecursionError:

    print(
        "\n[STOPPED] Graph execution limit reached"
    )

except Exception as exc:

    # This demonstration catches the exception at the
    # application boundary and reports a controlled stop.
    print(
        "\n[STOPPED] Agent execution failed or "
        "an execution limit was reached."
    )
    print("Error type:", type(exc).__name__)
