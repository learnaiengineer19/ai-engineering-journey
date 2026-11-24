
"""
LESSON 17 - PART 3: LLM-GENERATED PLANNING
==========================================

OBJECTIVE:
1. Give Qwen2 a goal.
2. Ask Qwen2 to generate a structured plan.
3. Validate the plan using Pydantic.
4. Check task dependencies.
5. Execute approved Python actions.
6. Track state and verify the final result.

IMPORTANT:
Qwen2 proposes WHAT actions should happen.
Python controls HOW actions execute.

RUN:
python -m agents.llm_planning_agent
"""

import json
import sqlite3
from math import isclose
from typing import Literal

from ollama import chat
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MODEL = "qwen2:7b"

GOAL = (
    "Find all approved insurance claims worth at least "
    "$1000, calculate their total amount and summarize."
)

# Trusted application-level filters.
# The LLM does not control these values.
STATUS = "Approved"
MIN_AMOUNT = 1000


# --------------------------------------------------
# STEP 2: DEFINE ALLOWED PLAN STRUCTURE
# --------------------------------------------------

class PlanStep(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    id: int = Field(ge=1)

    action: Literal[
        "find_claims",
        "calculate_total",
        "generate_summary"
    ]

    depends_on: list[int]


class AgentPlan(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=5
    )


# --------------------------------------------------
# STEP 3: ASK LLM TO GENERATE A PLAN
# --------------------------------------------------

def generate_plan(goal: str) -> AgentPlan:

    print("\n[PLANNER] Asking Qwen2 to create a plan")

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI workflow planner. "
                    "Your job is to create an execution plan. "
                    "Available actions are: "
                    "find_claims, calculate_total, "
                    "generate_summary. "
                    "Use each required action exactly once. "
                    "Task IDs must start at 1 and increase. "
                    "Dependencies must reference earlier IDs. "
                    "First find claims, then calculate total, "
                    "then generate the summary. "
                    "Return only the requested JSON structure."
                )
            },
            {
                "role": "user",
                "content": goal
            }
        ],
        format=AgentPlan.model_json_schema(),
        options={"temperature": 0}
    )

    # Validate the LLM-generated JSON.
    return AgentPlan.model_validate_json(
        response.message.content
    )


# --------------------------------------------------
# STEP 4: VALIDATE PLAN LOGIC
# --------------------------------------------------

def validate_plan(plan: AgentPlan) -> None:
    """
    Pydantic checks data types.

    This function checks BUSINESS RULES and
    task dependencies.

    For this first exercise, we require exactly
    the three essential actions in their dependency
    order, with no missing or extra tasks.
    """

    required_actions = [
        "find_claims",
        "calculate_total",
        "generate_summary"
    ]

    actual_actions = [
        step.action
        for step in plan.steps
    ]

    if actual_actions != required_actions:
        raise ValueError(
            "Plan has missing, duplicated "
            "or incorrectly ordered actions"
        )

    completed_ids = set()
    action_to_id = {}

    for step in plan.steps:

        if step.id in completed_ids:
            raise ValueError("Duplicate step ID")

        # A dependency must already exist.
        if not set(step.depends_on).issubset(
            completed_ids
        ):
            raise ValueError(
                f"Invalid dependencies for step {step.id}"
            )

        if len(step.depends_on) != len(
            set(step.depends_on)
        ):
            raise ValueError("Duplicate dependency")

        completed_ids.add(step.id)
        action_to_id[step.action] = step.id

    # Verify essential dependencies.
    search_id = action_to_id["find_claims"]
    total_id = action_to_id["calculate_total"]

    if search_id not in plan.steps[1].depends_on:
        raise ValueError(
            "calculate_total must depend on find_claims"
        )

    # The summary must depend directly on calculate_total.
    # calculate_total already depends on find_claims,
    # so find_claims is an indirect (transitive) dependency.

    if total_id not in plan.steps[2].depends_on:
        raise ValueError(
            "generate_summary must depend on calculate_total"
        )


# --------------------------------------------------
# STEP 5: APPROVED DATABASE OPERATION
# --------------------------------------------------

def find_claims() -> list[dict]:

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
            (STATUS, MIN_AMOUNT)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


# --------------------------------------------------
# STEP 6: APPROVED CALCULATION OPERATION
# --------------------------------------------------

def calculate_total(claims: list[dict]) -> float:

    return round(
        sum(claim["amount"] for claim in claims),
        2
    )


# --------------------------------------------------
# STEP 7: APPROVED SUMMARY OPERATION
# --------------------------------------------------

def generate_summary(
    claims: list[dict],
    total: float
) -> str:

    return (
        f"Found {len(claims)} approved claims "
        f"worth at least ${MIN_AMOUNT:,.2f}. "
        f"Total exposure: ${total:,.2f}."
    )


# --------------------------------------------------
# STEP 8: VERIFY FINAL GOAL
# --------------------------------------------------

def verify_goal(state: dict) -> bool:

    if state["errors"]:
        return False

    if not all(
        step["status"] == "completed"
        for step in state["plan"]
    ):
        return False

    results = state["results"]

    if not all(
        key in results
        for key in ("claims", "total", "summary")
    ):
        return False

    try:
        claims = results["claims"]

        if not isinstance(claims, list):
            return False

        if not all(
            claim["status"] == STATUS
            and claim["amount"] >= MIN_AMOUNT
            for claim in claims
        ):
            return False

        return (
            isclose(
                calculate_total(claims),
                results["total"],
                rel_tol=0,
                abs_tol=0.001
            )
            and isinstance(results["summary"], str)
            and bool(results["summary"].strip())
            and results["summary"]
                == generate_summary(
                    claims, results["total"]
                )
        )

    except (KeyError, TypeError, ValueError):
        return False


# --------------------------------------------------
# STEP 9: EXECUTE THE LLM-GENERATED PLAN
# --------------------------------------------------

def run_agent() -> dict:

    state = {
        "goal": GOAL,
        "status": "planning",
        "plan": [],
        "results": {},
        "errors": []
    }

    try:

        # ------------------------------------------
        # 9.1: GENERATE PLAN
        # ------------------------------------------

        plan = generate_plan(GOAL)

        print("\nLLM-GENERATED PLAN:")
        print(
            json.dumps(
                plan.model_dump(),
                indent=2
            )
        )

        # ------------------------------------------
        # 9.2: VALIDATE THE PROPOSED PLAN
        # ------------------------------------------

        validate_plan(plan)

        print("\n[VALIDATOR] Plan approved")

        # Convert validated plan into runtime state.
        state["plan"] = [
            {
                **step.model_dump(),
                "status": "pending"
            }
            for step in plan.steps
        ]

        state["status"] = "executing"

        completed = set()

        # ------------------------------------------
        # 9.3: EXECUTE EACH TASK
        # ------------------------------------------

        for step in state["plan"]:

            if not set(step["depends_on"]).issubset(
                completed
            ):
                raise RuntimeError(
                    "Required dependency not completed"
                )

            action = step["action"]

            print(f"\n[EXECUTOR] Running {action}")

            step["status"] = "in_progress"

            if action == "find_claims":

                claims = find_claims()

                state["results"]["claims"] = claims

            elif action == "calculate_total":

                claims = state["results"]["claims"]

                total = calculate_total(claims)

                state["results"]["total"] = total

            elif action == "generate_summary":

                claims = state["results"]["claims"]
                total = state["results"]["total"]

                summary = generate_summary(
                    claims,
                    total
                )

                state["results"]["summary"] = summary

            else:
                raise ValueError("Unauthorized action")

            step["status"] = "completed"

            completed.add(step["id"])

        # ------------------------------------------
        # 9.4: INDEPENDENT GOAL VERIFICATION
        # ------------------------------------------

        print("\n[VERIFIER] Checking results")

        if verify_goal(state):
            state["status"] = "completed"
        else:
            raise ValueError(
                "Goal verification failed"
            )

    except Exception as exc:

        state["status"] = "failed"
        state["errors"].append(str(exc))

        for step in state["plan"]:
            if step["status"] == "in_progress":
                step["status"] = "failed"

    return state


# --------------------------------------------------
# STEP 10: RUN THE AGENT
# --------------------------------------------------

if __name__ == "__main__":

    result = run_agent()

    print("\n========== FINAL AGENT STATE ==========")
    print(json.dumps(result, indent=2))

    print("\nGOAL STATUS:", result["status"])
