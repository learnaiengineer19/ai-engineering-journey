
"""
LESSON 17 - PART 2: AGENT PLANNING AND STATE
============================================

OBJECTIVE:
Build an agent workflow with explicit planning,
state tracking and independent goal verification.

CONCEPTS:
1. Agent state
2. Execution plans
3. Task dependencies
4. Completed and failed steps
5. Goal verification
6. Separating deterministic calculations from LLMs

RUN:
python -m agents.planning_agent

PREREQUISITE:
python -m tool_calling.setup_claims_db
"""

import json
import sqlite3
from math import isclose

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: DEFINE THE GOAL
# --------------------------------------------------

GOAL = (
    "Find approved insurance claims worth at least "
    "$1000, calculate their total and summarize them."
)


# --------------------------------------------------
# STEP 2: INITIALISE AGENT STATE
# --------------------------------------------------

def create_agent_state(goal: str) -> dict:

    return {
        "goal": goal,
        "status": "in_progress",
        "plan": [
            {"step": "find_claims", "status": "pending"},
            {"step": "calculate_total", "status": "pending"},
            {"step": "generate_summary", "status": "pending"}
        ],
        "results": {},
        "errors": []
    }


# --------------------------------------------------
# STEP 3: UPDATE TASK STATUS
# --------------------------------------------------

def update_step(
    state: dict,
    step_name: str,
    status: str
) -> None:

    for step in state["plan"]:

        if step["step"] == step_name:
            step["status"] = status
            return

    raise ValueError(f"Unknown step: {step_name}")


# --------------------------------------------------
# STEP 4: RETRIEVE CLAIMS
# --------------------------------------------------

def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:

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
# STEP 5: CALCULATE TOTAL
# --------------------------------------------------

def calculate_total(claims: list[dict]) -> float:

    return round(
        sum(claim["amount"] for claim in claims),
        2
    )


# --------------------------------------------------
# STEP 6: BUILD SUMMARY
# --------------------------------------------------

def generate_summary(
    claims: list[dict],
    total: float
) -> str:

    return (
        f"Found {len(claims)} approved claims "
        f"with a combined amount of ${total:,.2f}."
    )


# --------------------------------------------------
# STEP 7: VERIFY GOAL COMPLETION
# --------------------------------------------------

def verify_goal(state: dict) -> bool:
    """
    Verify:
    - Every planned step is complete.
    - Required results exist.
    - Claim filters are satisfied.
    - Calculated total matches the claim records.
    """

    all_completed = all(
        step["status"] == "completed"
        for step in state["plan"]
    )

    if not all_completed:
        return False

    results = state["results"]

    if not all(
        key in results
        for key in ("claims", "total", "summary")
    ):
        return False

    claims = results["claims"]

    if not isinstance(claims, list):
        return False

    try:
        filters_valid = all(
            claim["status"] == "Approved"
            and claim["amount"] >= 1000
            for claim in claims
        )

        expected_total = calculate_total(claims)

        total_valid = isclose(
            expected_total,
            results["total"],
            abs_tol=0.001,
            rel_tol=0
        )

    except (KeyError, TypeError, ValueError):
        return False

    return (
        filters_valid
        and total_valid
        and isinstance(results["summary"], str)
        and bool(results["summary"].strip())
    )


# --------------------------------------------------
# STEP 8: EXECUTE THE PLAN
# --------------------------------------------------

def run_agent() -> dict:

    state = create_agent_state(GOAL)

    try:

        # TASK 1: FIND CLAIMS

        print("\nExecuting: find_claims")

        update_step(state, "find_claims", "in_progress")

        claims = find_claims(
            status="Approved",
            min_amount=1000
        )

        state["results"]["claims"] = claims

        update_step(state, "find_claims", "completed")

        # TASK 2: CALCULATE TOTAL

        print("Executing: calculate_total")

        update_step(state, "calculate_total", "in_progress")

        total = calculate_total(claims)

        state["results"]["total"] = total

        update_step(state, "calculate_total", "completed")

        # TASK 3: GENERATE SUMMARY

        print("Executing: generate_summary")

        update_step(state, "generate_summary", "in_progress")

        summary = generate_summary(claims, total)

        state["results"]["summary"] = summary

        update_step(state, "generate_summary", "completed")

        # FINAL VERIFICATION

        if verify_goal(state):
            state["status"] = "completed"
        else:
            state["status"] = "failed"
            state["errors"].append(
                "Goal verification failed"
            )

    except Exception as exc:

        state["status"] = "failed"
        state["errors"].append(str(exc))

        # Mark an interrupted step as failed.
        for step in state["plan"]:
            if step["status"] == "in_progress":
                step["status"] = "failed"

    return state


# --------------------------------------------------
# STEP 9: RUN THE AGENT
# --------------------------------------------------

if __name__ == "__main__":

    final_state = run_agent()

    print("\n========== FINAL AGENT STATE ==========")

    print(json.dumps(final_state, indent=2))

    print("\nGOAL STATUS:", final_state["status"])
