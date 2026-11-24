
"""
LESSON 17 - PART 6: AGENT FAILURE RECOVERY
=========================================

OBJECTIVES:
1. Recover from transient tool failures.
2. Stop on permanent failures.
3. Limit retry attempts.
4. Detect repeated unsuccessful actions.
5. Track tool execution history.
6. Verify goal completion.

RUN:
python -m agents.recovery_agent
"""

import json
from collections import Counter

from agents.llm_planning_agent import (
    find_claims,
    calculate_total,
    generate_summary,
)


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

MAX_RETRIES = 2
MAX_TOOL_ATTEMPTS = 5

# Try:
# "transient"  -> first query fails, then succeeds
# "permanent"  -> query always fails
# "normal"     -> query succeeds immediately
SCENARIO = "normal"


# --------------------------------------------------
# STEP 2: CUSTOM ERROR TYPES
# --------------------------------------------------

class TransientToolError(Exception):
    """An error that might succeed if retried."""


class PermanentToolError(Exception):
    """An error that should not be retried."""


# --------------------------------------------------
# STEP 3: SIMULATED TOOL FAILURES
# --------------------------------------------------

class ClaimsTool:
    """
    Wrap the real find_claims function.

    Simulate different failure conditions without
    changing our database or its records.
    """

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.attempt_count = 0

    def execute(self) -> list[dict]:

        self.attempt_count += 1

        print(
            f"[DATABASE] Attempt {self.attempt_count}"
        )

        if self.scenario == "transient":

            if self.attempt_count == 1:
                raise TransientToolError(
                    "Temporary database connection failure"
                )

        elif self.scenario == "permanent":

            raise PermanentToolError(
                "Database access denied"
            )

        elif self.scenario != "normal":

            raise PermanentToolError(
                "Unknown test scenario"
            )

        elif self.scenario == "always_transient":
            raise TransientToolError(
                "Database connection unavailable"
             )

        # Real database lookup.
        return find_claims()


# --------------------------------------------------
# STEP 4: CREATE AGENT STATE
# --------------------------------------------------

def create_state() -> dict:

    return {
        "status": "in_progress",
        "results": {},
        "tool_attempts": 0,
        "retry_count": 0,
        "failed_action_counts": {},
        "history": [],
        "errors": []
    }


# --------------------------------------------------
# STEP 5: RECORD EXECUTION EVENTS
# --------------------------------------------------

def record_event(
    state: dict,
    action: str,
    outcome: str,
    details: str
) -> None:

    state["history"].append({
        "action": action,
        "outcome": outcome,
        "details": details
    })


# --------------------------------------------------
# STEP 6: EXECUTE WITH RECOVERY
# --------------------------------------------------

def execute_with_recovery(
    state: dict,
    action: str,
    tool
):
    """
    Execute one operation safely.

    - Retry only transient failures.
    - Stop on permanent failures.
    - Enforce global tool attempt limit.
    - Track repeated failed actions.

    The tool must be safe to retry.
    """

    max_attempts_for_action = MAX_RETRIES + 1

    for attempt in range(1, max_attempts_for_action + 1):

        if state["tool_attempts"] >= MAX_TOOL_ATTEMPTS:
            raise RuntimeError(
                "Global tool execution budget exceeded"
            )

        # Count actual tool attempts, not successes.
        state["tool_attempts"] += 1

        print(f"\n[AGENT] Executing: {action}")

        try:

            result = tool()

            record_event(
                state,
                action,
                "success",
                f"Succeeded on attempt {attempt}"
            )

            # Successful action ends the retry loop.
            return result

        except TransientToolError as exc:

            print("[RECOVERABLE ERROR]", str(exc))

            state["retry_count"] += 1

            counts = state["failed_action_counts"]

            counts[action] = counts.get(action, 0) + 1

            record_event(
                state,
                action,
                "transient_failure",
                str(exc)
            )

            if attempt == max_attempts_for_action:
                raise RuntimeError(
                    f"Retry limit reached for {action}"
                ) from exc

            # Repeated identical failures are bounded.
            if counts[action] > MAX_RETRIES:
                raise RuntimeError(
                    f"Repeated unsuccessful action: {action}"
                ) from exc

            print("[AGENT] Retrying the tool")

        except PermanentToolError as exc:

            record_event(
                state,
                action,
                "permanent_failure",
                str(exc)
            )

            raise RuntimeError(
                f"Non-recoverable error in {action}: {exc}"
            ) from exc


# --------------------------------------------------
# STEP 7: EXECUTE THE AGENT WORKFLOW
# --------------------------------------------------

def run_agent(scenario: str) -> dict:

    state = create_state()

    claims_tool = ClaimsTool(scenario)

    print("\nSCENARIO:", scenario)

    try:

        # ACTION 1: RETRIEVE CLAIMS

        claims = execute_with_recovery(
            state=state,
            action="find_claims",
            tool=claims_tool.execute
        )

        state["results"]["claims"] = claims

        # ACTION 2: CALCULATE EXPOSURE

        total = execute_with_recovery(
            state=state,
            action="calculate_total",
            tool=lambda: calculate_total(claims)
        )

        state["results"]["total"] = total

        # ACTION 3: GENERATE SUMMARY

        summary = execute_with_recovery(
            state=state,
            action="generate_summary",
            tool=lambda: generate_summary(claims, total)
        )

        state["results"]["summary"] = summary

        # FINAL VERIFICATION

        if total != calculate_total(claims):
            raise RuntimeError(
                "Goal verification failed: incorrect total"
            )

        if not all(
            c["status"] == "Approved"
            and c["amount"] >= 1000
            for c in claims
        ):
            raise RuntimeError(
                "Goal verification failed: invalid claims"
            )

        state["status"] = "completed"

    except Exception as exc:

        state["status"] = "failed"
        state["errors"].append(str(exc))

    return state


# --------------------------------------------------
# STEP 8: RUN
# --------------------------------------------------

if __name__ == "__main__":

    final_state = run_agent(SCENARIO)

    print("\n========== FINAL STATE ==========")

    print(json.dumps(final_state, indent=2))

    print("\nGOAL STATUS:", final_state["status"])
