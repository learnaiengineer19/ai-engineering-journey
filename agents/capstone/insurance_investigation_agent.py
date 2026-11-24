
"""
LESSON 17 - PART 8: INSURANCE AGENT CAPSTONE
============================================

Integrates:
- LLM planning
- Pydantic plan validation
- Agent state
- Tool execution and retries
- Adaptive decision-making
- Human-in-the-loop approval
- Checkpointing
- Final verification

Existing source code is reused from Parts 3, 5 and 6.

No claim database modifications are performed.

RUN:
python -m agents.capstone.insurance_investigation_agent
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.llm_planning_agent import (
    GOAL,
    generate_plan,
    validate_plan,
    find_claims,
    calculate_total,
    generate_summary,
)

from agents.adaptive_agent import (
    decide_next_action,
    validate_decision,
    review_high_value,
    EXPOSURE_THRESHOLD,
)

from agents.recovery_agent import (
    create_state,
    execute_with_recovery,
)


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "investigation_checkpoint.json"
)

# Set True to simulate one recoverable retrieval failure.
SIMULATE_TRANSIENT_FAILURE = False


# --------------------------------------------------
# STEP 2: STATE AND EVENT HISTORY
# --------------------------------------------------

def new_state() -> dict:

    state = create_state()

    state.update({
        "goal": GOAL,
        "plan": [],
        "decision": None,
        "proposals": [],
        "approvals": {},
        "review_flags": {},
        "summary": None,
    })

    return state


def record_event(state, event, details):

    state["history"].append({
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event": event,
        "details": details,
    })


def save_checkpoint(state):

    temporary = CHECKPOINT_PATH.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    temporary.replace(CHECKPOINT_PATH)

    print("[CHECKPOINT] State saved")


# --------------------------------------------------
# STEP 3: SIMULATED TRANSIENT FAILURE
# --------------------------------------------------

class ClaimsRetriever:
    """
    Optional first-attempt failure to demonstrate
    reuse of our retry controller.
    """

    def __init__(self):
        self.attempts = 0

    def execute(self):

        self.attempts += 1

        if (
            SIMULATE_TRANSIENT_FAILURE
            and self.attempts == 1
        ):
            from agents.recovery_agent import (
                TransientToolError
            )

            raise TransientToolError(
                "Simulated temporary database error"
            )

        return find_claims()


# --------------------------------------------------
# STEP 4: HUMAN APPROVAL
# --------------------------------------------------

def ask_for_approval(proposal: dict) -> bool:

    print("\n========== HUMAN APPROVAL ==========")
    print("Claim:", proposal["claim_id"])
    print("Amount:", proposal["amount"])
    print("Reason:", proposal["review_reason"])

    answer = input(
        "Flag this claim for manual review? (yes/no): "
    ).strip().lower()

    return answer == "yes"


# --------------------------------------------------
# STEP 5: FINAL GOAL VERIFICATION
# --------------------------------------------------

def verify_investigation(state: dict) -> bool:

    results = state["results"]

    required = {"claims", "total", "summary"}

    if not required.issubset(results):
        return False

    if state["errors"]:
        return False

    claims = results["claims"]
    total = results["total"]

    # Verify financial calculation.
    if total != calculate_total(claims):
        return False

    # Verify the query filters.
    if not all(
        claim["status"] == "Approved"
        and claim["amount"] >= 1000
        for claim in claims
    ):
        return False

    # Verify the summary matches source data.
    if results["summary"] != generate_summary(
        claims, total
    ):
        return False

    # Check that the chosen branch was permitted.
    decision = state["decision"]

    if decision is None:
        return False

    expected_action = (
        "review_high_value"
        if total > EXPOSURE_THRESHOLD
        else "generate_summary"
    )

    if decision["next_action"] != expected_action:
        return False

    # Every recorded approval must correspond to
    # exactly one proposed claim.
    proposed_ids = {
        p["claim_id"]
        for p in state["proposals"]
    }

    if not set(state["approvals"]).issubset(
        proposed_ids
    ):
        return False

    approved_ids = {
        claim_id
        for claim_id, approved
        in state["approvals"].items()
        if approved
    }

    # Only human-approved claims can be flagged.
    if set(state["review_flags"]) != approved_ids:
        return False

    if expected_action == "review_high_value":

        expected_proposals = review_high_value(
            claims
        )

        if state["proposals"] != expected_proposals:
            return False

        if set(state["approvals"]) != proposed_ids:
            return False

    else:
        if state["proposals"] or state["review_flags"]:
            return False

    return True


# --------------------------------------------------
# STEP 6: MAIN AGENT ORCHESTRATOR
# --------------------------------------------------

def run_agent() -> dict:

    state = new_state()

    try:

        # ------------------------------------------
        # PHASE 1: CREATE AND VALIDATE PLAN
        # ------------------------------------------

        print("\n[PHASE 1] LLM planning")

        plan = generate_plan(GOAL)

        validate_plan(plan)

        state["plan"] = [
            {
                **step.model_dump(),
                "status": "pending"
            }
            for step in plan.steps
        ]

        record_event(
            state,
            "plan_approved",
            "LLM-generated plan validated"
        )

        save_checkpoint(state)

        # ------------------------------------------
        # PHASE 2: EXECUTE DATA-RETRIEVAL TASK
        # ------------------------------------------

        print("\n[PHASE 2] Retrieve claims")

        retriever = ClaimsRetriever()

        claims = execute_with_recovery(
            state,
            "find_claims",
            retriever.execute
        )

        state["results"]["claims"] = claims
        state["plan"][0]["status"] = "completed"

        record_event(
            state,
            "claims_retrieved",
            f"{len(claims)} matching claims found"
        )

        save_checkpoint(state)

        # ------------------------------------------
        # PHASE 3: CALCULATE AND SUMMARIZE
        # ------------------------------------------

        print("\n[PHASE 3] Calculate exposure")

        total = execute_with_recovery(
            state,
            "calculate_total",
            lambda: calculate_total(claims)
        )

        state["results"]["total"] = total
        state["plan"][1]["status"] = "completed"

        summary = execute_with_recovery(
            state,
            "generate_summary",
            lambda: generate_summary(claims, total)
        )

        state["results"]["summary"] = summary
        state["summary"] = summary
        state["plan"][2]["status"] = "completed"

        print("TOTAL EXPOSURE:", total)

        record_event(
            state,
            "exposure_calculated",
            f"Total exposure: {total}"
        )

        save_checkpoint(state)

        # ------------------------------------------
        # PHASE 4: ADAPTIVE DECISION
        # ------------------------------------------

        print("\n[PHASE 4] Adaptive decision")

        decision = decide_next_action(total)

        validate_decision(decision, total)

        state["decision"] = decision.model_dump()

        print("ACTION:", decision.next_action)

        record_event(
            state,
            "decision_validated",
            decision.next_action
        )

        save_checkpoint(state)

        # ------------------------------------------
        # PHASE 5: HUMAN APPROVAL
        # ------------------------------------------

        if decision.next_action == "review_high_value":

            proposals = review_high_value(claims)

            state["proposals"] = proposals

            state["status"] = "awaiting_approval"

            save_checkpoint(state)

            for proposal in proposals:

                claim_id = proposal["claim_id"]

                approved = ask_for_approval(
                    proposal
                )

                state["approvals"][claim_id] = approved

                record_event(
                    state,
                    "human_decision",
                    f"{claim_id}: {approved}"
                )

                if approved:

                    # Simulation only. No database write.
                    state["review_flags"][claim_id] = {
                        "manual_review_required": True,
                        "reason": proposal["review_reason"]
                    }

                save_checkpoint(state)

        # ------------------------------------------
        # PHASE 6: VERIFY GOAL
        # ------------------------------------------

        print("\n[PHASE 6] Goal verification")

        if not verify_investigation(state):
            raise RuntimeError(
                "Investigation verification failed"
            )

        state["status"] = "completed"

        record_event(
            state,
            "goal_completed",
            "Investigation verified"
        )

    except (KeyboardInterrupt, EOFError):

        state["status"] = "cancelled"

        record_event(
            state,
            "interrupted",
            "Execution interrupted"
        )

    except Exception as exc:

        state["status"] = "failed"
        state["errors"].append(str(exc))

        record_event(
            state,
            "goal_failed",
            str(exc)
        )

    save_checkpoint(state)

    return state


# --------------------------------------------------
# STEP 7: RUN
# --------------------------------------------------

if __name__ == "__main__":

    final_state = run_agent()

    print("\n========== INVESTIGATION REPORT ==========")

    print(
        json.dumps(
            {
                "status": final_state["status"],
                "summary": final_state["summary"],
                "decision": final_state["decision"],
                "proposals": final_state["proposals"],
                "approvals": final_state["approvals"],
                "review_flags": final_state["review_flags"],
                "tool_attempts": final_state["tool_attempts"],
                "retry_count": final_state["retry_count"],
                "errors": final_state["errors"],
            },
            indent=2,
        )
    )
