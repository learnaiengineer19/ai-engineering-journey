
"""
LESSON 17 - PART 7: HUMAN-IN-THE-LOOP
=====================================

GOAL:
Investigate high-value insurance claims and request
human approval before marking them for manual review.

CONCEPTS:
1. Human-in-the-loop approval
2. Separating proposals from execution
3. Approval gates
4. Authorized tool execution
5. Audit history
6. Safe rejection and cancellation
7. Completion verification

IMPORTANT:
This exercise modifies only an in-memory dictionary.
The original SQLite database remains unchanged.

RUN:
python -m agents.human_approval_agent
"""

import json
from datetime import datetime, timezone

from agents.llm_planning_agent import find_claims


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

HIGH_VALUE_THRESHOLD = 3000


# --------------------------------------------------
# STEP 2: INITIALISE AGENT STATE
# --------------------------------------------------

def create_state() -> dict:

    return {
        "status": "in_progress",
        "claims": [],
        "proposed_actions": [],
        "approved_actions": [],
        "rejected_actions": [],
        "review_flags": {},
        "approval_status": "not_requested",
        "history": [],
        "errors": []
    }


# --------------------------------------------------
# STEP 3: RECORD AUDIT EVENTS
# --------------------------------------------------

def record_event(
    state: dict,
    event: str,
    details: str
) -> None:

    state["history"].append({
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event": event,
        "details": details
    })


# --------------------------------------------------
# STEP 4: IDENTIFY HIGH-VALUE CLAIMS
# --------------------------------------------------

def identify_high_value_claims(
    claims: list[dict]
) -> list[dict]:

    return [
        claim
        for claim in claims
        if claim["amount"] >= HIGH_VALUE_THRESHOLD
    ]


# --------------------------------------------------
# STEP 5: PROPOSE ACTIONS
# --------------------------------------------------

def propose_review_actions(
    state: dict,
    claims: list[dict]
) -> None:

    high_value_claims = identify_high_value_claims(
        claims
    )

    for claim in high_value_claims:

        proposal = {
            "claim_id": claim["claim_id"],
            "action": "flag_for_manual_review",
            "reason": (
                "Claim amount meets the configured "
                "high-value threshold."
            ),
            "amount": claim["amount"]
        }

        state["proposed_actions"].append(proposal)

        record_event(
            state,
            "action_proposed",
            f"Review flag proposed for {claim['claim_id']}"
        )


# --------------------------------------------------
# STEP 6: HUMAN APPROVAL GATE
# --------------------------------------------------

def request_approval(
    state: dict,
    proposal: dict
) -> bool:
    """
    Only an explicit 'yes' grants approval.

    Empty input, 'no' and all other values
    are treated as rejection.

    This is a local CLI demonstration, not
    production-grade user authentication.
    """

    print("\n========== APPROVAL REQUEST ==========")

    print("Claim:", proposal["claim_id"])
    print("Amount:", proposal["amount"])
    print("Action:", proposal["action"])
    print("Reason:", proposal["reason"])

    answer = input(
        "\nApprove this action? (yes/no): "
    ).strip().lower()

    approved = answer == "yes"

    record_event(
        state,
        "human_decision",
        (
            f"{proposal['claim_id']}: "
            f"{'approved' if approved else 'rejected'}"
        )
    )

    return approved


# --------------------------------------------------
# STEP 7: EXECUTE AN APPROVED ACTION
# --------------------------------------------------

def execute_approved_action(
    state: dict,
    proposal: dict
) -> None:
    """
    Apply only explicitly approved, known actions.

    The update is stored in memory, not SQLite.
    """

    if proposal not in state["approved_actions"]:
        raise PermissionError(
            "Action has not received approval"
        )

    if proposal["action"] != "flag_for_manual_review":
        raise PermissionError(
            "Unauthorized action type"
        )

    claim_id = proposal["claim_id"]

    # Prevent duplicate updates.
    if claim_id in state["review_flags"]:
        raise ValueError(
            "Claim has already been updated"
        )

    # Simulated update; no database write.
    state["review_flags"][claim_id] = {
        "manual_review_required": True,
        "reason": proposal["reason"]
    }

    record_event(
        state,
        "action_executed",
        f"Manual review flag added for {claim_id}"
    )


# --------------------------------------------------
# STEP 8: RUN AGENT
# --------------------------------------------------

def run_agent() -> dict:

    state = create_state()

    try:

        # ------------------------------------------
        # 8.1: RETRIEVE CLAIMS
        # ------------------------------------------

        print("\n[AGENT] Retrieving insurance claims")

        claims = find_claims()

        state["claims"] = claims

        record_event(
            state,
            "claims_retrieved",
            f"Retrieved {len(claims)} claims"
        )

        # ------------------------------------------
        # 8.2: GENERATE ACTION PROPOSALS
        # ------------------------------------------

        propose_review_actions(state, claims)

        if not state["proposed_actions"]:

            state["status"] = "completed"

            record_event(
                state,
                "goal_completed",
                "No high-value claims require review"
            )

            return state

        print(
            "\n[AGENT] Found",
            len(state["proposed_actions"]),
            "actions requiring approval"
        )

        # ------------------------------------------
        # 8.3: WAIT FOR APPROVAL
        # ------------------------------------------

        state["status"] = "awaiting_approval"
        state["approval_status"] = "pending"

        for proposal in state["proposed_actions"]:

            approved = request_approval(
                state,
                proposal
            )

            if approved:

                state["approved_actions"].append(
                    proposal
                )

                # ----------------------------------
                # 8.4: EXECUTE APPROVED ACTION
                # ----------------------------------

                execute_approved_action(
                    state,
                    proposal
                )

            else:

                print(
                    "[AGENT] Action rejected:",
                    proposal["claim_id"]
                )

                state["rejected_actions"].append(
                    proposal
                )

        # ------------------------------------------
        # 8.5: VERIFY EXECUTED ACTIONS
        # ------------------------------------------

        approved_ids = {
            action["claim_id"]
            for action in state["approved_actions"]
        }

        executed_ids = set(
            state["review_flags"].keys()
        )

        rejected_ids = {
            action["claim_id"]
            for action in state["rejected_actions"]
        }

        if (
            approved_ids != executed_ids
            or executed_ids & rejected_ids
        ):
            raise RuntimeError(
                "Approval verification failed"
            )

        state["approval_status"] = "resolved"

        state["status"] = "completed"

        record_event(
            state,
            "goal_completed",
            "All human decisions were processed"
        )

    except (KeyboardInterrupt, EOFError):

        state["status"] = "cancelled"
        state["approval_status"] = "interrupted"

        record_event(
            state,
            "approval_interrupted",
            "No further actions were executed"
        )

    except Exception as exc:

        state["status"] = "failed"

        state["errors"].append(str(exc))

    return state



def test_unauthorized_action():

    state = create_state()

    proposal = {
        "claim_id": "CLM1004",
        "action": "flag_for_manual_review",
        "reason": "High-value claim",
        "amount": 3200
    }

    try:
        execute_approved_action(
            state,
            proposal
        )

    except PermissionError as exc:
        print(
            "\n[SECURITY TEST] Blocked:",
            str(exc)
        )
# --------------------------------------------------
# STEP 9: RUN AND DISPLAY FINAL STATE
# --------------------------------------------------

if __name__ == "__main__":
    test_unauthorized_action()

    final_state = run_agent()

    print("\n========== FINAL AGENT STATE ==========")

    print(json.dumps(final_state, indent=2))

    print("\nGOAL STATUS:", final_state["status"])

    print(
        "FLAGS APPLIED:",
        list(final_state["review_flags"].keys())
    )
