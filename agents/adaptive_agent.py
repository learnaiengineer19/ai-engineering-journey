
"""
LESSON 17 - PART 5: ADAPTIVE DECISION-MAKING
============================================

OBJECTIVE:
An agent observes real data and chooses its
next action based on that observation.

WORKFLOW:
1. Retrieve approved claims.
2. Calculate the total exposure.
3. Ask Qwen2 to select the next action.
4. Validate the proposed action.
5. Execute the approved branch.
6. Verify completion.

RUN:
python -m agents.adaptive_agent
"""

import json
from typing import Literal

from ollama import chat
from pydantic import BaseModel, ConfigDict, Field

from agents.llm_planning_agent import (
    MODEL,
    find_claims,
    calculate_total,
    generate_summary,
)


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

EXPOSURE_THRESHOLD = 6000
HIGH_VALUE_THRESHOLD = 3000


# --------------------------------------------------
# STEP 2: STRUCTURE FOR AN AGENT DECISION
# --------------------------------------------------

class AgentDecision(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    next_action: Literal[
        "review_high_value",
        "generate_summary"
    ]

    reason: str = Field(min_length=1)


# --------------------------------------------------
# STEP 3: ASK QWEN2 TO DECIDE
# --------------------------------------------------

def decide_next_action(total: float) -> AgentDecision:
    """
    Ask Qwen2 to propose a next action based
    on the observed total claim exposure.
    """

    observation = {
        "total_exposure": total,
        "exposure_threshold": EXPOSURE_THRESHOLD,
        "rule": (
            "If total_exposure exceeds exposure_threshold, "
            "choose review_high_value. Otherwise choose "
            "generate_summary."
        )
    }

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an insurance investigation agent. "
                    "Select exactly one next action based "
                    "on the observation and business rule. "
                    "Return a structured decision. "
                    "Explain your reason briefly."
                )
            },
            {
                "role": "user",
                "content": json.dumps(observation)
            }
        ],
        format=AgentDecision.model_json_schema(),
        options={"temperature": 0}
    )

    return AgentDecision.model_validate_json(
        response.message.content
    )


# --------------------------------------------------
# STEP 4: REVIEW HIGH-VALUE CLAIMS
# --------------------------------------------------

def review_high_value(claims: list[dict]) -> list[dict]:
    """
    Flag individual claims above the configured
    high-value threshold.

    This does not make a fraud determination.
    """

    flagged = []

    for claim in claims:
        if claim["amount"] >= HIGH_VALUE_THRESHOLD:
            flagged.append({
                **claim,
                "review_reason": (
                    "Amount meets high-value threshold"
                )
            })

    return flagged


# --------------------------------------------------
# STEP 5: VALIDATE THE LLM DECISION
# --------------------------------------------------

def validate_decision(
    decision: AgentDecision,
    total: float
) -> None:
    """
    Python independently enforces the business rule.

    The model can propose an action but cannot
    override this rule.
    """

    if total > EXPOSURE_THRESHOLD:
        expected = "review_high_value"
    else:
        expected = "generate_summary"

    if decision.next_action != expected:
        raise ValueError(
            f"Invalid decision: {decision.next_action}. "
            f"Expected: {expected}"
        )


# --------------------------------------------------
# STEP 6: EXECUTE THE ADAPTIVE WORKFLOW
# --------------------------------------------------

def run_agent() -> dict:

    state = {
        "status": "in_progress",
        "claims": [],
        "total": None,
        "decision": None,
        "flagged_claims": [],
        "summary": None,
        "completed_actions": [],
        "errors": []
    }

    try:
        # Retrieve actual data.
        claims = find_claims()

        state["claims"] = claims
        state["completed_actions"].append("find_claims")

        # Calculate exposure.
        total = calculate_total(claims)

        state["total"] = total
        state["completed_actions"].append("calculate_total")

        print("\n[OBSERVATION] Total exposure:", total)

        # Ask the LLM to select the next action.
        decision = decide_next_action(total)

        state["decision"] = decision.model_dump()

        print("\n[AGENT DECISION]")
        print(json.dumps(state["decision"], indent=2))

        # Enforce the actual business rule.
        validate_decision(decision, total)

        print("\n[VALIDATOR] Decision approved")

        # Execute only the selected branch.
        if decision.next_action == "review_high_value":

            print("\n[ACTION] Reviewing high-value claims")

            state["flagged_claims"] = review_high_value(
                claims
            )

        elif decision.next_action == "generate_summary":

            print("\n[ACTION] Generating normal summary")

        # Every successful branch receives a summary.
        state["summary"] = generate_summary(claims, total)

        state["completed_actions"].append(
            decision.next_action
        )

        # Verify the data and the branch outcome.
        valid_claims = all(
            claim["status"] == "Approved"
            and claim["amount"] >= 1000
            for claim in claims
        )

        expected_flags = review_high_value(claims)

        branch_valid = (
            state["flagged_claims"] == expected_flags
            if total > EXPOSURE_THRESHOLD
            else state["flagged_claims"] == []
        )

        results_valid = (
            valid_claims
            and total == calculate_total(claims)
            and branch_valid
            and state["summary"]
                == generate_summary(claims, total)
        )

        if not results_valid:
            raise ValueError("Result verification failed")

        state["status"] = "completed"

    except Exception as exc:
        state["status"] = "failed"
        state["errors"].append(str(exc))

    return state


# --------------------------------------------------
# STEP 7: RUN AND DISPLAY
# --------------------------------------------------

if __name__ == "__main__":

    result = run_agent()

    print("\n========== FINAL AGENT STATE ==========")
    print(json.dumps(result, indent=2))

    print("\nGOAL STATUS:", result["status"])
