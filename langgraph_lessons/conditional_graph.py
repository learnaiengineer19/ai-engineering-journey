
"""
LESSON 19 - PART 2: CONDITIONAL EDGES
====================================

Concepts:
- Conditional routing
- State-driven decisions
- Recovery loops
- Bounded retries
- Explicit success/failure outcomes

RUN:
python -m langgraph_lessons.conditional_graph

No LLM or external API required.
"""

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# STEP 1: SAMPLE INSURANCE DATA
# --------------------------------------------------

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM1002",
        "status": "Approved",
        "amount": 850.0,
    },
    {
        "claim_id": "CLM1004",
        "status": "Approved",
        "amount": 3200.0,
    },
    {
        "claim_id": "CLM1005",
        "status": "Approved",
        "amount": 1800.0,
    },
]


MAX_RECOVERY_ATTEMPTS = 2


# --------------------------------------------------
# STEP 2: DEFINE SHARED STATE
# --------------------------------------------------

class ClaimState(TypedDict):

    status: str
    min_amount: float

    claims: list[dict]
    total_amount: float

    verified: bool
    recovery_attempts: int

    # Used only to simulate calculation failures.
    failure_mode: Literal["none", "once", "always"]

    outcome: str


# --------------------------------------------------
# STEP 3: FIND CLAIMS
# --------------------------------------------------

def find_claims(state: ClaimState) -> dict:

    print("\n[NODE] find_claims")

    matching = [
        claim
        for claim in SAMPLE_CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    ]

    print("Claims found:", len(matching))

    return {
        "claims": matching
    }


# --------------------------------------------------
# STEP 4: CALCULATE TOTAL
# --------------------------------------------------

def calculate_total(state: ClaimState) -> dict:

    print("\n[NODE] calculate_total")

    total = sum(
        claim["amount"]
        for claim in state["claims"]
    )

    # Simulated calculation error for experiments.
    if state["failure_mode"] != "none":
        total += 500

    print("Calculated total:", total)

    return {
        "total_amount": round(total, 2)
    }


# --------------------------------------------------
# STEP 5: VERIFY TOTAL
# --------------------------------------------------

def verify_total(state: ClaimState) -> dict:

    print("\n[NODE] verify_total")

    # Independently recompute the expected amount
    # from the original sample records.
    expected = sum(
        claim["amount"]
        for claim in SAMPLE_CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    )

    is_verified = (
        abs(state["total_amount"] - expected) < 0.01
    )

    print("Expected:", expected)
    print("Actual:", state["total_amount"])
    print("Verified:", is_verified)

    return {
        "verified": is_verified
    }


# --------------------------------------------------
# STEP 6: RECOVER FROM VERIFICATION FAILURE
# --------------------------------------------------

def recover_total(state: ClaimState) -> dict:

    attempt = state["recovery_attempts"] + 1

    print(f"\n[NODE] recover_total - Attempt {attempt}")

    # Recompute from the selected claims.
    corrected = sum(
        claim["amount"]
        for claim in state["claims"]
    )

    # Simulate a persistent failure.
    if state["failure_mode"] == "always":
        corrected += 500

    print("Recovered total:", corrected)

    return {
        "total_amount": round(corrected, 2),
        "recovery_attempts": attempt
    }


# --------------------------------------------------
# STEP 7: SUCCESS AND FAILURE NODES
# --------------------------------------------------

def mark_success(state: ClaimState) -> dict:

    print("\n[NODE] mark_success")

    return {
        "outcome": "SUCCESS"
    }


def mark_failure(state: ClaimState) -> dict:

    print("\n[NODE] mark_failure")

    return {
        "outcome": "FAILED: recovery limit reached"
    }


# --------------------------------------------------
# STEP 8: ROUTING FUNCTION
# --------------------------------------------------

def route_after_verification(
    state: ClaimState
) -> Literal["success", "recover", "failed"]:

    print("\n[ROUTER] Checking graph state")

    if state["verified"]:

        print("Route: SUCCESS")
        return "success"

    if (
        state["recovery_attempts"]
        < MAX_RECOVERY_ATTEMPTS
    ):

        print("Route: RECOVER")
        return "recover"

    print("Route: FAILED")
    return "failed"


# --------------------------------------------------
# STEP 9: CREATE THE GRAPH
# --------------------------------------------------

builder = StateGraph(ClaimState)

builder.add_node("find_claims", find_claims)
builder.add_node("calculate_total", calculate_total)
builder.add_node("verify_total", verify_total)
builder.add_node("recover_total", recover_total)
builder.add_node("mark_success", mark_success)
builder.add_node("mark_failure", mark_failure)


# --------------------------------------------------
# STEP 10: FIXED EDGES
# --------------------------------------------------

builder.add_edge(START, "find_claims")

builder.add_edge(
    "find_claims",
    "calculate_total"
)

builder.add_edge(
    "calculate_total",
    "verify_total"
)


# --------------------------------------------------
# STEP 11: CONDITIONAL EDGES
# --------------------------------------------------

builder.add_conditional_edges(
    "verify_total",

    route_after_verification,

    {
        "success": "mark_success",
        "recover": "recover_total",
        "failed": "mark_failure",
    }
)


# --------------------------------------------------
# STEP 12: RECOVERY LOOP AND EXIT EDGES
# --------------------------------------------------

# After recovery, verify the new total again.
builder.add_edge(
    "recover_total",
    "verify_total"
)

builder.add_edge("mark_success", END)
builder.add_edge("mark_failure", END)


# --------------------------------------------------
# STEP 13: COMPILE THE GRAPH
# --------------------------------------------------

graph = builder.compile()


# --------------------------------------------------
# STEP 14: EXECUTE TEST SCENARIOS
# --------------------------------------------------

def run_scenario(failure_mode: str):

    print("\n" + "=" * 55)
    print("SCENARIO:", failure_mode.upper())
    print("=" * 55)

    initial_state: ClaimState = {
        "status": "Approved",
        "min_amount": 1000.0,
        "claims": [],
        "total_amount": 0.0,
        "verified": False,
        "recovery_attempts": 0,
        "failure_mode": failure_mode,
        "outcome": "PENDING",
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 20}
    )

    print("\n========== FINAL STATE ==========")
    print("Total:", result["total_amount"])
    print("Verified:", result["verified"])
    print(
        "Recovery attempts:",
        result["recovery_attempts"]
    )
    print("Outcome:", result["outcome"])

    return result


if __name__ == "__main__":

    run_scenario("none")
    run_scenario("once")
    run_scenario("always")
