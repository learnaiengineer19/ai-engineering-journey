
"""
LESSON 19 - PART 5: PERSISTENT LANGGRAPH
=======================================

GOAL:
1. Run find_claims.
2. Save the checkpoint to SQLite.
3. Pause before calculate_total.
4. Exit Python completely.
5. Restart and resume the same workflow.

USAGE:
python -m langgraph_lessons.persistent_graph start
python -m langgraph_lessons.persistent_graph status
python -m langgraph_lessons.persistent_graph resume
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

DB_PATH = Path(
    "langgraph_lessons/checkpoints.sqlite"
)

THREAD_ID = "insurance-job-001"

CONFIG = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


# --------------------------------------------------
# STEP 2: SAMPLE INSURANCE DATA
# --------------------------------------------------

CLAIMS = [
    {
        "claim_id": "CLM1002",
        "status": "Approved",
        "amount": 850.0
    },
    {
        "claim_id": "CLM1004",
        "status": "Approved",
        "amount": 3200.0
    },
    {
        "claim_id": "CLM1005",
        "status": "Approved",
        "amount": 1800.0
    }
]


# --------------------------------------------------
# STEP 3: DEFINE SHARED STATE
# --------------------------------------------------

class ClaimState(TypedDict):

    status: str
    min_amount: float

    claims: list[dict]
    total_amount: float
    verified: bool


# --------------------------------------------------
# STEP 4: NODE 1 - FIND CLAIMS
# --------------------------------------------------

def find_claims(state: ClaimState) -> dict:

    print("\n[NODE EXECUTED] find_claims")

    matching = [
        claim
        for claim in CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    ]

    return {
        "claims": matching
    }


# --------------------------------------------------
# STEP 5: NODE 2 - CALCULATE TOTAL
# --------------------------------------------------

def calculate_total(state: ClaimState) -> dict:

    print("\n[NODE EXECUTED] calculate_total")

    total = sum(
        claim["amount"]
        for claim in state["claims"]
    )

    return {
        "total_amount": round(total, 2)
    }


# --------------------------------------------------
# STEP 6: NODE 3 - VERIFY
# --------------------------------------------------

def verify_total(state: ClaimState) -> dict:

    print("\n[NODE EXECUTED] verify_total")

    expected = sum(
        claim["amount"]
        for claim in CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    )

    verified = (
        abs(state["total_amount"] - expected) < 0.01
    )

    return {
        "verified": verified
    }


# --------------------------------------------------
# STEP 7: DEFINE THE GRAPH
# --------------------------------------------------

builder = StateGraph(ClaimState)

builder.add_node("find_claims", find_claims)
builder.add_node("calculate_total", calculate_total)
builder.add_node("verify_total", verify_total)

builder.add_edge(START, "find_claims")
builder.add_edge("find_claims", "calculate_total")
builder.add_edge("calculate_total", "verify_total")
builder.add_edge("verify_total", END)


# --------------------------------------------------
# STEP 8: INSPECT SAVED STATE
# --------------------------------------------------

def show_state(graph):

    snapshot = graph.get_state(CONFIG)

    print("\n========== SAVED STATE ==========")
    print("Thread ID:", THREAD_ID)
    print("State:", snapshot.values)
    print("Next node(s):", snapshot.next)


# --------------------------------------------------
# STEP 9: START THE WORKFLOW
# --------------------------------------------------

def start_workflow(graph):

    print("\n========== STARTING WORKFLOW ==========")

    initial_state: ClaimState = {
        "status": "Approved",
        "min_amount": 1000.0,
        "claims": [],
        "total_amount": 0.0,
        "verified": False
    }

    result = graph.invoke(
        initial_state,
        CONFIG,
        durability="sync"
    )

    print("\nExecution paused or completed.")
    print("Returned state:", result)

    show_state(graph)


# --------------------------------------------------
# STEP 10: RESUME THE WORKFLOW
# --------------------------------------------------

def resume_workflow(graph):

    print("\n========== RESUMING WORKFLOW ==========")

    snapshot = graph.get_state(CONFIG)

    if not snapshot.values:
        raise RuntimeError(
            "No saved workflow state found. "
            "Run 'start' first."
        )

    if not snapshot.next:
        print("Workflow is already complete.")
        show_state(graph)
        return

    print("Resuming from:", snapshot.next)

    # None means continue the saved execution.
    result = graph.invoke(
        None,
        CONFIG,
        durability="sync"
    )

    print("\n========== FINAL RESULT ==========")
    print(result)

    show_state(graph)


# --------------------------------------------------
# STEP 11: APPLICATION ENTRY POINT
# --------------------------------------------------

def main():

    if len(sys.argv) != 2 or sys.argv[1] not in {
        "start",
        "status",
        "resume"
    }:
        raise SystemExit(
            "Usage: python -m "
            "langgraph_lessons.persistent_graph "
            "[start|status|resume]"
        )

    action = sys.argv[1]

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # SQLite saver remains open while we use graph.
    with SqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as checkpointer:

        # This demo always pauses before this node.
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["calculate_total"]
        )

        if action == "start":
            start_workflow(graph)

        elif action == "status":
            show_state(graph)

        elif action == "resume":
            resume_workflow(graph)


if __name__ == "__main__":
    main()
