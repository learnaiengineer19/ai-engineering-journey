
"""
LESSON 19 - PART 6: HUMAN-IN-THE-LOOP
====================================

Concepts:
- interrupt()
- Command(resume=...)
- SQLite checkpoints
- Approval and rejection routing
- Resuming across Python processes

This lesson SIMULATES claim changes.
It does not update any real database.

Usage:
python -m langgraph_lessons.human_approval start
python -m langgraph_lessons.human_approval status
python -m langgraph_lessons.human_approval approve
python -m langgraph_lessons.human_approval reject

Use a fresh THREAD_ID for each new experiment.
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

CHECKPOINT_DB = Path(
    "langgraph_lessons/approval_checkpoints.sqlite"
)

THREAD_ID = "approval-experiment-002"

CONFIG = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


# --------------------------------------------------
# STEP 2: DEFINE SHARED STATE
# --------------------------------------------------

class ApprovalState(TypedDict):

    claim_id: str
    current_status: str
    requested_status: str

    proposal: dict
    approved: bool | None
    outcome: str


# --------------------------------------------------
# STEP 3: PREPARE THE ACTION
# --------------------------------------------------

def prepare_proposal(state: ApprovalState) -> dict:

    print("\n[NODE] prepare_proposal")

    proposal = {
        "claim_id": state["claim_id"],
        "action": "change_claim_status",
        "from_status": state["current_status"],
        "to_status": state["requested_status"],
    }

    print("Proposal:", proposal)

    return {
        "proposal": proposal,
        "outcome": "AWAITING_APPROVAL"
    }


# --------------------------------------------------
# STEP 4: HUMAN APPROVAL NODE
# --------------------------------------------------

def request_approval(state: ApprovalState) -> dict:

    print("\n[NODE] request_approval")

    # This information is shown to the caller.
    # interrupt() pauses the graph here.
    decision = interrupt({
        "question": "Approve the proposed claim update?",
        "proposal": state["proposal"],
    })

    # This section runs only after resumption.
    # Require an actual boolean, not a string.
    if type(decision) is not bool:
        raise ValueError(
            "Approval decision must be True or False"
        )

    print("Human decision received:", decision)

    return {
        "approved": decision
    }


# --------------------------------------------------
# STEP 5: ROUTE USING THE HUMAN DECISION
# --------------------------------------------------

def route_decision(state: ApprovalState) -> str:

    if state["approved"] is True:
        return "apply_change"

    return "cancel_change"


# --------------------------------------------------
# STEP 6: EXECUTE APPROVED ACTION
# --------------------------------------------------

def apply_change(state: ApprovalState) -> dict:

    print("\n[NODE] apply_change")

    print("SIMULATED approved action:")
    print(state["proposal"])

    # No real database mutation occurs.
    return {
        "outcome": "APPROVED_AND_SIMULATED"
    }


# --------------------------------------------------
# STEP 7: CANCEL REJECTED ACTION
# --------------------------------------------------

def cancel_change(state: ApprovalState) -> dict:

    print("\n[NODE] cancel_change")
    print("Proposed action rejected.")

    return {
        "outcome": "REJECTED_NO_ACTION"
    }


# --------------------------------------------------
# STEP 8: BUILD GRAPH
# --------------------------------------------------

builder = StateGraph(ApprovalState)

builder.add_node(
    "prepare_proposal",
    prepare_proposal
)

builder.add_node(
    "request_approval",
    request_approval
)

builder.add_node(
    "apply_change",
    apply_change
)

builder.add_node(
    "cancel_change",
    cancel_change
)

builder.add_edge(
    START,
    "prepare_proposal"
)

builder.add_edge(
    "prepare_proposal",
    "request_approval"
)

builder.add_conditional_edges(
    "request_approval",
    route_decision,
    {
        "apply_change": "apply_change",
        "cancel_change": "cancel_change",
    }
)

builder.add_edge("apply_change", END)
builder.add_edge("cancel_change", END)


# --------------------------------------------------
# STEP 9: START AND PAUSE
# --------------------------------------------------

def start_workflow(graph):

    existing = graph.get_state(CONFIG)

    if existing.values:
        raise RuntimeError(
            "This thread already has saved state. "
            "Use a new THREAD_ID for another experiment."
        )

    initial_state: ApprovalState = {
        "claim_id": "CLM1004",
        "current_status": "Approved",
        "requested_status": "Under Review",
        "proposal": {},
        "approved": None,
        "outcome": "PENDING",
    }

    print("\n========== START ==========")

    result = graph.invoke(
        initial_state,
        config=CONFIG,
        durability="sync"
    )

    print("\n========== RETURNED INTERRUPT ==========")

    for pending in result.get("__interrupt__", ()):
        print(pending.value)

    show_state(graph)


# --------------------------------------------------
# STEP 10: INSPECT CHECKPOINT
# --------------------------------------------------

def show_state(graph):

    snapshot = graph.get_state(CONFIG)

    print("\n========== CHECKPOINT ==========")
    print("Thread ID:", THREAD_ID)
    print("State:", snapshot.values)
    print("Next nodes:", snapshot.next)

    for task in snapshot.tasks:
        for pending in task.interrupts:
            print("Pending approval:", pending.value)


# --------------------------------------------------
# STEP 11: RESUME WITH HUMAN DECISION
# --------------------------------------------------

def resume_workflow(graph, approved: bool):

    snapshot = graph.get_state(CONFIG)

    if not snapshot.values:
        raise RuntimeError(
            "No checkpoint found. Run start first."
        )

    # A pending interrupt must exist before
    # accepting a resume decision.
    pending_interrupts = [
        item
        for task in snapshot.tasks
        for item in task.interrupts
    ]

    if not pending_interrupts:
        raise RuntimeError(
            "No pending human approval exists "
            "for this thread."
        )

    print("\n========== RESUME ==========")
    print("Human approved:", approved)

    result = graph.invoke(
        Command(resume=approved),
        config=CONFIG,
        durability="sync"
    )

    print("\n========== FINAL STATE ==========")
    print(result)

    show_state(graph)


# --------------------------------------------------
# STEP 12: COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def main():

    valid_commands = {
        "start",
        "status",
        "approve",
        "reject"
    }

    if (
        len(sys.argv) != 2
        or sys.argv[1] not in valid_commands
    ):
        raise SystemExit(
            "Usage: python -m "
            "langgraph_lessons.human_approval "
            "[start|status|approve|reject]"
        )

    action = sys.argv[1]

    CHECKPOINT_DB.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:

        graph = builder.compile(
            checkpointer=checkpointer
        )

        if action == "start":
            start_workflow(graph)

        elif action == "status":
            show_state(graph)

        elif action == "approve":
            resume_workflow(graph, approved=True)

        elif action == "reject":
            resume_workflow(graph, approved=False)


if __name__ == "__main__":
    main()
