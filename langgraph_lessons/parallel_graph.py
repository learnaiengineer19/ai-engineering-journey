
"""
LESSON 19 - PART 3: PARALLEL EXECUTION
=====================================

CONCEPTS:
- Fan-out and fan-in
- Concurrent graph nodes
- State reducers
- Annotated[list, operator.add]
- Asynchronous graph execution

RUN:
python -m langgraph_lessons.parallel_graph
"""

import asyncio
import operator
import time

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# STEP 1: SAMPLE INSURANCE DATA
# --------------------------------------------------

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM1001",
        "status": "Under Review",
        "amount": 4500.0,
    },
    {
        "claim_id": "CLM1002",
        "status": "Approved",
        "amount": 850.0,
    },
    {
        "claim_id": "CLM1003",
        "status": "Rejected",
        "amount": 2200.0,
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


# --------------------------------------------------
# STEP 2: DEFINE STATE WITH A REDUCER
# --------------------------------------------------

class AnalysisState(TypedDict):

    min_amount: float

    # Each parallel node contributes a list.
    # The reducer combines those lists.
    findings: Annotated[list[dict], operator.add]

    report: dict


# --------------------------------------------------
# STEP 3: INITIAL PREPARATION NODE
# --------------------------------------------------

def prepare_analysis(state: AnalysisState) -> dict:

    print("\n[NODE] prepare_analysis")
    print("Minimum claim amount:", state["min_amount"])

    # We do not need to change any state fields.
    return {}


# --------------------------------------------------
# STEP 4: HELPER FUNCTION FOR CLAIM ANALYSIS
# --------------------------------------------------

def analyse_status(
    status: str,
    min_amount: float,
) -> dict:

    matching = [
        claim
        for claim in SAMPLE_CLAIMS
        if claim["status"] == status
        and claim["amount"] >= min_amount
    ]

    return {
        "status": status,
        "claim_ids": [
            claim["claim_id"]
            for claim in matching
        ],
        "claim_count": len(matching),
        "total_amount": round(
            sum(
                claim["amount"]
                for claim in matching
            ),
            2
        ),
    }


# --------------------------------------------------
# STEP 5: PARALLEL NODE A
# --------------------------------------------------

async def analyse_approved(
    state: AnalysisState
) -> dict:

    print("\n[NODE] analyse_approved START")

    # Simulate an asynchronous service request.
    await asyncio.sleep(2)

    result = analyse_status(
        "Approved",
        state["min_amount"]
    )

    print("[NODE] analyse_approved FINISHED")

    return {
        "findings": [result]
    }


# --------------------------------------------------
# STEP 6: PARALLEL NODE B
# --------------------------------------------------

async def analyse_under_review(
    state: AnalysisState
) -> dict:

    print("\n[NODE] analyse_under_review START")

    await asyncio.sleep(3)

    result = analyse_status(
        "Under Review",
        state["min_amount"]
    )

    print("[NODE] analyse_under_review FINISHED")

    return {
        "findings": [result]
    }


# --------------------------------------------------
# STEP 7: FAN-IN NODE
# --------------------------------------------------

def combine_results(
    state: AnalysisState
) -> dict:

    print("\n[NODE] combine_results")

    # Both branches have finished before this
    # node executes.
    findings = state["findings"]

    # Sort for deterministic report presentation.
    findings = sorted(
        findings,
        key=lambda item: item["status"]
    )

    combined_total = round(
        sum(
            item["total_amount"]
            for item in findings
        ),
        2
    )

    return {
        "report": {
            "findings": findings,
            "combined_total": combined_total,
        }
    }


# --------------------------------------------------
# STEP 8: BUILD THE GRAPH
# --------------------------------------------------

builder = StateGraph(AnalysisState)

builder.add_node(
    "prepare_analysis",
    prepare_analysis
)

builder.add_node(
    "analyse_approved",
    analyse_approved
)

builder.add_node(
    "analyse_under_review",
    analyse_under_review
)

builder.add_node(
    "combine_results",
    combine_results
)


# --------------------------------------------------
# STEP 9: DEFINE FAN-OUT
# --------------------------------------------------

builder.add_edge(
    START,
    "prepare_analysis"
)

# Both branches start after preparation.
builder.add_edge(
    "prepare_analysis",
    "analyse_approved"
)

builder.add_edge(
    "prepare_analysis",
    "analyse_under_review"
)


# --------------------------------------------------
# STEP 10: DEFINE FAN-IN
# --------------------------------------------------

# Wait for BOTH analysis nodes to complete.
builder.add_edge(
    [
        "analyse_approved",
        "analyse_under_review",
    ],
    "combine_results"
)

builder.add_edge(
    "combine_results",
    END
)


# --------------------------------------------------
# STEP 11: COMPILE
# --------------------------------------------------

graph = builder.compile()


# --------------------------------------------------
# STEP 12: EXECUTE ASYNCHRONOUSLY
# --------------------------------------------------

async def main():

    initial_state: AnalysisState = {
        "min_amount": 1000.0,
        "findings": [],
        "report": {},
    }

    print("\n========== PARALLEL GRAPH ==========")

    start = time.perf_counter()

    # Nodes are async, so use ainvoke().
    result = await graph.ainvoke(initial_state)

    elapsed = time.perf_counter() - start

    print("\n========== FINAL REPORT ==========")

    for finding in result["report"]["findings"]:
        print(
            f"{finding['status']}: "
            f"{finding['claim_count']} claims, "
            f"${finding['total_amount']:.2f}"
        )

    print(
        "Combined total:",
        result["report"]["combined_total"]
    )

    print("\n========== STATE ==========")
    print("Findings:", result["findings"])
    print("Report:", result["report"])

    print(f"\nTotal execution time: {elapsed:.2f}s")

    print("\n========== GRAPH STRUCTURE ==========")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
